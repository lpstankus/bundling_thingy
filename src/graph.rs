use color_eyre::eyre::{OptionExt, Result, WrapErr, bail, ensure};
use std::{fs, path::Path};

pub use glam::Vec2 as Point;
#[derive(Debug)]
pub struct Graph {
    pub points: Vec<Point>,
    pub starts: Vec<usize>,

    min: Point,
    max: Point,
}

impl Graph {
    pub fn load(path: &Path) -> Result<Self> {
        let graph = Self::parse(
            &fs::read_to_string(path).wrap_err_with(|| format!("read {}", path.display()))?,
        )?;
        println!(
            "Read {}: {} edges, {} control points",
            path.display(),
            graph.starts.len().saturating_sub(1),
            graph.points.len()
        );
        Ok(graph)
    }

    pub fn parse(text: &str) -> Result<Self> {
        let mut g = Self {
            points: vec![],
            starts: vec![],
            min: Point::MAX,
            max: Point::MIN,
        };

        for (line_idx, line) in text.lines().enumerate() {
            // Each line is "edge-index x y [x y ...]", separated by spaces.
            // Tokens beginning with '(' or '<' are metadata.
            let mut tokens = line.split(' ');
            tokens.next();

            let start = g.points.len();
            while let Some(token) = tokens.next() {
                if token.is_empty() || token.starts_with(['(', '<']) {
                    continue;
                }

                let p = Point::new(
                    coordinate(Some(token), line_idx, 'x')?,
                    coordinate(tokens.next(), line_idx, 'y')?,
                );

                ensure!(p.is_finite(), "non-finite TRL coordinate");

                g.min = Point::min(g.min, p);
                g.max = Point::max(g.max, p);

                g.points.push(p);
            }

            if start != g.points.len() {
                g.starts.push(start);
            }
        }

        g.starts.push(g.points.len());
        if g.points.is_empty() {
            bail!("empty graph");
        }
        Ok(g)
    }

    pub fn normalize(&mut self) -> Result<()> {
        let range = self.max - self.min;
        let span = f32::max(range.x, range.y);
        ensure!(
            span > 0.0 && span.is_finite(),
            "graph has zero or invalid extent"
        );

        let scale = (1.0_f32 - 0.05) / span;
        let translation = (Point::ONE - scale * range) / 2.0;

        for point in &mut self.points {
            *point = ((*point - self.min) * scale + translation - Point::splat(0.5)) * 2.0;
        }

        self.min = Point::NEG_ONE;
        self.max = Point::ONE;
        Ok(())
    }

    pub fn resample(&mut self, spacing: f32) -> Result<()> {
        ensure!(
            spacing > 0.0 && spacing.is_finite(),
            "invalid resample spacing"
        );

        let mut points = Vec::new();
        let mut starts = Vec::new();

        for &[start, end] in self.starts.array_windows::<2>() {
            let edge = self
                .points
                .get(start..end)
                .ok_or_eyre("invalid edge bounds")?;

            starts.push(points.len());

            let mut remaining = spacing;
            let mut previous = *edge.first().ok_or_eyre("empty edge")?;
            points.push(previous);

            let mut pending = edge.iter().skip(1).peekable();
            while let Some(&&current) = pending.peek() {
                // Keep the squared-component sum; hypot changes f32 rounding.
                let distance = (previous - current).length();
                if distance < remaining {
                    remaining -= distance;
                    previous = current;
                    pending.next();
                    continue;
                }

                let t = remaining / distance;
                let next = previous * (1.0 - t) + current * t;
                // A tiny remainder may round to the previous point. Append it,
                // reset the spacing, then advance on the next iteration.
                ensure!(
                    next != previous || remaining < spacing,
                    "resampling made no progress"
                );

                previous = next;
                remaining = spacing;
                points.push(previous);
            }

            if remaining < spacing {
                points.push(*edge.last().ok_or_eyre("empty edge")?);
            }
        }

        starts.push(points.len());

        println!(
            "Resampled graph: {} -> {} edges, {} -> {} control points",
            self.starts.len().saturating_sub(1),
            starts.len().saturating_sub(1),
            self.points.len(),
            points.len()
        );

        self.points = points;
        self.starts = starts;
        Ok(())
    }

    pub fn mesh(&self, width: f32) -> Result<Vec<Point>> {
        let mut mesh = Vec::with_capacity(self.points.len().saturating_sub(1).saturating_mul(2));
        let half = width / 2.0;

        for &[start, end] in self.starts.array_windows::<2>() {
            for &[first, second] in self
                .points
                .get(start..end)
                .ok_or_eyre("invalid edge bounds")?
                .array_windows::<2>()
            {
                let dir = second - first;
                let len = dir.length();
                if len == 0.0 {
                    continue;
                }

                let dir = dir / len;
                let normal = Point::new(-dir.y, dir.x) * half;
                let v0 = first + normal;
                let v1 = first - normal;
                let v2 = second + normal;
                let v3 = second - normal;

                mesh.extend([v0, v1, v2, v3, v2, v1]);
            }
        }
        Ok(mesh)
    }
}

fn coordinate(token: Option<&str>, line: usize, axis: char) -> Result<f32> {
    token
        .ok_or_eyre("unexpected end of TRL line")
        .and_then(|token| token.parse::<f32>().wrap_err("invalid coordinate"))
        .wrap_err_with(|| format!("line {} {axis}", line.saturating_add(1)))
}
