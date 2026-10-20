use color_eyre::eyre::{OptionExt, Result, WrapErr, bail, ensure};
use std::{fs, path::Path};

pub type Point = [f32; 2];
#[derive(Debug)]
pub struct Graph {
    pub points: Vec<Point>,
    pub starts: Vec<usize>,

    min: Point,
    max: Point,
}

impl Graph {
    pub fn load(path: &Path) -> Result<Self> {
        Self::parse(&fs::read_to_string(path).wrap_err_with(|| format!("read {}", path.display()))?)
    }

    pub fn parse(text: &str) -> Result<Self> {
        let mut g = Self {
            points: vec![],
            starts: vec![],
            min: [f32::MAX; 2],
            max: [f32::MIN; 2],
        };

        for (line_idx, line) in text.lines().enumerate() {
            // The Odin reader splits on literal spaces, discards the first token,
            // and skips tokens beginning with '(' or '<'. Preserve that grammar.
            let mut tokens = line.split(' ');
            tokens.next();

            let start = g.points.len();
            while let Some(token) = tokens.next() {
                if token.is_empty() || token.starts_with(['(', '<']) {
                    continue;
                }

                let p: Point = [
                    coordinate(Some(token))
                        .wrap_err_with(|| format!("line {} x", line_idx.saturating_add(1)))?,
                    coordinate(tokens.next())
                        .wrap_err_with(|| format!("line {} y", line_idx.saturating_add(1)))?,
                ];

                ensure!(p.iter().all(|v| v.is_finite()), "non-finite TRL coordinate");

                for ((min, max), v) in g.min.iter_mut().zip(&mut g.max).zip(p) {
                    *min = min.min(v);
                    *max = max.max(v);
                }

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
        let range = sub(self.max, self.min);
        let span = range[0].max(range[1]);
        ensure!(
            span > 0.0 && span.is_finite(),
            "graph has zero or invalid extent"
        );

        let scale = (1.0_f32 - 0.05) / span;
        let translation = [
            (1.0 - scale * range[0]) / 2.0,
            (1.0 - scale * range[1]) / 2.0,
        ];

        self.points.iter_mut().for_each(|p| {
            *p = mul(
                sub(add(mul(sub(*p, self.min), scale), translation), [0.5; 2]),
                2.0,
            );
        });

        self.min = [-1.0; 2];
        self.max = [1.0; 2];
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
                let distance = length(sub(previous, current));
                if distance < remaining {
                    remaining -= distance;
                    previous = current;
                    pending.next();
                    continue;
                }

                let t = remaining / distance;
                let next = add(mul(previous, 1.0 - t), mul(current, t));
                // A tiny remainder may round to the previous point. Odin appends
                // it, resets the spacing, then advances on the next iteration.
                ensure!(
                    next.iter()
                        .zip(previous)
                        .any(|(a, b)| a.partial_cmp(&b) != Some(std::cmp::Ordering::Equal))
                        || remaining < spacing,
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
                let dir = sub(second, first);
                let len = length(dir);
                if len == 0.0 {
                    continue;
                }

                let dir = [dir[0] / len, dir[1] / len];
                let normal = mul([-dir[1], dir[0]], half);
                let v0 = add(first, normal);
                let v1 = sub(first, normal);
                let v2 = add(second, normal);
                let v3 = sub(second, normal);

                mesh.extend([v0, v1, v2, v3, v2, v1]);
            }
        }
        Ok(mesh)
    }
}

fn coordinate(token: Option<&str>) -> Result<f32> {
    token
        .ok_or_eyre("unexpected end of TRL line")?
        .parse::<f32>()
        .wrap_err("invalid coordinate")
}

fn add(a: Point, b: Point) -> Point {
    [a[0] + b[0], a[1] + b[1]]
}

fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1]]
}

fn mul(a: Point, k: f32) -> Point {
    [a[0] * k, a[1] * k]
}

fn length(a: Point) -> f32 {
    // Keep separate squared components and the original f32 reduction order.
    // hypot uses different rounding, which changes resampling and rasterization.
    a.map(|component| component * component)
        .into_iter()
        .sum::<f32>()
        .sqrt()
}
