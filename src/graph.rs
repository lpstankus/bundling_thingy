use color_eyre::eyre::{OptionExt, Result, WrapErr, ensure};
use glam::DVec2;
use std::{fs, path::Path};

pub use glam::Vec2 as Point;

#[derive(Debug)]
pub struct Graph {
    points: Vec<Point>,
    starts: Vec<usize>,
    min: Point,
    max: Point,
}

/// Only `Graph::resample` can construct the geometry accepted by GPU uploads.
#[derive(Debug)]
pub struct ResampledGraph {
    points: Vec<Point>,
    segments: Vec<u32>,
    dimensions: [u32; 2],
}

impl Graph {
    #[tracing::instrument(name = "graph.load", skip_all, fields(input = %path.display()), err)]
    pub fn load(path: &Path) -> Result<Self> {
        let graph = Self::parse(
            &fs::read_to_string(path).wrap_err_with(|| format!("read {}", path.display()))?,
        )?;
        println!(
            "Read {}: {} drawable edges, {} control points",
            path.display(),
            graph.starts.len().saturating_sub(1),
            graph.points.len()
        );
        Ok(graph)
    }

    pub fn parse(text: &str) -> Result<Self> {
        let mut graph = Self {
            points: Vec::new(),
            starts: Vec::new(),
            min: Point::MAX,
            max: Point::MIN,
        };
        for (line, text) in text.lines().enumerate() {
            let mut tokens = text.split(' ');
            tokens.next();
            let start = graph.points.len();
            while let Some(token) = tokens.next() {
                if token.is_empty() || token.starts_with(['(', '<']) {
                    continue;
                }
                let point = Point::new(
                    coordinate(Some(token), line, 'x')?,
                    coordinate(tokens.next(), line, 'y')?,
                );
                ensure!(point.is_finite(), "non-finite TRL coordinate");
                let previous = graph.points.get(start..).and_then(|edge| edge.last());
                if previous != Some(&point) {
                    graph.points.push(point);
                }
            }
            if graph.points.len().saturating_sub(start) < 2 {
                graph.points.truncate(start);
                continue;
            }
            graph.starts.push(start);
            for &point in graph
                .points
                .get(start..)
                .ok_or_eyre("invalid edge bounds")?
            {
                graph.min = graph.min.min(point);
                graph.max = graph.max.max(point);
            }
        }
        graph.starts.push(graph.points.len());
        ensure!(!graph.points.is_empty(), "graph has no drawable edges");
        Ok(graph)
    }

    #[tracing::instrument(name = "graph.normalize", skip_all, fields(points = self.points.len()), err)]
    pub fn normalize(&mut self) -> Result<()> {
        let range = self.max - self.min;
        let span = range.max_element();
        ensure!(span > 0.0 && span.is_finite(), "graph has invalid extent");
        let scale = 0.95 / span;
        ensure!(scale.is_finite(), "graph normalization scale is not finite");
        let translation = (Point::ONE - scale * range) / 2.0;
        for point in &mut self.points {
            *point = ((*point - self.min) * scale + translation - Point::splat(0.5)) * 2.0;
        }
        self.min = Point::NEG_ONE;
        self.max = Point::ONE;
        Ok(())
    }

    /// Sample at intervals of one texel, retaining each edge's final endpoint.
    #[expect(
        clippy::while_float,
        reason = "Each segment has at most u32::MAX unit steps, which f64 represents exactly."
    )]
    #[tracing::instrument(name = "graph.resample", skip_all, fields(width = width, height = height), err)]
    pub fn resample(&self, width: u32, height: u32) -> Result<ResampledGraph> {
        ensure!(width > 0 && height > 0, "render dimensions must be nonzero");
        let scale = DVec2::new(
            f64::from(width.saturating_sub(1)) / 2.0,
            f64::from(height.saturating_sub(1)) / 2.0,
        );
        let mut points = Vec::new();
        let mut segments = Vec::new();
        for &[start, end] in self.starts.array_windows::<2>() {
            let edge = self
                .points
                .get(start..end)
                .ok_or_eyre("invalid edge bounds")?;
            let first = *edge.first().ok_or_eyre("empty edge")?;
            let output_start = points.len();
            points.push(first);
            let mut previous = first;
            let mut remaining = 1.0;
            for &current in edge.iter().skip(1) {
                let delta = current.as_dvec2() - previous.as_dvec2();
                let distance = (delta * scale).length();
                ensure!(
                    distance <= f64::from(u32::MAX),
                    "edge exceeds the resampled point index limit"
                );
                let mut along = remaining;
                while along <= distance {
                    let next = (previous.as_dvec2() + delta * (along / distance)).as_vec2();
                    push_distinct(&mut points, next);
                    along += 1.0;
                }
                remaining = along - distance;
                previous = current;
            }
            push_distinct(&mut points, previous);
            // Keep short closed paths drawable when regular samples coincide.
            if points.len().saturating_sub(output_start) == 1
                && let Some(&next) = edge.iter().find(|&&point| (point - first).length() > 0.0)
            {
                push_distinct(&mut points, next);
                push_distinct(&mut points, previous);
            }
            if points.len().saturating_sub(output_start) < 2 {
                points.truncate(output_start);
                continue;
            }
            for index in output_start..points.len().saturating_sub(1) {
                segments.push(u32::try_from(index)?);
            }
        }
        ensure!(
            !segments.is_empty(),
            "resampled graph has no drawable edges"
        );
        println!(
            "Resampled for {width}x{height}: {} control points -> {} samples, {} segments",
            self.points.len(),
            points.len(),
            segments.len()
        );
        Ok(ResampledGraph {
            points,
            segments,
            dimensions: [width, height],
        })
    }
}

impl ResampledGraph {
    pub fn points(&self) -> &[Point] {
        &self.points
    }

    pub fn segments(&self) -> &[u32] {
        &self.segments
    }

    pub const fn dimensions(&self) -> [u32; 2] {
        self.dimensions
    }
}

fn push_distinct(points: &mut Vec<Point>, point: Point) {
    if points
        .last()
        .is_none_or(|&previous| (point - previous).length() > 0.0)
    {
        points.push(point);
    }
}

fn coordinate(token: Option<&str>, line: usize, axis: char) -> Result<f32> {
    token
        .ok_or_eyre("unexpected end of TRL line")
        .and_then(|token| token.parse::<f32>().wrap_err("invalid coordinate"))
        .wrap_err_with(|| format!("line {} {axis}", line.saturating_add(1)))
}
