use color_eyre::eyre::{OptionExt, Result, WrapErr, ensure};
use std::{fs, path::Path};

pub use glam::Vec2 as Point;

#[derive(Debug)]
pub struct Graph {
    points: Vec<Point>,
    starts: Vec<u32>,
    min: Point,
    max: Point,
}

pub struct Normalization {
    pub minimum: Point,
    pub scale: f32,
    pub translation: Point,
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
            graph.starts.push(u32::try_from(start)?);
            for &point in graph
                .points
                .get(start..)
                .ok_or_eyre("invalid edge bounds")?
            {
                graph.min = graph.min.min(point);
                graph.max = graph.max.max(point);
            }
        }
        graph.starts.push(u32::try_from(graph.points.len())?);
        ensure!(!graph.points.is_empty(), "graph has no drawable edges");
        Ok(graph)
    }

    pub fn normalization(&self) -> Result<Normalization> {
        let range = self.max - self.min;
        let span = range.max_element();
        ensure!(span > 0.0 && span.is_finite(), "graph has invalid extent");
        let scale = 0.95 / span;
        ensure!(scale.is_finite(), "graph normalization scale is not finite");
        let translation = (Point::ONE - scale * range) / 2.0;
        Ok(Normalization {
            minimum: self.min,
            scale,
            translation,
        })
    }

    pub fn points(&self) -> &[Point] {
        &self.points
    }

    pub fn starts(&self) -> &[u32] {
        &self.starts
    }
}

fn coordinate(token: Option<&str>, line: usize, axis: char) -> Result<f32> {
    token
        .ok_or_eyre("unexpected end of TRL line")
        .and_then(|token| token.parse::<f32>().wrap_err("invalid coordinate"))
        .wrap_err_with(|| format!("line {} {axis}", line.saturating_add(1)))
}
