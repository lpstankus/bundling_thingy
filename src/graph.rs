use color_eyre::eyre::{OptionExt, Result, WrapErr, bail, ensure};
use rayon::prelude::*;
use std::{fs, path::Path};

pub type Point = [f32; 2];
#[derive(Debug)]
pub struct Graph {
    pub points: Vec<Point>,
    pub starts: Vec<usize>,
    min: Point,
    max: Point,
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
            let mut tokens = SpaceTokens { rest: Some(line) };
            tokens.next();
            let start = g.points.len();
            while let Some(rest) = tokens.rest {
                if rest.is_empty() || rest.starts_with([' ', '(', '<']) {
                    tokens.next();
                    continue;
                }
                let p: Point = [
                    tokens
                        .number()
                        .wrap_err_with(|| format!("line {} x", line_idx.saturating_add(1)))?,
                    tokens
                        .number()
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
        // Parallelize strictly above 1,024 edges; cheap edges may still favor serial work.
        if self.starts.len().saturating_sub(1) > 1024 {
            return self.resample_parallel(spacing);
        }
        let mut points = Vec::new();
        let mut starts = Vec::new();
        for &[start, end] in self.starts.array_windows::<2>() {
            let edge = self
                .points
                .get(start..end)
                .ok_or_eyre("invalid edge bounds")?;
            starts.push(points.len());
            resample_edge(edge, spacing, &mut points)?;
        }
        starts.push(points.len());
        self.points = points;
        self.starts = starts;
        Ok(())
    }
    fn resample_parallel(&mut self, spacing: f32) -> Result<()> {
        let bounds: Vec<_> = self.starts.array_windows::<2>().copied().collect();
        // Indexed collection preserves edge order, including which error is reported first.
        let pieces: Vec<Result<Vec<Point>>> = bounds
            .par_iter()
            .map(|&[start, end]| {
                let edge = self
                    .points
                    .get(start..end)
                    .ok_or_eyre("invalid edge bounds")?;
                let mut points = Vec::new();
                resample_edge(edge, spacing, &mut points)?;
                Ok(points)
            })
            .collect();
        let pieces: Vec<_> = pieces.into_iter().collect::<Result<_>>()?;
        let capacity = pieces.iter().try_fold(0_usize, |total, edge| {
            total
                .checked_add(edge.len())
                .ok_or_eyre("resampled graph too large")
        })?;
        let mut points = Vec::with_capacity(capacity);
        let mut starts = Vec::with_capacity(self.starts.len());
        for edge in pieces {
            starts.push(points.len());
            points.extend(edge);
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

struct SpaceTokens<'a> {
    rest: Option<&'a str>,
}
impl<'a> Iterator for SpaceTokens<'a> {
    type Item = &'a str;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let rest = self.rest?;
        if let Some(i) = rest.as_bytes().iter().position(|&b| b == b' ') {
            let (token, tail) = rest.split_at(i);
            self.rest = tail.strip_prefix(' ');
            Some(token)
        } else {
            self.rest = None;
            Some(rest)
        }
    }
}

impl SpaceTokens<'_> {
    #[inline]
    fn number(&mut self) -> Result<f32> {
        self.next()
            .ok_or_eyre("unexpected end of TRL line")?
            .parse::<f32>()
            .wrap_err("invalid coordinate")
    }
}

fn resample_edge(edge: &[Point], spacing: f32, points: &mut Vec<Point>) -> Result<()> {
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
    Ok(())
}
