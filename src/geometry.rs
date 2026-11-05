//! CPU snapshots of GPU geometry. Reservations must remain immutable across captures.

use crate::distortion::Curve;
use color_eyre::eyre::{Result, ensure, eyre};
use std::ops::Range;

#[derive(Clone, Copy)]
pub struct Reservation {
    pub point_offset: usize,
    pub segment_offset: usize,
    pub count: usize,
}

struct Edge {
    id: u32,
    points: Range<usize>,
}

pub struct Layout {
    edges: Vec<Edge>,
    samples: usize,
    skipped: usize,
}

impl Layout {
    /// # Errors
    /// Rejects invalid counts, arithmetic overflow, or ranges that do not tile.
    pub fn new(paths: &[Reservation], samples: usize) -> Result<Self> {
        let mut edges = Vec::new();
        let mut segments = Vec::new();
        edges.try_reserve_exact(paths.len())?;
        segments.try_reserve_exact(paths.len())?;
        let mut skipped = 0usize;
        for (id, path) in paths.iter().enumerate() {
            if path.count == 0 {
                skipped = skipped
                    .checked_add(1)
                    .ok_or_else(|| eyre!("skipped count overflow"))?;
                continue;
            }
            ensure!(path.count >= 2, "edge {id} has only one sample");
            let end = path
                .point_offset
                .checked_add(path.count)
                .ok_or_else(|| eyre!("point range overflow"))?;
            let length = path
                .count
                .checked_sub(1)
                .ok_or_else(|| eyre!("segment count underflow"))?;
            let segment_end = path
                .segment_offset
                .checked_add(length)
                .ok_or_else(|| eyre!("segment range overflow"))?;
            edges.push(Edge {
                id: u32::try_from(id)?,
                points: path.point_offset..end,
            });
            segments.push(path.segment_offset..segment_end);
        }
        ensure!(!edges.is_empty(), "no eligible edges");
        let segment_count = samples
            .checked_sub(edges.len())
            .ok_or_else(|| eyre!("sample count below edge count"))?;
        edges.sort_unstable_by_key(|edge| edge.points.start);
        segments.sort_unstable_by_key(|range| range.start);
        tile(edges.iter().map(|edge| &edge.points), samples)?;
        tile(segments.iter(), segment_count)?;
        Ok(Self {
            edges,
            samples,
            skipped,
        })
    }

    #[must_use]
    pub const fn sample_count(&self) -> usize {
        self.samples
    }
    #[must_use]
    pub const fn edge_count(&self) -> usize {
        self.edges.len()
    }
    #[must_use]
    pub const fn skipped_count(&self) -> usize {
        self.skipped
    }

    /// Each capture must use the same reservations and active sample count.
    /// # Errors
    /// Rejects changed counts or failed allocation.
    pub fn decoder(&self, active_samples: usize) -> Result<Decoder<'_>> {
        ensure!(
            active_samples == self.samples,
            "active sample count changed"
        );
        let mut positions = Vec::new();
        positions.try_reserve_exact(self.samples)?;
        Ok(Decoder {
            layout: self,
            positions,
            owner: 0,
            failed: false,
        })
    }
}

fn tile<'a>(ranges: impl Iterator<Item = &'a Range<usize>>, expected: usize) -> Result<()> {
    let mut end = 0;
    for range in ranges {
        ensure!(
            range.start == end && range.end <= expected,
            "geometry ranges have a gap, overlap or exceed active count"
        );
        end = range.end;
    }
    ensure!(end == expected, "geometry ranges do not cover active count");
    Ok(())
}

pub struct Decoder<'a> {
    layout: &'a Layout,
    positions: Vec<[f32; 2]>,
    owner: usize,
    failed: bool,
}

impl<'a> Decoder<'a> {
    /// Append whole little-endian 12-byte GPU records. Any failure poisons this decoder.
    /// # Errors
    /// Rejects partial or excess records, incorrect IDs and nonfinite positions.
    pub fn push(&mut self, bytes: &[u8]) -> Result<()> {
        ensure!(!self.failed, "decoder previously failed");
        self.failed = true;
        ensure!(bytes.len().is_multiple_of(12), "partial GPU point record");
        for record in bytes.as_chunks::<12>().0 {
            let index = self.positions.len();
            ensure!(index < self.layout.samples, "excess GPU point records");
            while self
                .layout
                .edges
                .get(self.owner)
                .is_some_and(|edge| index >= edge.points.end)
            {
                self.owner = self
                    .owner
                    .checked_add(1)
                    .ok_or_else(|| eyre!("owner overflow"))?;
            }
            let edge = self
                .layout
                .edges
                .get(self.owner)
                .ok_or_else(|| eyre!("missing point owner"))?;
            let mut words = record.as_chunks::<4>().0.iter();
            let mut word = || -> Result<[u8; 4]> {
                words
                    .next()
                    .copied()
                    .ok_or_else(|| eyre!("missing record word"))
            };
            let x = f32::from_le_bytes(word()?);
            let y = f32::from_le_bytes(word()?);
            let id = u32::from_le_bytes(word()?);
            ensure!(id == edge.id, "point {index} has incorrect edge ID");
            ensure!(
                x.is_finite() && y.is_finite(),
                "point {index} has nonfinite coordinates"
            );
            self.positions.push([x, y]);
        }
        self.failed = false;
        Ok(())
    }

    /// # Errors
    /// Rejects failed decoders or missing records.
    pub fn finish(self) -> Result<Snapshot<'a>> {
        ensure!(!self.failed, "decoder previously failed");
        ensure!(
            self.positions.len() == self.layout.samples,
            "missing GPU point records"
        );
        Ok(Snapshot {
            layout: self.layout,
            positions: self.positions,
        })
    }
}

pub struct Snapshot<'a> {
    layout: &'a Layout,
    positions: Vec<[f32; 2]>,
}

impl Snapshot<'_> {
    #[must_use]
    pub fn edges(&self) -> impl ExactSizeIterator<Item = (u32, &[[f32; 2]])> {
        // Layout validates the ranges and Decoder verifies the final length.
        // Both remain private and unchanged after construction.
        self.layout.edges.iter().map(|edge| {
            (
                edge.id,
                self.positions.get(edge.points.clone()).unwrap_or_default(),
            )
        })
    }

    #[must_use]
    pub const fn layout(&self) -> &Layout {
        self.layout
    }

    /// # Errors
    /// Rejects zero-length baseline edges using the numerical curve validator.
    pub fn validate_baseline(&self) -> Result<()> {
        for (id, points) in self.edges() {
            ensure!(
                Curve::new(points)?.length() > 0.0,
                "baseline edge {id} has zero length"
            );
        }
        Ok(())
    }
}
