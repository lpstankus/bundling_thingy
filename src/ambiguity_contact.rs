//! Sustained contact for one pair of current curves in input coordinates.
//!
//! Geometry uses inclusive f64 predicates without a tolerance halo. Sort comparisons
//! are separate from charged geometric, lookup, cell, and union-find operations.
//! Segment-local projected unions avoid subtraction of large cumulative arcs.
//! Ordinary f64 rounding can still affect a classification exactly at a threshold.
//! Inconsistent projection existence or endpoint membership returns an error.
//! Workspace estimates count requested capacities, including both buffers during
//! growth. Allocator overhead and caller-owned candidate storage are excluded.
//! Radius and minimum come from a validated `ambiguity::Config`.
use crate::{
    ambiguity::Cause as Error,
    weak_frechet::{Error as BudgetError, Storage, Workspace},
};

type Point = [f64; 2];
#[derive(Clone, Copy)]
pub struct Segment {
    pub start: Point,
    pub end: Point,
    length: f64,
}
#[derive(Clone, Copy)]
struct Interval {
    low: f64,
    high: f64,
}
impl Interval {
    const UNIT: Self = Self {
        low: 0.0,
        high: 1.0,
    };
    fn intersect(self, other: Self) -> Option<Self> {
        let result = Self {
            low: self.low.max(other.low),
            high: self.high.min(other.high),
        };
        (result.low <= result.high).then_some(result)
    }
    const fn include(&mut self, other: Self) {
        self.low = self.low.min(other.low);
        self.high = self.high.max(other.high);
    }
}
#[derive(Clone, Copy)]
struct Patch {
    cell: (usize, usize),
    along_a: Interval,
    along_b: Interval,
}
#[derive(Clone, Copy)]
struct Node {
    parent: usize,
    size: usize,
}
#[derive(Clone, Copy)]
struct Participation {
    root: usize,
    along_a: Projection,
    along_b: Projection,
}
#[derive(Clone, Copy)]
// Keep segment-local parameters. Summing each segment's interval union avoids
// subtracting large cumulative arcs or measuring a span across a periodic cut.
struct Projection {
    segment: usize,
    interval: Interval,
    length: f64,
}

fn same_point(first: Point, second: Point) -> bool {
    let delta = difference(first, second);
    dot(delta, delta) <= 0.0
}
fn difference([ax, ay]: Point, [bx, by]: Point) -> Point {
    [ax - bx, ay - by]
}
fn dot([ax, ay]: Point, [bx, by]: Point) -> f64 {
    ax.mul_add(bx, ay * by)
}
fn cross([ax, ay]: Point, [bx, by]: Point) -> f64 {
    // Compensate product rounding to reduce cancellation near collinearity.
    let product = ay * bx;
    ax.mul_add(by, -product) + (-ay).mul_add(bx, product)
}
fn increment(value: usize) -> Result<usize, Error> {
    Ok(value.checked_add(1).ok_or(BudgetError::SizeOverflow)?)
}
fn at<T: Copy>(values: &[T], index: usize) -> Result<T, Error> {
    values
        .get(index)
        .copied()
        .ok_or_else(|| BudgetError::InvalidIndex.into())
}

#[derive(Clone, Copy)]
struct PositiveSegment {
    end: usize,
    length: f64,
}
/// Snapshot-local geometry borrowing finite snapshot points.
///
/// Its storage is charged to the workspace passed to `prepare` and must drop
/// before that workspace.
pub struct PreparedCurve<'a> {
    points: &'a [[f32; 2]],
    segments: Vec<PositiveSegment>,
    total_length: f64,
}
impl<'a> PreparedCurve<'a> {
    /// Retain positive-point indices and exact kernel lengths.
    /// # Errors
    /// Empty curves, arc precision failures and shared allocation/resource limits.
    pub fn prepare(points: &'a [[f32; 2]], workspace: &mut Workspace<'_>) -> Result<Self, Error> {
        let first = points.first().ok_or(BudgetError::InvalidCurve)?;
        let mut previous = first.map(f64::from);
        let mut count = 0;
        for point in points {
            workspace.charge(1)?;
            let current = point.map(f64::from);
            if !same_point(current, previous) {
                count = increment(count)?;
                previous = current;
            }
        }
        let mut segments = workspace.vector(count, Storage::Bytes)?;
        previous = first.map(f64::from);
        let mut total_length = 0.0;
        for (end, point) in points.iter().enumerate() {
            workspace.charge(1)?;
            let current = point.map(f64::from);
            if same_point(current, previous) {
                continue;
            }
            let delta = difference(current, previous);
            let length = dot(delta, delta).sqrt();
            let next = total_length + length;
            if !next.is_finite() || next <= total_length {
                return Err(Error::NumericalFailure);
            }
            segments.push(PositiveSegment { end, length });
            total_length = next;
            previous = current;
        }
        Ok(Self {
            points,
            segments,
            total_length,
        })
    }
    #[must_use]
    pub const fn total_length(&self) -> f64 {
        self.total_length
    }
    #[must_use]
    pub const fn segment_count(&self) -> usize {
        self.segments.len()
    }
    /// # Errors
    /// Rejects an index outside the positive segments.
    pub fn segment_length(&self, index: usize) -> Result<f64, Error> {
        Ok(at(&self.segments, index)?.length)
    }
    /// # Errors
    /// Rejects an index outside the positive segments.
    pub fn segment(&self, index: usize) -> Result<Segment, Error> {
        let metadata = at(&self.segments, index)?;
        let start = if let Some(previous) = index.checked_sub(1) {
            at(&self.segments, previous)?.end
        } else {
            0
        };
        Ok(Segment {
            start: at(self.points, start)?.map(f64::from),
            end: at(self.points, metadata.end)?.map(f64::from),
            length: metadata.length,
        })
    }
    fn closed(&self) -> bool {
        self.points.first() == self.points.last()
    }
}

fn point_distance_squared(point: Point, segment: Segment) -> f64 {
    let direction = difference(segment.end, segment.start);
    let relative = difference(point, segment.start);
    let length_squared = dot(direction, direction);
    let projected = dot(relative, direction);
    if projected <= 0.0 {
        return dot(relative, relative);
    }
    if projected >= length_squared {
        let residual = difference(point, segment.end);
        return dot(residual, residual);
    }
    let perpendicular = cross(relative, direction);
    perpendicular * perpendicular / length_squared
}
fn clip_linear(
    interval: Interval,
    value: f64,
    slope: f64,
    low: f64,
    high: f64,
) -> Option<Interval> {
    if slope == 0.0 {
        return (value >= low && value <= high).then_some(interval);
    }
    let first = (low - value) / slope;
    let last = (high - value) / slope;
    interval.intersect(Interval {
        low: first.min(last),
        high: first.max(last),
    })
}
const fn include(result: &mut Option<Interval>, interval: Interval) {
    if let Some(current) = result {
        current.include(interval);
    } else {
        *result = Some(interval);
    }
}
fn projection(
    source: Segment,
    target: Segment,
    radius: f64,
    squared_radius: f64,
) -> Result<Option<Interval>, Error> {
    let direction = difference(source.end, source.start);
    let target_direction = difference(target.end, target.start);
    let relative = difference(source.start, target.start);
    let target_squared = dot(target_direction, target_direction);
    let mut result = None;
    let strip = clip_linear(
        Interval::UNIT,
        dot(relative, target_direction),
        dot(direction, target_direction),
        0.0,
        target_squared,
    );
    if let Some(strip) = strip
        && let Some(interval) = clip_linear(
            strip,
            cross(relative, target_direction),
            cross(direction, target_direction),
            -radius * target.length,
            radius * target.length,
        )
    {
        include(&mut result, interval);
    }
    for point in [target.start, target.end] {
        let offset = difference(point, source.start);
        let center = dot(offset, direction) / dot(direction, direction);
        let perpendicular = cross(offset, direction);
        let distance_squared = perpendicular * perpendicular / dot(direction, direction);
        let remaining = squared_radius - distance_squared;
        if remaining >= 0.0 {
            let half = remaining.sqrt() / source.length;
            if let Some(interval) = Interval::UNIT.intersect(Interval {
                low: center - half,
                high: center + half,
            }) {
                include(&mut result, interval);
            }
        }
    }
    // Use exactly the boundary predicate for endpoint participation and cell joins.
    for (parameter, point) in [(0.0, source.start), (1.0, source.end)] {
        let inside = point_distance_squared(point, target) <= squared_radius;
        let included =
            result.is_some_and(|interval| parameter >= interval.low && parameter <= interval.high);
        if included && !inside {
            return Err(Error::NumericalFailure);
        }
        if inside {
            include(
                &mut result,
                Interval {
                    low: parameter,
                    high: parameter,
                },
            );
        }
    }
    Ok(result)
}
fn insert_patch(
    patches: &mut Vec<Patch>,
    capacity: &mut usize,
    cell: (usize, usize),
    curves: (&PreparedCurve<'_>, &PreparedCurve<'_>),
    radius: f64,
    workspace: &mut Workspace<'_>,
) -> Result<(), Error> {
    workspace.charge(3)?; // candidate and two capsule projections
    let (row, column) = cell;
    let source = curves.0.segment(row)?;
    let target = curves.1.segment(column)?;
    let along_a = projection(source, target, radius, radius * radius)?;
    let along_b = projection(target, source, radius, radius * radius)?;
    match (along_a, along_b) {
        (Some(along_a), Some(along_b)) => Ok(workspace.push(
            patches,
            capacity,
            Patch {
                cell,
                along_a,
                along_b,
            },
            Storage::Bytes,
        )?),
        (None, None) => Ok(()),
        _ => Err(Error::NumericalFailure),
    }
}

fn root(nodes: &mut [Node], index: usize, workspace: &mut Workspace<'_>) -> Result<usize, Error> {
    let mut current = index;
    loop {
        workspace.charge(1)?;
        let node = at(nodes, current)?;
        if node.parent == current {
            break;
        }
        current = node.parent;
    }
    let mut compress = index;
    while compress != current {
        workspace.charge(1)?;
        let node = nodes.get_mut(compress).ok_or(BudgetError::InvalidIndex)?;
        compress = node.parent;
        node.parent = current;
    }
    Ok(current)
}
fn join(
    nodes: &mut [Node],
    first: usize,
    second: usize,
    workspace: &mut Workspace<'_>,
) -> Result<(), Error> {
    let mut left = root(nodes, first, workspace)?;
    let mut right = root(nodes, second, workspace)?;
    if left == right {
        return Ok(());
    }
    workspace.charge(1)?;
    if at(nodes, left)?.size < at(nodes, right)?.size {
        std::mem::swap(&mut left, &mut right);
    }
    let size = at(nodes, left)?
        .size
        .checked_add(at(nodes, right)?.size)
        .ok_or(BudgetError::SizeOverflow)?;
    nodes.get_mut(left).ok_or(BudgetError::InvalidIndex)?.size = size;
    nodes
        .get_mut(right)
        .ok_or(BudgetError::InvalidIndex)?
        .parent = left;
    Ok(())
}
// Every positive cell checks both directed forward boundaries. Row joins sweep
// sorted column lists. Column joins use consecutive cells within each row. The
// periodic pass is separate, so two-segment retraces retain both cut boundaries.
fn connect(
    patches: &[Patch],
    nodes: &mut [Node],
    curves: (&PreparedCurve<'_>, &PreparedCurve<'_>),
    closed: (bool, bool),
    squared_radius: f64,
    workspace: &mut Workspace<'_>,
) -> Result<(), Error> {
    let mut row_start = 0;
    let mut next_row = 0;
    while row_start < patches.len() {
        let row = at(patches, row_start)?.cell.0;
        let mut row_end = row_start;
        while row_end < patches.len() && at(patches, row_end)?.cell.0 == row {
            workspace.charge(1)?;
            row_end = increment(row_end)?;
        }
        let forward = increment(row)?;
        let target_row = if forward < curves.0.segment_count() {
            Some(forward)
        } else {
            closed.0.then_some(0)
        };
        if target_row == Some(0) {
            next_row = 0;
        } else {
            next_row = next_row.max(row_end);
        }
        let mut column_cursor = next_row;
        for index in row_start..row_end {
            workspace.charge(1)?;
            let column = at(patches, index)?.cell.1;
            if let Some(target_row) = target_row {
                let point = curves.0.segment(row)?.end;
                workspace.charge(1)?;
                if point_distance_squared(point, curves.1.segment(column)?) <= squared_radius {
                    let cell = (target_row, column);
                    while column_cursor < patches.len() && at(patches, column_cursor)?.cell < cell {
                        workspace.charge(1)?;
                        column_cursor = increment(column_cursor)?;
                    }
                    if patches.get(column_cursor).is_none_or(|p| p.cell != cell) {
                        return Err(Error::IncompleteCandidates);
                    }
                    join(nodes, index, column_cursor, workspace)?;
                }
            }
            let forward_column = increment(column)?;
            let target = if forward_column < curves.1.segment_count() {
                Some((forward_column, increment(index)?))
            } else {
                closed.1.then_some((0, row_start))
            };
            if let Some((target_column, other)) = target {
                workspace.charge(1)?;
                let point = curves.1.segment(column)?.end;
                if point_distance_squared(point, curves.0.segment(row)?) <= squared_radius {
                    if patches
                        .get(other)
                        .is_none_or(|p| p.cell != (row, target_column))
                    {
                        return Err(Error::IncompleteCandidates);
                    }
                    join(nodes, index, other, workspace)?;
                }
            }
        }
        row_start = row_end;
    }
    Ok(())
}
fn component_lengths(
    records: &mut [Participation],
    minimum: f64,
    workspace: &mut Workspace<'_>,
) -> Result<bool, Error> {
    records.sort_unstable_by(|left, right| {
        left.root
            .cmp(&right.root)
            .then_with(|| left.along_a.segment.cmp(&right.along_a.segment))
            .then_with(|| {
                left.along_a
                    .interval
                    .low
                    .total_cmp(&right.along_a.interval.low)
            })
    });
    let mut offset = 0;
    while offset < records.len() {
        let component = at(records, offset)?.root;
        let mut end = increment(offset)?;
        while end < records.len() && at(records, end)?.root == component {
            workspace.charge(1)?;
            end = increment(end)?;
        }
        let group = records
            .get_mut(offset..end)
            .ok_or(BudgetError::InvalidIndex)?;
        let length_a = merged_length(group.iter().map(|record| record.along_a), workspace)?;
        group.sort_unstable_by(|left, right| {
            left.along_b
                .segment
                .cmp(&right.along_b.segment)
                .then_with(|| {
                    left.along_b
                        .interval
                        .low
                        .total_cmp(&right.along_b.interval.low)
                })
        });
        let length_b = merged_length(group.iter().map(|record| record.along_b), workspace)?;
        if length_a >= minimum && length_b >= minimum {
            return Ok(true);
        }
        offset = end;
    }
    Ok(false)
}
fn merged_length(
    intervals: impl Iterator<Item = Projection>,
    workspace: &mut Workspace<'_>,
) -> Result<f64, Error> {
    let mut end = 0.0_f64;
    let mut total = 0.0;
    let mut segment = None;
    for projection in intervals {
        workspace.charge(1)?;
        if segment != Some(projection.segment) {
            segment = Some(projection.segment);
            end = 0.0;
        }
        let interval = projection.interval;
        total =
            ((interval.high - interval.low.max(end)).max(0.0)).mul_add(projection.length, total);
        end = end.max(interval.high);
    }
    if !total.is_finite() {
        return Err(Error::NumericalFailure);
    }
    Ok(total)
}

/// Sustained contact from candidate cells indexing positive segments in path order.
///
/// The caller must include every nonempty cell, including tangent connectors.
/// Duplicates and arbitrary candidate order do not change the result. Caller-owned
/// iteration/storage is outside this workspace, and each yield is charged.
/// Constant curves have no contact. Radius zero is supported.
/// # Errors
/// Invalid indices, inconsistent f64 geometry, detected missing neighbors and
/// shared resource failures. Completeness of disconnected candidate regions is
/// the caller's responsibility.
pub fn contact(
    a: &PreparedCurve<'_>,
    b: &PreparedCurve<'_>,
    radius: f64,
    minimum: f64,
    candidates: impl IntoIterator<Item = (usize, usize)>,
    workspace: &mut Workspace<'_>,
) -> Result<bool, Error> {
    let source = a;
    let target = b;
    if source.segment_count() == 0 || target.segment_count() == 0 {
        return Ok(false);
    }
    let mut patches = Vec::new();
    let mut capacity = 0;
    for cell in candidates {
        insert_patch(
            &mut patches,
            &mut capacity,
            cell,
            (source, target),
            radius,
            workspace,
        )?;
    }
    patches.sort_unstable_by_key(|patch| patch.cell);
    workspace.charge(u64::try_from(patches.len()).map_err(|_| BudgetError::SizeOverflow)?)?;
    patches.dedup_by_key(|patch| patch.cell);
    let mut nodes = workspace.vector(patches.len(), Storage::Bytes)?;
    for index in 0..patches.len() {
        workspace.charge(1)?;
        nodes.push(Node {
            parent: index,
            size: 1,
        });
    }
    connect(
        &patches,
        &mut nodes,
        (source, target),
        (a.closed(), b.closed()),
        radius * radius,
        workspace,
    )?;
    let mut records = workspace.vector(patches.len(), Storage::Bytes)?;
    for (index, patch) in patches.iter().enumerate() {
        let root = root(&mut nodes, index, workspace)?;
        records.push(Participation {
            root,
            along_a: Projection {
                segment: patch.cell.0,
                interval: patch.along_a,
                length: source.segment_length(patch.cell.0)?,
            },
            along_b: Projection {
                segment: patch.cell.1,
                interval: patch.along_b,
                length: target.segment_length(patch.cell.1)?,
            },
        });
    }
    component_lengths(&mut records, minimum, workspace)
}
