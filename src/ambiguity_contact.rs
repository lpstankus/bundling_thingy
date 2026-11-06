//! Sustained contact for one pair of current curves in input coordinates.
//!
//! Geometry uses inclusive f64 predicates without a tolerance halo. Sort comparisons
//! are separate from charged geometric, lookup, cell, and union-find operations.
//! Segment-local projected unions avoid subtraction of large cumulative arcs.
//! Ordinary f64 rounding can still affect a classification exactly at a threshold.
//! Inconsistent projection existence or endpoint membership returns an error.
//! Workspace estimates count requested capacities, including both buffers during
//! growth. Allocator overhead and caller-owned candidate storage are excluded.
use crate::weak_frechet::{Budget, Error as BudgetError};
use std::{fmt, mem::size_of};

/// Contact-specific failures preserve the shared kernel's resource errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Budget(BudgetError),
    InvalidParameters,
    NumericalFailure,
    IncompleteCandidates,
}
impl From<BudgetError> for Error {
    fn from(value: BudgetError) -> Self {
        Self::Budget(value)
    }
}
impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Budget(error) => error.fmt(formatter),
            Self::InvalidParameters => formatter.write_str("contact requires finite nonnegative radius with representable square and finite positive minimum length"),
            Self::NumericalFailure => formatter.write_str("contact geometry exceeds f64 arc precision or has inconsistent free-space predicates"),
            Self::IncompleteCandidates => formatter.write_str("contact candidates omit a nonempty neighboring cell"),
        }
    }
}
impl std::error::Error for Error {}

type Point = [f64; 2];
#[derive(Clone, Copy)]
struct Segment {
    start: Point,
    end: Point,
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

/// Owns only this invocation's reservations. All vectors drop before this ledger.
struct Workspace<'a> {
    budget: &'a mut Budget,
    reserved: usize,
}
impl Drop for Workspace<'_> {
    fn drop(&mut self) {
        // Only this owner changes `reserved`; every addition follows a successful reserve.
        let _ = self.budget.release(self.reserved);
    }
}
impl Workspace<'_> {
    fn charge(&mut self, units: u64) -> Result<(), Error> {
        Ok(self.budget.charge(units)?)
    }
    fn reserve(&mut self, bytes: usize) -> Result<(), Error> {
        let total = self
            .reserved
            .checked_add(bytes)
            .ok_or(BudgetError::SizeOverflow)?;
        self.budget.reserve(bytes)?;
        self.reserved = total;
        Ok(())
    }
    fn vector<T>(&mut self, capacity: usize) -> Result<Vec<T>, Error> {
        let bytes = capacity
            .checked_mul(size_of::<T>())
            .ok_or(BudgetError::SizeOverflow)?;
        self.reserve(bytes)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(capacity)
            .map_err(|_| BudgetError::Allocation)?;
        Ok(values)
    }
    fn push<T>(
        &mut self,
        values: &mut Vec<T>,
        requested: &mut usize,
        value: T,
    ) -> Result<(), Error> {
        if values.len() == *requested {
            let capacity = requested
                .checked_mul(2)
                .ok_or(BudgetError::SizeOverflow)?
                .max(8);
            // Allocate a separate buffer to account for simultaneous old/new capacities.
            let mut replacement = self.vector(capacity)?;
            self.charge(u64::try_from(values.len()).map_err(|_| BudgetError::SizeOverflow)?)?;
            replacement.append(values);
            *values = replacement;
            let bytes = requested
                .checked_mul(size_of::<T>())
                .ok_or(BudgetError::SizeOverflow)?;
            self.budget.release(bytes)?;
            self.reserved = self
                .reserved
                .checked_sub(bytes)
                .ok_or(BudgetError::InvalidRelease)?;
            *requested = capacity;
        }
        values.push(value);
        Ok(())
    }
}

fn segments(points: &[[f32; 2]], workspace: &mut Workspace<'_>) -> Result<Vec<Segment>, Error> {
    let first = points.first().ok_or(BudgetError::InvalidCurve)?;
    let mut previous = first.map(f64::from);
    let mut count = 0_usize;
    for point in points {
        workspace.charge(1)?;
        if point.iter().any(|value| !value.is_finite()) {
            return Err(BudgetError::InvalidCurve.into());
        }
        let current = point.map(f64::from);
        if !same_point(current, previous) {
            count = increment(count)?;
            previous = current;
        }
    }
    let mut result = workspace.vector(count)?;
    previous = first.map(f64::from);
    let mut arc = 0.0;
    for point in points {
        workspace.charge(1)?;
        let current = point.map(f64::from);
        if same_point(current, previous) {
            continue;
        }
        let delta = difference(current, previous);
        let length = dot(delta, delta).sqrt();
        let next_arc = arc + length;
        if !next_arc.is_finite() || next_arc <= arc {
            return Err(Error::NumericalFailure);
        }
        result.push(Segment {
            start: previous,
            end: current,
            length,
        });
        arc = next_arc;
        previous = current;
    }
    Ok(result)
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
    curves: (&[Segment], &[Segment]),
    radius: f64,
    workspace: &mut Workspace<'_>,
) -> Result<(), Error> {
    workspace.charge(3)?; // candidate and two capsule projections
    let (row, column) = cell;
    let source = at(curves.0, row)?;
    let target = at(curves.1, column)?;
    let along_a = projection(source, target, radius, radius * radius)?;
    let along_b = projection(target, source, radius, radius * radius)?;
    match (along_a, along_b) {
        (Some(along_a), Some(along_b)) => workspace.push(
            patches,
            capacity,
            Patch {
                cell,
                along_a,
                along_b,
            },
        ),
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
fn find(
    patches: &[Patch],
    cell: (usize, usize),
    workspace: &mut Workspace<'_>,
) -> Result<Option<usize>, Error> {
    let mut low = 0;
    let mut high = patches.len();
    while low < high {
        workspace.charge(1)?;
        let middle = low
            .checked_add(high.checked_sub(low).ok_or(BudgetError::SizeOverflow)? / 2)
            .ok_or(BudgetError::SizeOverflow)?;
        match at(patches, middle)?.cell.cmp(&cell) {
            std::cmp::Ordering::Less => low = increment(middle)?,
            std::cmp::Ordering::Greater => high = middle,
            std::cmp::Ordering::Equal => return Ok(Some(middle)),
        }
    }
    Ok(None)
}
fn forward_neighbor(
    segments: &[Segment],
    index: usize,
    closed: bool,
) -> Result<Option<(usize, Point)>, Error> {
    let next = increment(index)?;
    if next < segments.len() {
        return Ok(Some((next, at(segments, index)?.end)));
    }
    Ok(closed.then_some((0, at(segments, index)?.end)))
}
fn connect(
    patches: &[Patch],
    nodes: &mut [Node],
    curves: (&[Segment], &[Segment]),
    closed: (bool, bool),
    squared_radius: f64,
    workspace: &mut Workspace<'_>,
) -> Result<(), Error> {
    for (index, patch) in patches.iter().enumerate() {
        workspace.charge(1)?;
        let (row, column) = patch.cell;
        // Each directed forward boundary is visited, including both boundaries of
        // two-segment closed retraces. Looking only at unordered neighbors loses one.
        let neighbors = [
            forward_neighbor(curves.0, row, closed.0)?
                .map(|(next, point)| ((next, column), point, at(curves.1, column))),
            forward_neighbor(curves.1, column, closed.1)?
                .map(|(next, point)| ((row, next), point, at(curves.0, row))),
        ];
        for neighbor in neighbors.into_iter().flatten() {
            let (cell, point, segment) = neighbor;
            workspace.charge(1)?;
            if point_distance_squared(point, segment?) > squared_radius {
                continue;
            }
            let other = find(patches, cell, workspace)?.ok_or(Error::IncompleteCandidates)?;
            join(nodes, index, other, workspace)?;
        }
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

/// Exhaustive segment enumeration followed by the same sparse narrow phase.
/// Constant curves have no contact. Radius zero is supported.
/// # Errors
/// Rejects empty/nonfinite curves, invalid parameters, inconsistent f64 geometry,
/// and shared resource failures. Caller reservations survive every exit.
pub fn contact(
    a: &[[f32; 2]],
    b: &[[f32; 2]],
    radius: f64,
    minimum_length: f64,
    budget: &mut Budget,
) -> Result<bool, Error> {
    evaluate(
        a,
        b,
        radius,
        minimum_length,
        None::<std::iter::Empty<(usize, usize)>>,
        budget,
    )
}
/// Candidate indices refer to positive segments in path order after duplicate removal.
///
/// The caller must include every nonempty cell, including tangent connectors.
/// Duplicates and arbitrary candidate order do not change the result. Caller-owned
/// iteration/storage is outside this function's workspace, and each yield is charged.
/// # Errors
/// Same failures as `contact`, plus invalid indices and detected missing neighbors.
/// Completeness of disconnected candidate regions is the caller's responsibility.
pub fn contact_candidates(
    a: &[[f32; 2]],
    b: &[[f32; 2]],
    radius: f64,
    minimum_length: f64,
    candidates: impl IntoIterator<Item = (usize, usize)>,
    budget: &mut Budget,
) -> Result<bool, Error> {
    evaluate(
        a,
        b,
        radius,
        minimum_length,
        Some(candidates.into_iter()),
        budget,
    )
}
fn evaluate(
    a: &[[f32; 2]],
    b: &[[f32; 2]],
    radius: f64,
    minimum: f64,
    candidates: Option<impl Iterator<Item = (usize, usize)>>,
    budget: &mut Budget,
) -> Result<bool, Error> {
    let squared_radius = radius * radius;
    if !radius.is_finite()
        || radius < 0.0
        || !squared_radius.is_finite()
        || radius > 0.0 && squared_radius == 0.0
        || !minimum.is_finite()
        || minimum <= 0.0
    {
        return Err(Error::InvalidParameters);
    }
    let mut workspace = Workspace {
        budget,
        reserved: 0,
    };
    let source = segments(a, &mut workspace)?;
    let target = segments(b, &mut workspace)?;
    if source.is_empty() || target.is_empty() {
        return Ok(false);
    }
    let mut patches = Vec::new();
    let mut capacity = 0;
    if let Some(candidates) = candidates {
        for cell in candidates {
            insert_patch(
                &mut patches,
                &mut capacity,
                cell,
                (&source, &target),
                radius,
                &mut workspace,
            )?;
        }
    } else {
        for row in 0..source.len() {
            for column in 0..target.len() {
                insert_patch(
                    &mut patches,
                    &mut capacity,
                    (row, column),
                    (&source, &target),
                    radius,
                    &mut workspace,
                )?;
            }
        }
    }
    patches.sort_unstable_by_key(|patch| patch.cell);
    workspace.charge(u64::try_from(patches.len()).map_err(|_| BudgetError::SizeOverflow)?)?;
    patches.dedup_by_key(|patch| patch.cell);
    let mut nodes = workspace.vector(patches.len())?;
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
        (&source, &target),
        (a.first() == a.last(), b.first() == b.last()),
        squared_radius,
        &mut workspace,
    )?;
    let mut records = workspace.vector(patches.len())?;
    for (index, patch) in patches.iter().enumerate() {
        let root = root(&mut nodes, index, &mut workspace)?;
        records.push(Participation {
            root,
            along_a: Projection {
                segment: patch.cell.0,
                interval: patch.along_a,
                length: at(&source, patch.cell.0)?.length,
            },
            along_b: Projection {
                segment: patch.cell.1,
                interval: patch.along_b,
                length: at(&target, patch.cell.1)?.length,
            },
        });
    }
    component_lengths(&mut records, minimum, &mut workspace)
}
