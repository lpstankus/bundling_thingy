//! Conservative piece bounds and streaming whole-drawing contact enumeration.
//!
//! Long-segment AABBs split into closed rectangles whose union covers the
//! original bound. Shared split boundaries prevent gaps without interpolating
//! segment geometry. Empty parts of those rectangles may add false candidates.
//! Sort comparisons are outside the work counter. Requested capacities include
//! simultaneous growth buffers, excluding allocator overhead and borrowed points.
//! The callback owns any evidence it retains on success. On failure the invocation
//! rolls back all new byte and record reservations, including callback reservations.
//! Callers must discard partial callback evidence after any failure.
//! Callbacks must not release reservations owned before this invocation.
use crate::{
    ambiguity_contact,
    weak_frechet::{Budget, Error as ResourceError},
};
use std::{fmt, mem::size_of};

#[derive(Clone, Copy)]
pub struct Path<'a> {
    pub id: u32,
    pub points: &'a [[f32; 2]],
}
#[derive(Debug, Default, Clone, Copy)]
pub struct Statistics {
    pub constant_paths: usize,
    pub short_paths: usize,
    pub pieces_generated: u64,
    pub entries_visited: u64,
    pub candidate_examinations: u64,
    pub candidate_records: u64,
    pub narrow_pairs: u64,
    pub contacts: u64,
}
#[derive(Debug)]
pub enum Error<E> {
    Resource(ResourceError, Statistics),
    InvalidParameters,
    InvalidPath(u32),
    DuplicateId(u32),
    Contact {
        source: u32,
        partner: u32,
        error: ambiguity_contact::Error,
    },
    Callback {
        source: u32,
        partner: u32,
        error: E,
    },
}
impl<E: fmt::Display> fmt::Display for Error<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resource(error, statistics) => {
                write!(f, "{error}; index statistics {statistics:?}")
            }
            Self::InvalidParameters => f.write_str("invalid contact index parameters"),
            Self::InvalidPath(id) => write!(f, "empty, nonfinite or unrepresentable path {id}"),
            Self::DuplicateId(id) => write!(f, "duplicate path ID {id}"),
            Self::Contact {
                source,
                partner,
                error,
            } => write!(f, "contact {source}/{partner}: {error}"),
            Self::Callback {
                source,
                partner,
                error,
            } => write!(f, "callback {source}/{partner}: {error}"),
        }
    }
}
impl<E: fmt::Debug + fmt::Display> std::error::Error for Error<E> {}
impl<E> From<ResourceError> for Error<E> {
    fn from(error: ResourceError) -> Self {
        Self::Resource(error, Statistics::default())
    }
}
#[derive(Clone, Copy)]
struct Piece {
    edge: usize,
    segment: usize,
    low: [f64; 2],
    high: [f64; 2],
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Candidate {
    partner: usize,
    cell: (usize, usize),
}
struct Workspace<'a> {
    budget: &'a mut Budget,
    bytes: usize,
    records: usize,
}
impl Drop for Workspace<'_> {
    fn drop(&mut self) {
        let _ = self.budget.release(self.bytes);
        let _ = self.budget.release_records(self.records);
    }
}
impl Workspace<'_> {
    fn vector<T>(&mut self, capacity: usize, records: bool) -> Result<Vec<T>, ResourceError> {
        let bytes = capacity
            .checked_mul(size_of::<T>())
            .ok_or(ResourceError::SizeOverflow)?;
        let total = self
            .bytes
            .checked_add(bytes)
            .ok_or(ResourceError::SizeOverflow)?;
        self.budget.reserve(bytes)?;
        self.bytes = total;
        if records {
            self.budget.reserve_records(capacity)?;
            self.records = self
                .records
                .checked_add(capacity)
                .ok_or(ResourceError::SizeOverflow)?;
        }
        let mut result = Vec::new();
        result
            .try_reserve_exact(capacity)
            .map_err(|_| ResourceError::Allocation)?;
        Ok(result)
    }
    fn push<T>(
        &mut self,
        values: &mut Vec<T>,
        capacity: &mut usize,
        value: T,
        records: bool,
    ) -> Result<(), ResourceError> {
        if values.len() == *capacity {
            let next = capacity
                .checked_mul(2)
                .ok_or(ResourceError::SizeOverflow)?
                .max(8);
            let mut replacement = self.vector(next, records)?;
            self.budget
                .charge(u64::try_from(values.len()).map_err(|_| ResourceError::SizeOverflow)?)?;
            replacement.append(values);
            *values = replacement;
            self.free::<T>(*capacity, records)?;
            *capacity = next;
        }
        values.push(value);
        Ok(())
    }
    fn free<T>(&mut self, capacity: usize, records: bool) -> Result<(), ResourceError> {
        let bytes = capacity
            .checked_mul(size_of::<T>())
            .ok_or(ResourceError::SizeOverflow)?;
        self.budget.release(bytes)?;
        self.bytes = self
            .bytes
            .checked_sub(bytes)
            .ok_or(ResourceError::InvalidRelease)?;
        if records {
            self.budget.release_records(capacity)?;
            self.records = self
                .records
                .checked_sub(capacity)
                .ok_or(ResourceError::InvalidRelease)?;
        }
        Ok(())
    }
}
fn tick(budget: &mut Budget, count: &mut u64) -> Result<(), ResourceError> {
    budget.charge(1)?;
    *count = count.checked_add(1).ok_or(ResourceError::SizeOverflow)?;
    Ok(())
}
const fn bounds(edge: usize, segment: usize, a: [f64; 2], b: [f64; 2]) -> Piece {
    Piece {
        edge,
        segment,
        low: [a[0].min(b[0]).next_down(), a[1].min(b[1]).next_down()],
        high: [a[0].max(b[0]).next_up(), a[1].max(b[1]).next_up()],
    }
}
fn pieces(
    identity: (usize, usize),
    endpoints: ([f64; 2], [f64; 2]),
    maximum: f64,
    output: &mut Vec<Piece>,
    capacity: &mut usize,
    workspace: &mut Workspace<'_>,
    stats: &mut Statistics,
) -> Result<(), ResourceError> {
    split_bounds(
        bounds(identity.0, identity.1, endpoints.0, endpoints.1),
        maximum,
        output,
        capacity,
        workspace,
        stats,
    )
}
fn split_bounds(
    piece: Piece,
    maximum: f64,
    output: &mut Vec<Piece>,
    capacity: &mut usize,
    workspace: &mut Workspace<'_>,
    stats: &mut Statistics,
) -> Result<(), ResourceError> {
    tick(workspace.budget, &mut stats.entries_visited)?;
    let width = piece.high[0] - piece.low[0];
    let height = piece.high[1] - piece.low[1];
    if width.max(height) <= maximum {
        tick(workspace.budget, &mut stats.pieces_generated)?;
        return workspace.push(output, capacity, piece, false);
    }
    let axis = usize::from(height > width);
    let low = *piece.low.get(axis).ok_or(ResourceError::InvalidIndex)?;
    let high = *piece.high.get(axis).ok_or(ResourceError::InvalidIndex)?;
    let middle = high.mul_add(0.5, low * 0.5);
    if middle <= low || middle >= high {
        return Err(ResourceError::SizeOverflow);
    }
    let mut left = piece;
    let mut right = piece;
    *left.high.get_mut(axis).ok_or(ResourceError::InvalidIndex)? = middle;
    *right.low.get_mut(axis).ok_or(ResourceError::InvalidIndex)? = middle;
    split_bounds(left, maximum, output, capacity, workspace, stats)?;
    split_bounds(right, maximum, output, capacity, workspace, stats)
}
/// Stream each qualifying unordered pair once, in stable ID order.
///
/// `piece_length` bounds each piece's axis extent and changes only acceleration.
/// The index is sorted by minimum x. Dense drawings can require quadratic work.
/// Callback reservations survive success and share the index's global ledger.
/// # Errors
/// Invalid geometry/IDs/parameters, global resource exhaustion, narrow-phase
/// numerical errors and callback failures abort without a successful result.
pub fn enumerate<E>(
    paths: &[Path<'_>],
    radius: f64,
    minimum: f64,
    piece_length: f64,
    budget: &mut Budget,
    mut callback: impl FnMut(u32, u32, &mut Budget) -> Result<(), E>,
) -> Result<Statistics, Error<E>> {
    let before = budget.statistics();
    let mut stats = Statistics::default();
    let result = run(
        paths,
        radius,
        minimum,
        piece_length,
        budget,
        &mut callback,
        &mut stats,
    )
    .map_err(|error| match error {
        Error::Resource(resource, _) => Error::Resource(resource, stats),
        other => other,
    });
    if result.is_err() {
        let after = budget.statistics();
        budget.release(
            after
                .workspace_reserved
                .checked_sub(before.workspace_reserved)
                .ok_or(ResourceError::InvalidRelease)?,
        )?;
        budget.release_records(
            after
                .records_reserved
                .checked_sub(before.records_reserved)
                .ok_or(ResourceError::InvalidRelease)?,
        )?;
    }
    result
}
fn run<E>(
    paths: &[Path<'_>],
    radius: f64,
    minimum: f64,
    piece_length: f64,
    budget: &mut Budget,
    callback: &mut impl FnMut(u32, u32, &mut Budget) -> Result<(), E>,
    stats: &mut Statistics,
) -> Result<Statistics, Error<E>> {
    validate_parameters(radius, minimum, piece_length)?;
    let mut workspace = Workspace {
        budget,
        bytes: 0,
        records: 0,
    };
    let mut ordered = workspace.vector(paths.len(), false)?;
    for path in paths {
        workspace.budget.charge(1)?;
        ordered.push(*path);
    }
    ordered.sort_unstable_by_key(|path| path.id);
    for pair in ordered.windows(2) {
        workspace.budget.charge(1)?;
        if let [a, b] = pair
            && a.id == b.id
        {
            return Err(Error::DuplicateId(a.id));
        }
    }
    let mut index = Vec::new();
    let mut index_capacity = 0;
    build(
        &ordered,
        minimum,
        piece_length,
        &mut index,
        &mut index_capacity,
        &mut workspace,
        stats,
    )?;
    index.sort_unstable_by(|a, b| a.low[0].total_cmp(&b.low[0]));
    let mut candidates = Vec::<Candidate>::new();
    let mut capacity = 0;
    for (source, path) in ordered.iter().enumerate() {
        collect(
            source,
            &index,
            radius,
            &mut candidates,
            &mut capacity,
            &mut workspace,
            stats,
        )?;
        candidates.sort_unstable();
        workspace
            .budget
            .charge(u64::try_from(candidates.len()).map_err(|_| ResourceError::SizeOverflow)?)?;
        candidates.dedup();
        let mut start = 0;
        while let Some(first) = candidates.get(start) {
            let partner = first.partner;
            let mut end = start;
            while candidates
                .get(end)
                .is_some_and(|entry| entry.partner == partner)
            {
                workspace.budget.charge(1)?;
                end = end.checked_add(1).ok_or(ResourceError::SizeOverflow)?;
            }
            let other = ordered.get(partner).ok_or(ResourceError::InvalidIndex)?;
            tick(workspace.budget, &mut stats.narrow_pairs)?;
            let cells = candidates
                .get(start..end)
                .ok_or(ResourceError::InvalidIndex)?
                .iter()
                .map(|entry| entry.cell);
            let contact = ambiguity_contact::contact_candidates(
                path.points,
                other.points,
                radius,
                minimum,
                cells,
                workspace.budget,
            )
            .map_err(|error| Error::Contact {
                source: path.id,
                partner: other.id,
                error,
            })?;
            if contact {
                tick(workspace.budget, &mut stats.contacts)?;
                callback(path.id, other.id, workspace.budget).map_err(|error| Error::Callback {
                    source: path.id,
                    partner: other.id,
                    error,
                })?;
            }
            start = end;
        }
    }
    Ok(*stats)
}

fn build<E>(
    ordered: &[Path<'_>],
    minimum: f64,
    piece_length: f64,
    index: &mut Vec<Piece>,
    index_capacity: &mut usize,
    workspace: &mut Workspace<'_>,
    stats: &mut Statistics,
) -> Result<(), Error<E>> {
    for (edge, path) in ordered.iter().enumerate() {
        let first = path.points.first().ok_or(Error::InvalidPath(path.id))?;
        let mut previous = first.map(f64::from);
        let mut length = 0.0;
        for point in path.points {
            workspace.budget.charge(1)?;
            if point.iter().any(|v| !v.is_finite()) {
                return Err(Error::InvalidPath(path.id));
            }
            let current = point.map(f64::from);
            let delta = [current[0] - previous[0], current[1] - previous[1]];
            let step = delta[0].mul_add(delta[0], delta[1] * delta[1]).sqrt();
            if step > 0.0 {
                let next = length + step;
                if !next.is_finite() || next <= length {
                    return Err(Error::InvalidPath(path.id));
                }
                length = next;
            }
            previous = current;
        }
        if length == 0.0 {
            stats.constant_paths = stats
                .constant_paths
                .checked_add(1)
                .ok_or(ResourceError::SizeOverflow)?;
            continue;
        }
        if length < minimum {
            stats.short_paths = stats
                .short_paths
                .checked_add(1)
                .ok_or(ResourceError::SizeOverflow)?;
            continue;
        }
        previous = first.map(f64::from);
        let mut segment = 0;
        for point in path.points {
            workspace.budget.charge(1)?;
            let current = point.map(f64::from);
            if (current[0] - previous[0]).hypot(current[1] - previous[1]) <= 0.0 {
                continue;
            }
            pieces(
                (edge, segment),
                (previous, current),
                piece_length,
                index,
                index_capacity,
                workspace,
                stats,
            )?;
            segment = segment.checked_add(1).ok_or(ResourceError::SizeOverflow)?;
            previous = current;
        }
    }
    Ok(())
}

fn collect(
    source: usize,
    index: &[Piece],
    radius: f64,
    candidates: &mut Vec<Candidate>,
    capacity: &mut usize,
    workspace: &mut Workspace<'_>,
    stats: &mut Statistics,
) -> Result<(), ResourceError> {
    candidates.clear();
    for piece in index {
        tick(workspace.budget, &mut stats.entries_visited)?;
        if piece.edge != source {
            continue;
        }
        let high = (piece.high[0] + radius).next_up();
        let low = (piece.low[0] - radius).next_down();
        for target in index {
            tick(workspace.budget, &mut stats.entries_visited)?;
            if target.low[0] > high {
                break;
            }
            if target.edge <= source || target.high[0] < low {
                continue;
            }
            tick(workspace.budget, &mut stats.candidate_examinations)?;
            if target.low[1] > (piece.high[1] + radius).next_up()
                || target.high[1] < (piece.low[1] - radius).next_down()
            {
                continue;
            }
            tick(workspace.budget, &mut stats.candidate_records)?;
            workspace.push(
                candidates,
                capacity,
                Candidate {
                    partner: target.edge,
                    cell: (piece.segment, target.segment),
                },
                true,
            )?;
        }
    }
    Ok(())
}

fn validate_parameters<E>(radius: f64, minimum: f64, piece_length: f64) -> Result<(), Error<E>> {
    if !radius.is_finite()
        || radius < 0.0
        || !(radius * radius).is_finite()
        || radius > 0.0 && radius * radius == 0.0
        || !minimum.is_finite()
        || minimum <= 0.0
        || !piece_length.is_finite()
        || piece_length <= 0.0
    {
        return Err(Error::InvalidParameters);
    }
    Ok(())
}
