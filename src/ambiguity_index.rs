//! Conservative piece bounds and streaming whole-drawing contact enumeration.
//!
//! Long-segment AABBs split into closed rectangles whose union covers the
//! original bound. Shared split boundaries prevent gaps without interpolating
//! segment geometry. Empty parts of those rectangles may add false candidates.
//! Sort comparisons are outside the work counter. Requested capacities include
//! simultaneous growth buffers, excluding allocator overhead and borrowed points.
//! The index releases only its own reservations on every exit. Callbacks own
//! their retained reservations on success and failure and must clean them up.
//! The crate's verified GPU frame may replace the CPU index. It decides each
//! complete source/partner run as certified contact, certified no contact or
//! complete CPU fallback through the same private pair route, in the same
//! source/partner order, emitting through the same contact helper. No other
//! external candidate or certificate provider is accepted.
use crate::{
    ambiguity_contact, ambiguity_gpu,
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
    /// Positive paths with a rounded total below the minimum; they remain indexed.
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
    /// GPU contact acceleration failed (resource, device or protocol); no CPU
    /// index replaces it.
    Gpu(ambiguity_gpu::Error),
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
            Self::Gpu(error) => write!(f, "GPU contact: {error}"),
        }
    }
}
impl<E: fmt::Debug + fmt::Display> std::error::Error for Error<E> {}
impl<E> From<ResourceError> for Error<E> {
    fn from(error: ResourceError) -> Self {
        Self::Resource(error, Statistics::default())
    }
}
impl<E> From<ambiguity_gpu::Error> for Error<E> {
    fn from(error: ambiguity_gpu::Error) -> Self {
        match error {
            ambiguity_gpu::Error::Workspace(resource) => resource.into(),
            other => Self::Gpu(other),
        }
    }
}
/// One candidate segment-pair cell of a GPU fallback run.
///
/// Path fields are ordinals in increasing ID order. Segment fields are positive
/// segment indices after duplicate removal, as in `PreparedCurve`. Field order
/// matches the CPU candidate order: source, partner, then cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct CandidateCell {
    pub source: u32,
    pub partner: u32,
    pub source_segment: u32,
    pub partner_segment: u32,
}

/// Candidate enumeration used before the unchanged per-source narrow phase.
pub enum BroadPhase<'a> {
    /// Streaming CPU AABB hierarchy over bounded segment pieces.
    Cpu,
    /// GPU candidates and certified contact decisions on resident geometry,
    /// verified bit for bit against the enumerated paths and using this
    /// enumeration's radius, minimum and preparation. Uncertain pairs take the
    /// complete CPU pair route.
    Gpu(&'a mut ambiguity_gpu::Frame),
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
/// Candidate view shared by CPU and external records of one source.
trait CellRecord: Copy {
    fn partner(&self) -> usize;
    fn cell(&self) -> (usize, usize);
}
impl CellRecord for Candidate {
    fn partner(&self) -> usize {
        self.partner
    }
    fn cell(&self) -> (usize, usize) {
        self.cell
    }
}
// Out-of-range sentinels cannot occur for u32 on supported targets; they would
// fail the later index lookups rather than alias a valid segment.
fn index(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}
impl CellRecord for CandidateCell {
    fn partner(&self) -> usize {
        index(self.partner)
    }
    fn cell(&self) -> (usize, usize) {
        (index(self.source_segment), index(self.partner_segment))
    }
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
        let total_records = self
            .records
            .checked_add(if records { capacity } else { 0 })
            .ok_or(ResourceError::SizeOverflow)?;
        self.budget.reserve(bytes)?;
        if records && let Err(error) = self.budget.reserve_records(capacity) {
            // Roll back so batch retries start from the same ledger.
            self.budget.release(bytes)?;
            return Err(error);
        }
        self.bytes = total;
        self.records = total_records;
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
    if let Some((left, right)) = split(piece, maximum)? {
        split_bounds(left, maximum, output, capacity, workspace, stats)?;
        split_bounds(right, maximum, output, capacity, workspace, stats)
    } else {
        tick(workspace.budget, &mut stats.pieces_generated)?;
        workspace.push(output, capacity, piece, false)
    }
}
fn split(piece: Piece, maximum: f64) -> Result<Option<(Piece, Piece)>, ResourceError> {
    let width = piece.high[0] - piece.low[0];
    let height = piece.high[1] - piece.low[1];
    if width.max(height) <= maximum {
        return Ok(None);
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
    Ok(Some((left, right)))
}
/// Stream each qualifying unordered pair once, in stable ID order.
///
/// `piece_length` bounds each piece's axis extent and changes only acceleration.
/// A balanced AABB hierarchy prunes both coordinates. Dense contact remains quadratic.
/// Callback reservations survive success and failure on the shared ledger.
/// The callback owner must release them after either result.
/// # Errors
/// Invalid geometry/IDs/parameters, global resource exhaustion, narrow-phase
/// numerical errors and callback failures abort without a successful result.
pub fn enumerate<E>(
    paths: &[Path<'_>],
    radius: f64,
    minimum: f64,
    piece_length: f64,
    budget: &mut Budget,
    callback: impl FnMut(u32, u32, &mut Budget) -> Result<(), E>,
) -> Result<Statistics, Error<E>> {
    enumerate_with(
        BroadPhase::Cpu,
        paths,
        radius,
        minimum,
        piece_length,
        budget,
        callback,
    )
}
/// `enumerate` with a selectable broad phase and identical contact decisions.
///
/// GPU batches are whole increasing source ranges sized by the joint device,
/// record and workspace plan; multi-source ranges halve before fill and a
/// single source that still does not fit is an error. CPU fallback pairs are
/// charged as on the CPU path; GPU generation and certification are not.
/// # Errors
/// As `enumerate`, plus external source failures and contract violations.
pub fn enumerate_with<E>(
    broad_phase: BroadPhase<'_>,
    paths: &[Path<'_>],
    radius: f64,
    minimum: f64,
    piece_length: f64,
    budget: &mut Budget,
    mut callback: impl FnMut(u32, u32, &mut Budget) -> Result<(), E>,
) -> Result<Statistics, Error<E>> {
    let mut stats = Statistics::default();
    run(
        broad_phase,
        paths,
        (radius, minimum, piece_length),
        budget,
        &mut callback,
        &mut stats,
    )
    .map_err(|error| match error {
        Error::Resource(resource, _) => Error::Resource(resource, stats),
        other => other,
    })
}
fn run<E>(
    broad_phase: BroadPhase<'_>,
    paths: &[Path<'_>],
    (radius, minimum, piece_length): (f64, f64, f64),
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
    let ordered = ordered_paths(paths, &mut workspace)?;
    let prepared = prepare_paths(&ordered, &mut workspace)?;
    let narrow = Narrow {
        ordered: &ordered,
        prepared: &prepared,
        radius,
        minimum,
    };
    match broad_phase {
        BroadPhase::Cpu => cpu_candidates(&narrow, piece_length, &mut workspace, stats, callback)?,
        BroadPhase::Gpu(frame) => {
            gpu_candidates(&narrow, frame, &mut workspace, stats, callback)?;
        }
    }
    Ok(*stats)
}
fn cpu_candidates<E>(
    narrow: &Narrow<'_, '_>,
    piece_length: f64,
    workspace: &mut Workspace<'_>,
    stats: &mut Statistics,
    callback: &mut impl FnMut(u32, u32, &mut Budget) -> Result<(), E>,
) -> Result<(), Error<E>> {
    let mut index = build(
        narrow.ordered,
        narrow.prepared,
        narrow.minimum,
        piece_length,
        workspace,
        stats,
    )?;
    let mut hierarchy = workspace.vector(node_count(index.len())?, false)?;
    if !index.is_empty() {
        build_tree(&mut index, 0, &mut hierarchy, workspace.budget, stats)?;
    }
    let mut candidates = Vec::<Candidate>::new();
    let mut capacity = 0;
    for (source, path) in narrow.ordered.iter().enumerate() {
        collect(
            (source, path.points),
            (&index, &hierarchy),
            (narrow.radius, piece_length),
            &mut candidates,
            &mut capacity,
            workspace,
            stats,
        )?;
        candidates.sort_unstable();
        workspace
            .budget
            .charge(u64::try_from(candidates.len()).map_err(|_| ResourceError::SizeOverflow)?)?;
        candidates.dedup();
        narrow.source(source, &mut candidates, workspace.budget, stats, callback)?;
    }
    Ok(())
}
/// Run GPU batches. The session releases the frame's device buffers and its
/// own reservations on every exit, including unwinding.
fn gpu_candidates<E>(
    narrow: &Narrow<'_, '_>,
    frame: &mut ambiguity_gpu::Frame,
    workspace: &mut Workspace<'_>,
    stats: &mut Statistics,
    callback: &mut impl FnMut(u32, u32, &mut Budget) -> Result<(), E>,
) -> Result<(), Error<E>> {
    let tables = narrow
        .ordered
        .len()
        .checked_add(1)
        .ok_or(ResourceError::SizeOverflow)?;
    let mut segment_offsets = workspace.vector::<u32>(tables, false)?;
    segment_offsets.push(0);
    for curve in narrow.prepared {
        workspace.budget.charge(1)?;
        classify(curve, narrow.minimum, stats)?;
        let previous = segment_offsets.last().copied().unwrap_or(0);
        let count =
            u32::try_from(curve.segment_count()).map_err(|_| ResourceError::SizeOverflow)?;
        segment_offsets.push(
            previous
                .checked_add(count)
                .ok_or(ResourceError::SizeOverflow)?,
        );
    }
    let request = ambiguity_gpu::Request {
        paths: narrow.ordered,
        prepared: narrow.prepared,
        segment_offsets: &segment_offsets,
        radius: narrow.radius,
        minimum: narrow.minimum,
    };
    let mut session = frame.session(&request, workspace.budget)?;
    while let Some(mut batch) = session.next_batch()? {
        stats.candidate_records = stats
            .candidate_records
            .checked_add(u64::from(batch.cells()))
            .ok_or(ResourceError::SizeOverflow)?;
        batch.decide(|decision, budget| match decision {
            ambiguity_gpu::Decision::Contact { source, partner } => {
                narrow.emit((source, partner), budget, stats, callback)
            }
            ambiguity_gpu::Decision::Fallback {
                source,
                partner,
                cells,
            } => {
                if narrow.pair((source, partner), cells, budget, stats)? {
                    narrow.emit((source, partner), budget, stats, callback)?;
                }
                Ok(())
            }
        })?;
    }
    Ok(())
}
fn classify(
    curve: &ambiguity_contact::PreparedCurve<'_>,
    minimum: f64,
    stats: &mut Statistics,
) -> Result<(), ResourceError> {
    let length = curve.total_length();
    let counter = if length == 0.0 {
        &mut stats.constant_paths
    } else if length < minimum {
        &mut stats.short_paths
    } else {
        return Ok(());
    };
    *counter = counter.checked_add(1).ok_or(ResourceError::SizeOverflow)?;
    Ok(())
}

/// Frame geometry shared by every per-source narrow phase.
struct Narrow<'p, 'a> {
    ordered: &'p [Path<'a>],
    prepared: &'p [ambiguity_contact::PreparedCurve<'a>],
    radius: f64,
    minimum: f64,
}
impl Narrow<'_, '_> {
    /// Stream sorted, deduplicated candidates of one source by increasing partner.
    fn source<E, T: CellRecord>(
        &self,
        source: usize,
        candidates: &mut [T],
        budget: &mut Budget,
        stats: &mut Statistics,
        callback: &mut impl FnMut(u32, u32, &mut Budget) -> Result<(), E>,
    ) -> Result<(), Error<E>> {
        let mut start = 0;
        while let Some(first) = candidates.get(start) {
            let partner = first.partner();
            let mut end = start;
            while candidates
                .get(end)
                .is_some_and(|entry| entry.partner() == partner)
            {
                budget.charge(1)?;
                end = end.checked_add(1).ok_or(ResourceError::SizeOverflow)?;
            }
            let pair = candidates
                .get_mut(start..end)
                .ok_or(ResourceError::InvalidIndex)?;
            if self.pair((source, partner), pair, budget, stats)? {
                self.emit((source, partner), budget, stats, callback)?;
            }
            start = end;
        }
        Ok(())
    }
    /// The exact CPU route of one complete pair run, shared by CPU enumeration
    /// and GPU fallback: the participation bound first (source side, then the
    /// partner side after its in-place partner sort), then the contact kernel
    /// only when participation remains possible. Returns whether they contact.
    fn pair<E, T: CellRecord>(
        &self,
        (source, partner): (usize, usize),
        cells: &mut [T],
        budget: &mut Budget,
        stats: &mut Statistics,
    ) -> Result<bool, Error<E>> {
        let path = self
            .ordered
            .get(source)
            .ok_or(ResourceError::InvalidIndex)?;
        let other = self
            .ordered
            .get(partner)
            .ok_or(ResourceError::InvalidIndex)?;
        if !participation_possible(
            cells,
            (source, partner),
            self.prepared,
            self.minimum,
            budget,
        )? {
            return Ok(false);
        }
        tick(budget, &mut stats.narrow_pairs)?;
        ambiguity_contact::contact_prepared_candidates(
            self.prepared
                .get(source)
                .ok_or(ResourceError::InvalidIndex)?,
            self.prepared
                .get(partner)
                .ok_or(ResourceError::InvalidIndex)?,
            self.radius,
            self.minimum,
            cells.iter().map(CellRecord::cell),
            budget,
        )
        .map_err(|error| Error::Contact {
            source: path.id,
            partner: other.id,
            error,
        })
    }
    /// The one canonical contact emission: count it, then call the callback once.
    fn emit<E>(
        &self,
        (source, partner): (usize, usize),
        budget: &mut Budget,
        stats: &mut Statistics,
        callback: &mut impl FnMut(u32, u32, &mut Budget) -> Result<(), E>,
    ) -> Result<(), Error<E>> {
        let path = self
            .ordered
            .get(source)
            .ok_or(ResourceError::InvalidIndex)?;
        let other = self
            .ordered
            .get(partner)
            .ok_or(ResourceError::InvalidIndex)?;
        tick(budget, &mut stats.contacts)?;
        callback(path.id, other.id, budget).map_err(|error| Error::Callback {
            source: path.id,
            partner: other.id,
            error,
        })
    }
}

fn build<E>(
    ordered: &[Path<'_>],
    prepared: &[ambiguity_contact::PreparedCurve<'_>],
    minimum: f64,
    piece_length: f64,
    workspace: &mut Workspace<'_>,
    stats: &mut Statistics,
) -> Result<Vec<Piece>, Error<E>> {
    let mut eligible = workspace.vector(ordered.len(), false)?;
    let mut count = 0_usize;
    for (edge, path) in ordered.iter().enumerate() {
        let first = path.points.first().ok_or(Error::InvalidPath(path.id))?;
        let length = prepared
            .get(edge)
            .ok_or(ResourceError::InvalidIndex)?
            .total_length();
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
        }
        eligible.push(edge);
        let mut previous = first.map(f64::from);
        for point in path.points {
            workspace.budget.charge(1)?;
            let current = point.map(f64::from);
            if (current[0] - previous[0]).hypot(current[1] - previous[1]) <= 0.0 {
                continue;
            }
            count = count
                .checked_add(count_pieces(
                    bounds(edge, 0, previous, current),
                    piece_length,
                    workspace.budget,
                    stats,
                )?)
                .ok_or(ResourceError::SizeOverflow)?;
            previous = current;
        }
    }
    let mut index = workspace.vector(count, false)?;
    let mut index_capacity = count;
    for edge in eligible {
        let path = ordered.get(edge).ok_or(ResourceError::InvalidIndex)?;
        let first = path.points.first().ok_or(ResourceError::InvalidCurve)?;
        let mut previous = first.map(f64::from);
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
                &mut index,
                &mut index_capacity,
                workspace,
                stats,
            )?;
            segment = segment.checked_add(1).ok_or(ResourceError::SizeOverflow)?;
            previous = current;
        }
    }
    Ok(index)
}

#[derive(Clone, Copy)]
struct Node {
    low: [f64; 2],
    high: [f64; 2],
    maximum_edge: usize,
    start: usize,
    end: usize,
    right: usize,
}
const LEAF_SIZE: usize = 16;
fn node_count(count: usize) -> Result<usize, ResourceError> {
    if count <= LEAF_SIZE {
        return Ok(usize::from(count > 0));
    }
    // At the last possibly mixed level, balanced ranges have q or q+1
    // elements. Only ranges larger than LEAF_SIZE split once more.
    let groups = 1_usize
        .checked_shl((count / LEAF_SIZE).ilog2())
        .ok_or(ResourceError::SizeOverflow)?;
    let quotient = count
        .checked_div(groups)
        .ok_or(ResourceError::SizeOverflow)?;
    let extra = if quotient > LEAF_SIZE {
        groups
    } else {
        count
            .checked_rem(groups)
            .ok_or(ResourceError::SizeOverflow)?
    };
    groups
        .checked_add(extra)
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_sub(1))
        .ok_or(ResourceError::SizeOverflow)
}
fn build_tree(
    pieces: &mut [Piece],
    offset: usize,
    nodes: &mut Vec<Node>,
    budget: &mut Budget,
    stats: &mut Statistics,
) -> Result<usize, ResourceError> {
    let first = pieces.first().ok_or(ResourceError::InvalidIndex)?;
    let mut node = Node {
        low: first.low,
        high: first.high,
        maximum_edge: first.edge,
        start: offset,
        end: offset
            .checked_add(pieces.len())
            .ok_or(ResourceError::SizeOverflow)?,
        right: 0,
    };
    for piece in pieces.iter() {
        tick(budget, &mut stats.entries_visited)?;
        node.low = [node.low[0].min(piece.low[0]), node.low[1].min(piece.low[1])];
        node.high = [
            node.high[0].max(piece.high[0]),
            node.high[1].max(piece.high[1]),
        ];
        node.maximum_edge = node.maximum_edge.max(piece.edge);
    }
    let index = nodes.len();
    budget.charge(1)?;
    nodes.push(node);
    if pieces.len() > LEAF_SIZE {
        let axis = usize::from(node.high[1] - node.low[1] > node.high[0] - node.low[0]);
        let middle = pieces.len() / 2;
        // In-place partition comparisons, like sort comparisons, are separate from work units.
        pieces.select_nth_unstable_by(middle, |a, b| {
            let center = |p: &Piece| {
                if axis == 0 {
                    p.high[0].mul_add(0.5, p.low[0] * 0.5)
                } else {
                    p.high[1].mul_add(0.5, p.low[1] * 0.5)
                }
            };
            center(a)
                .total_cmp(&center(b))
                .then_with(|| a.edge.cmp(&b.edge))
                .then_with(|| a.segment.cmp(&b.segment))
        });
        let (left, right) = pieces.split_at_mut(middle);
        build_tree(left, offset, nodes, budget, stats)?;
        let right_index = build_tree(
            right,
            offset
                .checked_add(middle)
                .ok_or(ResourceError::SizeOverflow)?,
            nodes,
            budget,
            stats,
        )?;
        nodes
            .get_mut(index)
            .ok_or(ResourceError::InvalidIndex)?
            .right = right_index;
    }
    Ok(index)
}
fn overlaps(low: [f64; 2], high: [f64; 2], other_low: [f64; 2], other_high: [f64; 2]) -> bool {
    low[0] <= other_high[0]
        && high[0] >= other_low[0]
        && low[1] <= other_high[1]
        && high[1] >= other_low[1]
}
struct Query<'a> {
    index: &'a [Piece],
    nodes: &'a [Node],
    candidates: &'a mut Vec<Candidate>,
    capacity: &'a mut usize,
    stats: &'a mut Statistics,
}
impl Query<'_> {
    fn visit(
        &mut self,
        piece: Piece,
        radius: f64,
        node_index: usize,
        workspace: &mut Workspace<'_>,
    ) -> Result<(), ResourceError> {
        tick(workspace.budget, &mut self.stats.entries_visited)?;
        let node = self
            .nodes
            .get(node_index)
            .ok_or(ResourceError::InvalidIndex)?;
        let low = [
            (piece.low[0] - radius).next_down(),
            (piece.low[1] - radius).next_down(),
        ];
        let high = [
            (piece.high[0] + radius).next_up(),
            (piece.high[1] + radius).next_up(),
        ];
        if node.maximum_edge <= piece.edge || !overlaps(low, high, node.low, node.high) {
            return Ok(());
        }
        if node.right != 0 {
            let right = node.right;
            self.visit(
                piece,
                radius,
                node_index
                    .checked_add(1)
                    .ok_or(ResourceError::SizeOverflow)?,
                workspace,
            )?;
            return self.visit(piece, radius, right, workspace);
        }
        for target in self
            .index
            .get(node.start..node.end)
            .ok_or(ResourceError::InvalidIndex)?
        {
            tick(workspace.budget, &mut self.stats.entries_visited)?;
            if target.edge <= piece.edge {
                continue;
            }
            tick(workspace.budget, &mut self.stats.candidate_examinations)?;
            if !overlaps(low, high, target.low, target.high) {
                continue;
            }
            tick(workspace.budget, &mut self.stats.candidate_records)?;
            workspace.push(
                self.candidates,
                self.capacity,
                Candidate {
                    partner: target.edge,
                    cell: (piece.segment, target.segment),
                },
                true,
            )?;
        }
        Ok(())
    }
    fn split_query(
        &mut self,
        piece: Piece,
        parameters: (f64, f64),
        workspace: &mut Workspace<'_>,
    ) -> Result<(), ResourceError> {
        tick(workspace.budget, &mut self.stats.entries_visited)?;
        if let Some((left, right)) = split(piece, parameters.1)? {
            self.split_query(left, parameters, workspace)?;
            self.split_query(right, parameters, workspace)
        } else if self.nodes.is_empty() {
            Ok(())
        } else {
            self.visit(piece, parameters.0, 0, workspace)
        }
    }
}
fn collect(
    source: (usize, &[[f32; 2]]),
    index: (&[Piece], &[Node]),
    parameters: (f64, f64),
    candidates: &mut Vec<Candidate>,
    capacity: &mut usize,
    workspace: &mut Workspace<'_>,
    stats: &mut Statistics,
) -> Result<(), ResourceError> {
    candidates.clear();
    let mut query = Query {
        index: index.0,
        nodes: index.1,
        candidates,
        capacity,
        stats,
    };
    let Some(first) = source.1.first() else {
        return Err(ResourceError::InvalidCurve);
    };
    let mut previous = first.map(f64::from);
    let mut segment = 0;
    for point in source.1 {
        workspace.budget.charge(1)?;
        let current = point.map(f64::from);
        if (current[0] - previous[0]).hypot(current[1] - previous[1]) <= 0.0 {
            continue;
        }
        query.split_query(
            bounds(source.0, segment, previous, current),
            parameters,
            workspace,
        )?;
        previous = current;
        segment = segment.checked_add(1).ok_or(ResourceError::SizeOverflow)?;
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

fn prepare_paths<'a, E>(
    paths: &[Path<'a>],
    workspace: &mut Workspace<'_>,
) -> Result<Vec<ambiguity_contact::PreparedCurve<'a>>, Error<E>> {
    let mut prepared = workspace.vector(paths.len(), false)?;
    for path in paths {
        let curve = ambiguity_contact::PreparedCurve::prepare(path.points, workspace.budget)
            .map_err(|error| match error {
                ambiguity_contact::Error::Budget(ResourceError::InvalidCurve) => {
                    Error::InvalidPath(path.id)
                }
                ambiguity_contact::Error::Budget(error) => Error::from(error),
                error => Error::Contact {
                    source: path.id,
                    partner: path.id,
                    error,
                },
            })?;
        workspace.bytes = workspace
            .bytes
            .checked_add(curve.workspace_bytes())
            .ok_or(ResourceError::SizeOverflow)?;
        prepared.push(curve);
    }
    Ok(prepared)
}
// A component's projection union cannot exceed the sum of lengths of all
// participating original segments. Count each distinct segment once. Round each
// addition upward, and add one upward step per candidate to cover rounding in
// the narrow phase's interval-union multiply-adds. This is rejection only.
fn participation_possible<T: CellRecord>(
    candidates: &mut [T],
    edges: (usize, usize),
    prepared: &[ambiguity_contact::PreparedCurve<'_>],
    minimum: f64,
    budget: &mut Budget,
) -> Result<bool, ResourceError> {
    let source = prepared.get(edges.0).ok_or(ResourceError::InvalidIndex)?;
    let partner = prepared.get(edges.1).ok_or(ResourceError::InvalidIndex)?;
    if !participation_bound(
        candidates.iter().map(|c| c.cell().0),
        source,
        minimum,
        budget,
    )? {
        return Ok(false);
    }
    candidates.sort_unstable_by_key(|c| (c.cell().1, c.cell().0));
    participation_bound(
        candidates.iter().map(|c| c.cell().1),
        partner,
        minimum,
        budget,
    )
}
fn participation_bound(
    segments: impl Iterator<Item = usize>,
    curve: &ambiguity_contact::PreparedCurve<'_>,
    minimum: f64,
    budget: &mut Budget,
) -> Result<bool, ResourceError> {
    let mut previous = None;
    let mut total = 0.0_f64;
    let mut count = 0_usize;
    for segment in segments {
        budget.charge(1)?;
        if previous != Some(segment) {
            let length = curve
                .segment_length(segment)
                .map_err(|_| ResourceError::InvalidIndex)?;
            total = (total + length).next_up();
            if total >= minimum {
                return Ok(true);
            }
            previous = Some(segment);
        }
        count = count.checked_add(1).ok_or(ResourceError::SizeOverflow)?;
    }
    for _ in 0..count {
        budget.charge(1)?;
        total = total.next_up();
        if total >= minimum {
            return Ok(true);
        }
    }
    Ok(false)
}

fn ordered_paths<'a, E>(
    paths: &[Path<'a>],
    workspace: &mut Workspace<'_>,
) -> Result<Vec<Path<'a>>, Error<E>> {
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
    Ok(ordered)
}

fn count_pieces(
    piece: Piece,
    maximum: f64,
    budget: &mut Budget,
    stats: &mut Statistics,
) -> Result<usize, ResourceError> {
    tick(budget, &mut stats.entries_visited)?;
    if let Some((left, right)) = split(piece, maximum)? {
        count_pieces(left, maximum, budget, stats)?
            .checked_add(count_pieces(right, maximum, budget, stats)?)
            .ok_or(ResourceError::SizeOverflow)
    } else {
        Ok(1)
    }
}
