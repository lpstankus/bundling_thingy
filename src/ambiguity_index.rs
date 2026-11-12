//! Conservative segment bounds and streaming whole-drawing contact enumeration.
//!
//! Each positive segment has one rounded AABB entry and one rounded query box.
//! Sort comparisons are outside the work counter. Requested capacities include
//! simultaneous growth buffers, excluding allocator overhead and borrowed points.
//! The index releases only its own temporary reservations on every exit.
//! Callback reservations are the callback owner's.
use crate::{
    ambiguity::Cause,
    ambiguity_contact::{self, PreparedCurve},
    geometry::Snapshot,
    weak_frechet::{Budget, Error as ResourceError, Storage, Workspace},
};

#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct Statistics {
    pub constant_paths: usize,
    /// Positive paths with a rounded total below the minimum; they remain indexed.
    pub short_paths: usize,
    /// Positive-segment AABB entries.
    pub pieces_generated: u64,
    pub entries_visited: u64,
    pub candidate_examinations: u64,
    pub candidate_records: u64,
    pub narrow_pairs: u64,
    pub contacts: u64,
}
/// An enumeration failure with the source/partner edge IDs it occurred for.
pub struct Failure {
    pub pair: Option<(u32, u32)>,
    pub cause: Cause,
}
impl From<Cause> for Failure {
    fn from(cause: Cause) -> Self {
        Self { pair: None, cause }
    }
}
impl From<ResourceError> for Failure {
    fn from(error: ResourceError) -> Self {
        Cause::from(error).into()
    }
}
#[derive(Clone, Copy)]
struct Path<'a> {
    id: u32,
    points: &'a [[f32; 2]],
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
/// Stream each qualifying unordered pair once, in stable ID order.
///
/// Edges are visited in ID order, independently of snapshot storage order.
/// A balanced AABB hierarchy prunes both coordinates. Dense contact remains quadratic.
/// Callback reservations survive success and failure on the shared ledger.
/// # Errors
/// Global resource exhaustion, narrow-phase numerical errors and callback
/// failures abort without a successful result.
pub fn enumerate(
    snapshot: &Snapshot<'_>,
    radius: f64,
    minimum: f64,
    budget: &mut Budget,
    mut callback: impl FnMut(u32, u32, &mut Budget) -> Result<(), Cause>,
) -> Result<Statistics, Failure> {
    let mut stats = Statistics::default();
    let mut workspace = Workspace::new(budget);
    let ordered = ordered_paths(snapshot, &mut workspace)?;
    let prepared = prepare_paths(&ordered, &mut workspace)?;
    let mut index = build(&prepared, minimum, &mut workspace, &mut stats)?;
    let mut hierarchy = workspace.vector(node_count(index.len())?, Storage::Bytes)?;
    if !index.is_empty() {
        build_tree(&mut index, 0, &mut hierarchy, workspace.budget, &mut stats)?;
    }
    let mut candidates = Vec::<Candidate>::new();
    let mut capacity = 0;
    for (source, path) in ordered.iter().enumerate() {
        candidates.clear();
        let curve = prepared.get(source).ok_or(ResourceError::InvalidIndex)?;
        if !hierarchy.is_empty() {
            let mut query = Query {
                index: &index,
                nodes: &hierarchy,
                candidates: &mut candidates,
                capacity: &mut capacity,
                stats: &mut stats,
            };
            for segment in 0..curve.segment_count() {
                let endpoints = curve.segment(segment)?;
                query.visit(
                    bounds(source, segment, endpoints.start, endpoints.end),
                    radius,
                    0,
                    &mut workspace,
                )?;
            }
        }
        // One stored entry and one query per positive segment emit each cell once.
        candidates.sort_unstable();
        let mut start = 0;
        while let Some(first) = candidates.get(start) {
            let partner = first.partner;
            let mut end = start;
            while candidates
                .get(end)
                .is_some_and(|entry| entry.partner == partner)
            {
                workspace.charge(1)?;
                end = end.checked_add(1).ok_or(ResourceError::SizeOverflow)?;
            }
            let other = ordered.get(partner).ok_or(ResourceError::InvalidIndex)?;
            let pair = candidates
                .get_mut(start..end)
                .ok_or(ResourceError::InvalidIndex)?;
            if !participation_possible(
                pair,
                (source, partner),
                &prepared,
                minimum,
                workspace.budget,
            )? {
                start = end;
                continue;
            }
            tick(workspace.budget, &mut stats.narrow_pairs)?;
            let cells = candidates
                .get(start..end)
                .ok_or(ResourceError::InvalidIndex)?
                .iter()
                .map(|entry| entry.cell);
            let located = |cause| Failure {
                pair: Some((path.id, other.id)),
                cause,
            };
            let contact = ambiguity_contact::contact(
                curve,
                prepared.get(partner).ok_or(ResourceError::InvalidIndex)?,
                radius,
                minimum,
                cells,
                &mut workspace.nested(),
            )
            .map_err(located)?;
            if contact {
                tick(workspace.budget, &mut stats.contacts)?;
                callback(path.id, other.id, workspace.budget).map_err(located)?;
            }
            start = end;
        }
    }
    Ok(stats)
}

fn build(
    prepared: &[PreparedCurve<'_>],
    minimum: f64,
    workspace: &mut Workspace<'_>,
    stats: &mut Statistics,
) -> Result<Vec<Piece>, ResourceError> {
    let mut count = 0_usize;
    for curve in prepared {
        workspace.charge(1)?;
        if curve.segment_count() == 0 {
            stats.constant_paths = stats
                .constant_paths
                .checked_add(1)
                .ok_or(ResourceError::SizeOverflow)?;
            continue;
        }
        if curve.total_length() < minimum {
            stats.short_paths = stats
                .short_paths
                .checked_add(1)
                .ok_or(ResourceError::SizeOverflow)?;
        }
        count = count
            .checked_add(curve.segment_count())
            .ok_or(ResourceError::SizeOverflow)?;
    }
    let mut index = workspace.vector(count, Storage::Bytes)?;
    for (edge, curve) in prepared.iter().enumerate() {
        for segment in 0..curve.segment_count() {
            tick(workspace.budget, &mut stats.pieces_generated)?;
            let endpoints = curve
                .segment(segment)
                .map_err(|_| ResourceError::InvalidIndex)?;
            index.push(bounds(edge, segment, endpoints.start, endpoints.end));
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
                Storage::Records,
            )?;
        }
        Ok(())
    }
}

fn prepare_paths<'a>(
    paths: &[Path<'a>],
    workspace: &mut Workspace<'_>,
) -> Result<Vec<PreparedCurve<'a>>, Failure> {
    let mut prepared = workspace.vector(paths.len(), Storage::Bytes)?;
    for path in paths {
        let curve = PreparedCurve::prepare(path.points, workspace).map_err(|cause| Failure {
            pair: Some((path.id, path.id)),
            cause,
        })?;
        prepared.push(curve);
    }
    Ok(prepared)
}
// A component's projection union cannot exceed the sum of lengths of all
// participating original segments. Count each distinct segment once. Round each
// addition upward, and add one upward step per candidate to cover rounding in
// the narrow phase's interval-union multiply-adds. This is rejection only.
fn participation_possible(
    candidates: &mut [Candidate],
    edges: (usize, usize),
    prepared: &[PreparedCurve<'_>],
    minimum: f64,
    budget: &mut Budget,
) -> Result<bool, ResourceError> {
    let source = prepared.get(edges.0).ok_or(ResourceError::InvalidIndex)?;
    let partner = prepared.get(edges.1).ok_or(ResourceError::InvalidIndex)?;
    if !participation_bound(candidates.iter().map(|c| c.cell.0), source, minimum, budget)? {
        return Ok(false);
    }
    candidates.sort_unstable_by_key(|c| (c.cell.1, c.cell.0));
    participation_bound(
        candidates.iter().map(|c| c.cell.1),
        partner,
        minimum,
        budget,
    )
}
fn participation_bound(
    segments: impl Iterator<Item = usize>,
    curve: &PreparedCurve<'_>,
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

// Snapshot storage order need not be ID order; this charged view restores it.
fn ordered_paths<'a>(
    snapshot: &'a Snapshot<'_>,
    workspace: &mut Workspace<'_>,
) -> Result<Vec<Path<'a>>, ResourceError> {
    let mut ordered = workspace.vector(snapshot.layout().edge_count(), Storage::Bytes)?;
    for (id, points) in snapshot.edges() {
        workspace.charge(1)?;
        ordered.push(Path { id, points });
    }
    ordered.sort_unstable_by_key(|path| path.id);
    Ok(ordered)
}
