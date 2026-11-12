//! Continuous endpoint-constrained weak Frechet distance in input coordinates.
use std::{cmp::Ordering, collections::BinaryHeap, fmt, mem::size_of};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    InvalidCurve,
    InvalidScale,
    WorkLimit,
    WorkspaceLimit,
    RecordLimit,
    InvalidRelease,
    SizeOverflow,
    Allocation,
    InvalidIndex,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidCurve => "weak Frechet requires nonempty finite curves",
            Self::InvalidScale => "evaluation tolerances produce nonfinite normalized distances",
            Self::WorkLimit => "evaluation work limit exceeded",
            Self::RecordLimit => "evaluation stored pair record limit exceeded",
            Self::WorkspaceLimit => "evaluation requested workspace limit exceeded",
            Self::InvalidRelease => "workspace release exceeds outstanding reservations",
            Self::SizeOverflow => "evaluation size arithmetic overflow",
            Self::Allocation => "evaluation workspace allocation failed",
            Self::InvalidIndex => "invalid weak Frechet cell index",
        })
    }
}
impl std::error::Error for Error {}

/// Disjoint classifications of existing charged work units, not elapsed time.
#[derive(Debug, Clone, Copy)]
pub enum WorkCategory {
    Other,
    WeakCellVisits,
    HeapOperations,
    SimplifierPointVisits,
}

#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct WorkBreakdown {
    pub other: u64,
    pub weak_cell_visits: u64,
    pub heap_operations: u64,
    pub simplifier_point_visits: u64,
}
impl WorkBreakdown {
    const ZERO: Self = Self {
        other: 0,
        weak_cell_visits: 0,
        heap_operations: 0,
        simplifier_point_visits: 0,
    };
}

#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct Statistics {
    pub work_consumed: u64,
    pub work_breakdown: WorkBreakdown,
    pub workspace_reserved: usize,
    pub workspace_peak: usize,
    pub records_reserved: usize,
    pub records_peak: usize,
}
/// A sequential ledger. Reservations made directly here are persistent; temporary
/// storage reserves through a `Workspace`, which releases its own counters.
#[derive(Debug)]
pub struct Budget {
    work_limit: u64,
    workspace_limit: usize,
    record_limit: usize,
    statistics: Statistics,
}
impl Budget {
    /// Begin a preparation or snapshot phase.
    /// Outstanding storage reservations and lifetime peaks remain unchanged.
    pub const fn begin_phase(&mut self) {
        self.statistics.work_consumed = 0;
        self.statistics.work_breakdown = WorkBreakdown::ZERO;
    }
    #[must_use]
    pub const fn new(work: u64, workspace: usize) -> Self {
        Self {
            work_limit: work,
            workspace_limit: workspace,
            record_limit: usize::MAX,
            statistics: Statistics {
                work_consumed: 0,
                work_breakdown: WorkBreakdown::ZERO,
                workspace_reserved: 0,
                workspace_peak: 0,
                records_reserved: 0,
                records_peak: 0,
            },
        }
    }
    /// Construct a ledger with a global stored-pair capacity limit.
    #[must_use]
    pub const fn with_record_limit(work: u64, workspace: usize, records: usize) -> Self {
        let mut budget = Self::new(work, workspace);
        budget.record_limit = records;
        budget
    }
    /// Reserve requested record capacity before allocating its storage.
    /// # Errors
    /// Returns `RecordLimit` without changing the ledger on failure.
    pub fn reserve_records(&mut self, records: usize) -> Result<(), Error> {
        reserve_capacity(
            &mut self.statistics.records_reserved,
            &mut self.statistics.records_peak,
            records,
            self.record_limit,
            Error::RecordLimit,
        )
    }
    /// Release capacity when its owner drops the corresponding storage.
    /// # Errors
    /// Rejects a release larger than outstanding reservations.
    pub fn release_records(&mut self, records: usize) -> Result<(), Error> {
        self.statistics.records_reserved = self
            .statistics
            .records_reserved
            .checked_sub(records)
            .ok_or(Error::InvalidRelease)?;
        Ok(())
    }
    #[must_use]
    pub const fn statistics(&self) -> Statistics {
        self.statistics
    }
    /// # Errors
    /// Returns `WorkLimit` without changing the ledger if the limit would be exceeded.
    pub fn charge(&mut self, units: u64) -> Result<(), Error> {
        self.charge_classified(units, WorkCategory::Other)
    }
    /// Charge existing work to one disjoint category before its operation.
    /// # Errors
    /// Returns `WorkLimit` without changing any counter on failure.
    pub fn charge_classified(&mut self, units: u64, category: WorkCategory) -> Result<(), Error> {
        let total = self
            .statistics
            .work_consumed
            .checked_add(units)
            .ok_or(Error::WorkLimit)?;
        if total > self.work_limit {
            return Err(Error::WorkLimit);
        }
        let counter = match category {
            WorkCategory::Other => &mut self.statistics.work_breakdown.other,
            WorkCategory::WeakCellVisits => &mut self.statistics.work_breakdown.weak_cell_visits,
            WorkCategory::HeapOperations => &mut self.statistics.work_breakdown.heap_operations,
            WorkCategory::SimplifierPointVisits => {
                &mut self.statistics.work_breakdown.simplifier_point_visits
            }
        };
        let classified = counter.checked_add(units).ok_or(Error::WorkLimit)?;
        *counter = classified;
        self.statistics.work_consumed = total;
        Ok(())
    }
    /// # Errors
    /// Returns `WorkspaceLimit` without changing outstanding reservations on failure.
    pub fn reserve(&mut self, bytes: usize) -> Result<(), Error> {
        reserve_capacity(
            &mut self.statistics.workspace_reserved,
            &mut self.statistics.workspace_peak,
            bytes,
            self.workspace_limit,
            Error::WorkspaceLimit,
        )
    }
    /// # Errors
    /// Rejects releases larger than the outstanding ledger, leaving it unchanged.
    pub fn release(&mut self, bytes: usize) -> Result<(), Error> {
        self.statistics.workspace_reserved = self
            .statistics
            .workspace_reserved
            .checked_sub(bytes)
            .ok_or(Error::InvalidRelease)?;
        Ok(())
    }
}
/// Ledger counters occupied by a temporary vector's requested capacity.
#[derive(Clone, Copy)]
pub(crate) enum Storage {
    Bytes,
    /// Bytes plus one stored-pair record per element.
    Records,
}

/// Owns only its own temporary reservations on a borrowed ledger.
///
/// Each counter increase follows a successful reservation by this owner, so the
/// ledger always holds at least these amounts and the releases in `drop` cannot
/// fail. Vectors allocated here must drop before their workspace. `commit`
/// instead leaves successfully retained storage reserved on the ledger.
pub(crate) struct Workspace<'a> {
    pub(crate) budget: &'a mut Budget,
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
    pub(crate) const fn new(budget: &mut Budget) -> Workspace<'_> {
        Workspace {
            budget,
            bytes: 0,
            records: 0,
        }
    }
    /// A workspace whose reservations end before this one's.
    pub(crate) const fn nested(&mut self) -> Workspace<'_> {
        Workspace::new(self.budget)
    }
    pub(crate) fn charge(&mut self, units: u64) -> Result<(), Error> {
        self.budget.charge(units)
    }
    /// Transfer this owner's reservations to the ledger's persistent total.
    pub(crate) fn commit(mut self) {
        self.bytes = 0;
        self.records = 0;
    }
    pub(crate) fn vector<T>(&mut self, capacity: usize, storage: Storage) -> Result<Vec<T>, Error> {
        let bytes = capacity
            .checked_mul(size_of::<T>())
            .ok_or(Error::SizeOverflow)?;
        let total = self.bytes.checked_add(bytes).ok_or(Error::SizeOverflow)?;
        self.budget.reserve(bytes)?;
        self.bytes = total;
        if matches!(storage, Storage::Records) {
            let total = self
                .records
                .checked_add(capacity)
                .ok_or(Error::SizeOverflow)?;
            self.budget.reserve_records(capacity)?;
            self.records = total;
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(capacity)
            .map_err(|_| Error::Allocation)?;
        Ok(values)
    }
    /// Push into a vector with requested capacity `requested`, growing through a
    /// separate buffer so old and new capacities are reserved simultaneously.
    pub(crate) fn push<T>(
        &mut self,
        values: &mut Vec<T>,
        requested: &mut usize,
        value: T,
        storage: Storage,
    ) -> Result<(), Error> {
        if values.len() == *requested {
            let capacity = requested.checked_mul(2).ok_or(Error::SizeOverflow)?.max(8);
            let mut replacement = self.vector(capacity, storage)?;
            self.charge(u64::try_from(values.len()).map_err(|_| Error::SizeOverflow)?)?;
            replacement.append(values);
            *values = replacement;
            self.free::<T>(*requested, storage)?;
            *requested = capacity;
        }
        values.push(value);
        Ok(())
    }
    fn free<T>(&mut self, capacity: usize, storage: Storage) -> Result<(), Error> {
        let bytes = capacity
            .checked_mul(size_of::<T>())
            .ok_or(Error::SizeOverflow)?;
        self.budget.release(bytes)?;
        self.bytes = self.bytes.checked_sub(bytes).ok_or(Error::InvalidRelease)?;
        if matches!(storage, Storage::Records) {
            self.budget.release_records(capacity)?;
            self.records = self
                .records
                .checked_sub(capacity)
                .ok_or(Error::InvalidRelease)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub enum Orientation {
    Forward,
    Reversed,
}
#[derive(Debug, Clone, Copy)]
pub struct Distance {
    pub route: f64,
    pub endpoints: f64,
}
#[derive(Clone, Copy)]
struct Entry {
    cost: f64,
    cell: usize,
}
impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl Eq for Entry {}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            // With equal bottleneck costs, favor progress toward the final cell.
            .then_with(|| self.cell.cmp(&other.cell))
    }
}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
fn squared([ax, ay]: [f32; 2], [bx, by]: [f32; 2]) -> f64 {
    let dx = f64::from(ax) - f64::from(bx);
    let dy = f64::from(ay) - f64::from(by);
    dx.mul_add(dx, dy * dy)
}
fn boundary([px, py]: [f32; 2], [ax, ay]: [f32; 2], [bx, by]: [f32; 2]) -> f64 {
    let dx = f64::from(bx) - f64::from(ax);
    let dy = f64::from(by) - f64::from(ay);
    let x = f64::from(px) - f64::from(ax);
    let y = f64::from(py) - f64::from(ay);
    let length = dx.mul_add(dx, dy * dy);
    let t = if length > 0.0 {
        (x.mul_add(dx, y * dy) / length).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let rx = (-t).mul_add(dx, x);
    let ry = (-t).mul_add(dy, y);
    rx.mul_add(rx, ry * ry)
}
struct Curve<'a> {
    points: &'a [[f32; 2]],
    reversed: bool,
    segments: usize,
    constant: bool,
}
impl<'a> Curve<'a> {
    fn new(points: &'a [[f32; 2]], reversed: bool, budget: &mut Budget) -> Result<Self, Error> {
        budget.charge(u64::try_from(points.len()).map_err(|_| Error::SizeOverflow)?)?;
        let first = points.first().ok_or(Error::InvalidCurve)?;
        let mut constant = true;
        for point in points {
            if point.iter().any(|v| !v.is_finite()) {
                return Err(Error::InvalidCurve);
            }
            // Numeric equality intentionally identifies signed zeros as the same position.
            constant &= squared(*first, *point) <= 0.0;
        }
        Ok(Self {
            points,
            reversed,
            segments: points.len().checked_sub(1).ok_or(Error::InvalidCurve)?,
            constant,
        })
    }
    fn at(&self, index: usize) -> Result<[f32; 2], Error> {
        let index = if self.reversed {
            self.segments
                .checked_sub(index)
                .ok_or(Error::InvalidIndex)?
        } else {
            index
        };
        self.points.get(index).copied().ok_or(Error::InvalidIndex)
    }
}
fn point_distance(point: [f32; 2], curve: &Curve<'_>, budget: &mut Budget) -> Result<f64, Error> {
    budget.charge(u64::try_from(curve.points.len()).map_err(|_| Error::SizeOverflow)?)?;
    Ok(curve
        .points
        .iter()
        .fold(0.0_f64, |cost, &p| cost.max(squared(point, p))))
}
fn increment(value: usize) -> Result<usize, Error> {
    value.checked_add(1).ok_or(Error::SizeOverflow)
}
struct Grid {
    columns: usize,
    cells: usize,
    entries: usize,
    bytes: usize,
}
impl Grid {
    fn new(rows: usize, columns: usize) -> Result<Self, Error> {
        let cells = rows.checked_mul(columns).ok_or(Error::SizeOverflow)?;
        let entries = cells
            .checked_mul(4)
            .and_then(|n| n.checked_add(1))
            .ok_or(Error::SizeOverflow)?;
        let bytes = cells
            .checked_mul(size_of::<f64>())
            .and_then(|n| {
                entries
                    .checked_mul(size_of::<Entry>())
                    .and_then(|h| n.checked_add(h))
            })
            .ok_or(Error::SizeOverflow)?;
        Ok(Self {
            columns,
            cells,
            entries,
            bytes,
        })
    }
    fn index(&self, row: usize, column: usize) -> Result<usize, Error> {
        row.checked_mul(self.columns)
            .and_then(|n| n.checked_add(column))
            .filter(|&n| n < self.cells)
            .ok_or(Error::InvalidIndex)
    }
}
struct Search {
    labels: Vec<f64>,
    heap: BinaryHeap<Entry>,
}
impl Search {
    fn new(grid: &Grid, initial: f64, budget: &mut Budget) -> Result<Self, Error> {
        // Initialization touches every label; charge before allocation and initialization.
        budget.charge(u64::try_from(grid.cells).map_err(|_| Error::SizeOverflow)?)?;
        let mut labels = Vec::new();
        labels
            .try_reserve_exact(grid.cells)
            .map_err(|_| Error::Allocation)?;
        labels.resize(grid.cells, f64::INFINITY);
        let mut heap = BinaryHeap::new();
        heap.try_reserve_exact(grid.entries)
            .map_err(|_| Error::Allocation)?;
        budget.charge_classified(1, WorkCategory::HeapOperations)?;
        *labels.first_mut().ok_or(Error::InvalidIndex)? = initial;
        heap.push(Entry {
            cost: initial,
            cell: 0,
        });
        Ok(Self { labels, heap })
    }
    fn relax(&mut self, cell: usize, cost: f64, budget: &mut Budget) -> Result<(), Error> {
        let label = self.labels.get_mut(cell).ok_or(Error::InvalidIndex)?;
        if cost < *label {
            budget.charge_classified(1, WorkCategory::HeapOperations)?;
            *label = cost;
            self.heap.push(Entry { cost, cell });
        }
        Ok(())
    }
}
fn traverse(
    a: &Curve<'_>,
    b: &Curve<'_>,
    initial: f64,
    grid: &Grid,
    budget: &mut Budget,
) -> Result<f64, Error> {
    let mut search = Search::new(grid, initial, budget)?;
    while !search.heap.is_empty() {
        budget.charge_classified(1, WorkCategory::HeapOperations)?;
        let Some(Entry { cost, cell }) = search.heap.pop() else {
            break;
        };
        if cost
            .total_cmp(search.labels.get(cell).ok_or(Error::InvalidIndex)?)
            .is_ne()
        {
            continue;
        }
        budget.charge_classified(1, WorkCategory::WeakCellVisits)?;
        if increment(cell)? == grid.cells {
            return Ok(cost);
        }
        let row = cell.checked_div(grid.columns).ok_or(Error::InvalidIndex)?;
        let column = cell.checked_rem(grid.columns).ok_or(Error::InvalidIndex)?;
        let neighbors = [
            (row.checked_sub(1), Some(column)),
            (row.checked_add(1), Some(column)),
            (Some(row), column.checked_sub(1)),
            (Some(row), column.checked_add(1)),
        ];
        for (r, c) in neighbors {
            let (Some(r), Some(c)) = (r, c) else { continue };
            if r >= a.segments || c >= b.segments {
                continue;
            }
            budget.charge(1)?;
            let weight = if r == row {
                boundary(b.at(column.max(c))?, a.at(row)?, a.at(increment(row)?)?)
            } else {
                boundary(a.at(row.max(r))?, b.at(column)?, b.at(increment(column)?)?)
            };
            search.relax(grid.index(r, c)?, cost.max(weight), budget)?;
        }
    }
    Err(Error::InvalidIndex)
}
// A one-segment grid is a chain, so every interior boundary is mandatory.
fn chain(a: &Curve<'_>, b: &Curve<'_>, initial: f64, budget: &mut Budget) -> Result<f64, Error> {
    let (line, other) = if a.segments == 1 { (a, b) } else { (b, a) };
    let mut cost = initial;
    for i in 1..other.segments {
        budget.charge(1)?;
        cost = cost.max(boundary(other.at(i)?, line.at(0)?, line.at(1)?));
    }
    Ok(cost)
}

// Each step crosses exactly one shared cell boundary. If all weights are at
// most the endpoint lower bound, this path certifies the exact minimax cost.
fn staircase(
    a: &Curve<'_>,
    b: &Curve<'_>,
    initial: f64,
    budget: &mut Budget,
) -> Result<bool, Error> {
    let mut row = 0;
    let mut column = 0;
    while increment(row)? < a.segments || increment(column)? < b.segments {
        budget.charge(1)?;
        let next_row = increment(row)?;
        let next_column = increment(column)?;
        let advance_row = next_row < a.segments
            && (next_column == b.segments
                || next_row
                    .checked_mul(b.segments)
                    .ok_or(Error::SizeOverflow)?
                    <= next_column
                        .checked_mul(a.segments)
                        .ok_or(Error::SizeOverflow)?);
        let weight = if advance_row {
            let weight = boundary(a.at(next_row)?, b.at(column)?, b.at(next_column)?);
            row = next_row;
            weight
        } else {
            let weight = boundary(b.at(next_column)?, a.at(row)?, a.at(next_row)?);
            column = next_column;
            weight
        };
        if weight > initial {
            return Ok(false);
        }
    }
    Ok(true)
}
/// Backtracking is allowed; closed routes retain their stored starts.
/// Squared f64 costs preserve input units after the final square root.
/// # Errors
/// Rejects empty/nonfinite input, exhausted budgets, size overflow and allocation failure.
pub fn distance(
    a: &[[f32; 2]],
    b: &[[f32; 2]],
    orientation: Orientation,
    budget: &mut Budget,
) -> Result<Distance, Error> {
    let a = Curve::new(a, false, budget)?;
    let b = Curve::new(b, matches!(orientation, Orientation::Reversed), budget)?;
    budget.charge(2)?;
    let endpoints = squared(a.at(0)?, b.at(0)?).max(squared(a.at(a.segments)?, b.at(b.segments)?));
    let cost = if a.constant {
        point_distance(a.at(0)?, &b, budget)?
    } else if b.constant {
        point_distance(b.at(0)?, &a, budget)?
    } else if a.segments == 1 && b.segments == 1 {
        endpoints
    } else if a.segments == 1 || b.segments == 1 {
        chain(&a, &b, endpoints, budget)?
    } else {
        // Equal vertex counts provide a strong synchronized correspondence bound.
        let mut upper = f64::INFINITY;
        if a.points.len() == b.points.len() {
            budget.charge(u64::try_from(a.points.len()).map_err(|_| Error::SizeOverflow)?)?;
            upper = 0.0_f64;
            for i in 0..a.points.len() {
                upper = upper.max(squared(a.at(i)?, b.at(i)?));
            }
        }
        if upper <= endpoints || staircase(&a, &b, endpoints, budget)? {
            endpoints
        } else {
            let grid = Grid::new(a.segments, b.segments)?;
            budget.reserve(grid.bytes)?;
            let result = traverse(&a, &b, endpoints, &grid, budget);
            budget.release(grid.bytes)?;
            result?
        }
    };
    Ok(Distance {
        route: cost.sqrt(),
        endpoints: endpoints.sqrt(),
    })
}

fn reserve_capacity(
    current: &mut usize,
    peak: &mut usize,
    added: usize,
    limit: usize,
    error: Error,
) -> Result<(), Error> {
    let total = current.checked_add(added).ok_or(error)?;
    if total > limit {
        return Err(error);
    }
    *current = total;
    *peak = (*peak).max(total);
    Ok(())
}
