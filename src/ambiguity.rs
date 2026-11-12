//! CPU ambiguity evaluation in original input units, with frozen baseline evidence.
//!
//! General RMS excess can exceed one.
//! Empty contact sets have zero ratios. Bounds describe original route distance
//! approximation under ordinary f64 arithmetic, without exact-real guarantees.
//! Canonical original order is frozen. Closed routes retain starts.
//! Contact measures connected continuous participation along both routes, which
//! can include shallow crossings and backtracking rather than a monotone match.
//! All owned heap storage is charged to the evaluator's exclusively owned ledger.
//! Fixed-capacity tables never grow, baseline evidence never evicts, and only
//! cached assessments evict. Index and kernel temporaries release after each frame.
pub use crate::{
    ambiguity_index::Statistics as IndexStatistics,
    weak_frechet::{Budget, Statistics},
};
use crate::{
    ambiguity_index::{self, Failure},
    ambiguity_originals::{Assessment, Config as Similarity, Originals},
    geometry::{Layout, Snapshot},
    weak_frechet::Error as Kernel,
};
use std::{fmt, mem::size_of, ptr};

#[derive(Debug, Clone, Copy)]
enum Phase {
    Configuration,
    OriginalPreparation,
    Baseline,
    Snapshot,
}

/// Distinguishable failure causes, reported with phase and edge-pair context.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Cause {
    /// Resource limits, size arithmetic, allocation and kernel input failures.
    Kernel(Kernel),
    InvalidConfig,
    LayoutMismatch,
    /// Contact geometry exceeds f64 arc precision or has inconsistent predicates.
    NumericalFailure,
    IncompleteCandidates,
    InvalidState,
}
impl From<Kernel> for Cause {
    fn from(error: Kernel) -> Self {
        Self::Kernel(error)
    }
}
impl fmt::Display for Cause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kernel(error) => error.fmt(f),
            Self::InvalidConfig => f.write_str("invalid configuration: thresholds must be finite and positive, sigma and radius finite and nonnegative, and a positive radius must have a positive finite square"),
            Self::LayoutMismatch => f.write_str("snapshot layout differs from the evaluator's layout"),
            Self::NumericalFailure => f.write_str("contact geometry exceeds f64 arc precision or has inconsistent free-space predicates"),
            Self::IncompleteCandidates => f.write_str("contact candidates omit a nonempty neighboring cell"),
            Self::InvalidState => f.write_str("evaluator is invalid after a failed snapshot"),
        }
    }
}

#[derive(Debug)]
pub struct Error {
    phase: Phase,
    pair: Option<(u32, u32)>,
    cause: Cause,
}
impl Error {
    const fn new(phase: Phase, cause: Cause) -> Self {
        Self {
            phase,
            pair: None,
            cause,
        }
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ambiguity {:?}", self.phase)?;
        if let Some((source, partner)) = self.pair {
            write!(f, " edges {source}/{partner}")?;
        }
        write!(f, ": {}", self.cause)
    }
}
impl std::error::Error for Error {}

/// Thresholds in input units and fixed table capacities, before validation.
#[derive(Clone, Copy)]
pub struct Settings {
    pub route_tolerance: f64,
    pub endpoint_tolerance: f64,
    pub sigma: f64,
    pub radius: f64,
    pub minimum_contact: f64,
    /// Maximum retained incompatible baseline edge pairs.
    pub baseline_capacity: usize,
    /// Direct-mapped recomputable assessment slots. Zero disables caching.
    pub cache_capacity: usize,
}
pub struct Config {
    similarity: Similarity,
    radius: f64,
    minimum_contact: f64,
    baseline_capacity: usize,
    cache_capacity: usize,
}
impl Config {
    /// # Errors
    /// Rejects nonfinite or nonpositive route/endpoint tolerances and minimum
    /// contact, negative or nonfinite sigma/radius, and a positive radius whose
    /// square is nonfinite or zero. Zero capacities are valid.
    pub fn new(settings: Settings) -> Result<Self, Error> {
        let positive = |value: f64| value.is_finite() && value > 0.0;
        let nonnegative = |value: f64| value.is_finite() && value >= 0.0;
        let squared_radius = settings.radius * settings.radius;
        let radius_square_valid =
            squared_radius.is_finite() && (settings.radius == 0.0 || squared_radius > 0.0);
        let is_valid = positive(settings.route_tolerance)
            && positive(settings.endpoint_tolerance)
            && positive(settings.minimum_contact)
            && nonnegative(settings.sigma)
            && nonnegative(settings.radius)
            && radius_square_valid;
        if !is_valid {
            return Err(Error::new(Phase::Configuration, Cause::InvalidConfig));
        }
        Ok(Self {
            similarity: Similarity {
                route_tolerance: settings.route_tolerance,
                endpoint_tolerance: settings.endpoint_tolerance,
                sigma: settings.sigma,
            },
            radius: settings.radius,
            minimum_contact: settings.minimum_contact,
            baseline_capacity: settings.baseline_capacity,
            cache_capacity: settings.cache_capacity,
        })
    }
}
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct Bounds {
    pub lower: f64,
    pub upper: f64,
}
impl Bounds {
    fn add(&mut self, value: Self) -> Result<(), Kernel> {
        self.lower += value.lower;
        self.upper += value.upper;
        if !self.lower.is_finite() || !self.upper.is_finite() {
            return Err(Kernel::InvalidScale);
        }
        Ok(())
    }
    fn excess(lower: f64, upper: f64, tolerance: f64) -> Self {
        Self {
            lower: (lower - tolerance).max(0.0).powi(2),
            upper: (upper - tolerance).max(0.0).powi(2),
        }
    }
}
#[derive(Debug, Default, Clone, Copy)]
pub struct View {
    pub squared_excess: Bounds,
    pub rms_excess: Bounds,
    pub ratio: Bounds,
    pub current: usize,
    pub baseline: usize,
    pub introduced: usize,
    pub resolved: usize,
    pub introduced_squared_excess: Bounds,
    pub resolved_squared_excess: Bounds,
}
#[derive(Debug, Default, Clone, Copy)]
pub struct Row {
    pub contacts: u64,
    pub collapsed: usize,
    pub general: View,
    pub resources: Statistics,
    pub index: IndexStatistics,
}
#[derive(Clone, Copy)]
struct Cached {
    key: (u32, u32),
    assessment: Assessment,
}
/// A frozen incompatible baseline pair.
struct Evidence {
    key: (u32, u32),
    general: Bounds,
    seen: bool,
}
fn vector<T>(capacity: usize) -> Result<Vec<T>, Kernel> {
    let mut v = Vec::new();
    v.try_reserve_exact(capacity)
        .map_err(|_| Kernel::Allocation)?;
    Ok(v)
}
fn route(lookup: &[(u32, usize)], id: u32) -> Result<usize, Kernel> {
    let index = lookup
        .binary_search_by_key(&id, |p| p.0)
        .map_err(|_| Kernel::InvalidIndex)?;
    Ok(lookup.get(index).ok_or(Kernel::InvalidIndex)?.1)
}

/// Owns canonical originals, preparations, tables and the ledger charged for them.
///
/// Every snapshot must decode against the construction `Layout`.
/// Failed frames permanently invalidate this evaluator.
pub struct Evaluator<'l> {
    layout: &'l Layout,
    config: Config,
    originals: Originals,
    lookup: Vec<(u32, usize)>,
    baseline: Vec<Evidence>,
    cache: Vec<Option<Cached>>,
    preparation: Statistics,
    baseline_row: Row,
    valid: bool,
    // Persistent reservations die with this exclusively owned ledger, not individually.
    budget: Budget,
}
impl<'l> Evaluator<'l> {
    /// Prepare canonical originals and evaluate the baseline snapshot.
    ///
    /// Reservations made before construction, such as report storage, remain on
    /// `budget`. Persistent storage reserves its full requested capacity first.
    /// # Errors
    /// A different baseline layout, exhausted resources or a failed baseline
    /// frame. No evaluator or partial evidence is returned.
    pub fn new(
        layout: &'l Layout,
        baseline: &Snapshot<'_>,
        config: Config,
        mut budget: Budget,
    ) -> Result<Self, Error> {
        let preparation_error = |cause| Error::new(Phase::OriginalPreparation, cause);
        if !ptr::eq(baseline.layout(), layout) {
            return Err(preparation_error(Cause::LayoutMismatch));
        }
        budget.begin_phase();
        let prepared = (|| -> Result<_, Kernel> {
            let originals = Originals::new(baseline, &mut budget)?;
            let (bytes, records) = storage_requirements(&config, originals.len())?;
            budget.reserve(bytes)?;
            budget.reserve_records(records)?;
            let mut lookup = vector(originals.len())?;
            for index in 0..originals.len() {
                budget.charge(1)?;
                lookup.push((originals.edge_id(index).ok_or(Kernel::InvalidIndex)?, index));
            }
            lookup.sort_unstable();
            let mut cache = vector(config.cache_capacity)?;
            budget
                .charge(u64::try_from(config.cache_capacity).map_err(|_| Kernel::SizeOverflow)?)?;
            cache.resize(config.cache_capacity, None);
            Ok((originals, lookup, cache, vector(config.baseline_capacity)?))
        })();
        let (originals, lookup, cache, evidence) =
            prepared.map_err(|error| preparation_error(error.into()))?;
        let mut evaluator = Self {
            layout,
            config,
            originals,
            lookup,
            baseline: evidence,
            cache,
            preparation: budget.statistics(),
            baseline_row: Row::default(),
            valid: false,
            budget,
        };
        evaluator.budget.begin_phase();
        evaluator.baseline_row = evaluator.frame(baseline, Phase::Baseline)?;
        evaluator.valid = true;
        Ok(evaluator)
    }
    #[must_use]
    pub const fn preparation_resources(&self) -> Statistics {
        self.preparation
    }
    /// Current ledger statistics. Persistent storage stays reserved until drop.
    #[must_use]
    pub const fn resources(&self) -> Statistics {
        self.budget.statistics()
    }
    #[must_use]
    pub const fn original_constant_count(&self) -> usize {
        self.originals.constant_route_count()
    }
    #[must_use]
    pub const fn closed_route_count(&self) -> usize {
        self.originals.closed_route_count()
    }
    #[must_use]
    pub const fn unique_route_count(&self) -> usize {
        self.originals.unique_route_count()
    }
    #[must_use]
    pub const fn baseline(&self) -> Row {
        self.baseline_row
    }
    /// Evaluate one snapshot of the construction layout.
    /// # Errors
    /// Any failure invalidates this evaluator. No partial row is returned.
    pub fn evaluate(&mut self, snapshot: &Snapshot<'_>) -> Result<Row, Error> {
        if !self.valid {
            return Err(Error::new(Phase::Snapshot, Cause::InvalidState));
        }
        self.valid = false;
        if !ptr::eq(snapshot.layout(), self.layout) {
            return Err(Error::new(Phase::Snapshot, Cause::LayoutMismatch));
        }
        self.budget.begin_phase();
        let row = self.frame(snapshot, Phase::Snapshot)?;
        self.valid = true;
        Ok(row)
    }
    fn frame(&mut self, snapshot: &Snapshot<'_>, phase: Phase) -> Result<Row, Error> {
        let located = |failure: Failure| Error {
            phase,
            pair: failure.pair,
            cause: failure.cause,
        };
        let is_baseline = matches!(phase, Phase::Baseline);
        for evidence in &mut self.baseline {
            self.budget
                .charge(1)
                .map_err(Failure::from)
                .map_err(located)?;
            evidence.seen = false;
        }
        let mut row = Row::default();
        let mut accumulator = Accumulator {
            originals: &mut self.originals,
            config: &self.config,
            lookup: &self.lookup,
            baseline: &mut self.baseline,
            cache: &mut self.cache,
            row: &mut row,
            is_baseline,
        };
        let stats = ambiguity_index::enumerate(
            snapshot,
            self.config.radius,
            self.config.minimum_contact,
            &mut self.budget,
            |a, b, budget| accumulator.contact(a, b, budget),
        )
        .map_err(located)?;
        row.index = stats;
        row.contacts = stats.contacts;
        row.collapsed = stats.constant_paths;
        self.resolve(&mut row, is_baseline)
            .map_err(Failure::from)
            .map_err(located)?;
        row.general.baseline = if is_baseline {
            row.general.current
        } else {
            self.baseline_row.general.current
        };
        row.resources = self.budget.statistics();
        Ok(row)
    }
    fn resolve(&mut self, row: &mut Row, is_baseline: bool) -> Result<(), Kernel> {
        for evidence in &self.baseline {
            self.budget.charge(1)?;
            if !is_baseline && !evidence.seen {
                row.general.resolved = row
                    .general
                    .resolved
                    .checked_add(1)
                    .ok_or(Kernel::SizeOverflow)?;
                row.general.resolved_squared_excess.add(evidence.general)?;
            }
        }
        finish(
            &mut row.general,
            row.contacts,
            self.config.similarity.route_tolerance,
        )
    }
}
struct Accumulator<'a> {
    originals: &'a mut Originals,
    config: &'a Config,
    lookup: &'a [(u32, usize)],
    baseline: &'a mut Vec<Evidence>,
    cache: &'a mut [Option<Cached>],
    row: &'a mut Row,
    is_baseline: bool,
}
impl Accumulator<'_> {
    fn assess(
        &mut self,
        a: u32,
        b: u32,
        ai: usize,
        bi: usize,
        budget: &mut Budget,
    ) -> Result<Assessment, Kernel> {
        let slot = if self.cache.is_empty() {
            None
        } else {
            budget.charge(1)?;
            Some(
                usize::try_from(a)
                    .map_err(|_| Kernel::SizeOverflow)?
                    .wrapping_mul(0x9e37_79b9)
                    .wrapping_add(usize::try_from(b).map_err(|_| Kernel::SizeOverflow)?)
                    .checked_rem(self.cache.len())
                    .ok_or(Kernel::InvalidIndex)?,
            )
        };
        if let Some(index) = slot
            && let Some(cached) = self.cache.get(index).ok_or(Kernel::InvalidIndex)?
            && cached.key == (a, b)
        {
            return Ok(cached.assessment);
        }
        let assessment = self
            .originals
            .assess(ai, bi, &self.config.similarity, budget)?;
        if let Some(index) = slot {
            budget.charge(1)?;
            *self.cache.get_mut(index).ok_or(Kernel::InvalidIndex)? = Some(Cached {
                key: (a, b),
                assessment,
            });
        }
        Ok(assessment)
    }
    fn contact(&mut self, a: u32, b: u32, budget: &mut Budget) -> Result<(), Cause> {
        budget.charge(
            u64::from(self.lookup.len().checked_ilog2().unwrap_or(0))
                .saturating_add(2)
                .saturating_mul(2),
        )?;
        let key = (a, b);
        let search_work = u64::from(self.baseline.len().checked_ilog2().unwrap_or(0));
        budget.charge(search_work.saturating_add(2))?;
        let previous = self.baseline.binary_search_by_key(&key, |e| e.key).ok();
        let ai = route(self.lookup, a)?;
        let bi = route(self.lookup, b)?;
        // Stored baseline evidence is always incompatible.
        let (incompatible, weight) = if let Some(index) = previous {
            let evidence = self.baseline.get_mut(index).ok_or(Kernel::InvalidIndex)?;
            evidence.seen = true;
            (true, evidence.general)
        } else {
            let assessment = self.assess(a, b, ai, bi, budget)?;
            (
                assessment.incompatible,
                Bounds::excess(
                    assessment.lower,
                    assessment.upper,
                    self.config.similarity.route_tolerance,
                ),
            )
        };
        let view = &mut self.row.general;
        view.squared_excess.add(weight)?;
        if incompatible {
            view.current = view.current.checked_add(1).ok_or(Kernel::SizeOverflow)?;
            if !self.is_baseline && previous.is_none() {
                view.introduced = view.introduced.checked_add(1).ok_or(Kernel::SizeOverflow)?;
                view.introduced_squared_excess.add(weight)?;
            }
        }
        if self.is_baseline && incompatible {
            if self.baseline.len() == self.config.baseline_capacity {
                return Err(Kernel::RecordLimit.into());
            }
            self.baseline.push(Evidence {
                key,
                general: weight,
                seen: true,
            });
        }
        Ok(())
    }
}
fn count_value(value: u64) -> Result<f64, Kernel> {
    let high = f64::from(u32::try_from(value >> 32).map_err(|_| Kernel::SizeOverflow)?);
    let low =
        f64::from(u32::try_from(value & u64::from(u32::MAX)).map_err(|_| Kernel::SizeOverflow)?);
    Ok(high.mul_add(4_294_967_296.0, low))
}
fn finish(view: &mut View, count: u64, tolerance: f64) -> Result<(), Kernel> {
    if count > 0 {
        let count = count_value(count)?;
        view.rms_excess = Bounds {
            lower: (view.squared_excess.lower / count).sqrt(),
            upper: (view.squared_excess.upper / count).sqrt(),
        };
        view.ratio = Bounds {
            lower: view.rms_excess.lower / tolerance,
            upper: view.rms_excess.upper / tolerance,
        };
        if !view.ratio.upper.is_finite() {
            return Err(Kernel::InvalidScale);
        }
    }
    Ok(())
}
fn storage_requirements(config: &Config, route_count: usize) -> Result<(usize, usize), Kernel> {
    let records = config
        .baseline_capacity
        .checked_add(config.cache_capacity)
        .ok_or(Kernel::SizeOverflow)?;
    let bytes = config
        .baseline_capacity
        .checked_mul(size_of::<Evidence>())
        .and_then(|n| {
            n.checked_add(
                config
                    .cache_capacity
                    .checked_mul(size_of::<Option<Cached>>())?,
            )
        })
        .and_then(|n| n.checked_add(route_count.checked_mul(size_of::<(u32, usize)>())?))
        .ok_or(Kernel::SizeOverflow)?;

    Ok((bytes, records))
}
