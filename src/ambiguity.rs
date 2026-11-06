//! CPU ambiguity evaluation in original input units, with frozen baseline evidence.
//!
//! General and endpoint ratios are separate RMS excesses and can exceed one.
//! Empty contact sets have zero ratios. Bounds describe original route distance
//! approximation under ordinary f64 arithmetic, without exact-real guarantees.
//! Canonical original order is frozen. Closed routes retain starts.
//! Contact measures connected continuous participation along both routes, which
//! can include shallow crossings and backtracking rather than a monotone match.
//! All owned heap storage uses the caller's shared ledger. Fixed-capacity tables
//! never grow, baseline evidence never evicts, and only cached assessments evict.
//! Index and kernel temporary storage remain on that ledger during callbacks.
use crate::{
    ambiguity_index::{self, Path},
    ambiguity_originals::{self, Originals, Route},
    ambiguity_similarity::{self, Config as Similarity},
    weak_frechet::{Budget, Error as Resource, Statistics},
};
use std::{fmt, mem::size_of};

#[derive(Debug, Clone, Copy)]
pub enum Phase {
    OriginalPreparation,
    Baseline,
    Snapshot,
}

#[derive(Debug)]
pub enum Error {
    Resource(Resource),
    Index {
        phase: Phase,
        error: ambiguity_index::Error<Resource>,
    },
    PhaseResource(Phase, Resource),
    OriginalGeometry(u32),
    InvalidState,
    InvalidIds,
}
impl From<Resource> for Error {
    fn from(value: Resource) -> Self {
        Self::Resource(value)
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resource(error) => write!(f, "ambiguity storage: {error}"),
            Self::PhaseResource(phase, error) => write!(f, "ambiguity {phase:?}: {error}"),
            Self::Index { phase, error } => write!(f, "ambiguity {phase:?}: {error}"),
            Self::OriginalGeometry(id) => write!(
                f,
                "ambiguity original preparation: edge {id} has empty or nonfinite geometry"
            ),
            Self::InvalidState => {
                f.write_str("ambiguity evaluator is invalid after a failed snapshot")
            }
            Self::InvalidIds => {
                f.write_str("ambiguity snapshot IDs differ from eligible original IDs")
            }
        }
    }
}
impl std::error::Error for Error {}

pub struct Config {
    pub similarity: Similarity,
    pub radius: f64,
    pub minimum_contact: f64,
    pub piece_length: f64,
    /// Maximum retained incompatible baseline edges.
    pub baseline_capacity: usize,
    /// Direct-mapped recomputable assessment slots. Zero disables caching.
    pub cache_capacity: usize,
}
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct Bounds {
    pub lower: f64,
    pub upper: f64,
}
impl Bounds {
    fn add(&mut self, value: Self) -> Result<(), Resource> {
        self.lower += value.lower;
        self.upper += value.upper;
        if !self.lower.is_finite() || !self.upper.is_finite() {
            return Err(Resource::InvalidScale);
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
    pub endpoints: View,
    pub resources: Statistics,
    pub index: ambiguity_index::Statistics,
}
#[derive(Clone, Copy)]
struct Cached {
    key: (u32, u32),
    assessment: ambiguity_similarity::Assessment,
}
struct Evidence {
    key: (u32, u32),
    general: Bounds,
    endpoints: Bounds,
    flags: [bool; 2],
    seen: bool,
}
fn vector<T>(capacity: usize) -> Result<Vec<T>, Resource> {
    let mut v = Vec::new();
    v.try_reserve_exact(capacity)
        .map_err(|_| Resource::Allocation)?;
    Ok(v)
}

/// Scoped evaluator. Canonical geometry remains borrowed from
/// original preparation, with their reservations live throughout this scope.
/// Failed frames permanently invalidate this evaluator.
pub struct Evaluator<'a, 'b> {
    originals: Originals<'a>,
    config: &'b Config,
    budget: &'b mut Budget,
    lookup: Vec<(u32, usize)>,
    baseline: Vec<Evidence>,
    cache: Vec<Option<Cached>>,
    preparations: &'b mut [Option<ambiguity_similarity::Prepared<'a>>],
    preparation: Statistics,
    original_constants: usize,
    baseline_row: Row,
    valid: bool,
}
impl Evaluator<'_, '_> {
    #[must_use]
    pub const fn originals(&self) -> &Originals<'_> {
        &self.originals
    }
    #[must_use]
    pub const fn preparation_resources(&self) -> Statistics {
        self.preparation
    }
    #[must_use]
    pub const fn config(&self) -> &Config {
        self.config
    }
    #[must_use]
    pub const fn original_constant_count(&self) -> usize {
        self.original_constants
    }
    #[must_use]
    pub const fn baseline(&self) -> Row {
        self.baseline_row
    }
    fn route(&self, id: u32) -> Result<usize, Resource> {
        let index = self
            .lookup
            .binary_search_by_key(&id, |p| p.0)
            .map_err(|_| Resource::InvalidIndex)?;
        Ok(self.lookup.get(index).ok_or(Resource::InvalidIndex)?.1)
    }
    /// Evaluate one snapshot with exactly the frozen eligible IDs.
    /// # Errors
    /// Any failure invalidates this evaluator. No partial row is returned.
    pub fn evaluate(&mut self, paths: &[Path<'_>]) -> Result<Row, Error> {
        if !self.valid {
            return Err(Error::InvalidState);
        }
        self.valid = false;
        self.budget.begin_phase();
        if paths.len() != self.lookup.len() {
            return Err(Error::InvalidIds);
        }
        // The index independently rejects duplicate IDs. Equal lengths and all
        // IDs present then establish exact membership without a frame ID table.
        for path in paths {
            self.budget.charge(
                u64::from(self.lookup.len().checked_ilog2().unwrap_or(0)).saturating_add(2),
            )?;
            self.route(path.id).map_err(|_| Error::InvalidIds)?;
        }
        let row = self.frame(paths, false)?;
        self.valid = true;
        Ok(row)
    }
    fn frame(&mut self, paths: &[Path<'_>], baseline: bool) -> Result<Row, Error> {
        let phase = if baseline {
            Phase::Baseline
        } else {
            Phase::Snapshot
        };
        self.frame_inner(paths, baseline, phase)
            .map_err(|error| match error {
                Error::Resource(resource) => Error::PhaseResource(phase, resource),
                other => other,
            })
    }
    fn frame_inner(
        &mut self,
        paths: &[Path<'_>],
        baseline: bool,
        phase: Phase,
    ) -> Result<Row, Error> {
        for evidence in &mut self.baseline {
            self.budget.charge(1)?;
            evidence.seen = false;
        }
        let mut row = Row::default();
        let mut accumulator = Accumulator {
            originals: &self.originals,
            config: self.config,
            lookup: &self.lookup,
            baseline: &mut self.baseline,
            cache: &mut self.cache,
            preparations: self.preparations,
            row: &mut row,
            is_baseline: baseline,
        };
        let stats = ambiguity_index::enumerate(
            paths,
            self.config.radius,
            self.config.minimum_contact,
            self.config.piece_length,
            self.budget,
            |a, b, budget| accumulator.contact(a, b, budget),
        )
        .map_err(|error| Error::Index { phase, error })?;
        row.index = stats;
        row.contacts = stats.contacts;
        row.collapsed = stats.constant_paths;
        for evidence in &self.baseline {
            self.budget.charge(1)?;
            if !baseline && !evidence.seen {
                for (view, flag, weight) in [
                    (&mut row.general, evidence.flags[0], evidence.general),
                    (&mut row.endpoints, evidence.flags[1], evidence.endpoints),
                ] {
                    if flag {
                        view.resolved =
                            view.resolved.checked_add(1).ok_or(Resource::SizeOverflow)?;
                        view.resolved_squared_excess.add(weight)?;
                    }
                }
            }
        }
        finish(
            &mut row.general,
            row.contacts,
            self.config.similarity.route_tolerance,
        )?;
        finish(
            &mut row.endpoints,
            row.contacts,
            self.config.similarity.endpoint_tolerance,
        )?;
        row.general.baseline = if baseline {
            row.general.current
        } else {
            self.baseline_row.general.current
        };
        row.endpoints.baseline = if baseline {
            row.endpoints.current
        } else {
            self.baseline_row.endpoints.current
        };
        row.resources = self.budget.statistics();
        Ok(row)
    }
}
struct Accumulator<'a, 'b> {
    originals: &'a Originals<'b>,
    config: &'a Config,
    lookup: &'a [(u32, usize)],
    baseline: &'a mut Vec<Evidence>,
    cache: &'a mut [Option<Cached>],
    preparations: &'a mut [Option<ambiguity_similarity::Prepared<'b>>],
    row: &'a mut Row,
    is_baseline: bool,
}
impl Accumulator<'_, '_> {
    fn assess(
        &mut self,
        a: u32,
        b: u32,
        ai: usize,
        bi: usize,
        budget: &mut Budget,
    ) -> Result<ambiguity_similarity::Assessment, Resource> {
        let slot = if self.cache.is_empty() {
            None
        } else {
            budget.charge(1)?;
            Some(
                usize::try_from(a)
                    .map_err(|_| Resource::SizeOverflow)?
                    .wrapping_mul(0x9e37_79b9)
                    .wrapping_add(usize::try_from(b).map_err(|_| Resource::SizeOverflow)?)
                    .checked_rem(self.cache.len())
                    .ok_or(Resource::InvalidIndex)?,
            )
        };
        if let Some(index) = slot
            && let Some(cached) = self.cache.get(index).ok_or(Resource::InvalidIndex)?
            && cached.key == (a, b)
        {
            return Ok(cached.assessment);
        }
        // Use the later canonical original route first.
        for index in [ai, bi] {
            budget.charge(1)?;
            let slot = self
                .preparations
                .get_mut(index)
                .ok_or(Resource::InvalidIndex)?;
            if slot.is_none() {
                let points = self
                    .originals
                    .canonical_points(index)
                    .ok_or(Resource::InvalidIndex)?;
                *slot = Some(ambiguity_similarity::prepare(
                    points,
                    self.config.similarity.sigma,
                    budget,
                )?);
            }
        }
        let assessment = ambiguity_similarity::assess_prepared(
            self.preparations
                .get(ai.max(bi))
                .and_then(Option::as_ref)
                .ok_or(Resource::InvalidIndex)?,
            self.preparations
                .get(ai.min(bi))
                .and_then(Option::as_ref)
                .ok_or(Resource::InvalidIndex)?,
            &self.config.similarity,
            budget,
        )?;
        if let Some(index) = slot {
            budget.charge(1)?;
            *self.cache.get_mut(index).ok_or(Resource::InvalidIndex)? = Some(Cached {
                key: (a, b),
                assessment,
            });
        }
        Ok(assessment)
    }
    fn contact(&mut self, a: u32, b: u32, budget: &mut Budget) -> Result<(), Resource> {
        budget.charge(
            u64::from(self.lookup.len().checked_ilog2().unwrap_or(0))
                .saturating_add(2)
                .saturating_mul(2),
        )?;
        let key = (a, b);
        let search_work = u64::from(self.baseline.len().checked_ilog2().unwrap_or(0));
        budget.charge(search_work.saturating_add(2))?;
        let previous = self.baseline.binary_search_by_key(&key, |e| e.key).ok();
        let ai = self
            .lookup
            .binary_search_by_key(&a, |p| p.0)
            .map_err(|_| Resource::InvalidIndex)?;
        let bi = self
            .lookup
            .binary_search_by_key(&b, |p| p.0)
            .map_err(|_| Resource::InvalidIndex)?;
        let ai = self.lookup.get(ai).ok_or(Resource::InvalidIndex)?.1;
        let bi = self.lookup.get(bi).ok_or(Resource::InvalidIndex)?.1;
        let (flags, weights) = if let Some(index) = previous {
            let evidence = self.baseline.get(index).ok_or(Resource::InvalidIndex)?;
            (evidence.flags, [evidence.general, evidence.endpoints])
        } else {
            let assessment = self.assess(a, b, ai, bi, budget)?;
            (
                [
                    assessment.general_incompatible,
                    assessment.endpoint_incompatible,
                ],
                [
                    Bounds::excess(
                        assessment.route_lower,
                        assessment.route_upper,
                        self.config.similarity.route_tolerance,
                    ),
                    Bounds::excess(
                        assessment.endpoints,
                        assessment.endpoints,
                        self.config.similarity.endpoint_tolerance,
                    ),
                ],
            )
        };
        if let Some(index) = previous {
            self.baseline
                .get_mut(index)
                .ok_or(Resource::InvalidIndex)?
                .seen = true;
        }
        for (i, view) in [&mut self.row.general, &mut self.row.endpoints]
            .into_iter()
            .enumerate()
        {
            let weight = *weights.get(i).ok_or(Resource::InvalidIndex)?;
            view.squared_excess.add(weight)?;
            if *flags.get(i).ok_or(Resource::InvalidIndex)? {
                view.current = view.current.checked_add(1).ok_or(Resource::SizeOverflow)?;
                if !self.is_baseline && previous.is_none() {
                    view.introduced = view
                        .introduced
                        .checked_add(1)
                        .ok_or(Resource::SizeOverflow)?;
                    view.introduced_squared_excess.add(weight)?;
                }
            }
        }
        if self.is_baseline && (flags[0] || flags[1]) {
            if self.baseline.len() == self.config.baseline_capacity {
                return Err(Resource::RecordLimit);
            }
            self.baseline.push(Evidence {
                key,
                general: weights[0],
                endpoints: weights[1],
                flags,
                seen: true,
            });
        }
        Ok(())
    }
}
fn count_value(value: u64) -> Result<f64, Resource> {
    let high = f64::from(u32::try_from(value >> 32).map_err(|_| Resource::SizeOverflow)?);
    let low =
        f64::from(u32::try_from(value & u64::from(u32::MAX)).map_err(|_| Resource::SizeOverflow)?);
    Ok(high.mul_add(4_294_967_296.0, low))
}
fn finish(view: &mut View, count: u64, tolerance: f64) -> Result<(), Resource> {
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
            return Err(Resource::InvalidScale);
        }
    }
    Ok(())
}
/// Prepare originals, compute the original baseline, then evaluate snapshots in
/// the callback.
///
/// All owned storage is released on success or failure. Original
/// canonical storage remains reserved by `with_originals`; no source copies exist.
/// Cache and baseline evidence reserve their full configured capacities
/// before allocation. The evaluator releases lazy preparations on every exit.
/// # Errors
/// Invalid geometry/configuration, resource limits and callback errors abort.
pub fn with_evaluator<R>(
    original: &[Route<'_>],
    config: &Config,
    budget: &mut Budget,
    operation: impl FnOnce(&mut Evaluator<'_, '_>) -> Result<R, Error>,
) -> Result<R, Error> {
    budget.begin_phase();
    for route in original {
        budget.charge(u64::try_from(route.points.len()).map_err(|_| Resource::SizeOverflow)?)?;
        if route.points.is_empty() || route.points.iter().flatten().any(|x| !x.is_finite()) {
            return Err(Error::OriginalGeometry(route.edge_id));
        }
    }
    let mut output = None;
    let preparation = ambiguity_originals::with_originals(
        original,
        &config.similarity,
        budget,
        |originals, preparations, budget| {
            let (bytes, records) = storage_requirements(config, original.len())?;
            budget.reserve(bytes)?;
            if let Err(error) = budget.reserve_records(records) {
                budget.release(bytes)?;
                return Err(error);
            }
            let result = (|| -> Result<R, Error> {
                let mut lookup = vector(original.len())?;
                let mut original_constants = 0usize;
                for index in 0..original.len() {
                    budget.charge(1)?;
                    lookup.push((
                        originals.edge_id(index).ok_or(Resource::InvalidIndex)?,
                        index,
                    ));
                    if originals
                        .canonical_points(index)
                        .ok_or(Resource::InvalidIndex)?
                        .len()
                        == 1
                    {
                        original_constants = original_constants
                            .checked_add(1)
                            .ok_or(Resource::SizeOverflow)?;
                    }
                }
                lookup.sort_unstable();
                let mut paths = vector(original.len())?;
                for r in original {
                    budget.charge(1)?;
                    paths.push(Path {
                        id: r.edge_id,
                        points: r.points,
                    });
                }
                let mut cache = vector(config.cache_capacity)?;
                budget.charge(
                    u64::try_from(config.cache_capacity).map_err(|_| Resource::SizeOverflow)?,
                )?;
                cache.resize(config.cache_capacity, None);
                let preparation = budget.statistics();
                let mut evaluator = Evaluator {
                    originals,
                    config,
                    budget,
                    lookup,
                    cache,
                    preparations,
                    baseline: vector(config.baseline_capacity)?,
                    preparation,
                    original_constants,
                    baseline_row: Row::default(),
                    valid: false,
                };
                evaluator.budget.begin_phase();
                evaluator.baseline_row = evaluator.frame(&paths, true)?;
                evaluator.valid = true;
                operation(&mut evaluator)
            })();
            budget.release_records(records)?;
            budget.release(bytes)?;
            output = Some(result);
            Ok(())
        },
    );
    preparation.map_err(|error| Error::PhaseResource(Phase::OriginalPreparation, error))?;
    output.ok_or(Error::InvalidState)?
}

fn storage_requirements(config: &Config, route_count: usize) -> Result<(usize, usize), Resource> {
    let records = config
        .baseline_capacity
        .checked_add(config.cache_capacity)
        .ok_or(Resource::SizeOverflow)?;
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
        .and_then(|n| {
            n.checked_add(
                route_count
                    .checked_mul(size_of::<(u32, usize)>().checked_add(size_of::<Path<'_>>())?)?,
            )
        })
        .ok_or(Resource::SizeOverflow)?;

    Ok((bytes, records))
}
