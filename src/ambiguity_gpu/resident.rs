//! GPU-resident f32 ambiguity evaluation (backend `gpu-f32`).
//!
//! Every numerical stage runs in WGSL f32 on the device: the frozen baseline
//! copy, canonical originals and their order, simplification, first-fit
//! complete-link cohorts, continuous sustained contact of the current
//! geometry, whole-route endpoint-constrained weak Frechet distances with
//! coherent orientation, baseline evidence, cohort associations, counts and
//! reductions. The host uploads parameters, sizes and schedules bounded
//! dispatches, enforces limits, and reads counts, the membership table and the
//! finished scalars. There is no CPU geometry, distance, cohort or scoring
//! path in this mode and no fallback to one: range, resource, convergence and
//! protocol failures abort without a result.
//!
//! Numerical contract: ordinary f32 point estimates, not certified enclosures
//! and not bit-compatible with the CPU f64 reference. Lower and upper general
//! values are the f32 simplification estimates `d - e` and `d + e`; they
//! coincide after full-curve refinement. Classifications at a threshold can
//! differ from the reference. Coordinates must be finite and below 2^20 in
//! magnitude and tolerances within [2^-20, 2^20] clip units; anything else is
//! rejected before use.
use super::{Error, Geometry, Result, Scope, Work, nearest_f32};
use std::fmt;

mod contact;
mod distance;
mod originals;
mod runner;
mod score;

pub use distance::{SEARCH_CELLS_LIMIT, Statistics as DistanceStatistics};
use originals::Originals;
use runner::{Bound, Entry, Module, P, Runner, Store, add, histogram_words, mul, scan_words};
pub use runner::{INVOCATION_TARGET, WORK_TARGET};

/// Version of the f32 numerical contract reported with every result.
pub const CONTRACT_VERSION: u32 = 1;
const MINIMUM_SCALE: f64 = 9.536_743_164_062_5e-7;
const MAXIMUM_SCALE: f64 = 1_048_576.0;
const STATUS_NAMES: [(u32, &str); 11] = [
    (1, "nonfinite coordinate"),
    (2, "coordinate outside the supported f32 range"),
    (4, "points disagree with path reservations"),
    (8, "invalid candidate"),
    (16, "duplicate free-space cell"),
    (32, "missing neighbor cell"),
    (64, "cell capacity exceeded"),
    (128, "route distance search exceeds the cell limit"),
    (256, "internal invariant"),
    (512, "route exceeds the supported point count"),
    (1024, "u32 prefix sum overflow"),
];

/// Failures specific to the resident backend. None of them has a fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    /// Device status bits raised by the named stage.
    Status {
        stage: &'static str,
        bits: u32,
    },
    /// A bounded iteration did not converge within its limit.
    Rounds {
        stage: &'static str,
        limit: u64,
    },
    /// A parameter or result outside the supported f32 range.
    Range(&'static str),
    /// A device schedule or count that violates the protocol.
    Schedule(&'static str),
    Kernel(&'static str),
    HostAllocation,
    /// An earlier frame failed; the session cannot be used.
    Invalid,
}
impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Status { stage, bits } => {
                write!(f, "gpu-f32 {stage} failed (status {bits:#x}):")?;
                for (bit, name) in STATUS_NAMES {
                    if bits & bit != 0 {
                        write!(f, " {name};")?;
                    }
                }
                Ok(())
            }
            Self::Rounds { stage, limit } => write!(
                f,
                "gpu-f32 {stage} did not finish within {limit} bounded rounds"
            ),
            Self::Range(what) => write!(f, "gpu-f32 range: {what}"),
            Self::Schedule(what) => write!(f, "gpu-f32 protocol: {what}"),
            Self::Kernel(module) => write!(f, "gpu-f32 kernel missing from {module}"),
            Self::HostAllocation => f.write_str("gpu-f32 host table allocation failed"),
            Self::Invalid => f.write_str("gpu-f32 session is invalid after a failed frame"),
        }
    }
}

/// Run `operation` inside device error scopes. A captured validation or
/// out-of-memory error is the root cause of whatever the operation then
/// observed, so it takes precedence over the operation's own failure.
fn scoped<T>(device: &wgpu::Device, operation: impl FnOnce() -> Result<T>) -> Result<T> {
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let mut scope = Scope { device, open: true };
    let result = operation();
    scope.pop().map_or(result, Err)
}

/// Thresholds in clip units and resource limits.
#[derive(Debug, Clone, Copy)]
pub struct Settings {
    pub route_tolerance: f64,
    pub endpoint_tolerance: f64,
    pub sigma: f64,
    pub radius: f64,
    pub minimum_contact: f64,
    /// Combined bytes of every buffer this backend allocates, including the
    /// frozen baseline copy. Renderer-owned geometry is borrowed and excluded.
    pub memory_bytes: u64,
    /// Contact pairs one frame may log.
    pub pair_capacity: u64,
    /// Bytes of host metadata this backend may allocate from input or device
    /// counts (readbacks, membership and ID tables, batch and chunk plans).
    pub host_bytes: u64,
    /// Largest free-space cell count of one whole-route distance search.
    pub search_cells: u64,
}

/// Validated f32 parameters.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Config {
    route_tolerance: f32,
    endpoint_tolerance: f32,
    sigma: f32,
    radius: f32,
    minimum_contact: f32,
    pair_capacity: u64,
    search_cells: u64,
}

/// Convert one threshold; positive values must stay within the supported scale
/// so that no normalized key, square or sum can overflow or vanish.
fn threshold(value: f64, allow_zero: bool, name: &'static str) -> Result<f32> {
    if value == 0.0 && allow_zero {
        return Ok(0.0);
    }
    if !(MINIMUM_SCALE..=MAXIMUM_SCALE).contains(&value) {
        return Err(Error::Fault(Fault::Range(name)));
    }
    Ok(nearest_f32(value))
}

impl Config {
    fn new(settings: &Settings) -> Result<Self> {
        if settings.memory_bytes == 0 || settings.pair_capacity == 0 || settings.host_bytes == 0 {
            return Err(Error::Parameters(
                "gpu-f32 ambiguity limits must be positive",
            ));
        }
        Ok(Self {
            route_tolerance: threshold(
                settings.route_tolerance,
                false,
                "route tolerance must be within [2^-20, 2^20] clip units",
            )?,
            endpoint_tolerance: threshold(
                settings.endpoint_tolerance,
                false,
                "endpoint tolerance must be within [2^-20, 2^20] clip units",
            )?,
            sigma: threshold(
                settings.sigma,
                true,
                "simplification must be zero or within [2^-20, 2^20] clip units",
            )?,
            radius: threshold(
                settings.radius,
                true,
                "contact radius must be zero or within [2^-20, 2^20] clip units",
            )?,
            minimum_contact: threshold(
                settings.minimum_contact,
                false,
                "minimum contact must be within [2^-20, 2^20] clip units",
            )?,
            pair_capacity: settings.pair_capacity,
            search_cells: settings.search_cells,
        })
    }
}

/// f32 estimates from the lower and upper route distance.
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct Interval {
    pub lower: f32,
    pub upper: f32,
}
/// One view in clip units; the endpoint view has equal lower and upper values.
#[derive(Debug, Default, Clone, Copy)]
pub struct View {
    pub current: u64,
    pub introduced: u64,
    pub resolved: u64,
    pub squared_excess: Interval,
    pub introduced_squared_excess: Interval,
    pub resolved_squared_excess: Interval,
    pub rms_excess: Interval,
    pub ratio: Interval,
}
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct CohortScores {
    pub current: u64,
    pub baseline: u64,
    pub introduced: u64,
    pub resolved: u64,
    pub contacting_cross_cohort: u64,
    pub incompatible_fraction: f32,
}

/// Device counts and accounting of one frame.
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct FrameStatistics {
    pub validated_points: u64,
    pub positive_segments: u64,
    pub collapsed_paths: u64,
    pub short_paths: u64,
    pub grid_cells_per_axis: u32,
    pub footprint_entries: u64,
    pub work_items: u64,
    /// Free-space cells accepted by the f32 predicates.
    pub cells: u64,
    /// Connected components that reach the minimum on both routes.
    pub sustained_components: u64,
    pub batches: u64,
    pub batch_capacity_cells: u64,
    pub largest_batch_cells: u64,
    pub label_rounds: u64,
    pub largest_batch_label_rounds: u64,
    /// Contact pairs that needed an original distance assessment.
    pub assessed_pairs: u64,
    pub distance: DistanceStatistics,
    pub work: Work,
    pub allocated_bytes: u64,
    pub peak_allocated_bytes: u64,
    pub memory_limit_bytes: u64,
    pub readback_bytes: u64,
    /// Largest host metadata bytes counted so far, and their limit.
    pub host_peak_bytes: u64,
    pub host_limit_bytes: u64,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Frame {
    pub contacts: u64,
    pub general: View,
    pub endpoints: View,
    pub cohorts: CohortScores,
    pub statistics: FrameStatistics,
}

/// Counts and accounting of the original preparation.
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct Preparation {
    pub eligible_routes: u64,
    pub skipped_reservations: u64,
    pub canonical_points: u64,
    pub longest_canonical_route: u64,
    pub closed_stored_start_routes: u64,
    pub original_constants: u64,
    pub unique_canonical_routes: u64,
    pub cohorts: u64,
    /// Earlier unique routes with compatible endpoints, all assessed.
    pub endpoint_candidate_pairs: u64,
    pub distance: DistanceStatistics,
    pub work: Work,
    pub frozen_baseline_bytes: u64,
    pub allocated_bytes: u64,
    pub peak_allocated_bytes: u64,
    pub readback_bytes: u64,
    pub host_peak_bytes: u64,
}

/// The failure of a bounded iteration that used up its `limit`.
const fn unfinished(stage: &'static str, limit: u64) -> Error {
    Error::Fault(Fault::Rounds { stage, limit })
}

pub(crate) struct Core {
    runner: Runner,
    store: Store,
    /// Every kernel of the backend; entry names are unique.
    kernels: Module,
}
impl Core {
    /// Encode one invocation of `entry` per element at the kernel's fixed
    /// work bound.
    fn run(
        &mut self,
        entry: &str,
        binds: &[runner::Bind<'_>],
        registers: runner::P,
        invocations: u64,
    ) -> Result<()> {
        self.run_bounded(entry, binds, registers, invocations, Bound::Fixed)
    }
    /// Encode one invocation of `entry` per element at a work bound the
    /// caller derived from registers or counts.
    fn run_bounded(
        &mut self,
        entry: &str,
        binds: &[runner::Bind<'_>],
        registers: runner::P,
        invocations: u64,
        bound: Bound,
    ) -> Result<()> {
        self.runner
            .run(&self.kernels, entry, binds, registers, invocations, bound)
    }
    /// Encode a single-invocation kernel.
    fn run_once(
        &mut self,
        entry: &str,
        binds: &[runner::Bind<'_>],
        registers: runner::P,
        bound: Bound,
    ) -> Result<()> {
        self.runner
            .run_groups(&self.kernels, entry, binds, registers, 1, bound)
    }
    /// Zeroed storage of `count` words, released with the current scope.
    fn words(&mut self, label: &'static str, count: u64) -> Result<wgpu::Buffer> {
        self.store.words(label, count)
    }
    fn scan(
        &mut self,
        values: &wgpu::Buffer,
        range: (u64, u64),
        scratch: &wgpu::Buffer,
    ) -> Result<()> {
        self.runner.scan(&self.kernels, values, range, scratch)?;
        self.runner.status("prefix scan")?;
        Ok(())
    }
    /// Exclusive scan of `count` + 1 words from word 0, then the total.
    fn total(&mut self, values: &wgpu::Buffer, count: u64, scratch: &wgpu::Buffer) -> Result<u64> {
        self.count("prefix scan", values, count, scratch)
    }
    /// `total` whose overflow is reported as the named u32 count.
    fn count(
        &mut self,
        what: &'static str,
        values: &wgpu::Buffer,
        count: u64,
        scratch: &wgpu::Buffer,
    ) -> Result<u64> {
        self.runner
            .scan(&self.kernels, values, (0, add(count, 1)?), scratch)?;
        self.runner.status(what)?;
        self.word(values, count)
    }
    /// Submit pending work and read one word.
    fn word(&mut self, buffer: &wgpu::Buffer, index: u64) -> Result<u64> {
        let used = self.runner.host.used();
        let values = self.runner.read(buffer, index, 1);
        self.runner.host.restore(used);
        values?
            .first()
            .copied()
            .map(u64::from)
            .ok_or(Error::Protocol(super::Protocol::Readback))
    }
    /// Stable record sort; returns (sorted, other).
    fn sort<'a>(
        &mut self,
        (first, second): (&'a wgpu::Buffer, &'a wgpu::Buffer),
        scratch: (&wgpu::Buffer, &wgpu::Buffer),
        count: u64,
        key_bits: (u32, u32),
    ) -> Result<(&'a wgpu::Buffer, &'a wgpu::Buffer)> {
        let flipped = self
            .runner
            .sort(&self.kernels, (first, second), scratch, count, key_bits)?;
        Ok(if flipped {
            (second, first)
        } else {
            (first, second)
        })
    }
    /// Run `operation`, submit what it encoded, then release its unpromoted
    /// buffers. On failure everything encoded but not submitted is dropped
    /// before the buffers it names are destroyed.
    fn scoped<T>(&mut self, operation: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let mut scope = Scoped {
            mark: self.store.mark(),
            host: self.runner.host.used(),
            core: self,
            open: true,
        };
        let result = operation(scope.core).and_then(|value| {
            scope.core.runner.flush()?;
            Ok(value)
        });
        let released = scope.close(result.is_err());
        let value = result?;
        released.map(|()| value)
    }
    /// Bounded iteration: run `step` with the attempt index, then read the
    /// status of `stage`, until `done` accepts it. After `attempts` steps the
    /// stage fails as unfinished within `limit`, the caller's round count.
    /// `step` resets the status counters where its algorithm needs. Returns
    /// the attempts used.
    fn settle(
        &mut self,
        (stage, attempts, limit): (&'static str, u64, u64),
        mut step: impl FnMut(&mut Self, u64) -> Result<()>,
        done: impl Fn(&[u32]) -> bool,
    ) -> Result<u64> {
        for attempt in 0..attempts {
            step(self, attempt)?;
            if done(&self.runner.status(stage)?) {
                return add(attempt, 1);
            }
        }
        Err(unfinished(stage, limit))
    }
    /// Write host words to a buffer; nothing may be encoded for it yet.
    fn upload(&self, buffer: &wgpu::Buffer, bytes: &[u8]) {
        self.runner.write(buffer, bytes);
    }
}

/// Scratch element of the ranking allocation layout.
const RANK_SCRATCH_INDEX: usize = 3;

/// Word counts of a ranking workspace over `count` records: both record
/// buffers, the radix histogram, the scan scratch and the key marks. Contact
/// capacity planning uses the same counts.
fn ranked_words(count: u64) -> Result<[u64; 5]> {
    let histogram = histogram_words(count)?;
    let marks = add(count, 1)?;
    Ok([
        mul(count, 3)?,
        mul(count, 3)?,
        histogram,
        scan_words(marks.max(histogram))?,
        marks,
    ])
}

/// Labels of a ranking workspace's buffers.
struct Labels {
    records: &'static str,
    histogram: &'static str,
    marks: &'static str,
}
/// Where a ranking workspace's scan scratch comes from.
#[derive(Clone, Copy)]
enum Scratch<'a> {
    /// A new buffer with this label.
    Owned(&'static str),
    /// A live buffer of at least the workspace's scratch words.
    Borrowed(&'a wgpu::Buffer),
}
/// Which key words decide that two sorted records differ.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Key {
    Both,
    High,
}

/// Workspace of stable three-word record sorts with key-change ranks, released
/// with the scope that allocated it. Callers write records into `records[0]`.
struct Ranking {
    records: [wgpu::Buffer; 2],
    histogram: wgpu::Buffer,
    scratch: wgpu::Buffer,
    /// Scanned key-change marks of the last rank: dense rank from 1.
    marks: wgpu::Buffer,
}
impl Ranking {
    /// Allocate a workspace for at most `count` records.
    fn new(core: &mut Core, count: u64, labels: &Labels, scratch: Scratch<'_>) -> Result<Self> {
        let [records, _, histogram, scan, marks] = ranked_words(count)?;
        let first = core.words(labels.records, records)?;
        let second = core.words(labels.records, records)?;
        let histogram = core.words(labels.histogram, histogram)?;
        let scratch = match scratch {
            Scratch::Owned(label) => core.words(label, scan)?,
            Scratch::Borrowed(buffer) => buffer.clone(),
        };
        Ok(Self {
            records: [first, second],
            histogram,
            scratch,
            marks: core.words(labels.marks, marks)?,
        })
    }
    /// Sort `count` records by `key_bits`, then mark and scan the key changes.
    /// Returns the sorted records.
    fn rank(
        &self,
        core: &mut Core,
        count: u64,
        key_bits: (u32, u32),
        key: Key,
    ) -> Result<&wgpu::Buffer> {
        let [first, second] = &self.records;
        let (sorted, _) = core.sort(
            (first, second),
            (&self.histogram, &self.scratch),
            count,
            key_bits,
        )?;
        core.run(
            "key_marks",
            &[(4, sorted), (5, &self.marks)],
            P {
                a: u32::from(key == Key::High),
                ..P::count(count)?
            },
            add(count, 1)?,
        )?;
        core.scan(&self.marks, (0, add(count, 1)?), &self.scratch)?;
        Ok(sorted)
    }
}

/// One storage scope of a `Core`. It is closed explicitly on ordinary exits and
/// by `Drop` when the operation unwinds, so a caught panic cannot leave scoped
/// buffers or encoded commands behind.
struct Scoped<'a> {
    core: &'a mut Core,
    mark: usize,
    host: u64,
    open: bool,
}
impl Scoped<'_> {
    fn close(&mut self, failed: bool) -> Result<()> {
        self.open = false;
        if failed {
            self.core.runner.abandon();
        }
        self.core.runner.host.restore(self.host);
        self.core.store.release(self.mark)
    }
}
impl Drop for Scoped<'_> {
    fn drop(&mut self) {
        if self.open {
            let _ = self.close(true);
        }
    }
}

fn status_word(status: &[u32], index: usize) -> u64 {
    u64::from(status.get(index).copied().unwrap_or(0))
}

/// The device's f32 assessment of one original route pair (clip units).
#[derive(Debug, Clone, Copy)]
pub struct Assessment {
    pub route_lower: f32,
    pub route_upper: f32,
    pub endpoints: f32,
    pub reversed: bool,
    pub general_incompatible: bool,
    pub endpoint_incompatible: bool,
}

const fn decode(record: &[u32; 16]) -> Assessment {
    let [.., lower, upper, endpoints, flags] = *record;
    Assessment {
        route_lower: f32::from_bits(lower),
        route_upper: f32::from_bits(upper),
        endpoints: f32::from_bits(endpoints),
        reversed: flags & 1 != 0,
        general_incompatible: flags & 2 != 0,
        endpoint_incompatible: flags & 4 != 0,
    }
}

/// Read the logged contact pairs back as reservation IDs (diagnostic).
fn list(
    core: &mut Core,
    originals: &Originals,
    log: &wgpu::Buffer,
    contacts: u64,
    listing: &mut Vec<[u32; 2]>,
) -> Result<()> {
    let words = core.runner.read(log, 0, mul(contacts, 2)?)?;
    let id = |ordinal: u32| {
        usize::try_from(ordinal)
            .ok()
            .and_then(|index| originals.ids.get(index))
            .copied()
            .ok_or(Error::Fault(Fault::Schedule("contact pair ordinal")))
    };
    core.runner.host.reserve(mul(contacts, 8)?)?;
    listing
        .try_reserve_exact(usize::try_from(contacts).map_err(|_| Error::SizeOverflow)?)
        .map_err(|_| Error::Fault(Fault::HostAllocation))?;
    for [a, b] in words.as_chunks::<2>().0 {
        listing.push([id(*a)?, id(*b)?]);
    }
    Ok(())
}

/// Frozen originals, cohorts and baseline evidence on the device.
pub struct Session {
    device: wgpu::Device,
    core: Core,
    config: Config,
    originals: Originals,
    assessor: distance::Assessor,
    evidence: score::Evidence,
    baseline: Frame,
    memory_limit: u64,
    host_limit: u64,
    samples: u32,
    reservations: u32,
    valid: bool,
    listing: Option<Vec<[u32; 2]>>,
}

// Kernel entry points with their worst-case work units per invocation. One unit
// is one iteration of an innermost loop, or one straight-line block, touching
// at most 32 buffer or private-array elements; the counts come from the source
// loops (for example collinear_mark: 12 + 6 * 24 + 12 limb iterations plus its
// blocks). Zero marks a bound that depends on registers or data: its caller
// supplies it with each dispatch.
const UTIL: &[Entry] = &[
    ("scan_blocks", 10),
    ("scan_add", 2),
    ("radix_count", 1536),
    ("radix_scatter", 1280),
];
const ORIGINALS: &[Entry] = &[
    ("route_mark", 1),
    ("route_init", 0),
    ("dedup_write", 1),
    ("route_dedup", 1),
    ("collinear_mark", 192),
    ("canon_tmp", 1),
    ("route_canon", 1),
    ("route_orient", 0),
    ("canon_write", 1),
    ("name_entries0", 1),
    ("key_marks", 1),
    ("name_write", 1),
    ("name_entries", 1),
    ("route_entries", 1),
    ("route_place", 1),
    ("route_rep", 1),
    ("simplify_step", 0),
    ("endpoint_visit", 0),
    ("cohort_pairs", 40),
    ("cohort_collect", 1),
    ("first_fit", 0),
    ("cohort_flags", 1),
    ("cohort_write", 1),
    ("membership", 1),
];
const DISTANCE: &[Entry] = &[
    ("assess_begin", 16),
    ("task_fast", 0),
    ("grid_frontier", 0),
    ("grid_marks", 1),
    ("grid_list", 1),
    ("grid_chunks", 1),
    ("grid_init", 4128),
    ("grid_relax", 2080),
    ("grid_roll", 1),
    ("grid_finish", 1),
    ("assess_step", 4),
];
const FRAME: &[Entry] = &[
    ("point_mark", 1),
    ("frame_counts", 0),
    ("frame_segments", 1),
    ("frame_routes", 1),
    ("frame_short", 0),
];
const BROAD: &[Entry] = &[
    ("foot_count", 1),
    ("foot_emit", 40),
    ("cell_fill", 1),
    ("item_weights", 1),
    ("route_items", 1),
    ("count_cells", 928),
    ("fill_cells", 928),
];
const CONTACT: &[Entry] = &[
    ("chain", 2),
    ("links", 0),
    ("label_round", 4),
    ("measure", 0),
    ("sustained", 1),
    ("sustained_pairs", 1),
    ("pair_out", 1),
];
const SCORE: &[Entry] = &[
    ("score_init", 64),
    ("probe_pairs", 1),
    ("score_pairs", 1),
    ("score_collect", 1),
    ("score_rows", 96),
    ("reduce", 3100),
    ("record_marks", 1),
    ("evidence_write", 8),
    ("cross_keys", 1),
    ("witness_marks", 1),
    ("witness_list", 1),
    ("witness_difference", 64),
    ("score_finish", 16),
];

fn kernels(runner: &Runner) -> Result<Module> {
    runner.kernels(
        "ambiguity f32 kernels",
        &[
            (
                "ambiguity f32 util",
                include_str!("../shaders/ambiguity_f32_util.wgsl"),
                UTIL,
            ),
            (
                "ambiguity f32 originals",
                include_str!("../shaders/ambiguity_f32_originals.wgsl"),
                ORIGINALS,
            ),
            (
                "ambiguity f32 distance",
                include_str!("../shaders/ambiguity_f32_distance.wgsl"),
                DISTANCE,
            ),
            (
                "ambiguity f32 frame",
                include_str!("../shaders/ambiguity_f32_frame.wgsl"),
                FRAME,
            ),
            (
                "ambiguity f32 broad",
                include_str!("../shaders/ambiguity_f32_broad.wgsl"),
                BROAD,
            ),
            (
                "ambiguity f32 contact",
                include_str!("../shaders/ambiguity_f32_contact.wgsl"),
                CONTACT,
            ),
            (
                "ambiguity f32 score",
                include_str!("../shaders/ambiguity_f32_score.wgsl"),
                SCORE,
            ),
        ],
    )
}

impl Session {
    /// Freeze `geometry`'s current points as the baseline on the device, prepare
    /// originals and cohorts there, and score the baseline frame.
    /// # Errors
    /// Unsupported ranges, resource limits, device and protocol failures. No
    /// partial session is returned.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        geometry: &Geometry,
        settings: &Settings,
    ) -> Result<Self> {
        let config = Config::new(settings)?;
        scoped(device, || {
            let mut store = Store::new(device.clone(), settings.memory_bytes);
            let runner = Runner::new(
                device,
                queue,
                &mut store,
                runner::Host::new(settings.host_bytes),
            )?;
            let kernels = kernels(&runner)?;
            let mut core = Core {
                runner,
                store,
                kernels,
            };
            let (originals, assessor) = originals::prepare(&mut core, geometry, &config)?;
            let mut session = Self {
                device: device.clone(),
                core,
                config,
                originals,
                assessor,
                evidence: score::Evidence::default(),
                baseline: Frame::default(),
                memory_limit: settings.memory_bytes,
                host_limit: settings.host_bytes,
                samples: geometry.samples,
                reservations: geometry.reservations,
                valid: false,
                listing: None,
            };
            let preparation = &mut session.originals.preparation;
            preparation.distance = session.assessor.statistics;
            preparation.work = session.core.runner.work;
            preparation.frozen_baseline_bytes = mul(u64::from(geometry.samples), 12)?;
            preparation.allocated_bytes = session.core.store.allocated();
            preparation.peak_allocated_bytes = session.core.store.peak();
            preparation.readback_bytes = session.core.runner.readback_bytes;
            preparation.host_peak_bytes = session.core.runner.host.peak();
            let frozen = session.originals.frozen.clone();
            session.baseline = session.frame(geometry, &frozen, true)?;
            session.valid = true;
            Ok(session)
        })
    }

    /// Score the current resident points of `geometry` against the frozen
    /// originals and baseline evidence.
    /// # Errors
    /// Any failure invalidates the session. No partial frame is returned.
    pub fn evaluate(&mut self, geometry: &Geometry) -> Result<Frame> {
        if !std::mem::replace(&mut self.valid, false) {
            return Err(Error::Fault(Fault::Invalid));
        }
        if geometry.samples != self.samples || geometry.reservations != self.reservations {
            return Err(Error::Parameters(
                "gpu-f32 ambiguity frame counts differ from the frozen baseline",
            ));
        }
        let device = self.device.clone();
        let frame = scoped(&device, || {
            let points = geometry.points.clone();
            self.frame(geometry, &points, false)
        })?;
        self.valid = true;
        Ok(frame)
    }

    /// `evaluate`, also returning the frame's contact pairs as reservation IDs
    /// (lower ID first, in increasing order). Diagnostic output only.
    /// # Errors
    /// As `evaluate`.
    pub fn evaluate_listing(&mut self, geometry: &Geometry) -> Result<(Frame, Vec<[u32; 2]>)> {
        let mut pairs = Vec::new();
        self.listing = Some(Vec::new());
        let frame = self.evaluate(geometry);
        if let Some(listed) = self.listing.take() {
            pairs = listed;
        }
        Ok((frame?, pairs))
    }

    /// Diagnostic: the device's f32 assessment of original route pairs given
    /// as reservation IDs.
    /// # Errors
    /// Unknown IDs, more pairs than one assessment chunk, and device failures.
    pub fn assess(&mut self, pairs: &[[u32; 2]]) -> Result<Vec<Assessment>> {
        if !self.valid {
            return Err(Error::Fault(Fault::Invalid));
        }
        let count = u64::try_from(pairs.len()).map_err(|_| Error::SizeOverflow)?;
        if count == 0 {
            return Ok(Vec::new());
        }
        if count > distance::PAIRS {
            return Err(Error::Parameters(
                "gpu-f32 assessment listing exceeds one chunk",
            ));
        }
        let device = self.device.clone();
        let Self {
            core,
            originals,
            assessor,
            config,
            valid,
            ..
        } = self;
        let assessed = scoped(&device, || {
            core.scoped(|core| {
                let mut ordinals = core.runner.host.vector(mul(count, 2)?)?;
                for id in pairs.iter().flatten() {
                    let ordinal = originals.ids.binary_search(id).map_err(|_| {
                        Error::Parameters("gpu-f32 assessment of an unknown route ID")
                    })?;
                    ordinals.push(runner::word(
                        u64::try_from(ordinal).map_err(|_| Error::SizeOverflow)?,
                    )?);
                }
                // Input validation leaves the session healthy; device work may
                // invalidate it. The outer scope accounts for ordinals throughout.
                *valid = false;
                let log = core.words("ambiguity f32 probe pairs", mul(count, 2)?)?;
                core.runner.flush()?;
                core.upload(&log, bytemuck::cast_slice(&ordinals));
                let pair_buffer = assessor.pairs.clone();
                core.run(
                    "probe_pairs",
                    &[(1, &log), (3, &originals.routes), (6, &pair_buffer)],
                    runner::P::count(count)?,
                    count,
                )?;
                assessor.assess(
                    core,
                    originals,
                    count,
                    (config.route_tolerance, config.endpoint_tolerance),
                )?;
                let words = core.runner.read(&pair_buffer, 0, mul(count, 16)?)?;
                let mut assessments = core.runner.host.vector(count)?;
                assessments.extend(words.as_chunks::<16>().0.iter().map(decode));
                Ok(assessments)
            })
        });
        if assessed.is_ok() {
            *valid = true;
        }
        assessed
    }

    fn frame(
        &mut self,
        geometry: &Geometry,
        points: &wgpu::Buffer,
        baseline: bool,
    ) -> Result<Frame> {
        let mut statistics = FrameStatistics::default();
        let work = self.core.runner.work;
        let readback = self.core.runner.readback_bytes;
        let distance = self.assessor.statistics;
        let Self {
            core,
            originals,
            assessor,
            evidence,
            config,
            listing,
            ..
        } = self;
        let (contacts, scored) = core.scoped(|core| {
            let log = core.words("ambiguity f32 contact pairs", mul(config.pair_capacity, 2)?)?;
            let contacts = contact::contacts(
                core,
                originals,
                (geometry, points),
                config,
                contact::Log {
                    buffer: &log,
                    capacity: config.pair_capacity,
                },
                &mut statistics,
            )?;
            tracing::info!(contacts, "gpu-f32 contacts found");
            if let Some(listing) = listing {
                list(core, originals, &log, contacts, listing)?;
            }
            let scored = score::score(
                core,
                originals,
                assessor,
                evidence,
                (&log, contacts),
                config,
                baseline,
            )?;
            Ok((contacts, scored))
        })?;
        let now = self.core.runner.work;
        statistics.work = Work {
            dispatches: now.dispatches.saturating_sub(work.dispatches),
            submissions: now.submissions.saturating_sub(work.submissions),
            maximum_wait_seconds: now.maximum_wait_seconds,
            total_wait_seconds: now.total_wait_seconds - work.total_wait_seconds,
        };
        let after = self.assessor.statistics;
        statistics.distance = DistanceStatistics {
            assessed_pairs: after.assessed_pairs.saturating_sub(distance.assessed_pairs),
            frontier_searches: after
                .frontier_searches
                .saturating_sub(distance.frontier_searches),
            cell_searches: after.cell_searches.saturating_sub(distance.cell_searches),
            cell_search_chunks: after
                .cell_search_chunks
                .saturating_sub(distance.cell_search_chunks),
            relaxation_rounds: after
                .relaxation_rounds
                .saturating_sub(distance.relaxation_rounds),
            largest_search_cells_limit: after.largest_search_cells_limit,
        };
        statistics.assessed_pairs = scored.assessed_pairs;
        statistics.allocated_bytes = self.core.store.allocated();
        statistics.peak_allocated_bytes = self.core.store.peak();
        statistics.memory_limit_bytes = self.memory_limit;
        statistics.host_peak_bytes = self.core.runner.host.peak();
        statistics.host_limit_bytes = self.host_limit;
        statistics.readback_bytes = self.core.runner.readback_bytes.saturating_sub(readback);
        Ok(Frame {
            contacts,
            general: scored.general,
            endpoints: scored.endpoints,
            cohorts: scored.cohorts,
            statistics,
        })
    }

    /// Bytes of every buffer this session currently owns.
    #[must_use]
    pub const fn allocated_bytes(&self) -> u64 {
        self.core.store.allocated()
    }
    #[must_use]
    pub const fn baseline(&self) -> Frame {
        self.baseline
    }
    #[must_use]
    pub const fn preparation(&self) -> Preparation {
        self.originals.preparation
    }
    /// (reservation ID, cohort) in canonical geometry order, then ID.
    #[must_use]
    pub fn memberships(&self) -> &[[u32; 2]] {
        &self.originals.memberships
    }
}
