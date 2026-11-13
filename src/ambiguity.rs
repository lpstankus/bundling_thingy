//! Device-resident f32 general ambiguity evaluation.
//!
//! Every numerical stage runs in WGSL f32 on the device: the frozen baseline
//! copy, canonical originals and their order, simplification, continuous
//! sustained contact of the current geometry, whole-route endpoint-constrained
//! weak Frechet distances with joint orientation, baseline evidence, counts and
//! reductions. The host converts thresholds, sizes and schedules bounded
//! dispatches, enforces limits, and reads counts, status words and the finished
//! scalars. There is no CPU geometry, distance or scoring path and no fallback
//! to one: range, resource, convergence and protocol failures abort without a
//! result.
//!
//! Numerical contract: ordinary f32 estimates, not certified enclosures and not
//! bit-compatible with an f64 evaluation. Low and high general values are the
//! f32 simplification estimates `d - e` and `d + e`; they coincide after
//! full-curve refinement. Classifications at a threshold can differ from an f64
//! evaluation. Coordinates must be finite and below 2^20 in magnitude and
//! tolerances within [2^-20, 2^20] clip units; anything else is rejected before
//! use.
use std::{fmt, time::Duration};

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
/// Smallest positive threshold in clip units, 2^-20.
pub const MINIMUM_SCALE: f64 = 9.536_743_164_062_5e-7;
/// Largest threshold in clip units, 2^20.
pub const MAXIMUM_SCALE: f64 = 1_048_576.0;
const POLL_TIMEOUT: Duration = Duration::from_secs(60);
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

/// Resource whose capacity limited a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Limit {
    /// The adapter's buffer or binding size for the named buffer.
    AdapterBuffer(&'static str),
    /// The combined GPU byte limit, reached at the named buffer.
    DeviceMemory(&'static str),
    /// Workgroups per dispatch.
    Dispatch,
    /// A u32 candidate or work-item total.
    CountOverflow,
    /// Contact pairs one frame may log.
    ContactPairs,
    /// Host metadata bytes.
    HostMetadata,
    /// Declared kernel-work units of one phase.
    Work,
}
impl fmt::Display for Limit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AdapterBuffer(label) => write!(f, "adapter buffer size ({label})"),
            Self::DeviceMemory(label) => {
                write!(
                    f,
                    "combined GPU memory limit ({label}); increase --ambiguity-gpu-memory-mib"
                )
            }
            Self::Dispatch => f.write_str("workgroups per dispatch"),
            Self::CountOverflow => f.write_str("u32 candidate or work-item count"),
            Self::ContactPairs => f.write_str("--ambiguity-gpu-contact-pairs"),
            Self::HostMetadata => f.write_str("--ambiguity-gpu-host-mib"),
            Self::Work => f.write_str("--ambiguity-gpu-work-limit"),
        }
    }
}

/// A device result or schedule that violates the readback protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    Inventory,
    Readback,
}
impl fmt::Display for Protocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inventory => f.write_str("GPU buffer inventory is inconsistent"),
            Self::Readback => f.write_str("short GPU readback"),
        }
    }
}

/// An ambiguity failure. None of them has a fallback.
#[derive(Debug)]
pub enum Error {
    Parameters(&'static str),
    SizeOverflow,
    /// A fixed table, plan or phase budget exceeds a limit.
    Limit {
        limit: Limit,
        demand: u64,
        available: u64,
    },
    /// One source path's candidates alone exceed a limit; sources are never split.
    SourceCapacity {
        source: u32,
        limit: Limit,
        demand: u64,
        available: u64,
    },
    /// The packed sort key needs this many bits for one source.
    KeyWidth {
        required: u32,
    },
    Validation(String),
    OutOfMemory(String),
    Poll(String),
    Map(String),
    Protocol(Protocol),
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
    /// The renderer's path buffer, counts or point capacity differ from the
    /// frozen baseline's.
    GeometryChanged,
    /// An earlier frame failed; the session cannot be used.
    Invalid,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parameters(text) => f.write_str(text),
            Self::SizeOverflow => f.write_str("GPU ambiguity size arithmetic overflow"),
            Self::Limit {
                limit,
                demand,
                available,
            } => write!(
                f,
                "GPU ambiguity needs {demand} of {limit}; {available} available"
            ),
            Self::SourceCapacity {
                source,
                limit,
                demand,
                available,
            } => write!(
                f,
                "candidates of source path {source} alone need {demand} of {limit}; {available} available"
            ),
            Self::KeyWidth { required } => write!(
                f,
                "candidate sort key needs {required} bits for a single source, above 64"
            ),
            Self::Validation(text) => write!(f, "GPU validation error: {text}"),
            Self::OutOfMemory(text) => write!(f, "GPU out of memory: {text}"),
            Self::Poll(text) => write!(f, "GPU device poll failed: {text}"),
            Self::Map(text) => write!(f, "GPU readback mapping failed: {text}"),
            Self::Protocol(protocol) => write!(f, "GPU ambiguity protocol: {protocol}"),
            Self::Status { stage, bits } => {
                write!(f, "ambiguity {stage} failed (status {bits:#x}):")?;
                for (bit, name) in STATUS_NAMES {
                    if bits & bit != 0 {
                        write!(f, " {name};")?;
                    }
                }
                Ok(())
            }
            Self::Rounds { stage, limit } => write!(
                f,
                "ambiguity {stage} did not finish within {limit} bounded rounds"
            ),
            Self::Range(what) => write!(f, "ambiguity range: {what}"),
            Self::Schedule(what) => write!(f, "ambiguity protocol: {what}"),
            Self::Kernel(module) => write!(f, "ambiguity kernel missing from {module}"),
            Self::HostAllocation => f.write_str("ambiguity host table allocation failed"),
            Self::GeometryChanged => f.write_str(
                "ambiguity frame geometry differs from the frozen baseline's paths, counts or point capacity",
            ),
            Self::Invalid => f.write_str("ambiguity session is invalid after a failed frame"),
        }
    }
}
impl std::error::Error for Error {}
type Result<T> = std::result::Result<T, Error>;

const fn overflow() -> Error {
    Error::SizeOverflow
}

/// Geometry resident on the renderer's device, borrowed for one evaluation.
///
/// `points` is the renderer's current point buffer, which changes handle as
/// bundling alternates buffers; `paths` and the counts stay fixed. Neither is
/// written here, and neither may be mutated while a frame is evaluated.
#[derive(Clone)]
pub struct Geometry {
    pub points: wgpu::Buffer,
    pub paths: wgpu::Buffer,
    pub samples: u32,
    pub reservations: u32,
}

/// Host wait and transfer accounting of one phase's submissions.
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct Work {
    pub dispatches: u64,
    pub submissions: u64,
    /// Largest host submit-through-completion wait; not a per-pair duration.
    pub maximum_wait_seconds: f64,
    pub total_wait_seconds: f64,
}
impl Work {
    fn add_dispatch(&mut self) -> Result<()> {
        self.dispatches = self.dispatches.checked_add(1).ok_or_else(overflow)?;
        Ok(())
    }
    fn record_submission(&mut self, elapsed: Duration) -> Result<()> {
        self.submissions = self.submissions.checked_add(1).ok_or_else(overflow)?;
        let seconds = elapsed.as_secs_f64();
        self.maximum_wait_seconds = self.maximum_wait_seconds.max(seconds);
        self.total_wait_seconds += seconds;
        Ok(())
    }
}

/// Round-to-nearest f32 of `value`.
#[must_use]
pub const fn nearest_f32(value: f64) -> f32 {
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Rounded f64-to-f32 conversion; thresholds use the nearest f32."
    )]
    let rounded = value as f32;
    rounded
}

/// Validation and out-of-memory scopes popped on success, error and unwind.
struct Scope<'a> {
    device: &'a wgpu::Device,
    open: bool,
}
impl Scope<'_> {
    fn pop(&mut self) -> Option<Error> {
        if !std::mem::replace(&mut self.open, false) {
            return None;
        }
        let memory = pollster::block_on(self.device.pop_error_scope());
        let validation = pollster::block_on(self.device.pop_error_scope());
        memory
            .map(|error| Error::OutOfMemory(error.to_string()))
            .or_else(|| validation.map(|error| Error::Validation(error.to_string())))
    }
}
impl Drop for Scope<'_> {
    fn drop(&mut self) {
        let _ = self.pop();
    }
}

fn words(bytes: &[u8]) -> impl Iterator<Item = u32> + '_ {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
}

/// Unmaps a mapped staging buffer on success, error and unwind.
struct Mapped<'a>(&'a wgpu::Buffer);
impl Drop for Mapped<'_> {
    fn drop(&mut self) {
        self.0.unmap();
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
    /// Combined bytes of every buffer the session allocates, including the
    /// frozen baseline copy. Renderer-owned geometry is borrowed and excluded.
    pub memory_bytes: u64,
    /// Contact pairs one frame may log.
    pub pair_capacity: u64,
    /// Bytes of host metadata the session may allocate from input or device
    /// counts (readbacks, batch and chunk plans).
    pub host_bytes: u64,
    /// Largest free-space cell count of one whole-route distance search.
    pub search_cells: u64,
    /// Declared kernel-work units of one phase: the original preparation, the
    /// baseline frame, or one later frame.
    pub work_limit: u64,
}

/// Validated f32 parameters.
#[derive(Debug, Clone, Copy)]
struct Config {
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
        return Err(Error::Range(name));
    }
    Ok(nearest_f32(value))
}

impl Config {
    fn new(settings: &Settings) -> Result<Self> {
        let limits = [
            settings.memory_bytes,
            settings.pair_capacity,
            settings.host_bytes,
            settings.work_limit,
        ];
        if limits.contains(&0) {
            return Err(Error::Parameters("ambiguity limits must be positive"));
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

/// Low and high f32 estimates from the low and high route distance.
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct Estimate {
    pub low: f32,
    pub high: f32,
}
/// The general view of one frame in clip units.
#[derive(Debug, Default, Clone, Copy)]
pub struct View {
    pub current: u64,
    pub introduced: u64,
    pub resolved: u64,
    pub squared_excess: Estimate,
    pub introduced_squared_excess: Estimate,
    pub resolved_squared_excess: Estimate,
    pub rms_excess: Estimate,
    pub ratio: Estimate,
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
    /// Declared kernel-work units charged by this frame's phase.
    pub declared_work_units: u64,
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
    pub statistics: FrameStatistics,
}

/// Counts and accounting of the original preparation.
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct Preparation {
    pub active_samples: u64,
    pub eligible_routes: u64,
    pub skipped_reservations: u64,
    pub canonical_points: u64,
    pub longest_canonical_route: u64,
    pub closed_stored_start_routes: u64,
    pub original_constants: u64,
    pub unique_canonical_routes: u64,
    pub distance: DistanceStatistics,
    /// Declared kernel-work units charged by the preparation phase.
    pub declared_work_units: u64,
    pub work: Work,
    pub frozen_baseline_bytes: u64,
    pub allocated_bytes: u64,
    pub peak_allocated_bytes: u64,
    pub readback_bytes: u64,
    pub host_peak_bytes: u64,
}

/// The failure of a bounded iteration that used up its `limit`.
const fn unfinished(stage: &'static str, limit: u64) -> Error {
    Error::Rounds { stage, limit }
}

struct Core {
    runner: Runner,
    store: Store,
    /// Every kernel; entry names are unique.
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
            .ok_or(Error::Protocol(Protocol::Readback))
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
    scratch: &'static str,
    marks: &'static str,
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
    fn new(core: &mut Core, count: u64, labels: &Labels) -> Result<Self> {
        let [records, _, histogram, scan, marks] = ranked_words(count)?;
        let first = core.words(labels.records, records)?;
        let second = core.words(labels.records, records)?;
        Ok(Self {
            records: [first, second],
            histogram: core.words(labels.histogram, histogram)?,
            scratch: core.words(labels.scratch, scan)?,
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

/// Frozen originals and baseline evidence on the device.
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
    /// The renderer's fixed reservation buffer and counts.
    paths: wgpu::Buffer,
    samples: u32,
    reservations: u32,
    valid: bool,
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
// Recounted for the five-word records and six-column rows: score_init has five
// stores, a binary search of at most 33 iterations and three copies; score_rows
// six zeroing iterations, two stores, one 33-iteration binary search and, for
// evidence rows, one more, plus its blocks; reduce 256 rows of six columns,
// six initializations, six stores and four blocks; evidence_write five copies;
// score_finish six columns, two sides and four blocks.
const SCORE: &[Entry] = &[
    ("score_init", 64),
    ("score_pairs", 1),
    ("score_collect", 1),
    ("score_rows", 80),
    ("reduce", 1552),
    ("record_marks", 1),
    ("evidence_write", 5),
    ("score_finish", 12),
];

fn kernels(runner: &Runner) -> Result<Module> {
    runner.kernels(
        "ambiguity f32 kernels",
        &[
            (
                "ambiguity f32 util",
                include_str!("shaders/ambiguity_f32_util.wgsl"),
                UTIL,
            ),
            (
                "ambiguity f32 originals",
                include_str!("shaders/ambiguity_f32_originals.wgsl"),
                ORIGINALS,
            ),
            (
                "ambiguity f32 distance",
                include_str!("shaders/ambiguity_f32_distance.wgsl"),
                DISTANCE,
            ),
            (
                "ambiguity f32 frame",
                include_str!("shaders/ambiguity_f32_frame.wgsl"),
                FRAME,
            ),
            (
                "ambiguity f32 broad",
                include_str!("shaders/ambiguity_f32_broad.wgsl"),
                BROAD,
            ),
            (
                "ambiguity f32 contact",
                include_str!("shaders/ambiguity_f32_contact.wgsl"),
                CONTACT,
            ),
            (
                "ambiguity f32 score",
                include_str!("shaders/ambiguity_f32_score.wgsl"),
                SCORE,
            ),
        ],
    )
}

impl Session {
    /// Copy `geometry`'s current points to owned device storage as the frozen
    /// baseline, prepare the originals from that copy and score the baseline
    /// frame. Every submission, including the copy, has completed on return.
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
                (runner::Host::new(settings.host_bytes), settings.work_limit),
            )?;
            let kernels = kernels(&runner)?;
            let mut core = Core {
                runner,
                store,
                kernels,
            };
            core.runner.begin_phase();
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
                paths: geometry.paths.clone(),
                samples: geometry.samples,
                reservations: geometry.reservations,
                valid: false,
            };
            let preparation = &mut session.originals.preparation;
            preparation.active_samples = u64::from(geometry.samples);
            preparation.distance = session.assessor.statistics;
            preparation.declared_work_units = session.core.runner.phase_work();
            preparation.work = session.core.runner.work;
            preparation.frozen_baseline_bytes = mul(u64::from(geometry.samples), 12)?;
            preparation.allocated_bytes = session.core.store.allocated();
            preparation.peak_allocated_bytes = session.core.store.peak();
            preparation.readback_bytes = session.core.runner.readback_bytes;
            preparation.host_peak_bytes = session.core.runner.host.peak();
            session.core.runner.begin_phase();
            let frozen = session.originals.frozen.clone();
            session.baseline = session.frame(geometry, &frozen, true)?;
            session.valid = true;
            Ok(session)
        })
    }

    /// Score the renderer's current points against the frozen originals and
    /// baseline evidence. `geometry.points` may be a different buffer on every
    /// call; the paths buffer, counts and point capacity must match the frozen
    /// baseline's, and the device validates the point layout again.
    /// # Errors
    /// Any failure invalidates the session. No partial frame is returned.
    pub fn evaluate(&mut self, geometry: &Geometry) -> Result<Frame> {
        if !std::mem::replace(&mut self.valid, false) {
            return Err(Error::Invalid);
        }
        let points_fit = geometry.points.size() >= mul(u64::from(self.samples), 12)?;
        let same_layout = geometry.paths == self.paths
            && geometry.samples == self.samples
            && geometry.reservations == self.reservations;
        if !(points_fit && same_layout) {
            return Err(Error::GeometryChanged);
        }
        let device = self.device.clone();
        let frame = scoped(&device, || {
            self.core.runner.begin_phase();
            self.frame(geometry, &geometry.points, false)
        })?;
        self.valid = true;
        Ok(frame)
    }

    fn frame(
        &mut self,
        geometry: &Geometry,
        points: &wgpu::Buffer,
        baseline: bool,
    ) -> Result<Frame> {
        let mut statistics = FrameStatistics::default();
        let readback = self.core.runner.readback_bytes;
        let distance = self.assessor.statistics;
        let Self {
            core,
            originals,
            assessor,
            evidence,
            config,
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
            tracing::info!(contacts, "ambiguity contacts found");
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
        statistics.declared_work_units = self.core.runner.phase_work();
        statistics.work = self.core.runner.work;
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
}
