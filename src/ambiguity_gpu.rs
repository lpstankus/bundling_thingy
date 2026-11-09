//! GPU ambiguity contact acceleration: conservative candidate enumeration (S1)
//! and certified continuous-contact classification with complete per-pair CPU
//! fallback (S2).
//!
//! S1 reads the current resampled point buffer in place, derives positive
//! segments with exact bit comparisons, widens each segment box by half the
//! contact radius plus a proven f32 rounding margin, and enumerates box-overlap
//! pairs through a clamped uniform grid. Every partner pair is emitted once, by
//! its lower-ordinal source. S2 sorts each resident batch by an exact 64-bit
//! key, certifies every cell against the ordinary CPU-f64 contact kernel with
//! outward f32 intervals, and decides each complete source/partner run as
//! contact, no contact or complete CPU fallback. Fallback pairs take the exact
//! CPU pair route of `ambiguity_index`; resource, device and protocol failures
//! abort. Nothing is truncated and no backend is switched silently.
//!
//! A frame is only usable through `ambiguity_index::BroadPhase::Gpu`. Before
//! any candidate work it reads back the resident points and reservations and
//! compares them bit for bit with the CPU paths of that enumeration; radius,
//! minimum, ordered IDs, positive-segment offsets and CPU-f64 segment lengths
//! come from that same enumeration's preparation. Mismatches fail explicitly.
//! The renderer's geometry buffers are borrowed handle clones and must not be
//! mutated while an enumeration runs.
//!
//! One frame owner allocates every buffer from a single combined byte limit
//! (`Settings::memory_bytes`): geometry-derived tables first, count resources
//! next, then demand-sized resident candidates plus S2 workspace once a batch's
//! exact cell count is known. Batches are whole increasing source ranges, so a
//! pair never crosses batches. Work per invocation and per submission is
//! bounded; see `contact` for the classifier bounds.
use crate::{
    ambiguity_contact::PreparedCurve,
    ambiguity_index::Path,
    weak_frechet::{Budget, Error as ResourceError},
};
use std::{
    fmt,
    mem::size_of,
    ops::Range,
    sync::mpsc,
    time::{Duration, Instant},
};

mod contact;
mod resources;

pub(crate) use contact::Decision;
pub use contact::{Reasons, Statistics as ContactStatistics, Tally};
use resources::{Demand, Pool};

const BLOCK_SIZE: u32 = 256;
/// Partner entries examined by one candidate invocation (matches the WGSL).
pub const ITEM_ENTRIES: u32 = 64;
/// Candidate invocations per submission.
pub const ITEMS_PER_DISPATCH: u32 = 1 << 20;
const STATUS_WORDS: usize = 4;
const STATUS_BYTES: u64 = 16;
const POINT_BYTES: u64 = 12;
const PATH_BYTES: u64 = 12;
const SEGMENT_BYTES: u64 = 24;
const FOOTPRINT_BYTES: u64 = 8;
/// Uniform slots for every scan level encoded in one command buffer.
const UNIFORM_SLOTS: u32 = 16;
/// Storage offsets used by recursive scans are multiples of this.
const SCAN_OFFSET_ALIGNMENT: u32 = 256;
const SCAN_BLOCK: u32 = 512;
const INVALID_ORDINAL: u32 = u32::MAX;
/// Cell edges are powers of two from 2^-9 clip units; the grid covers [-2, 2]
/// and clamps everything outside to border cells.
const MINIMUM_CELL_SHIFT: i32 = -9;
const MAXIMUM_CELL_SHIFT: i32 = 2;
const POLL_TIMEOUT: Duration = Duration::from_secs(60);
const STATUS_NAMES: [&str; STATUS_WORDS] = [
    "a u32 prefix sum overflowed",
    "candidate count and fill passes disagree",
    "GPU points disagree with path reservations or eligible IDs",
    "a candidate batch exceeds the output buffer",
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
    /// The configured per-batch candidate record cap.
    BatchRecords,
    /// The shared stored-record ledger.
    StoredRecords,
    /// The shared host workspace ledger.
    HostWorkspace,
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
            Self::BatchRecords => f.write_str("--ambiguity-gpu-batch-records"),
            Self::StoredRecords => f.write_str("--ambiguity-stored-pair-limit"),
            Self::HostWorkspace => f.write_str("--ambiguity-workspace-bytes"),
        }
    }
}

/// A device result or schedule that violates the S1/S2 protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    NotBegun,
    Tables,
    Inventory,
    Readback,
    Token,
    FootprintOffsets,
    Status { words: [u32; STATUS_WORDS] },
    ScanOverflow,
    DuplicateCell,
    InvalidCandidate { status: u32 },
    PairCount { pairs: u32, cells: u32 },
    PairStarts,
    Schedule,
    Claims { errors: u32 },
    Execution { serial: u32, cooperative: u32 },
    LabelRange { failures: u32 },
    Outcome { pair: usize, word: u32 },
    Pair { pair: usize },
    Fallback { pair: usize },
    FallbackLength,
}
impl fmt::Display for Protocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotBegun => f.write_str("GPU frame was not begun"),
            Self::Tables => f.write_str("GPU frame tables disagree with the CPU preparation"),
            Self::Inventory => f.write_str("GPU buffer inventory is inconsistent"),
            Self::Readback => f.write_str("short GPU readback"),
            Self::Token => f.write_str("GPU batch filled without its matching count"),
            Self::FootprintOffsets => f.write_str("GPU footprint offsets omit positive segments"),
            Self::Status { words } => write!(f, "GPU contact status words {words:?}"),
            Self::ScanOverflow => f.write_str("GPU contact prefix sum overflowed"),
            Self::DuplicateCell => f.write_str("duplicate candidate cell"),
            Self::InvalidCandidate { status } => {
                write!(
                    f,
                    "invalid candidate fields, indices or order (status {status:#x})"
                )
            }
            Self::PairCount { pairs, cells } => write!(f, "{pairs} pairs over {cells} cells"),
            Self::PairStarts => f.write_str("pair starts are not strictly increasing from 0 to N"),
            Self::Schedule => f.write_str("classification schedule audit failed"),
            Self::Claims { errors } => write!(f, "classifier claim audit bits {errors:#x}"),
            Self::Execution {
                serial,
                cooperative,
            } => write!(
                f,
                "classifier executions {serial} serial and {cooperative} cooperative differ from the schedule"
            ),
            Self::Outcome { pair, word } => write!(f, "pair {pair}: invalid outcome {word:#x}"),
            Self::LabelRange { failures } => {
                write!(f, "classifier produced {failures} invalid label indices")
            }
            Self::Pair { pair } => {
                write!(f, "pair {pair}: invalid identity, count or contact route")
            }
            Self::Fallback { pair } => write!(f, "pair {pair}: invalid fallback run"),
            Self::FallbackLength => f.write_str("fallback runs do not cover the fallback copy"),
        }
    }
}

#[derive(Debug)]
pub enum Error {
    Parameters(&'static str),
    SizeOverflow,
    /// A fixed table or plan exceeds a limit.
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
    Geometry(String),
    Protocol(Protocol),
    /// The shared CPU evaluation ledger rejected a reservation.
    Workspace(ResourceError),
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
            Self::Geometry(text) => f.write_str(text),
            Self::Protocol(protocol) => write!(f, "GPU ambiguity protocol: {protocol}"),
            Self::Workspace(error) => write!(f, "GPU ambiguity host workspace: {error}"),
        }
    }
}
impl std::error::Error for Error {}
type Result<T> = std::result::Result<T, Error>;

const fn overflow() -> Error {
    Error::SizeOverflow
}

/// User-facing limits. The contact radius and minimum are not settings: each
/// enumeration uses its own narrow-phase values.
#[derive(Debug, Clone, Copy)]
pub struct Settings {
    /// Combined bytes of all GPU buffers S1 and S2 allocate for one frame.
    /// Renderer-owned geometry and driver overhead are excluded.
    pub memory_bytes: u64,
    /// Optional cap on candidate records per batch.
    pub batch_records: Option<u32>,
}

/// Current geometry already resident on the device. Its contents are verified
/// against the CPU paths of each enumeration before use and must not be
/// mutated while the enumeration runs.
#[derive(Clone)]
pub struct Geometry {
    pub points: wgpu::Buffer,
    pub paths: wgpu::Buffer,
    pub samples: u32,
    pub reservations: u32,
}

/// Combined device and host accounting of one frame.
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct Memory {
    pub limit_bytes: u64,
    /// Largest simultaneously allocated bytes of all owned buffers.
    pub peak_allocated_bytes: u64,
    pub fixed_bytes: u64,
    pub footprint_bytes: u64,
    pub peak_item_bytes: u64,
    pub peak_batch_bytes: u64,
    pub staging_cap_bytes: u64,
    /// Largest simultaneous stored-record slots of owned device buffers.
    pub peak_device_records: u64,
    /// Largest owner-local host bytes and stored records of the enumeration.
    pub peak_host_bytes: u64,
    pub peak_host_records: u64,
    /// Renderer-owned input geometry read in place; not allocated or limited here.
    pub borrowed_geometry_bytes: u64,
}

/// Host wait and transfer accounting of one stage's submissions.
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct Work {
    pub dispatches: u64,
    pub submissions: u64,
    /// Largest host submit-through-completion wait; not a per-pair duration.
    pub maximum_wait_seconds: f64,
    pub total_wait_seconds: f64,
}
impl Work {
    fn add_dispatches(&mut self, count: u64) -> Result<()> {
        self.dispatches = self.dispatches.checked_add(count).ok_or_else(overflow)?;
        Ok(())
    }
    fn add_dispatch(&mut self) -> Result<()> {
        self.add_dispatches(1)
    }
    fn record_submission(&mut self, elapsed: Duration) -> Result<()> {
        self.submissions = self.submissions.checked_add(1).ok_or_else(overflow)?;
        let seconds = elapsed.as_secs_f64();
        self.maximum_wait_seconds = self.maximum_wait_seconds.max(seconds);
        self.total_wait_seconds += seconds;
        Ok(())
    }
}

/// GPU-side counts and accounting, separate from CPU charged work units.
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct Statistics {
    pub validated_points: u64,
    pub positive_segments: u64,
    pub grid_cells_per_axis: u32,
    pub cell_size_clip: f64,
    pub footprint_entries: u64,
    pub count_passes: u64,
    pub filled_batches: u64,
    pub work_items: u64,
    pub largest_batch_items: u64,
    pub item_capacity: u64,
    pub partner_entries_per_item: u32,
    pub items_per_dispatch: u32,
    pub dispatches: u64,
    pub candidate_records: u64,
    pub largest_batch_records: u64,
    pub batch_capacity_records: u64,
    pub allocated_bytes: u64,
    pub memory_limit_bytes: u64,
    pub uploaded_bytes: u64,
    pub readback_bytes: u64,
    pub host_table_bytes: u64,
    /// S1 submissions and their host waits.
    pub candidate_work: Work,
    /// Multi-source ranges halved to fit a limit before fill.
    pub batch_splits: u64,
    pub memory: Memory,
    pub contact: ContactStatistics,
}

/// Host wall time spent waiting on GPU submissions, for logs only.
#[derive(Debug, Default, Clone, Copy)]
pub struct Timings {
    pub validation: Duration,
    pub setup: Duration,
    pub count: Duration,
    pub fill: Duration,
    pub contact: Duration,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Parameters {
    point_count: u32,
    path_count: u32,
    reservation_count: u32,
    grid_size: u32,
    cell_shift: i32,
    half_radius: f32,
    segment_capacity: u32,
    footprint_start: u32,
    footprint_count: u32,
    item_start: u32,
    item_count: u32,
    capacity: u32,
    padding: [u32; 2],
}

#[derive(Clone)]
struct Pipelines {
    mark_points: wgpu::ComputePipeline,
    count_segments: wgpu::ComputePipeline,
    emit_segments: wgpu::ComputePipeline,
    count_footprints: wgpu::ComputePipeline,
    gather_footprints: wgpu::ComputePipeline,
    emit_footprint: wgpu::ComputePipeline,
    fill_cells: wgpu::ComputePipeline,
    count_items: wgpu::ComputePipeline,
    count_candidates: wgpu::ComputePipeline,
    fill_candidates: wgpu::ComputePipeline,
    scan_blocks: wgpu::ComputePipeline,
    scan_add: wgpu::ComputePipeline,
}

/// Compiled kernels and limits, sharing the renderer's device and queue.
#[derive(Clone)]
pub struct Candidates {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Pipelines,
    contact: contact::Kernels,
    settings: Settings,
}

impl Candidates {
    /// # Errors
    /// Rejects zero limits, unsupported adapter alignment and shader failures.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, settings: Settings) -> Result<Self> {
        if settings.memory_bytes == 0 || settings.batch_records == Some(0) {
            return Err(Error::Parameters(
                "GPU ambiguity memory and batch records must be positive",
            ));
        }
        if device.limits().min_storage_buffer_offset_alignment > SCAN_OFFSET_ALIGNMENT {
            return Err(Error::Parameters(
                "GPU ambiguity scans need storage offsets aligned to at most 256 bytes",
            ));
        }
        let (pipelines, contact) = scoped(&device, || {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ambiguity candidates"),
                source: wgpu::ShaderSource::Wgsl(
                    format!(
                        "{}\n{}",
                        include_str!("shaders/graph.wgsl"),
                        include_str!("shaders/ambiguity_candidates.wgsl")
                    )
                    .into(),
                ),
            });
            let pipeline = |entry| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
            };
            Ok((
                Pipelines {
                    mark_points: pipeline("mark_points"),
                    count_segments: pipeline("count_segments"),
                    emit_segments: pipeline("emit_segments"),
                    count_footprints: pipeline("count_footprints"),
                    gather_footprints: pipeline("gather_footprints"),
                    emit_footprint: pipeline("emit_footprint"),
                    fill_cells: pipeline("fill_cells"),
                    count_items: pipeline("count_items"),
                    count_candidates: pipeline("count_candidates"),
                    fill_candidates: pipeline("fill_candidates"),
                    scan_blocks: pipeline("scan_blocks"),
                    scan_add: pipeline("scan_add"),
                },
                contact::Kernels::new(&device),
            ))
        })?;
        Ok(Self {
            device,
            queue,
            pipelines,
            contact,
            settings,
        })
    }

    /// Bind one frame's resident geometry. GPU work starts in the enumeration.
    #[must_use]
    pub fn frame(&self, geometry: Geometry) -> Frame {
        Frame {
            shared: self.clone(),
            geometry,
            state: None,
            statistics: Statistics::default(),
            timings: Timings::default(),
        }
    }
}

/// Smallest f32 at least half the radius, raised to the smallest normal value.
/// Bounded far below `f32::MAX` so a widened coordinate cannot round to infinity.
fn half_radius(radius: f64) -> Result<f32> {
    if !radius.is_finite() || radius < 0.0 {
        return Err(Error::Parameters(
            "GPU contact radius must be finite and nonnegative",
        ));
    }
    if radius == 0.0 {
        return Ok(0.0);
    }
    let half = radius / 2.0;
    if half > 2.0_f64.powi(64) {
        return Err(Error::Parameters(
            "GPU contact radius exceeds 2^65 input units",
        ));
    }
    let rounded = nearest_f32(half);
    let bound = if f64::from(rounded) < half {
        rounded.next_up()
    } else {
        rounded
    };
    Ok(bound.max(f32::MIN_POSITIVE))
}

/// Round-to-nearest f32 of `value`; callers bracket or correct the result.
const fn nearest_f32(value: f64) -> f32 {
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Rounded f64-to-f32 conversion; every caller corrects it outward."
    )]
    let rounded = value as f32;
    rounded
}

/// Cell shift and per-axis cell count. Cells are at least the contact radius.
fn grid(radius: f64) -> Result<(i32, u32)> {
    let mut shift = MINIMUM_CELL_SHIFT;
    while shift < MAXIMUM_CELL_SHIFT && 2.0_f64.powi(shift) < radius {
        shift = shift.checked_add(1).ok_or_else(overflow)?;
    }
    let exponent = u32::try_from(MAXIMUM_CELL_SHIFT.checked_sub(shift).ok_or_else(overflow)?)
        .map_err(|_| overflow())?;
    Ok((shift, 1_u32.checked_shl(exponent).ok_or_else(overflow)?))
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
/// Run `operation` inside device error scopes. The operation's own failure is
/// the primary error; a captured device error fails an otherwise successful run.
fn scoped<T>(device: &wgpu::Device, operation: impl FnOnce() -> Result<T>) -> Result<T> {
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let mut scope = Scope { device, open: true };
    let result = operation();
    let device_error = scope.pop();
    match (result, device_error) {
        (Err(error), _) | (Ok(_), Some(error)) => Err(error),
        (Ok(value), None) => Ok(value),
    }
}

fn words(bytes: &[u8]) -> impl Iterator<Item = u32> + '_ {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
}

fn check_status(status: &[u32]) -> Result<()> {
    let words: [u32; STATUS_WORDS] = status
        .try_into()
        .map_err(|_| Error::Protocol(Protocol::Readback))?;
    for (index, (word, name)) in words.iter().zip(STATUS_NAMES).enumerate() {
        if *word != 0 {
            if matches!(index, 1 | 3) {
                return Err(Error::Protocol(Protocol::Status { words }));
            }
            return Err(Error::Geometry(format!(
                "GPU candidate enumeration failed: {name}"
            )));
        }
    }
    Ok(())
}

fn plus_one(value: u32) -> Result<u32> {
    value.checked_add(1).ok_or_else(overflow)
}

fn add(total: &mut u64, value: u64) -> Result<()> {
    *total = total.checked_add(value).ok_or_else(overflow)?;
    Ok(())
}

/// Device and queue of one operation.
#[derive(Clone, Copy)]
struct Gpu<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
}
/// Unmaps a mapped staging buffer on success, error and unwind.
struct Mapped<'a>(&'a wgpu::Buffer);
impl Drop for Mapped<'_> {
    fn drop(&mut self) {
        self.0.unmap();
    }
}
impl Gpu<'_> {
    fn encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
    }
    fn bind<const N: usize>(
        &self,
        pipeline: &wgpu::ComputePipeline,
        group: u32,
        entries: [(u32, wgpu::BindingResource<'_>); N],
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ambiguity inputs"),
            layout: &pipeline.get_bind_group_layout(group),
            entries: &entries.map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource }),
        })
    }
    /// Encode `workgroups` workgroups as a 2D grid. Returns whether a pass was
    /// encoded.
    fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        groups: &[&wgpu::BindGroup],
        workgroups: u64,
    ) -> Result<bool> {
        if workgroups == 0 {
            return Ok(false);
        }
        let maximum = u64::from(self.device.limits().max_compute_workgroups_per_dimension);
        let rows = workgroups.div_ceil(maximum);
        let columns = workgroups.div_ceil(rows);
        if rows > maximum {
            return Err(Error::Limit {
                limit: Limit::Dispatch,
                demand: workgroups,
                available: maximum.saturating_mul(maximum),
            });
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ambiguity"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        for (index, group) in (0..).zip(groups) {
            pass.set_bind_group(index, *group, &[]);
        }
        pass.dispatch_workgroups(
            u32::try_from(columns).map_err(|_| overflow())?,
            u32::try_from(rows).map_err(|_| overflow())?,
            1,
        );
        Ok(true)
    }
    fn wait(&self) -> Result<()> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(POLL_TIMEOUT),
            })
            .map(|_| ())
            .map_err(|error| Error::Poll(error.to_string()))
    }
    /// Submit and wait for completion; returns the host wait.
    fn submit(&self, encoder: wgpu::CommandEncoder) -> Result<Duration> {
        let started = Instant::now();
        self.queue.submit([encoder.finish()]);
        self.wait()?;
        Ok(started.elapsed())
    }
    /// Submit, map `bytes` of `staging` and decode them; the mapping is
    /// released on every exit and no callback runs while it is mapped.
    fn submit_and_read(
        &self,
        encoder: wgpu::CommandEncoder,
        staging: &wgpu::Buffer,
        bytes: u64,
        consume: impl FnOnce(&[u8]) -> Result<()>,
    ) -> Result<Duration> {
        let started = Instant::now();
        self.queue.submit([encoder.finish()]);
        let slice = staging.slice(..bytes);
        let (send, receive) = mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
        self.wait()?;
        // A successful wait delivered the callback; never block on the channel.
        receive
            .try_recv()
            .map_err(|error| Error::Map(error.to_string()))?
            .map_err(|error| Error::Map(error.to_string()))?;
        let mapped = Mapped(staging);
        let view = slice.get_mapped_range();
        let result = consume(&view);
        drop(view);
        drop(mapped);
        result.map(|()| started.elapsed())
    }
}

/// Record layout of a staged read.
#[derive(Clone, Copy)]
struct Content {
    record_bytes: u64,
    /// Whether the staged bytes are candidate or pair records.
    records: bool,
}
/// Pool-owned staging for bounded chunked readbacks.
struct Transfer<'a> {
    pool: &'a mut Pool,
    budget: &'a mut Budget,
}
impl Transfer<'_> {
    /// Read `bytes` of `source` through staging sized by the read and capped,
    /// in record-aligned chunks. Staging is destroyed before returning.
    fn read(
        &mut self,
        gpu: &Gpu<'_>,
        work: &mut Work,
        (source, bytes): (&wgpu::Buffer, u64),
        content: Content,
        mut consume: impl FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let spec = resources::staging(bytes, content.records)?;
        let staging = self.pool.allocate(self.budget, &spec)?;
        let alignment = content.record_bytes.checked_mul(4).ok_or_else(overflow)?;
        let size = staging.buffer().size();
        let chunk_bytes = size
            .checked_sub(size.checked_rem(alignment).ok_or_else(overflow)?)
            .filter(|chunk| *chunk > 0)
            .ok_or(Error::Protocol(Protocol::Inventory))?;
        let result = (|| {
            let mut offset = 0_u64;
            while offset < bytes {
                let chunk = bytes.saturating_sub(offset).min(chunk_bytes);
                let mut encoder = gpu.encoder();
                encoder.copy_buffer_to_buffer(source, offset, staging.buffer(), 0, chunk);
                let elapsed =
                    gpu.submit_and_read(encoder, staging.buffer(), chunk, &mut consume)?;
                work.record_submission(elapsed)?;
                offset = offset.checked_add(chunk).ok_or_else(overflow)?;
            }
            Ok(())
        })();
        let freed = self.pool.free(self.budget, staging);
        result.and(freed)
    }
}

/// Scan kernels of one shader and their binding layout.
#[derive(Clone, Copy)]
enum ScanLayout {
    /// S1: values, sums and uniform in group 1 (bindings 7, 8, 6).
    Candidates,
    /// S2: values, sums and uniform in group 2 (bindings 1, 2, 0).
    Contact,
}
#[derive(Clone, Copy)]
struct ScanKernels<'a> {
    blocks: &'a wgpu::ComputePipeline,
    add: &'a wgpu::ComputePipeline,
    status: &'a wgpu::Buffer,
    uniforms: &'a wgpu::Buffer,
    stride: u64,
    layout: ScanLayout,
}
impl ScanKernels<'_> {
    /// Encode an exclusive in-place scan of `count` u32 values at `offset`.
    /// Each level takes the next uniform slot of `slot`, which counts slots
    /// across every scan encoded into the same command buffer: queue writes
    /// land before that buffer executes, so slots must never be reused within
    /// it. Returns the number of dispatches encoded.
    fn encode(
        &self,
        gpu: &Gpu<'_>,
        encoder: &mut wgpu::CommandEncoder,
        (values, scratch): ((&wgpu::Buffer, u64, u32), &wgpu::Buffer),
        slot: &mut u32,
    ) -> Result<u64> {
        self.level(gpu, encoder, values, (scratch, 0), slot)
    }
    /// Bind groups of one scan pipeline: status, then the scan group (S1) or
    /// an empty group 1 and the scan group (S2).
    fn groups(
        &self,
        gpu: &Gpu<'_>,
        pipeline: &wgpu::ComputePipeline,
        (uniform, values, sums): (
            wgpu::BindingResource<'_>,
            wgpu::BindingResource<'_>,
            wgpu::BindingResource<'_>,
        ),
    ) -> (wgpu::BindGroup, wgpu::BindGroup, Option<wgpu::BindGroup>) {
        let status = gpu.bind(pipeline, 0, [(7, self.status.as_entire_binding())]);
        match self.layout {
            ScanLayout::Candidates => (
                status,
                gpu.bind(pipeline, 1, [(6, uniform), (7, values), (8, sums)]),
                None,
            ),
            ScanLayout::Contact => (
                status,
                gpu.bind(pipeline, 1, []),
                Some(gpu.bind(pipeline, 2, [(0, uniform), (1, values), (2, sums)])),
            ),
        }
    }
    fn run(
        gpu: &Gpu<'_>,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        groups: &(wgpu::BindGroup, wgpu::BindGroup, Option<wgpu::BindGroup>),
        workgroups: u32,
    ) -> Result<u64> {
        let (first, second, third) = groups;
        let dispatched = match third {
            Some(third) => gpu.dispatch(
                encoder,
                pipeline,
                &[first, second, third],
                u64::from(workgroups),
            )?,
            None => gpu.dispatch(encoder, pipeline, &[first, second], u64::from(workgroups))?,
        };
        Ok(u64::from(dispatched))
    }
    fn level(
        &self,
        gpu: &Gpu<'_>,
        encoder: &mut wgpu::CommandEncoder,
        (buffer, offset, count): (&wgpu::Buffer, u64, u32),
        (scratch, scratch_offset): (&wgpu::Buffer, u64),
        slot: &mut u32,
    ) -> Result<u64> {
        if *slot >= UNIFORM_SLOTS {
            return Err(Error::Protocol(Protocol::Inventory));
        }
        let uniform_offset = u64::from(*slot)
            .checked_mul(self.stride)
            .ok_or_else(overflow)?;
        *slot = plus_one(*slot)?;
        gpu.queue.write_buffer(
            self.uniforms,
            uniform_offset,
            bytemuck::cast_slice(&[count, 0, 0, 0]),
        );
        let blocks = count.div_ceil(SCAN_BLOCK);
        let sums_bytes = resources::scan_region(blocks)?;
        let binding = |buffer, offset, size| {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset,
                size: std::num::NonZeroU64::new(size),
            })
        };
        let value_bytes = resources::words(count, 4)?;
        let make = |pipeline| {
            self.groups(
                gpu,
                pipeline,
                (
                    binding(self.uniforms, uniform_offset, 16),
                    binding(buffer, offset, value_bytes),
                    binding(scratch, scratch_offset, sums_bytes),
                ),
            )
        };
        let first = make(self.blocks);
        let mut dispatches = Self::run(gpu, encoder, self.blocks, &first, blocks)?;
        if blocks > 1 {
            let next = scratch_offset
                .checked_add(sums_bytes)
                .ok_or_else(overflow)?;
            let inner = self.level(
                gpu,
                encoder,
                (scratch, scratch_offset, blocks),
                (scratch, next),
                slot,
            )?;
            let second = make(self.add);
            let added = Self::run(gpu, encoder, self.add, &second, blocks)?;
            dispatches = dispatches
                .checked_add(inner)
                .and_then(|total| total.checked_add(added))
                .ok_or_else(overflow)?;
        }
        Ok(dispatches)
    }
}

// Fixed slot indices, in `resources::fixed` order.
const PARAMETERS: usize = 0;
const UNIFORMS: usize = 1;
const SUMMARY: usize = 2;
const ORDER: usize = 3;
const ORDINALS: usize = 4;
const POINT_SCAN: usize = 5;
const PATH_BASE: usize = 6;
const PATH_FOOTPRINTS: usize = 7;
const STATUS: usize = 8;
const SEGMENTS: usize = 9;
const FOOTPRINT_OFFSETS: usize = 10;
const CELL_COUNTS: usize = 11;
const CELL_CURSOR: usize = 12;
const SETUP_SCRATCH: usize = 13;
const PLACEHOLDER: usize = 14;
const CONTACT_PARAMETERS: usize = 15;
const CONTACT_STATUS: usize = 16;
const PATH_INFO: usize = 17;
const LENGTHS: usize = 18;
const SEGMENT_END: usize = 19;
// Footprint and item slot indices.
const FOOTPRINT: usize = 0;
const ENTRIES: usize = 1;
const WEIGHTS: usize = 2;
const FOOTPRINT_SCRATCH: usize = 3;
const ITEM_COUNTS: usize = 0;
const ITEM_SCRATCH: usize = 1;

/// The private count result that alone authorizes the matching fill.
struct Counted {
    generation: u64,
    sources: Range<usize>,
    footprints: Range<u32>,
    items: u32,
    records: u32,
}

/// Begun frame state; every buffer is owned by `pool`'s inventory.
struct State {
    pool: Pool,
    parameters: Parameters,
    fixed: resources::Slots<{ resources::FIXED }>,
    footprint: Option<resources::Slots<4>>,
    /// Item-count capacity class and its buffers.
    items: Option<(u32, resources::Slots<2>)>,
    /// Retained batch-class resident candidates and S2 workspace.
    batch: Option<contact::Buffers>,
    counted: Option<Counted>,
    generation: u64,
    keys: contact::Keys,
    contact: contact::Parameters,
    uniform_stride: u64,
}

/// Narrow-phase inputs the GPU frame must match exactly.
pub(crate) struct Request<'a, 'p> {
    pub paths: &'a [Path<'p>],
    pub prepared: &'a [PreparedCurve<'p>],
    /// Positive-segment offsets by ordinal, plus the total.
    pub segment_offsets: &'a [u32],
    pub radius: f64,
    pub minimum: f64,
}

/// One frame's GPU contact source, consumed by `BroadPhase::Gpu`.
pub struct Frame {
    shared: Candidates,
    geometry: Geometry,
    state: Option<State>,
    statistics: Statistics,
    timings: Timings,
}
impl Frame {
    #[must_use]
    pub const fn statistics(&self) -> Statistics {
        self.statistics
    }
    #[must_use]
    pub const fn timings(&self) -> Timings {
        self.timings
    }
    /// Destroy every owned buffer; statistics remain. Returns the stored-record
    /// slots the destroyed buffers held, for their owner to release.
    fn end(&mut self) -> u64 {
        let Some(state) = self.state.take() else {
            return 0;
        };
        let memory = &mut self.statistics.memory;
        memory.peak_allocated_bytes = memory.peak_allocated_bytes.max(state.pool.peak());
        memory.peak_device_records = memory.peak_device_records.max(state.pool.peak_records());
        self.statistics.allocated_bytes = memory.peak_allocated_bytes;
        let records = state.pool.records();
        drop(state);
        records
    }
    /// Begin an enumeration session: validate the resident geometry against
    /// `request`, build the frame tables and S2 tables.
    pub(crate) fn session<'s, 'p>(
        &'s mut self,
        request: &'s Request<'s, 'p>,
        budget: &'s mut Budget,
    ) -> Result<Session<'s, 'p>> {
        let mut session = Session {
            frame: self,
            budget,
            request,
            holdings: Holdings::default(),
            footprints: Vec::new(),
            host: None,
            start: 0,
            span: 1,
            frozen_until: 0,
        };
        session.begin()?;
        Ok(session)
    }
}

/// Owner-local host bytes and stored records on the shared ledger.
#[derive(Default)]
struct Holdings {
    bytes: usize,
    records: usize,
    peak_bytes: usize,
    peak_records: usize,
}
impl Holdings {
    /// Reserve bytes and records together; nothing changes on failure.
    fn reserve(&mut self, budget: &mut Budget, bytes: usize, records: usize) -> Result<()> {
        let total_bytes = self.bytes.checked_add(bytes).ok_or_else(overflow)?;
        let total_records = self.records.checked_add(records).ok_or_else(overflow)?;
        budget.reserve(bytes).map_err(Error::Workspace)?;
        if let Err(error) = budget.reserve_records(records) {
            budget.release(bytes).map_err(Error::Workspace)?;
            return Err(Error::Workspace(error));
        }
        self.bytes = total_bytes;
        self.records = total_records;
        self.peak_bytes = self.peak_bytes.max(total_bytes);
        self.peak_records = self.peak_records.max(total_records);
        Ok(())
    }
    fn release(&mut self, budget: &mut Budget, bytes: usize, records: usize) -> Result<()> {
        self.bytes = self
            .bytes
            .checked_sub(bytes)
            .ok_or(Error::Workspace(ResourceError::InvalidRelease))?;
        self.records = self
            .records
            .checked_sub(records)
            .ok_or(Error::Workspace(ResourceError::InvalidRelease))?;
        budget.release(bytes).map_err(Error::Workspace)?;
        budget.release_records(records).map_err(Error::Workspace)
    }
    /// A byte-only vector reserved before allocation, including any capacity
    /// surplus the allocator exposes.
    fn vector<T>(&mut self, budget: &mut Budget, capacity: usize) -> Result<Vec<T>> {
        let bytes = capacity.checked_mul(size_of::<T>()).ok_or_else(overflow)?;
        self.reserve(budget, bytes, 0)?;
        let mut values = Vec::new();
        if values.try_reserve_exact(capacity).is_err() {
            self.release(budget, bytes, 0)?;
            return Err(Error::Workspace(ResourceError::Allocation));
        }
        let surplus = values
            .capacity()
            .saturating_sub(capacity)
            .checked_mul(size_of::<T>())
            .ok_or_else(overflow)?;
        if let Err(error) = self.reserve(budget, surplus, 0) {
            drop(values);
            self.release(budget, bytes, 0)?;
            return Err(error);
        }
        Ok(values)
    }
    fn free<T>(&mut self, budget: &mut Budget, values: Vec<T>) -> Result<()> {
        let bytes = values
            .capacity()
            .checked_mul(size_of::<T>())
            .ok_or_else(overflow)?;
        drop(values);
        self.release(budget, bytes, 0)
    }
}

/// Planned limit that a batch range exceeded.
#[derive(Clone, Copy)]
struct Shortfall {
    limit: Limit,
    demand: u64,
    available: u64,
}
impl Shortfall {
    /// Only pure plan limits are retryable; execution errors remain fatal.
    fn planned(error: Error) -> Result<Self> {
        match error {
            Error::Limit {
                limit,
                demand,
                available,
            } => Ok(Self {
                limit,
                demand,
                available,
            }),
            other => Err(other),
        }
    }
}
fn number(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| overflow())
}

/// One enumeration's frame owner. It holds the shared ledger for its scope,
/// lends it to callbacks between batches, and on drop (success, error or
/// unwind) destroys every owned buffer and releases only its own reservations.
pub(crate) struct Session<'s, 'p> {
    frame: &'s mut Frame,
    budget: &'s mut Budget,
    request: &'s Request<'s, 'p>,
    holdings: Holdings,
    footprints: Vec<u32>,
    /// Host arrays of the current batch and their reserved bytes and records.
    host: Option<(contact::Host, (usize, usize))>,
    start: usize,
    span: usize,
    /// Growth stays disabled until batches pass the end of a range that had to
    /// be halved, so the same region is never retried at the failed size.
    frozen_until: usize,
}
impl Drop for Session<'_, '_> {
    fn drop(&mut self) {
        let _ = self.release_host();
        let footprints = std::mem::take(&mut self.footprints);
        let _ = self.holdings.free(self.budget, footprints);
        let memory = &mut self.frame.statistics.memory;
        memory.peak_host_bytes = memory
            .peak_host_bytes
            .max(number(self.holdings.peak_bytes).unwrap_or(u64::MAX));
        memory.peak_host_records = memory
            .peak_host_records
            .max(number(self.holdings.peak_records).unwrap_or(u64::MAX));
        let records = self.frame.end();
        let _ = self
            .budget
            .release_records(usize::try_from(records).unwrap_or(usize::MAX));
        let (bytes, records) = (self.holdings.bytes, self.holdings.records);
        let _ = self.holdings.release(self.budget, bytes, records);
    }
}

/// One validated batch, lent to the caller with the shared ledger.
pub(crate) struct Batch<'s> {
    host: &'s mut contact::Host,
    budget: &'s mut Budget,
    cells: u32,
}
impl Batch<'_> {
    /// Resident candidate cells of this batch.
    pub(crate) const fn cells(&self) -> u32 {
        self.cells
    }
    /// Visit decisions in increasing pair order. Every device result of this
    /// batch was read, its error scope popped and its protocol validated before
    /// this batch existed.
    pub(crate) fn decide<X: From<Error>>(
        &mut self,
        mut decide: impl FnMut(Decision<'_>, &mut Budget) -> std::result::Result<(), X>,
    ) -> std::result::Result<(), X> {
        let budget = &mut *self.budget;
        self.host.visit(|decision| decide(decision, budget))
    }
}

/// Borrowed pieces of a begun frame plus the session's ledger.
struct Parts<'a> {
    gpu: Gpu<'a>,
    pipelines: &'a Pipelines,
    kernels: &'a contact::Kernels,
    settings: Settings,
    geometry: &'a Geometry,
    state: &'a mut State,
    statistics: &'a mut Statistics,
    timings: &'a mut Timings,
    budget: &'a mut Budget,
    holdings: &'a mut Holdings,
}
fn parts<'a>(
    frame: &'a mut Frame,
    budget: &'a mut Budget,
    holdings: &'a mut Holdings,
) -> Result<Parts<'a>> {
    let Frame {
        shared,
        geometry,
        state,
        statistics,
        timings,
    } = frame;
    Ok(Parts {
        gpu: Gpu {
            device: &shared.device,
            queue: &shared.queue,
        },
        pipelines: &shared.pipelines,
        kernels: &shared.contact,
        settings: shared.settings,
        geometry,
        state: state.as_mut().ok_or(Error::Protocol(Protocol::NotBegun))?,
        statistics,
        timings,
        budget,
        holdings,
    })
}

/// Sizes the begun frame state determined.
struct Begun {
    samples: u32,
    cells: u32,
    segment_capacity: u32,
}

/// Largest positive-segment count of one path (at least one).
fn maximum_segments(offsets: &[u32]) -> Result<u32> {
    let mut maximum = 1_u32;
    for pair in offsets.windows(2) {
        let (Some(start), Some(end)) = (pair.first(), pair.last()) else {
            return Err(Error::Protocol(Protocol::Tables));
        };
        let count = end
            .checked_sub(*start)
            .ok_or(Error::Protocol(Protocol::Tables))?;
        maximum = maximum.max(count);
    }
    Ok(maximum)
}

/// Allocate the fixed inventory and install the begun state of `frame`.
fn initial_state(
    frame: &mut Frame,
    request: &Request<'_, '_>,
    budget: &mut Budget,
) -> Result<Begun> {
    frame.state = None;
    frame.timings = Timings::default();
    let half_radius = half_radius(request.radius)?;
    let (cell_shift, grid_size) = grid(request.radius)?;
    let path_count = u32::try_from(request.paths.len()).map_err(|_| overflow())?;
    if request.segment_offsets.len() != request.paths.len().saturating_add(1)
        || request.prepared.len() != request.paths.len()
    {
        return Err(Error::Protocol(Protocol::Tables));
    }
    let keys = contact::Keys::new(path_count, maximum_segments(request.segment_offsets)?)?;
    let samples = frame.geometry.samples;
    let cells = grid_size.checked_mul(grid_size).ok_or_else(overflow)?;
    // Positive segments never exceed points minus nonempty paths.
    let segment_capacity = samples.saturating_sub(path_count).max(1);
    let limits = frame.shared.device.limits();
    let uniform_stride = u64::from(limits.min_uniform_buffer_offset_alignment).max(16);
    let shape = resources::Shape {
        samples,
        paths: path_count,
        reservations: frame.geometry.reservations,
        segment_capacity,
        segments: *request
            .segment_offsets
            .last()
            .ok_or(Error::Protocol(Protocol::Tables))?,
        cells,
        uniform_stride,
        s1_parameters: resources::words(size_of::<Parameters>(), 1)?,
        s2_parameters: resources::words(size_of::<contact::Parameters>(), 1)?,
    };
    let mut pool = Pool::new(
        frame.shared.device.clone(),
        frame.shared.settings.memory_bytes,
    );
    let fixed = resources::Slots::allocate(&mut pool, budget, &resources::fixed(&shape)?)?;
    let scalars = contact::admit(request.radius, request.minimum);
    frame.statistics = Statistics {
        grid_cells_per_axis: grid_size,
        cell_size_clip: 2.0_f64.powi(cell_shift),
        memory_limit_bytes: frame.shared.settings.memory_bytes,
        partner_entries_per_item: ITEM_ENTRIES,
        items_per_dispatch: ITEMS_PER_DISPATCH,
        memory: Memory {
            limit_bytes: frame.shared.settings.memory_bytes,
            fixed_bytes: pool.allocated(),
            staging_cap_bytes: resources::MAXIMUM_STAGING_BYTES,
            borrowed_geometry_bytes: frame
                .geometry
                .points
                .size()
                .checked_add(frame.geometry.paths.size())
                .ok_or_else(overflow)?,
            ..Memory::default()
        },
        contact: ContactStatistics {
            admitted_scalars: scalars.is_some(),
            work_target: contact::WORK_TARGET,
            ..ContactStatistics::default()
        },
        ..Statistics::default()
    };
    frame.state = Some(State {
        pool,
        parameters: Parameters {
            point_count: samples,
            path_count,
            reservation_count: frame.geometry.reservations,
            grid_size,
            cell_shift,
            half_radius,
            segment_capacity,
            footprint_start: 0,
            footprint_count: 0,
            item_start: 0,
            item_count: 0,
            capacity: 0,
            padding: [0; 2],
        },
        fixed,
        footprint: None,
        items: None,
        batch: None,
        counted: None,
        generation: 0,
        keys,
        contact: contact::parameters(scalars, keys, path_count),
        uniform_stride,
    });
    Ok(Begun {
        samples,
        cells,
        segment_capacity,
    })
}

impl Session<'_, '_> {
    fn begin(&mut self) -> Result<()> {
        let device = self.frame.shared.device.clone();
        scoped(&device, || self.setup())
    }

    fn setup(&mut self) -> Result<()> {
        let request = self.request;
        let Begun {
            samples,
            cells,
            segment_capacity,
        } = initial_state(self.frame, request, self.budget)?;
        let mut p = parts(self.frame, self.budget, &mut self.holdings)?;
        p.validate(request.paths)?;
        p.upload_tables(request.paths)?;
        let tables = request.segment_offsets.len();
        let mut footprints = p.holdings.vector::<u32>(p.budget, tables)?;
        footprints.resize(tables, 0);
        let total = p.encode_geometry(request, &mut footprints, segment_capacity);
        self.footprints = footprints;
        let total = total?;
        let mut p = parts(self.frame, self.budget, &mut self.holdings)?;
        let slots =
            resources::Slots::allocate(&mut p.state.pool, p.budget, &resources::footprint(total)?)?;
        p.statistics.memory.footprint_bytes = slots.bytes();
        p.state.footprint = Some(slots);
        p.encode_cells(total, cells)?;
        p.upload_contact(request, samples)?;
        p.statistics.host_table_bytes = number(p.holdings.peak_bytes)?;
        Ok(())
    }

    fn release_host(&mut self) -> Result<()> {
        if let Some((host, (bytes, records))) = self.host.take() {
            drop(host);
            self.holdings.release(self.budget, bytes, records)?;
        }
        Ok(())
    }

    /// Count, plan, fill and classify the next whole-source batch. Returns
    /// `None` after the last source.
    pub(crate) fn next_batch(&mut self) -> Result<Option<Batch<'_>>> {
        self.release_host()?;
        if self.start >= self.request.paths.len() {
            return Ok(None);
        }
        let (sources, cells) = self.prepare()?;
        let device = self.frame.shared.device.clone();
        let started = Instant::now();
        scoped(&device, || {
            let mut p = parts(self.frame, self.budget, &mut self.holdings)?;
            let resident = p.fill(&sources, cells)?;
            let host = &mut self
                .host
                .as_mut()
                .ok_or(Error::Protocol(Protocol::Inventory))?
                .0;
            p.classify(&resident, host, self.request.segment_offsets)
        })?;
        self.frame.timings.contact = self.frame.timings.contact.saturating_add(started.elapsed());
        self.advance(&sources, cells)?;
        let host = &mut self
            .host
            .as_mut()
            .ok_or(Error::Protocol(Protocol::Inventory))?
            .0;
        Ok(Some(Batch {
            host,
            budget: &mut *self.budget,
            cells,
        }))
    }

    fn range_end(&self, span: usize) -> Result<usize> {
        let keys = self
            .frame
            .state
            .as_ref()
            .ok_or(Error::Protocol(Protocol::NotBegun))?
            .keys;
        Ok(self
            .start
            .saturating_add(span.clamp(1, keys.maximum_span))
            .min(self.request.paths.len()))
    }

    /// Count the next range and reserve its batch, halving multi-source ranges
    /// that exceed a limit. A single source that cannot fit is `SourceCapacity`.
    fn prepare(&mut self) -> Result<(Range<usize>, u32)> {
        let mut end = self.range_end(self.span)?;
        loop {
            let sources = self.start..end;
            let footprint = |index: usize| {
                self.footprints
                    .get(index)
                    .copied()
                    .ok_or(Error::Protocol(Protocol::FootprintOffsets))
            };
            let footprints = footprint(sources.start)?..footprint(sources.end)?;
            let device = self.frame.shared.device.clone();
            let counted = scoped(&device, || {
                parts(self.frame, self.budget, &mut self.holdings)?
                    .count(sources.clone(), footprints.clone())
            })?;
            let shortfall = match counted {
                Ok(cells) => match scoped(&device, || self.reserve(cells))? {
                    None => return Ok((sources, cells)),
                    Some(shortfall) => shortfall,
                },
                Err(shortfall) => shortfall,
            };
            let length = end.saturating_sub(self.start);
            if length > 1 {
                self.frozen_until = self.frozen_until.max(end);
                end = self.start.saturating_add(length / 2);
                self.span = end.saturating_sub(self.start);
                add(&mut self.frame.statistics.batch_splits, 1)?;
                continue;
            }
            let path = self
                .request
                .paths
                .get(self.start)
                .ok_or(Error::Protocol(Protocol::Tables))?;
            return Err(Error::SourceCapacity {
                source: path.id,
                limit: shortfall.limit,
                demand: shortfall.demand,
                available: shortfall.available,
            });
        }
    }

    /// Reserve host arrays and resident batch buffers for `cells` cells before
    /// fill. Releases the retained batch cache before declaring a shortfall.
    fn reserve(&mut self, cells: u32) -> Result<Option<Shortfall>> {
        let (bytes, records) = contact::host_demand(cells)?;
        loop {
            let cached = self
                .frame
                .state
                .as_ref()
                .and_then(|state| state.batch.as_ref())
                .is_some();
            if let Err(error) = self.holdings.reserve(self.budget, bytes, records) {
                if cached {
                    self.release_cache()?;
                    continue;
                }
                return match error {
                    Error::Workspace(ResourceError::RecordLimit) => Ok(Some(Shortfall {
                        limit: Limit::StoredRecords,
                        demand: number(records)?,
                        available: number(self.budget.available_records())?,
                    })),
                    Error::Workspace(ResourceError::WorkspaceLimit) => Ok(Some(Shortfall {
                        limit: Limit::HostWorkspace,
                        demand: number(bytes)?,
                        available: number(self.budget.available_workspace())?,
                    })),
                    other => Err(other),
                };
            }
            match self.allocate_batch(cells)? {
                None => break,
                Some(shortfall) => {
                    self.holdings.release(self.budget, bytes, records)?;
                    if cached {
                        self.release_cache()?;
                        continue;
                    }
                    return Ok(Some(shortfall));
                }
            }
        }
        let host = contact::Host::allocate(cells)?;
        let (actual_bytes, actual_records) = host.footprint()?;
        let surplus = (
            actual_bytes.saturating_sub(bytes),
            actual_records.saturating_sub(records),
        );
        if let Err(error) = self.holdings.reserve(self.budget, surplus.0, surplus.1) {
            drop(host);
            self.holdings.release(self.budget, bytes, records)?;
            return Err(error);
        }
        let reserved = (
            bytes.checked_add(surplus.0).ok_or_else(overflow)?,
            records.checked_add(surplus.1).ok_or_else(overflow)?,
        );
        self.host = Some((host, reserved));
        Ok(None)
    }

    fn release_cache(&mut self) -> Result<()> {
        let state = self
            .frame
            .state
            .as_mut()
            .ok_or(Error::Protocol(Protocol::NotBegun))?;
        if let Some(batch) = state.batch.take() {
            batch.owned.free(&mut state.pool, self.budget)?;
        }
        Ok(())
    }

    /// Keep a retained batch class that covers `cells`, or replace it with a
    /// demand-sized class (next power of two when it fits, else exact).
    fn allocate_batch(&mut self, cells: u32) -> Result<Option<Shortfall>> {
        if cells == 0 {
            return Ok(None);
        }
        let state = self
            .frame
            .state
            .as_mut()
            .ok_or(Error::Protocol(Protocol::NotBegun))?;
        let staging = state.pool.demand(&[resources::batch_staging(cells)?])?;
        let staging_fits = staging.bytes <= state.pool.available(0)
            && staging.records <= number(self.budget.available_records())?;
        if staging_fits
            && state
                .batch
                .as_ref()
                .is_some_and(|batch| batch.capacity >= cells)
        {
            return Ok(None);
        }
        if let Some(batch) = state.batch.take() {
            batch.owned.free(&mut state.pool, self.budget)?;
        }
        let mut shortfall = None;
        for class in [cells.checked_next_power_of_two().unwrap_or(cells), cells] {
            let plan = (|| {
                let specs = resources::batch(class)?;
                let demand = state
                    .pool
                    .demand(&specs)?
                    .plus(state.pool.demand(&[resources::batch_staging(class)?])?)?;
                Ok((specs, demand))
            })();
            let (specs, demand) = match plan {
                Ok(plan) => plan,
                Err(error) => {
                    shortfall = Some(Shortfall::planned(error)?);
                    continue;
                }
            };
            let available = Demand {
                bytes: state.pool.available(0),
                records: number(self.budget.available_records())?,
            };
            if demand.bytes > available.bytes {
                shortfall = Some(Shortfall {
                    limit: Limit::DeviceMemory("ambiguity batch"),
                    demand: demand.bytes,
                    available: available.bytes,
                });
                continue;
            }
            if demand.records > available.records {
                shortfall = Some(Shortfall {
                    limit: Limit::StoredRecords,
                    demand: demand.records,
                    available: available.records,
                });
                continue;
            }
            let owned = resources::Slots::allocate(&mut state.pool, self.budget, &specs)?;
            let memory = &mut self.frame.statistics.memory;
            memory.peak_batch_bytes = memory.peak_batch_bytes.max(owned.bytes());
            self.frame.statistics.batch_capacity_records = self
                .frame
                .statistics
                .batch_capacity_records
                .max(u64::from(class));
            state.batch = Some(contact::Buffers {
                capacity: class,
                owned,
            });
            return Ok(None);
        }
        Ok(shortfall)
    }

    /// Whether a batch of `cells` cells would fit with the retained cache and
    /// this batch's host arrays released.
    fn feasible(&self, cells: u32) -> Result<bool> {
        let state = self
            .frame
            .state
            .as_ref()
            .ok_or(Error::Protocol(Protocol::NotBegun))?;
        if cells > self.frame.shared.settings.batch_records.unwrap_or(u32::MAX) {
            return Ok(false);
        }
        let Ok(specs) = resources::batch(cells) else {
            return Ok(false);
        };
        let (Ok(device), Ok(staging)) = (
            state.pool.demand(&specs),
            state.pool.demand(&[resources::batch_staging(cells)?]),
        ) else {
            return Ok(false);
        };
        let device = device.plus(staging)?;
        let (cache_bytes, cache_records) = state
            .batch
            .as_ref()
            .map_or((0, 0), |batch| (batch.owned.bytes(), batch.owned.records()));
        let (held_bytes, held_records) = self.host.as_ref().map_or((0, 0), |host| host.1);
        let (host_bytes, host_records) = contact::host_demand(cells)?;
        let records = device
            .records
            .checked_add(number(host_records)?)
            .ok_or_else(overflow)?;
        let record_room = number(self.budget.available_records())?
            .saturating_add(cache_records)
            .saturating_add(number(held_records)?);
        let byte_room = self.budget.available_workspace().saturating_add(held_bytes);
        Ok(device.bytes <= state.pool.available(cache_bytes)
            && records <= record_room
            && host_bytes <= byte_room)
    }

    /// Advance past `sources`. A range doubles only when twice its cells fit
    /// the planner's feasible batch and it lies beyond any halved region.
    fn advance(&mut self, sources: &Range<usize>, cells: u32) -> Result<()> {
        let length = sources.len();
        let grow = sources.end >= self.frozen_until
            && (cells == 0 || self.feasible(cells.saturating_mul(2))?);
        self.span = if grow {
            length.saturating_mul(2)
        } else {
            length
        }
        .max(1);
        self.start = sources.end;
        Ok(())
    }
}

impl Parts<'_> {
    const fn transfer(&mut self) -> Transfer<'_> {
        Transfer {
            pool: &mut self.state.pool,
            budget: &mut *self.budget,
        }
    }
    fn fixed(&self, index: usize) -> Result<&wgpu::Buffer> {
        self.state.fixed.get(index)
    }
    fn footprint(&self, index: usize) -> Result<&wgpu::Buffer> {
        self.state
            .footprint
            .as_ref()
            .ok_or(Error::Protocol(Protocol::NotBegun))?
            .get(index)
    }
    fn scan_kernels(&self) -> Result<ScanKernels<'_>> {
        Ok(ScanKernels {
            blocks: &self.pipelines.scan_blocks,
            add: &self.pipelines.scan_add,
            status: self.fixed(STATUS)?,
            uniforms: self.fixed(UNIFORMS)?,
            stride: self.state.uniform_stride,
            layout: ScanLayout::Candidates,
        })
    }
    fn dispatch(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        groups: &[&wgpu::BindGroup],
        threads: u64,
    ) -> Result<()> {
        if self.gpu.dispatch(
            encoder,
            pipeline,
            groups,
            threads.div_ceil(u64::from(BLOCK_SIZE)),
        )? {
            add(&mut self.statistics.dispatches, 1)?;
            self.statistics.candidate_work.add_dispatch()?;
        }
        Ok(())
    }
    fn scan(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        (values, scratch): ((&wgpu::Buffer, u32), &wgpu::Buffer),
        slot: &mut u32,
    ) -> Result<()> {
        let dispatches = self.scan_kernels()?.encode(
            &self.gpu,
            encoder,
            ((values.0, 0, values.1), scratch),
            slot,
        )?;
        add(&mut self.statistics.dispatches, dispatches)?;
        self.statistics.candidate_work.add_dispatches(dispatches)
    }
    fn submit_and_read(
        &mut self,
        encoder: wgpu::CommandEncoder,
        staging: &wgpu::Buffer,
        bytes: u64,
        consume: impl FnOnce(&[u8]) -> Result<()>,
    ) -> Result<Duration> {
        let elapsed = self.gpu.submit_and_read(encoder, staging, bytes, consume)?;
        self.statistics.candidate_work.record_submission(elapsed)?;
        add(&mut self.statistics.readback_bytes, bytes)?;
        Ok(elapsed)
    }
    fn read(
        &mut self,
        source: &wgpu::Buffer,
        bytes: u64,
        record_bytes: u64,
        consume: impl FnMut(&[u8]) -> Result<()>,
    ) -> Result<Duration> {
        let started = Instant::now();
        let gpu = self.gpu;
        let mut work = self.statistics.candidate_work;
        let result = self.transfer().read(
            &gpu,
            &mut work,
            (source, bytes),
            Content {
                record_bytes,
                records: false,
            },
            consume,
        );
        self.statistics.candidate_work = work;
        result?;
        add(&mut self.statistics.readback_bytes, bytes)?;
        Ok(started.elapsed())
    }

    /// Bit-exact comparison of resident reservations and points with `paths`.
    fn validate(&mut self, paths: &[Path<'_>]) -> Result<()> {
        let reservation_count =
            usize::try_from(self.geometry.reservations).map_err(|_| overflow())?;
        let mut reservations: Vec<[u32; 3]> =
            self.holdings.vector(self.budget, reservation_count)?;
        let paths_buffer = self.geometry.paths.clone();
        let reservation_bytes = resources::words(self.geometry.reservations, PATH_BYTES)?;
        let mut elapsed = self.read(&paths_buffer, reservation_bytes, PATH_BYTES, |data| {
            for record in data.as_chunks::<12>().0 {
                let mut fields = words(record);
                let mut field = || fields.next().ok_or(Error::Protocol(Protocol::Readback));
                reservations.push([field()?, field()?, field()?]);
            }
            Ok(())
        })?;
        // (point offset, ordinal), sorted by offset; ranges must tile the buffer.
        let mut owners: Vec<(u32, u32)> = self.holdings.vector(self.budget, paths.len())?;
        for (ordinal, path) in (0_u32..).zip(paths) {
            let reservation = usize::try_from(path.id)
                .ok()
                .and_then(|id| reservations.get(id))
                .ok_or_else(|| {
                    Error::Geometry(format!("path {} has no GPU reservation", path.id))
                })?;
            if usize::try_from(reservation[2]).ok() != Some(path.points.len()) {
                return Err(Error::Geometry(format!(
                    "path {}: GPU reservation holds {} points, CPU path {}",
                    path.id,
                    reservation[2],
                    path.points.len()
                )));
            }
            owners.push((reservation[0], ordinal));
        }
        self.holdings.free(self.budget, reservations)?;
        owners.sort_unstable();
        let mut end = 0_usize;
        for &(offset, ordinal) in &owners {
            let path = paths
                .get(usize::try_from(ordinal).map_err(|_| overflow())?)
                .ok_or_else(overflow)?;
            if usize::try_from(offset).ok() != Some(end) {
                return Err(Error::Geometry(
                    "GPU reservations do not tile the CPU paths".to_owned(),
                ));
            }
            end = end.checked_add(path.points.len()).ok_or_else(overflow)?;
        }
        if usize::try_from(self.geometry.samples).ok() != Some(end) {
            return Err(Error::Geometry(
                "CPU paths do not cover the GPU point buffer exactly".to_owned(),
            ));
        }
        let points = self.geometry.points.clone();
        let mut cursor = (0_usize, 0_usize);
        elapsed = elapsed.saturating_add(self.read(&points, resources::words(end, POINT_BYTES)?, POINT_BYTES, |data| {
            for record in data.as_chunks::<12>().0 {
                let mut fields = words(record);
                let (x, y, id) = (fields.next(), fields.next(), fields.next());
                let (owner, index) = &mut cursor;
                let mut path = owners
                    .get(*owner)
                    .and_then(|(_, ordinal)| paths.get(usize::try_from(*ordinal).ok()?));
                while path.is_some_and(|path| *index >= path.points.len()) {
                    *owner = owner.saturating_add(1);
                    *index = 0;
                    path = owners
                        .get(*owner)
                        .and_then(|(_, ordinal)| paths.get(usize::try_from(*ordinal).ok()?));
                }
                let path = path.ok_or_else(|| Error::Geometry("excess GPU points".to_owned()))?;
                let point = path.points.get(*index).ok_or_else(overflow)?;
                let same = x == Some(point[0].to_bits()) && y == Some(point[1].to_bits()) && id == Some(path.id);
                if !same {
                    return Err(Error::Geometry(format!(
                        "path {} point {index}: resident GPU geometry differs from the CPU narrow-phase input",
                        path.id
                    )));
                }
                *index = index.saturating_add(1);
            }
            Ok(())
        })?);
        self.holdings.free(self.budget, owners)?;
        self.statistics.validated_points = number(end)?;
        self.timings.validation = elapsed;
        Ok(())
    }

    /// Upload ordinal-to-ID and ID-to-ordinal tables; queue writes copy them.
    fn upload_tables(&mut self, paths: &[Path<'_>]) -> Result<()> {
        let reservation_count =
            usize::try_from(self.geometry.reservations).map_err(|_| overflow())?;
        let mut order: Vec<u32> = self.holdings.vector(self.budget, paths.len())?;
        let mut ordinals: Vec<u32> = self.holdings.vector(self.budget, reservation_count)?;
        ordinals.resize(reservation_count, INVALID_ORDINAL);
        for (ordinal, path) in (0_u32..).zip(paths) {
            let slot = usize::try_from(path.id)
                .ok()
                .and_then(|id| ordinals.get_mut(id))
                .ok_or_else(|| {
                    Error::Geometry(format!("path {} has no GPU reservation", path.id))
                })?;
            *slot = ordinal;
            order.push(path.id);
        }
        let queue = self.gpu.queue;
        queue.write_buffer(
            self.fixed(PARAMETERS)?,
            0,
            bytemuck::bytes_of(&self.state.parameters),
        );
        queue.write_buffer(self.fixed(ORDER)?, 0, bytemuck::cast_slice(&order));
        queue.write_buffer(self.fixed(ORDINALS)?, 0, bytemuck::cast_slice(&ordinals));
        let uploaded = resources::words(size_of::<Parameters>(), 1)?
            .checked_add(resources::words(order.len(), 4)?)
            .and_then(|bytes| bytes.checked_add(resources::words(ordinals.len(), 4).ok()?))
            .ok_or_else(overflow)?;
        add(&mut self.statistics.uploaded_bytes, uploaded)?;
        self.holdings.free(self.budget, order)?;
        self.holdings.free(self.budget, ordinals)
    }

    /// Path metadata, length enclosures and positive segment end points.
    fn upload_contact(&mut self, request: &Request<'_, '_>, samples: u32) -> Result<()> {
        let paths = request.paths.len();
        let segments = usize::try_from(
            *request
                .segment_offsets
                .last()
                .ok_or(Error::Protocol(Protocol::Tables))?,
        )
        .map_err(|_| overflow())?;
        let mut data = contact::TableData {
            info: self.holdings.vector(self.budget, paths)?,
            lengths: self.holdings.vector(self.budget, segments)?,
        };
        let filled = contact::fill_tables(
            &mut data,
            request.paths,
            request.prepared,
            request.segment_offsets,
        );
        let uploaded = filled.and_then(|()| {
            let frame = contact_frame(&self.state.fixed, self.geometry)?;
            contact::upload_tables(
                &self.gpu,
                self.kernels,
                &frame,
                &data,
                (self.state.contact, samples),
                &mut self.statistics.contact,
            )
        });
        self.holdings.free(self.budget, data.info)?;
        self.holdings.free(self.budget, data.lengths)?;
        uploaded
    }

    fn geometry_groups(&self) -> Result<[wgpu::BindGroup; 8]> {
        let p = self.pipelines;
        let g = &self.gpu;
        let points = || self.geometry.points.as_entire_binding();
        let paths = || self.geometry.paths.as_entire_binding();
        let fixed = |index| self.fixed(index).map(wgpu::Buffer::as_entire_binding);
        Ok([
            g.bind(
                &p.mark_points,
                0,
                [
                    (0, fixed(PARAMETERS)?),
                    (1, points()),
                    (2, paths()),
                    (4, fixed(ORDINALS)?),
                    (5, fixed(POINT_SCAN)?),
                    (7, fixed(STATUS)?),
                ],
            ),
            g.bind(
                &p.count_segments,
                0,
                [
                    (0, fixed(PARAMETERS)?),
                    (2, paths()),
                    (3, fixed(ORDER)?),
                    (5, fixed(POINT_SCAN)?),
                    (6, fixed(PATH_BASE)?),
                    (7, fixed(STATUS)?),
                ],
            ),
            g.bind(
                &p.emit_segments,
                0,
                [
                    (0, fixed(PARAMETERS)?),
                    (1, points()),
                    (2, paths()),
                    (4, fixed(ORDINALS)?),
                    (5, fixed(POINT_SCAN)?),
                    (6, fixed(PATH_BASE)?),
                ],
            ),
            g.bind(&p.emit_segments, 1, [(0, fixed(SEGMENTS)?)]),
            g.bind(
                &p.count_footprints,
                0,
                [(0, fixed(PARAMETERS)?), (6, fixed(PATH_BASE)?)],
            ),
            g.bind(
                &p.count_footprints,
                1,
                [(0, fixed(SEGMENTS)?), (9, fixed(FOOTPRINT_OFFSETS)?)],
            ),
            g.bind(
                &p.gather_footprints,
                0,
                [(0, fixed(PARAMETERS)?), (6, fixed(PATH_BASE)?)],
            ),
            g.bind(
                &p.gather_footprints,
                1,
                [
                    (9, fixed(FOOTPRINT_OFFSETS)?),
                    (11, fixed(PATH_FOOTPRINTS)?),
                ],
            ),
        ])
    }

    /// Mark, scan and emit segments and their footprint offsets. Reads back and
    /// validates segment offsets against the CPU; returns the footprint total.
    fn encode_geometry(
        &mut self,
        request: &Request<'_, '_>,
        path_footprints: &mut [u32],
        segments: u32,
    ) -> Result<u32> {
        let [
            mark,
            count,
            emit,
            emit_segments,
            area,
            area_segments,
            gather,
            gather_output,
        ] = self.geometry_groups()?;
        let p = self.pipelines;
        let samples = self.geometry.samples;
        let path_count = u32::try_from(request.paths.len()).map_err(|_| overflow())?;
        let (status, scratch, summary) = (
            self.fixed(STATUS)?.clone(),
            self.fixed(SETUP_SCRATCH)?.clone(),
            self.fixed(SUMMARY)?.clone(),
        );
        let (point_scan, path_base, footprint_offsets, path_table) = (
            self.fixed(POINT_SCAN)?.clone(),
            self.fixed(PATH_BASE)?.clone(),
            self.fixed(FOOTPRINT_OFFSETS)?.clone(),
            self.fixed(PATH_FOOTPRINTS)?.clone(),
        );
        let mut encoder = self.gpu.encoder();
        let mut slots = 0_u32;
        encoder.clear_buffer(&status, 0, None);
        encoder.clear_buffer(self.fixed(CELL_COUNTS)?, 0, None);
        encoder.clear_buffer(self.fixed(CELL_CURSOR)?, 0, None);
        self.dispatch(
            &mut encoder,
            &p.mark_points,
            &[&mark],
            u64::from(plus_one(samples)?),
        )?;
        self.scan(
            &mut encoder,
            ((&point_scan, plus_one(samples)?), &scratch),
            &mut slots,
        )?;
        self.dispatch(
            &mut encoder,
            &p.count_segments,
            &[&count],
            u64::from(plus_one(path_count)?),
        )?;
        self.scan(
            &mut encoder,
            ((&path_base, plus_one(path_count)?), &scratch),
            &mut slots,
        )?;
        self.dispatch(
            &mut encoder,
            &p.emit_segments,
            &[&emit, &emit_segments],
            u64::from(samples),
        )?;
        self.dispatch(
            &mut encoder,
            &p.count_footprints,
            &[&area, &area_segments],
            u64::from(plus_one(segments)?),
        )?;
        self.scan(
            &mut encoder,
            ((&footprint_offsets, plus_one(segments)?), &scratch),
            &mut slots,
        )?;
        self.dispatch(
            &mut encoder,
            &p.gather_footprints,
            &[&gather, &gather_output],
            u64::from(plus_one(path_count)?),
        )?;
        let table_bytes = resources::words(plus_one(path_count)?, 4)?;
        encoder.copy_buffer_to_buffer(&path_base, 0, &summary, 0, table_bytes);
        encoder.copy_buffer_to_buffer(&path_table, 0, &summary, table_bytes, table_bytes);
        let status_offset = table_bytes.checked_mul(2).ok_or_else(overflow)?;
        encoder.copy_buffer_to_buffer(&status, 0, &summary, status_offset, STATUS_BYTES);
        let bytes = status_offset
            .checked_add(STATUS_BYTES)
            .ok_or_else(overflow)?;
        let tables = usize::try_from(plus_one(path_count)?).map_err(|_| overflow())?;
        let elapsed = self.submit_and_read(encoder, &summary, bytes, |data| {
            parse_summary(
                data,
                request.segment_offsets,
                path_footprints.get_mut(..tables),
            )
        })?;
        self.timings.setup = self.timings.setup.saturating_add(elapsed);
        self.footprint_total(request, path_footprints)
    }

    fn footprint_total(
        &mut self,
        request: &Request<'_, '_>,
        path_footprints: &[u32],
    ) -> Result<u32> {
        for (footprints, segments) in path_footprints
            .windows(2)
            .zip(request.segment_offsets.windows(2))
        {
            let footprint_count = footprints
                .last()
                .zip(footprints.first())
                .and_then(|(end, start)| end.checked_sub(*start));
            let segment_count = segments
                .last()
                .zip(segments.first())
                .and_then(|(end, start)| end.checked_sub(*start));
            if footprint_count
                .zip(segment_count)
                .is_none_or(|(footprints, segments)| footprints < segments)
            {
                return Err(Error::Protocol(Protocol::FootprintOffsets));
            }
        }
        self.statistics.positive_segments =
            u64::from(*request.segment_offsets.last().ok_or_else(overflow)?);
        let total = *path_footprints.last().ok_or_else(overflow)?;
        self.statistics.footprint_entries = u64::from(total);
        Ok(total)
    }

    /// Emit footprint entries, bucket them by cell, and check the status words.
    fn encode_cells(&mut self, footprints: u32, cells: u32) -> Result<()> {
        let p = self.pipelines;
        let g = &self.gpu;
        let fixed = |index| self.fixed(index).map(wgpu::Buffer::as_entire_binding);
        let footprint = |index| self.footprint(index).map(wgpu::Buffer::as_entire_binding);
        let groups = [
            g.bind(
                &p.emit_footprint,
                0,
                [(0, fixed(PARAMETERS)?), (6, fixed(PATH_BASE)?)],
            ),
            g.bind(
                &p.emit_footprint,
                1,
                [
                    (0, fixed(SEGMENTS)?),
                    (1, fixed(CELL_COUNTS)?),
                    (9, fixed(FOOTPRINT_OFFSETS)?),
                    (10, footprint(FOOTPRINT)?),
                ],
            ),
            g.bind(
                &p.fill_cells,
                0,
                [
                    (0, fixed(PARAMETERS)?),
                    (6, fixed(PATH_BASE)?),
                    (7, fixed(STATUS)?),
                ],
            ),
            g.bind(
                &p.fill_cells,
                1,
                [
                    (1, fixed(CELL_COUNTS)?),
                    (2, fixed(CELL_CURSOR)?),
                    (3, footprint(ENTRIES)?),
                    (9, fixed(FOOTPRINT_OFFSETS)?),
                    (10, footprint(FOOTPRINT)?),
                ],
            ),
        ];
        let [emit, emit_output, fill, fill_output] = groups;
        let (status, summary, scratch, cell_counts) = (
            self.fixed(STATUS)?.clone(),
            self.fixed(SUMMARY)?.clone(),
            self.fixed(SETUP_SCRATCH)?.clone(),
            self.fixed(CELL_COUNTS)?.clone(),
        );
        let mut encoder = self.gpu.encoder();
        let mut slots = 0_u32;
        self.dispatch(
            &mut encoder,
            &p.emit_footprint,
            &[&emit, &emit_output],
            u64::from(footprints),
        )?;
        self.scan(
            &mut encoder,
            ((&cell_counts, plus_one(cells)?), &scratch),
            &mut slots,
        )?;
        self.dispatch(
            &mut encoder,
            &p.fill_cells,
            &[&fill, &fill_output],
            u64::from(footprints),
        )?;
        encoder.copy_buffer_to_buffer(&status, 0, &summary, 0, STATUS_BYTES);
        let mut status_words = [0_u32; STATUS_WORDS];
        let elapsed = self.submit_and_read(encoder, &summary, STATUS_BYTES, |data| {
            for (slot, word) in status_words.iter_mut().zip(words(data)) {
                *slot = word;
            }
            Ok(())
        })?;
        self.timings.setup = self.timings.setup.saturating_add(elapsed);
        check_status(&status_words)
    }
}

/// S2 frame tables plus the S1 and borrowed resident buffers they read.
fn contact_frame<'a>(
    fixed: &'a resources::Slots<{ resources::FIXED }>,
    geometry: &'a Geometry,
) -> Result<contact::Frame<'a>> {
    Ok(contact::Frame {
        parameters: fixed.get(CONTACT_PARAMETERS)?,
        status: fixed.get(CONTACT_STATUS)?,
        path_info: fixed.get(PATH_INFO)?,
        lengths: fixed.get(LENGTHS)?,
        segment_end: fixed.get(SEGMENT_END)?,
        points: &geometry.points,
        reservations: &geometry.paths,
        ordinals: fixed.get(ORDINALS)?,
        point_scan: fixed.get(POINT_SCAN)?,
        path_base: fixed.get(PATH_BASE)?,
        summary: fixed.get(SUMMARY)?,
    })
}

/// Check GPU segment offsets against the CPU, copy footprint offsets, then
/// check the status words.
fn parse_summary(
    data: &[u8],
    segment_offsets: &[u32],
    footprints: Option<&mut [u32]>,
) -> Result<()> {
    let mut values = words(data);
    for expected in segment_offsets {
        if values.next() != Some(*expected) {
            return Err(Error::Geometry(
                "GPU positive-segment offsets differ from the CPU preparation".to_owned(),
            ));
        }
    }
    for slot in footprints.ok_or(Error::Protocol(Protocol::Readback))? {
        *slot = values.next().ok_or(Error::Protocol(Protocol::Readback))?;
    }
    let mut status = [0_u32; STATUS_WORDS];
    for slot in &mut status {
        *slot = values.next().ok_or(Error::Protocol(Protocol::Readback))?;
    }
    check_status(&status)
}

impl Parts<'_> {
    fn candidate_groups(&self, pipeline: &wgpu::ComputePipeline) -> Result<[wgpu::BindGroup; 2]> {
        let output = match self.state.batch.as_ref() {
            Some(batch) => batch.output()?,
            None => self.fixed(PLACEHOLDER)?,
        };
        let items = &self
            .state
            .items
            .as_ref()
            .ok_or(Error::Protocol(Protocol::NotBegun))?
            .1;
        Ok([
            self.gpu.bind(
                pipeline,
                0,
                [
                    (0, self.fixed(PARAMETERS)?.as_entire_binding()),
                    (7, self.fixed(STATUS)?.as_entire_binding()),
                ],
            ),
            self.gpu.bind(
                pipeline,
                1,
                [
                    (0, self.fixed(SEGMENTS)?.as_entire_binding()),
                    (1, self.fixed(CELL_COUNTS)?.as_entire_binding()),
                    (3, self.footprint(ENTRIES)?.as_entire_binding()),
                    (4, items.get(ITEM_COUNTS)?.as_entire_binding()),
                    (5, output.as_entire_binding()),
                    (10, self.footprint(FOOTPRINT)?.as_entire_binding()),
                    (12, self.footprint(WEIGHTS)?.as_entire_binding()),
                ],
            ),
        ])
    }

    fn write_parameters(&mut self, update: impl FnOnce(&mut Parameters)) -> Result<()> {
        update(&mut self.state.parameters);
        self.gpu.queue.write_buffer(
            self.fixed(PARAMETERS)?,
            0,
            bytemuck::bytes_of(&self.state.parameters),
        );
        add(
            &mut self.statistics.uploaded_bytes,
            resources::words(size_of::<Parameters>(), 1)?,
        )
    }

    /// Read the scanned total at `buffer[count]` and the status words.
    fn read_total(
        &mut self,
        mut encoder: wgpu::CommandEncoder,
        buffer: &wgpu::Buffer,
        count: u32,
    ) -> Result<(u32, [u32; STATUS_WORDS], Duration)> {
        let (status, summary) = (self.fixed(STATUS)?.clone(), self.fixed(SUMMARY)?.clone());
        encoder.copy_buffer_to_buffer(buffer, resources::words(count, 4)?, &summary, 0, 4);
        encoder.copy_buffer_to_buffer(&status, 0, &summary, 4, STATUS_BYTES);
        let mut values = [0_u32; 5];
        let elapsed = self.submit_and_read(
            encoder,
            &summary,
            4_u64.saturating_add(STATUS_BYTES),
            |data| {
                for (slot, word) in values.iter_mut().zip(words(data)) {
                    *slot = word;
                }
                Ok(())
            },
        )?;
        let [total, rest @ ..] = values;
        Ok((total, rest, elapsed))
    }

    /// Dispatch `pipeline` over items `0..=items` in bounded submissions.
    fn dispatch_items(&mut self, pipeline: &wgpu::ComputePipeline, items: u32) -> Result<()> {
        let groups = self.candidate_groups(pipeline)?;
        let mut start = 0_u32;
        while start <= items {
            self.write_parameters(|parameters| {
                parameters.item_start = start;
                parameters.item_count = items;
            })?;
            let chunk = items
                .saturating_sub(start)
                .saturating_add(1)
                .min(ITEMS_PER_DISPATCH);
            let mut encoder = self.gpu.encoder();
            self.dispatch(
                &mut encoder,
                pipeline,
                &[&groups[0], &groups[1]],
                u64::from(chunk),
            )?;
            let elapsed = self.gpu.submit(encoder)?;
            self.statistics.candidate_work.record_submission(elapsed)?;
            let Some(next) = start.checked_add(chunk) else {
                break;
            };
            start = next;
        }
        Ok(())
    }

    /// Exact cell count of `sources` (footprint range `footprints`), or the
    /// shortfall that makes the range unfillable. On success the private count
    /// token authorizes exactly one matching fill.
    fn count(
        &mut self,
        sources: Range<usize>,
        footprints: Range<u32>,
    ) -> Result<std::result::Result<u32, Shortfall>> {
        add(&mut self.statistics.count_passes, 1)?;
        self.state.counted = None;
        self.state.generation = self.state.generation.checked_add(1).ok_or_else(overflow)?;
        let generation = self.state.generation;
        let length = footprints
            .end
            .checked_sub(footprints.start)
            .ok_or(Error::Protocol(Protocol::FootprintOffsets))?;
        if length == 0 {
            self.state.counted = Some(Counted {
                generation,
                sources,
                footprints,
                items: 0,
                records: 0,
            });
            return Ok(Ok(0));
        }
        let started = Instant::now();
        let (items, overflowed) = self.count_items(footprints.start, length)?;
        if overflowed != 0 {
            return Ok(Err(Shortfall {
                limit: Limit::CountOverflow,
                demand: u64::from(u32::MAX).saturating_add(1),
                available: u64::from(u32::MAX),
            }));
        }
        if let Some(shortfall) = self.ensure_items(items)? {
            return Ok(Err(shortfall));
        }
        let p = self.pipelines;
        self.dispatch_items(&p.count_candidates, items)?;
        let items_buffer = self.items(ITEM_COUNTS)?.clone();
        let item_scratch = self.items(ITEM_SCRATCH)?.clone();
        let mut encoder = self.gpu.encoder();
        let mut slots = 0_u32;
        self.scan(
            &mut encoder,
            ((&items_buffer, plus_one(items)?), &item_scratch),
            &mut slots,
        )?;
        let (records, [overflowed, mismatch, geometry, exceeded], _) =
            self.read_total(encoder, &items_buffer, items)?;
        self.timings.count = self.timings.count.saturating_add(started.elapsed());
        check_status(&[0, mismatch, geometry, exceeded])?;
        add(&mut self.statistics.work_items, u64::from(items))?;
        if overflowed != 0 {
            return Ok(Err(Shortfall {
                limit: Limit::CountOverflow,
                demand: u64::from(u32::MAX).saturating_add(1),
                available: u64::from(u32::MAX),
            }));
        }
        let cap = self.settings.batch_records.unwrap_or(u32::MAX);
        if records > cap {
            return Ok(Err(Shortfall {
                limit: Limit::BatchRecords,
                demand: u64::from(records),
                available: u64::from(cap),
            }));
        }
        self.state.counted = Some(Counted {
            generation,
            sources,
            footprints,
            items,
            records,
        });
        Ok(Ok(records))
    }

    /// Count work items of the footprint range; returns the item total and
    /// the scan overflow word.
    fn count_items(&mut self, footprint_start: u32, length: u32) -> Result<(u32, u32)> {
        self.write_parameters(|parameters| {
            parameters.footprint_start = footprint_start;
            parameters.footprint_count = length;
            parameters.capacity = 0;
        })?;
        let p = self.pipelines;
        let groups = [
            self.gpu.bind(
                &p.count_items,
                0,
                [(0, self.fixed(PARAMETERS)?.as_entire_binding())],
            ),
            self.gpu.bind(
                &p.count_items,
                1,
                [
                    (1, self.fixed(CELL_COUNTS)?.as_entire_binding()),
                    (10, self.footprint(FOOTPRINT)?.as_entire_binding()),
                    (12, self.footprint(WEIGHTS)?.as_entire_binding()),
                ],
            ),
        ];
        let (status, weights, scratch) = (
            self.fixed(STATUS)?.clone(),
            self.footprint(WEIGHTS)?.clone(),
            self.footprint(FOOTPRINT_SCRATCH)?.clone(),
        );
        let mut encoder = self.gpu.encoder();
        let mut slots = 0_u32;
        // Setup status was already checked; this batch reports only its own overflow.
        encoder.clear_buffer(&status, 0, Some(4));
        self.dispatch(
            &mut encoder,
            &p.count_items,
            &[&groups[0], &groups[1]],
            u64::from(plus_one(length)?),
        )?;
        self.scan(
            &mut encoder,
            ((&weights, plus_one(length)?), &scratch),
            &mut slots,
        )?;
        let (items, [overflowed, mismatch, geometry, exceeded], _) =
            self.read_total(encoder, &weights, length)?;
        check_status(&[0, mismatch, geometry, exceeded])?;
        Ok((items, overflowed))
    }

    fn items(&self, index: usize) -> Result<&wgpu::Buffer> {
        self.state
            .items
            .as_ref()
            .ok_or(Error::Protocol(Protocol::NotBegun))?
            .1
            .get(index)
    }

    /// Keep exactly the current item-count demand. A larger retained count
    /// buffer could otherwise consume space needed by this batch's cells.
    /// Releases the batch cache before declaring a shortfall.
    fn ensure_items(&mut self, items: u32) -> Result<Option<Shortfall>> {
        if self
            .state
            .items
            .as_ref()
            .is_some_and(|(capacity, _)| *capacity == items)
        {
            return Ok(None);
        }
        if let Some((_, slots)) = self.state.items.take() {
            slots.free(&mut self.state.pool, self.budget)?;
        }
        loop {
            let plan = resources::items(items)
                .and_then(|specs| self.state.pool.demand(&specs).map(|demand| (specs, demand)));
            let shortfall = match plan {
                Ok((specs, demand)) if demand.bytes <= self.state.pool.available(0) => {
                    let slots =
                        resources::Slots::allocate(&mut self.state.pool, self.budget, &specs)?;
                    let memory = &mut self.statistics.memory;
                    memory.peak_item_bytes = memory.peak_item_bytes.max(slots.bytes());
                    self.statistics.item_capacity =
                        self.statistics.item_capacity.max(u64::from(items));
                    self.state.items = Some((items, slots));
                    return Ok(None);
                }
                Ok((_, demand)) => Shortfall {
                    limit: Limit::DeviceMemory("ambiguity item counts"),
                    demand: demand.bytes,
                    available: self.state.pool.available(0),
                },
                Err(error) => Shortfall::planned(error)?,
            };
            let Some(batch) = self.state.batch.take() else {
                return Ok(Some(shortfall));
            };
            batch.owned.free(&mut self.state.pool, self.budget)?;
        }
    }

    /// Fill exactly the counted cells into the resident batch output. Only the
    /// matching private count token authorizes this.
    fn fill(&mut self, sources: &Range<usize>, cells: u32) -> Result<contact::Resident> {
        let counted = self
            .state
            .counted
            .take()
            .ok_or(Error::Protocol(Protocol::Token))?;
        if counted.generation != self.state.generation
            || counted.sources != *sources
            || counted.records != cells
        {
            return Err(Error::Protocol(Protocol::Token));
        }
        add(&mut self.statistics.filled_batches, 1)?;
        let resident = contact::Resident {
            sources: sources.clone(),
            cells,
        };
        if cells == 0 {
            return Ok(resident);
        }
        let capacity = self
            .state
            .batch
            .as_ref()
            .ok_or(Error::Protocol(Protocol::Inventory))?
            .capacity;
        let length = counted
            .footprints
            .end
            .saturating_sub(counted.footprints.start);
        self.write_parameters(|parameters| {
            parameters.footprint_start = counted.footprints.start;
            parameters.footprint_count = length;
            parameters.capacity = capacity;
        })?;
        let started = Instant::now();
        let p = self.pipelines;
        self.dispatch_items(&p.fill_candidates, counted.items)?;
        let encoder = self.gpu.encoder();
        let status_buffer = self.fixed(STATUS)?.clone();
        let (_, status, _) = self.read_total(encoder, &status_buffer, 0)?;
        check_status(&status)?;
        self.timings.fill = self.timings.fill.saturating_add(started.elapsed());
        add(&mut self.statistics.candidate_records, u64::from(cells))?;
        self.statistics.largest_batch_records =
            self.statistics.largest_batch_records.max(u64::from(cells));
        self.statistics.largest_batch_items = self
            .statistics
            .largest_batch_items
            .max(u64::from(counted.items));
        Ok(resident)
    }

    /// Classify a resident batch into `host` (S2), validating everything.
    fn classify(
        &mut self,
        resident: &contact::Resident,
        host: &mut contact::Host,
        segment_offsets: &[u32],
    ) -> Result<()> {
        let State {
            pool,
            fixed,
            batch,
            contact: base,
            uniform_stride,
            ..
        } = &mut *self.state;
        let Some(batch) = batch.as_ref() else {
            host.clear();
            return if resident.cells == 0 {
                Ok(())
            } else {
                Err(Error::Protocol(Protocol::Inventory))
            };
        };
        let frame = contact_frame(fixed, self.geometry)?;
        let scan = ScanKernels {
            blocks: &self.kernels.scan_blocks,
            add: &self.kernels.scan_add,
            status: fixed.get(CONTACT_STATUS)?,
            uniforms: fixed.get(UNIFORMS)?,
            stride: *uniform_stride,
            layout: ScanLayout::Contact,
        };
        let mut run = contact::Run {
            gpu: self.gpu,
            kernels: self.kernels,
            frame,
            batch,
            transfer: Transfer {
                pool,
                budget: &mut *self.budget,
            },
            statistics: &mut self.statistics.contact,
            scan,
            base: *base,
            segment_offsets,
        };
        run.classify(resident, host)
    }
}
