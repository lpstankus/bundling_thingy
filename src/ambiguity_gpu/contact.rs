//! Certified continuous-contact classification of resident candidate batches
//! (stage S2) and decoding of its pair decisions.
//!
//! One exact row-major radix sort groups each complete source/partner run.
//! Each cell is certified in both projection directions with outward f32
//! intervals that enclose the ordinary CPU-f64 values. Pairs of at most
//! `SMALL_PAIR` cells, and pairs above the `PAIR_CAP` cooperative cap, use one
//! serial invocation (the latter returns oversized at once). Every other pair
//! is owned by one workgroup of 64 lanes (padded size at most 128) or 256 lanes.
//! Uncertain certificates, unsupported arithmetic, oversized pairs and labels
//! that do not settle within `ROUND_CAP` rounds become complete CPU fallback
//! runs. Schedules are audited on the host and by exclusive GPU claims.
//! Protocol violations abort; they never become ordinary uncertainty.
use super::{
    Content, Error, Gpu, Protocol, Result, ScanKernels, Transfer, Work, overflow,
    resources::{self, AUDIT_WORDS, RADIX_BINS, RADIX_BLOCK, Slots, words},
};
use crate::ambiguity_index::CandidateCell;
use std::{mem::size_of, ops::Range, time::Duration};

pub(super) const CERTIFICATE_BYTES: u64 = 36;
/// Pairs with at most this many cells use the serial classifier.
const SMALL_PAIR: u32 = 32;
/// Cooperative cap; larger pairs are oversized and take the CPU route.
const PAIR_CAP: u32 = 4096;
const ROUND_CAP: u32 = 128;
/// Padded pair sizes up to this use the 64-lane cooperative kernel.
const NARROW_PADDED: u64 = 128;
const SERIAL_LANES: u64 = 64;
const BLOCK_SIZE: u64 = 256;
const PAIRS_PER_SUBMISSION: u32 = 1 << 20;
const CELLS_PER_SUBMISSION: u64 = 1 << 23;
/// Cooperative submission grouping target in work units, with at least one
/// complete pair per submission. The effective bound is the larger of this
/// target and the largest scheduled pair's estimate. Not a wall-clock bound.
pub(super) const WORK_TARGET: u64 = 1 << 31;
const CLAIM_ERRORS: usize = 37;
const COOPERATIVE_CLAIMS: usize = 35;
const SERIAL_CLAIMS: usize = 36;
const ROUND_BINS: usize = 32;
const STATUS_WORDS: usize = 4;

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(super) struct Parameters {
    cells: u32,
    source_base: u32,
    segment_bits: u32,
    partner_bits: u32,
    shift: u32,
    blocks: u32,
    pairs: u32,
    chunk_start: u32,
    chunk_count: u32,
    pair_cap: u32,
    admitted: u32,
    source_bits: u32,
    radius: [f32; 2],
    squared_radius: [f32; 2],
    minimum: [f32; 2],
    radius_is_zero: u32,
    local_columns: u32,
    round_cap: u32,
    path_count: u32,
    source_count: u32,
    padding: u32,
}

/// f32 enclosure [lo, hi] of an f64 value: round, take the neighbors that
/// bracket it, step one more ulp outward, then widen nonzero endpoints below
/// 2^-125 to +-2^-125 so every endpoint is normal or exactly zero.
pub(super) fn enclose(value: f64) -> [f32; 2] {
    if value == 0.0 {
        return [0.0, 0.0];
    }
    let rounded = super::nearest_f32(value);
    let low = if f64::from(rounded) <= value {
        rounded
    } else {
        rounded.next_down()
    };
    let high = if f64::from(rounded) >= value {
        rounded
    } else {
        rounded.next_up()
    };
    let tiny = f32::from_bits(0x0100_0000);
    let low = low.next_down().min(low);
    let high = high.next_up().max(high);
    [
        if low != 0.0 && low.abs() < tiny {
            -tiny
        } else {
            low
        },
        if high != 0.0 && high.abs() < tiny {
            tiny
        } else {
            high
        },
    ]
}

/// Admitted scalar enclosures: radius, fl64(radius * radius) and minimum.
#[derive(Clone, Copy)]
pub(super) struct Scalars {
    radius: [f32; 2],
    squared_radius: [f32; 2],
    minimum: [f32; 2],
    radius_is_zero: bool,
}
/// Scalar admission. `None` keeps the frame running with every pair on the
/// complete CPU route.
pub(super) fn admit(radius: f64, minimum: f64) -> Option<Scalars> {
    let radius_ok =
        radius.is_finite() && (radius == 0.0 || (2.0_f64.powi(-40)..=256.0).contains(&radius));
    let minimum_ok = minimum.is_finite() && minimum > 0.0 && minimum <= 2.0_f64.powi(20);
    (radius_ok && minimum_ok).then(|| Scalars {
        radius: enclose(radius),
        squared_radius: enclose(radius * radius),
        minimum: enclose(minimum),
        radius_is_zero: radius == 0.0,
    })
}

/// Bits needed to represent `count` distinct values `0..count`.
pub(super) const fn bits_for(count: u32) -> u32 {
    u32::BITS.saturating_sub(count.saturating_sub(1).leading_zeros())
}

/// Packed sort-key widths fixed for the frame.
#[derive(Clone, Copy)]
pub(super) struct Keys {
    partner_bits: u32,
    segment_bits: u32,
    /// Largest source span whose batch-relative source field fits the key.
    pub maximum_span: usize,
}
impl Keys {
    /// Key widths of the frame, or the required width when even one source
    /// cannot fit a 64-bit key.
    pub(super) fn new(paths: u32, maximum_segments: u32) -> Result<Self> {
        let partner_bits = bits_for(paths);
        let segment_bits = bits_for(maximum_segments);
        let fixed = segment_bits
            .checked_mul(2)
            .and_then(|bits| bits.checked_add(partner_bits))
            .ok_or_else(overflow)?;
        let Some(source_bits) = 64_u32.checked_sub(fixed) else {
            return Err(Error::KeyWidth { required: fixed });
        };
        let maximum_span = 1_usize
            .checked_shl(source_bits.min(usize::BITS.saturating_sub(1)))
            .ok_or_else(overflow)?;
        Ok(Self {
            partner_bits,
            segment_bits,
            maximum_span,
        })
    }
}

/// Why a pair takes the complete CPU route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    Oversized,
    UncertainCell,
    PossibleError,
    MissingNeighbor,
    InconclusiveLength,
    UncertainGate,
    UnsupportedLength,
    Unconverged,
    UnsupportedScalars,
}
/// Why a certified pair has no contact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Negative {
    Gate,
    Empty,
    Length,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Contact,
    NoContact(Negative),
    Fallback(Reason),
}
impl Route {
    /// Match routes reachable under scalar admission and the host size schedule.
    fn valid_for(self, cells: u32, admitted: u32) -> bool {
        cells != 0
            && match admitted {
                0 => self == Self::Fallback(Reason::UnsupportedScalars),
                1 if cells > PAIR_CAP => self == Self::Fallback(Reason::Oversized),
                1 => !matches!(
                    self,
                    Self::Fallback(Reason::Oversized | Reason::UnsupportedScalars)
                ),
                _ => false,
            }
    }
}
/// One decoded, validated pair decision in canonical pair order.
#[derive(Clone, Copy)]
pub struct PairDecision {
    source: u32,
    partner: u32,
    cells: u32,
    route: Route,
}

/// Decode one classifier outcome word. Unknown combinations are protocol errors.
fn route(word: u32, pair: usize) -> Result<Route> {
    let outcome = word & 0xff;
    let reason = (word >> 8) & 0xff;
    let detail = word >> 16;
    let route = match (outcome, reason) {
        (1, 4) => Route::Contact,
        (2, 1) => Route::NoContact(Negative::Gate),
        (2, 2) => Route::NoContact(Negative::Empty),
        (2, 3) => Route::NoContact(Negative::Length),
        (3, 20) => Route::Fallback(Reason::Oversized),
        (3, 21) => Route::Fallback(Reason::UncertainCell),
        (3, 22) => Route::Fallback(Reason::PossibleError),
        (3, 23) => Route::Fallback(Reason::MissingNeighbor),
        (3, 25) => Route::Fallback(Reason::InconclusiveLength),
        (3, 26) => Route::Fallback(Reason::UncertainGate),
        (3, 27) => Route::Fallback(Reason::UnsupportedLength),
        (3, 28) => Route::Fallback(Reason::Unconverged),
        (3, 30) => Route::Fallback(Reason::UnsupportedScalars),
        _ => return Err(Error::Protocol(Protocol::Outcome { pair, word })),
    };
    // Only the listed certificate failures can reach the uncertain-cell route.
    // Possible errors and unsupported scalars have their own pair outcomes.
    let valid_detail = if route == Route::Fallback(Reason::UncertainCell) {
        matches!(
            detail,
            1 | 2 | 3 | 4 | 6 | 7 | 8 | 11 | 12 | 13 | 18 | 19 | 20 | 28
        )
    } else {
        detail == 0
    };
    if !valid_detail {
        return Err(Error::Protocol(Protocol::Outcome { pair, word }));
    }
    Ok(route)
}

/// Pair and cell totals of one route category.
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct Tally {
    pub pairs: u64,
    pub cells: u64,
}
impl Tally {
    fn add(&mut self, cells: u32) -> Result<()> {
        self.pairs = self.pairs.checked_add(1).ok_or_else(overflow)?;
        self.cells = self
            .cells
            .checked_add(u64::from(cells))
            .ok_or_else(overflow)?;
        Ok(())
    }
}
/// Complete CPU fallback by reason.
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct Reasons {
    pub oversized: Tally,
    pub uncertain_cell: Tally,
    pub possible_error: Tally,
    pub missing_neighbor: Tally,
    pub inconclusive_length: Tally,
    pub uncertain_gate: Tally,
    pub unsupported_length: Tally,
    pub unconverged: Tally,
    pub unsupported_scalars: Tally,
}
impl Reasons {
    const fn slot(&mut self, reason: Reason) -> &mut Tally {
        match reason {
            Reason::Oversized => &mut self.oversized,
            Reason::UncertainCell => &mut self.uncertain_cell,
            Reason::PossibleError => &mut self.possible_error,
            Reason::MissingNeighbor => &mut self.missing_neighbor,
            Reason::InconclusiveLength => &mut self.inconclusive_length,
            Reason::UncertainGate => &mut self.uncertain_gate,
            Reason::UnsupportedLength => &mut self.unsupported_length,
            Reason::Unconverged => &mut self.unconverged,
            Reason::UnsupportedScalars => &mut self.unsupported_scalars,
        }
    }
}

/// S2 route, schedule and transfer accounting for one frame.
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct Statistics {
    pub admitted_scalars: bool,
    pub batches: u64,
    pub cells: u64,
    pub pairs: u64,
    /// Certified contacts emitted without CPU contact work.
    pub certain_contacts: Tally,
    /// Certified negatives: participation gate certainly false.
    pub certain_no_contact_gate: Tally,
    /// Certified negatives: every cell certainly absent.
    pub certain_no_contact_empty: Tally,
    /// Certified negatives: every component certainly short, including pairs
    /// whose participation gate was uncertain.
    pub certain_no_contact_length: Tally,
    /// Pairs and cells sent to the complete CPU pair route.
    pub fallback: Tally,
    pub fallback_reasons: Reasons,
    pub serial_pairs: u64,
    pub cooperative_pairs: u64,
    pub oversized_pairs: u64,
    /// Cooperative pairs by log2 of their padded size.
    pub cooperative_buckets: [u64; 13],
    /// Cooperative pairs settled after r rounds (31: 31 or more).
    pub rounds: [u64; ROUND_BINS],
    pub rounds_unsettled: u64,
    pub audit_failures: u64,
    /// Always zero in successful reports: invalid label indices abort on the host.
    pub label_range_failures: u64,
    pub largest_pair_cells: u64,
    pub largest_fallback_pair_cells: u64,
    pub radix_passes: u64,
    pub work_target: u64,
    pub maximum_submitted_work: u64,
    pub cooperative_submissions: u64,
    pub maximum_cooperative_wait_seconds: f64,
    pub work: Work,
    pub uploaded_bytes: u64,
    pub readback_bytes: u64,
}

/// Compiled S2 kernels; small serial, 64- and 256-lane cooperative variants
/// are fixed private choices.
#[derive(Clone)]
pub(super) struct Kernels {
    segment_table: wgpu::ComputePipeline,
    make_keys: wgpu::ComputePipeline,
    radix_count: wgpu::ComputePipeline,
    radix_scatter: wgpu::ComputePipeline,
    gather_sorted: wgpu::ComputePipeline,
    certify_cells: wgpu::ComputePipeline,
    pair_marks: wgpu::ComputePipeline,
    pair_starts: wgpu::ComputePipeline,
    init_pair_results: wgpu::ComputePipeline,
    classify_pairs: wgpu::ComputePipeline,
    classify_cooperative: wgpu::ComputePipeline,
    classify_narrow: wgpu::ComputePipeline,
    fallback_counts: wgpu::ComputePipeline,
    fallback_copy: wgpu::ComputePipeline,
    pub(super) scan_blocks: wgpu::ComputePipeline,
    pub(super) scan_add: wgpu::ComputePipeline,
}
impl Kernels {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ambiguity contact"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/ambiguity_contact.wgsl").into(),
            ),
        });
        let pipeline = |entry, constants: &[(&str, f64)]| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants,
                    ..Default::default()
                },
                cache: None,
            })
        };
        Self {
            segment_table: pipeline("segment_table", &[]),
            make_keys: pipeline("make_keys", &[]),
            radix_count: pipeline("radix_count", &[]),
            radix_scatter: pipeline("radix_scatter", &[]),
            gather_sorted: pipeline("gather_sorted", &[]),
            certify_cells: pipeline("certify_cells", &[]),
            pair_marks: pipeline("pair_marks", &[]),
            pair_starts: pipeline("pair_starts", &[]),
            init_pair_results: pipeline("init_pair_results", &[]),
            classify_pairs: pipeline("classify_pairs", &[]),
            classify_cooperative: pipeline("classify_coop", &[]),
            classify_narrow: pipeline("classify_coop", &[("WG", 64.0)]),
            fallback_counts: pipeline("fallback_counts", &[]),
            fallback_copy: pipeline("fallback_copy", &[]),
            scan_blocks: pipeline("scan_blocks", &[]),
            scan_add: pipeline("scan_add", &[]),
        }
    }
}

/// Frame-lifetime S2 tables and the borrowed/S1 resident buffers they read.
#[derive(Clone, Copy)]
pub(super) struct Frame<'a> {
    pub parameters: &'a wgpu::Buffer,
    pub status: &'a wgpu::Buffer,
    pub path_info: &'a wgpu::Buffer,
    pub lengths: &'a wgpu::Buffer,
    pub segment_end: &'a wgpu::Buffer,
    pub points: &'a wgpu::Buffer,
    pub reservations: &'a wgpu::Buffer,
    pub ordinals: &'a wgpu::Buffer,
    pub point_scan: &'a wgpu::Buffer,
    pub path_base: &'a wgpu::Buffer,
    pub summary: &'a wgpu::Buffer,
}

/// Batch-class S2 buffers. The first buffer is the resident S1 output, which
/// later receives the fallback copy.
pub(super) struct Buffers {
    pub capacity: u32,
    pub owned: Slots<{ resources::BATCH }>,
}
impl Buffers {
    fn get(&self, index: usize) -> Result<&wgpu::Buffer> {
        self.owned.get(index)
    }
    pub(super) fn output(&self) -> Result<&wgpu::Buffer> {
        self.get(OUTPUT)
    }
    /// Radix ping-pong keys and permutation of one side.
    fn keys(&self, side: usize) -> Result<(&wgpu::Buffer, &wgpu::Buffer)> {
        let (Some(key), Some(permutation)) = (KEYS.get(side), PERMUTATIONS.get(side)) else {
            return Err(Error::Protocol(Protocol::Inventory));
        };
        Ok((self.get(*key)?, self.get(*permutation)?))
    }
}
const OUTPUT: usize = 0;
const KEYS: [usize; 2] = [1, 2];
const PERMUTATIONS: [usize; 2] = [3, 4];
const HISTOGRAM: usize = 5;
const MARKS: usize = 6;
const SCAN: usize = 7;
const SORTED: usize = 8;
const CERTIFICATES: usize = 9;
const STARTS: usize = 10;
const RESULTS: usize = 11;
const SCRATCH: usize = 12;
const AUDIT: usize = 13;
const OFFSETS: usize = 14;

/// Host arrays of one batch, reserved at the planned worst case before fill.
pub struct Host {
    starts: Vec<u32>,
    order: Vec<u32>,
    marks: Vec<u8>,
    decisions: Vec<PairDecision>,
    fallback: Vec<CandidateCell>,
}
/// Host bytes and stored records `Host::new` reserves for `cells` cells.
pub(super) fn host_demand(cells: u32) -> Result<(usize, usize)> {
    let cells = usize::try_from(cells).map_err(|_| overflow())?;
    let starts = cells.checked_add(1).ok_or_else(overflow)?;
    let parts = [
        starts.checked_mul(size_of::<u32>()),
        cells.checked_mul(size_of::<u32>()),
        cells.checked_mul(size_of::<u8>()),
        cells.checked_mul(size_of::<PairDecision>()),
        cells.checked_mul(size_of::<CandidateCell>()),
    ];
    let mut bytes = 0_usize;
    for part in parts {
        bytes = bytes
            .checked_add(part.ok_or_else(overflow)?)
            .ok_or_else(overflow)?;
    }
    Ok((bytes, cells.checked_mul(2).ok_or_else(overflow)?))
}
fn reserved<T>(capacity: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| Error::Workspace(crate::weak_frechet::Error::Allocation))?;
    Ok(values)
}
impl Host {
    /// Allocate the arrays `host_demand` reserved; the caller charges any
    /// capacity surplus reported by `footprint` before use.
    pub(super) fn allocate(cells: u32) -> Result<Self> {
        let count = usize::try_from(cells).map_err(|_| overflow())?;
        Ok(Self {
            starts: reserved(count.checked_add(1).ok_or_else(overflow)?)?,
            order: reserved(count)?,
            marks: reserved(count)?,
            decisions: reserved(count)?,
            fallback: reserved(count)?,
        })
    }
    /// Actual bytes and stored records of the allocated capacities.
    pub(super) fn footprint(&self) -> Result<(usize, usize)> {
        let parts = [
            self.starts.capacity().checked_mul(size_of::<u32>()),
            self.order.capacity().checked_mul(size_of::<u32>()),
            self.marks.capacity().checked_mul(size_of::<u8>()),
            self.decisions
                .capacity()
                .checked_mul(size_of::<PairDecision>()),
            self.fallback
                .capacity()
                .checked_mul(size_of::<CandidateCell>()),
        ];
        let mut bytes = 0_usize;
        for part in parts {
            bytes = bytes
                .checked_add(part.ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
        }
        let records = self
            .decisions
            .capacity()
            .checked_add(self.fallback.capacity())
            .ok_or_else(overflow)?;
        Ok((bytes, records))
    }
    pub(super) fn clear(&mut self) {
        self.starts.clear();
        self.order.clear();
        self.marks.clear();
        self.decisions.clear();
        self.fallback.clear();
    }
    /// Visit decisions in increasing pair order with each fallback pair's run.
    pub(super) fn visit<X>(
        &mut self,
        mut visit: impl FnMut(Decision<'_>) -> std::result::Result<(), X>,
    ) -> std::result::Result<(), X>
    where
        X: From<Error>,
    {
        let mut rest = self.fallback.as_mut_slice();
        for decision in &self.decisions {
            let source = usize::try_from(decision.source).map_err(|_| overflow())?;
            let partner = usize::try_from(decision.partner).map_err(|_| overflow())?;
            match decision.route {
                Route::Contact => visit(Decision::Contact { source, partner })?,
                Route::NoContact(_) => {}
                Route::Fallback(_) => {
                    let length = usize::try_from(decision.cells).map_err(|_| overflow())?;
                    let (cells, tail) = rest
                        .split_at_mut_checked(length)
                        .ok_or(Error::Protocol(Protocol::FallbackLength))?;
                    rest = tail;
                    visit(Decision::Fallback {
                        source,
                        partner,
                        cells,
                    })?;
                }
            }
        }
        Ok(())
    }
}

/// A validated decision handed to the CPU pair route or the contact callback.
pub enum Decision<'a> {
    Contact {
        source: usize,
        partner: usize,
    },
    Fallback {
        source: usize,
        partner: usize,
        cells: &'a mut [CandidateCell],
    },
}

/// The resident batch produced by one authorized fill.
pub(super) struct Resident {
    pub sources: Range<usize>,
    pub cells: u32,
}

/// Everything one batch classification borrows.
pub(super) struct Run<'a> {
    pub gpu: Gpu<'a>,
    pub kernels: &'a Kernels,
    pub frame: Frame<'a>,
    pub batch: &'a Buffers,
    pub transfer: Transfer<'a>,
    pub statistics: &'a mut Statistics,
    pub scan: ScanKernels<'a>,
    pub base: Parameters,
    /// Positive segments per ordinal, for host fallback validation.
    pub segment_offsets: &'a [u32],
}

/// Frame base parameters for the S2 kernels.
pub(super) fn parameters(scalars: Option<Scalars>, keys: Keys, paths: u32) -> Parameters {
    let admitted = u32::from(scalars.is_some());
    let scalars = scalars.unwrap_or(Scalars {
        radius: [0.0; 2],
        squared_radius: [0.0; 2],
        minimum: [0.0; 2],
        radius_is_zero: false,
    });
    Parameters {
        segment_bits: keys.segment_bits,
        partner_bits: keys.partner_bits,
        pair_cap: PAIR_CAP,
        admitted,
        radius: scalars.radius,
        squared_radius: scalars.squared_radius,
        minimum: scalars.minimum,
        radius_is_zero: u32::from(scalars.radius_is_zero),
        local_columns: 1,
        round_cap: ROUND_CAP,
        path_count: paths,
        ..Parameters::default()
    }
}

/// Path metadata and segment-length enclosures from the CPU preparation.
pub(super) struct TableData {
    pub info: Vec<[u32; 4]>,
    pub lengths: Vec<[f32; 2]>,
}

impl Run<'_> {
    fn write_parameters(&mut self, parameters: &Parameters) -> Result<()> {
        self.gpu
            .queue
            .write_buffer(self.frame.parameters, 0, bytemuck::bytes_of(parameters));
        self.statistics.uploaded_bytes = self
            .statistics
            .uploaded_bytes
            .checked_add(words(size_of::<Parameters>(), 1)?)
            .ok_or_else(overflow)?;
        Ok(())
    }
    fn group0(&self, pipeline: &wgpu::ComputePipeline, status: bool) -> wgpu::BindGroup {
        if status {
            self.gpu.bind(
                pipeline,
                0,
                [
                    (0, self.frame.parameters.as_entire_binding()),
                    (7, self.frame.status.as_entire_binding()),
                ],
            )
        } else {
            self.gpu.bind(
                pipeline,
                0,
                [(0, self.frame.parameters.as_entire_binding())],
            )
        }
    }
    fn submit(&mut self, encoder: wgpu::CommandEncoder) -> Result<Duration> {
        let elapsed = self.gpu.submit(encoder)?;
        self.statistics.work.record_submission(elapsed)?;
        Ok(elapsed)
    }
    fn dispatch(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        groups: &[&wgpu::BindGroup],
        workgroups: u64,
    ) -> Result<()> {
        if self.gpu.dispatch(encoder, pipeline, groups, workgroups)? {
            self.statistics.work.add_dispatch()?;
        }
        Ok(())
    }
    /// Read `N` words at `offset` words of `source` plus the S2 status words.
    fn read_small<const N: usize>(
        &mut self,
        source: &wgpu::Buffer,
        offset: u64,
    ) -> Result<([u32; N], [u32; STATUS_WORDS])> {
        let mut encoder = self.gpu.encoder();
        let bytes = words(N, 4)?;
        encoder.copy_buffer_to_buffer(source, words(offset, 4)?, self.frame.summary, 0, bytes);
        encoder.copy_buffer_to_buffer(
            self.frame.status,
            0,
            self.frame.summary,
            bytes,
            super::STATUS_BYTES,
        );
        let total = bytes
            .checked_add(super::STATUS_BYTES)
            .ok_or_else(overflow)?;
        let mut values = [0_u32; N];
        let mut status = [0_u32; STATUS_WORDS];
        let summary = self.frame.summary;
        let elapsed = self.gpu.submit_and_read(encoder, summary, total, |data| {
            let mut words = super::words(data);
            for slot in values.iter_mut().chain(status.iter_mut()) {
                *slot = words.next().ok_or(Error::Protocol(Protocol::Readback))?;
            }
            Ok(())
        })?;
        self.statistics.work.record_submission(elapsed)?;
        self.statistics.readback_bytes = self
            .statistics
            .readback_bytes
            .checked_add(total)
            .ok_or_else(overflow)?;
        Ok((values, status))
    }
    /// Read `count` records of `size` bytes from `source` through staging.
    fn read_records(
        &mut self,
        source: &wgpu::Buffer,
        count: usize,
        size: u64,
        records: bool,
        consume: impl FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        let bytes = words(count, size)?;
        self.transfer.read(
            &self.gpu,
            &mut self.statistics.work,
            (source, bytes),
            Content {
                record_bytes: size,
                records,
            },
            consume,
        )?;
        self.statistics.readback_bytes = self
            .statistics
            .readback_bytes
            .checked_add(bytes)
            .ok_or_else(overflow)?;
        Ok(())
    }
}

/// Build path metadata and length enclosures into reserved host vectors.
pub(super) fn fill_tables(
    data: &mut TableData,
    paths: &[crate::ambiguity_index::Path<'_>],
    prepared: &[crate::ambiguity_contact::PreparedCurve<'_>],
    segment_offsets: &[u32],
) -> Result<()> {
    for ((path, curve), offsets) in paths.iter().zip(prepared).zip(segment_offsets.windows(2)) {
        let (Some(&base), Some(&end)) = (offsets.first(), offsets.last()) else {
            return Err(Error::Protocol(Protocol::Tables));
        };
        let count = end
            .checked_sub(base)
            .ok_or(Error::Protocol(Protocol::Tables))?;
        if usize::try_from(count).ok() != Some(curve.segment_count()) {
            return Err(Error::Protocol(Protocol::Tables));
        }
        data.info.push([
            base,
            count,
            u32::from(path.points.first() == path.points.last()),
            0,
        ]);
        for segment in 0..curve.segment_count() {
            let length = curve
                .segment_length(segment)
                .map_err(|_| Error::Protocol(Protocol::Tables))?;
            data.lengths
                .push(if (2.0_f64.powi(-47)..=16.0).contains(&length) {
                    enclose(length)
                } else {
                    [-1.0, -1.0]
                });
        }
    }
    Ok(())
}

/// Upload frame tables and derive positive segment end points from S1 scans.
pub(super) fn upload_tables(
    gpu: &Gpu<'_>,
    kernels: &Kernels,
    frame: &Frame<'_>,
    data: &TableData,
    (base, samples): (Parameters, u32),
    statistics: &mut Statistics,
) -> Result<()> {
    gpu.queue
        .write_buffer(frame.path_info, 0, bytemuck::cast_slice(&data.info));
    gpu.queue
        .write_buffer(frame.lengths, 0, bytemuck::cast_slice(&data.lengths));
    let mut parameters = base;
    parameters.cells = samples;
    gpu.queue
        .write_buffer(frame.parameters, 0, bytemuck::bytes_of(&parameters));
    let uploaded = [
        words(data.info.len(), 16)?,
        words(data.lengths.len(), 8)?,
        words(size_of::<Parameters>(), 1)?,
    ];
    for bytes in uploaded {
        statistics.uploaded_bytes = statistics
            .uploaded_bytes
            .checked_add(bytes)
            .ok_or_else(overflow)?;
    }
    let group = gpu.bind(
        &kernels.segment_table,
        0,
        [
            (0, frame.parameters.as_entire_binding()),
            (1, frame.points.as_entire_binding()),
            (2, frame.reservations.as_entire_binding()),
            (3, frame.ordinals.as_entire_binding()),
            (4, frame.point_scan.as_entire_binding()),
            (5, frame.path_base.as_entire_binding()),
            (6, frame.segment_end.as_entire_binding()),
        ],
    );
    let mut encoder = gpu.encoder();
    if gpu.dispatch(
        &mut encoder,
        &kernels.segment_table,
        &[&group],
        u64::from(samples).div_ceil(BLOCK_SIZE),
    )? {
        statistics.work.add_dispatch()?;
    }
    let elapsed = gpu.submit(encoder)?;
    statistics.work.record_submission(elapsed)
}

/// O10 worst-case work estimate of one cooperative pair with padded size p,
/// l = log2(p): five bitonic sorts, four unions with two scans each, bounded
/// label rounds, and four union plus two graph binary searches per element.
pub(super) fn cooperative_work(cells: u32) -> Result<u64> {
    let p = u64::from(cells.max(1))
        .checked_next_power_of_two()
        .ok_or_else(overflow)?;
    let l = u64::from(p.trailing_zeros());
    let phases = l
        .checked_add(1)
        .and_then(|next| next.checked_mul(l))
        .map(|product| product / 2)
        .ok_or_else(overflow)?;
    let terms = [
        (p / 2).checked_mul(phases).and_then(|v| v.checked_mul(5)),
        l.checked_mul(p).and_then(|v| v.checked_mul(8)),
        u64::from(ROUND_CAP)
            .checked_add(1)
            .and_then(|v| v.checked_mul(10))
            .and_then(|v| v.checked_mul(p)),
        l.checked_add(1)
            .and_then(|v| v.checked_mul(p))
            .and_then(|v| v.checked_mul(6)),
        p.checked_mul(8),
    ];
    let mut total = 0_u64;
    for term in terms {
        total = total
            .checked_add(term.ok_or_else(overflow)?)
            .ok_or_else(overflow)?;
    }
    Ok(total)
}

impl Run<'_> {
    /// Classify one resident batch into `host`. Every device result, status
    /// and audit word is read and validated before this returns; no callback
    /// runs here.
    pub(super) fn classify(&mut self, resident: &Resident, host: &mut Host) -> Result<()> {
        host.clear();
        let cells = resident.cells;
        if cells == 0 {
            return Ok(());
        }
        let source_count = u32::try_from(resident.sources.len()).map_err(|_| overflow())?;
        let mut parameters = self.base;
        parameters.cells = cells;
        parameters.source_base = u32::try_from(resident.sources.start).map_err(|_| overflow())?;
        parameters.source_count = source_count;
        parameters.source_bits = bits_for(source_count);
        parameters.blocks = cells.div_ceil(RADIX_BLOCK).max(1);
        let pairs = self.sort_and_mark(&mut parameters)?;
        self.certify(cells)?;
        parameters.pairs = pairs;
        self.read_starts(cells, pairs, host)?;
        let serial = schedule(host, pairs)?;
        self.bucket_statistics(host, serial)?;
        self.classify_pairs(&mut parameters, host, serial)?;
        let fallback_cells = self.compact(&parameters)?;
        self.decode(host, pairs)?;
        self.read_fallback(host, fallback_cells)?;
        self.validate(resident, host, fallback_cells)?;
        self.statistics.batches = self
            .statistics
            .batches
            .checked_add(1)
            .ok_or_else(overflow)?;
        self.statistics.cells = self
            .statistics
            .cells
            .checked_add(u64::from(cells))
            .ok_or_else(overflow)?;
        self.statistics.pairs = self
            .statistics
            .pairs
            .checked_add(u64::from(pairs))
            .ok_or_else(overflow)?;
        Ok(())
    }

    /// Validate fields and indices, sort by the exact 64-bit key, mark pair runs
    /// and scan their starts. Returns the pair count after checking status.
    fn sort_and_mark(&mut self, parameters: &mut Parameters) -> Result<u32> {
        let cells = parameters.cells;
        let key_bits = parameters
            .segment_bits
            .checked_mul(2)
            .and_then(|bits| bits.checked_add(parameters.partner_bits))
            .and_then(|bits| bits.checked_add(parameters.source_bits))
            .ok_or_else(overflow)?;
        if key_bits > 64 {
            return Err(Error::KeyWidth { required: key_bits });
        }
        let threads = u64::from(cells).div_ceil(BLOCK_SIZE);
        self.write_parameters(parameters)?;
        let output = self.batch.output()?;
        self.make_keys(threads)?;
        let current = self.radix(parameters, key_bits)?;
        let k = self.kernels;
        let marks = self.batch.get(MARKS)?;
        let gather = [
            self.group0(&k.gather_sorted, false),
            self.gpu.bind(
                &k.gather_sorted,
                1,
                [
                    (2, output.as_entire_binding()),
                    (3, self.batch.get(SORTED)?.as_entire_binding()),
                    (6, self.batch.keys(current)?.1.as_entire_binding()),
                ],
            ),
        ];
        let mark = [
            self.group0(&k.pair_marks, true),
            self.gpu.bind(
                &k.pair_marks,
                1,
                [
                    (3, self.batch.get(SORTED)?.as_entire_binding()),
                    (9, marks.as_entire_binding()),
                ],
            ),
        ];
        let starts = [
            self.group0(&k.pair_starts, false),
            self.gpu.bind(
                &k.pair_starts,
                1,
                [
                    (9, marks.as_entire_binding()),
                    (10, self.batch.get(STARTS)?.as_entire_binding()),
                ],
            ),
        ];
        let next = cells.checked_add(1).ok_or_else(overflow)?;
        let mut encoder = self.gpu.encoder();
        self.dispatch(
            &mut encoder,
            &k.gather_sorted,
            &[&gather[0], &gather[1]],
            threads,
        )?;
        self.dispatch(
            &mut encoder,
            &k.pair_marks,
            &[&mark[0], &mark[1]],
            u64::from(next).div_ceil(BLOCK_SIZE),
        )?;
        let scan = self.batch.get(SCAN)?;
        let dispatches = self.scan.encode(
            &self.gpu,
            &mut encoder,
            ((marks, 0, next), scan),
            &mut 0_u32,
        )?;
        self.statistics.work.add_dispatches(dispatches)?;
        self.dispatch(
            &mut encoder,
            &k.pair_starts,
            &[&starts[0], &starts[1]],
            u64::from(next).div_ceil(BLOCK_SIZE),
        )?;
        self.submit(encoder)?;
        let ([pairs], status) = self.read_small::<1>(marks, u64::from(cells))?;
        check_sort_status(status)?;
        if pairs == 0 || pairs > cells {
            return Err(Error::Protocol(Protocol::PairCount { pairs, cells }));
        }
        Ok(pairs)
    }

    /// Field and index validation plus exact keys over the resident output.
    fn make_keys(&mut self, threads: u64) -> Result<()> {
        let (k, batch) = (self.kernels, self.batch);
        let keys = [
            self.group0(&k.make_keys, true),
            self.gpu.bind(
                &k.make_keys,
                1,
                [
                    (0, self.frame.path_info.as_entire_binding()),
                    (2, batch.output()?.as_entire_binding()),
                    (5, batch.keys(0)?.0.as_entire_binding()),
                    (6, batch.keys(0)?.1.as_entire_binding()),
                ],
            ),
        ];
        let mut encoder = self.gpu.encoder();
        encoder.clear_buffer(self.frame.status, 0, None);
        self.dispatch(&mut encoder, &k.make_keys, &[&keys[0], &keys[1]], threads)?;
        self.submit(encoder)?;
        Ok(())
    }

    /// Stable 6-bit LSD radix passes; returns the final permutation index.
    fn radix(&mut self, parameters: &mut Parameters, key_bits: u32) -> Result<usize> {
        let digit_bits = RADIX_BINS.trailing_zeros();
        let passes = key_bits.div_ceil(digit_bits);
        let histogram = self.batch.get(HISTOGRAM)?;
        let scan = self.batch.get(SCAN)?;
        let bins = parameters
            .blocks
            .checked_mul(RADIX_BINS)
            .ok_or_else(overflow)?;
        let mut current = 0_usize;
        for pass in 0..passes {
            parameters.shift = pass.checked_mul(digit_bits).ok_or_else(overflow)?;
            self.write_parameters(parameters)?;
            let other = 1_usize.saturating_sub(current);
            let input = self.batch.keys(current)?;
            let output = self.batch.keys(other)?;
            let k = self.kernels;
            let count = [
                self.group0(&k.radix_count, false),
                self.gpu.bind(
                    &k.radix_count,
                    1,
                    [
                        (5, input.0.as_entire_binding()),
                        (9, histogram.as_entire_binding()),
                    ],
                ),
            ];
            let scatter = [
                self.group0(&k.radix_scatter, false),
                self.gpu.bind(
                    &k.radix_scatter,
                    1,
                    [
                        (5, input.0.as_entire_binding()),
                        (6, input.1.as_entire_binding()),
                        (7, output.0.as_entire_binding()),
                        (8, output.1.as_entire_binding()),
                        (9, histogram.as_entire_binding()),
                    ],
                ),
            ];
            let mut encoder = self.gpu.encoder();
            let blocks = u64::from(parameters.blocks);
            self.dispatch(
                &mut encoder,
                &k.radix_count,
                &[&count[0], &count[1]],
                blocks,
            )?;
            let dispatches = self.scan.encode(
                &self.gpu,
                &mut encoder,
                ((histogram, 0, bins), scan),
                &mut 0_u32,
            )?;
            self.statistics.work.add_dispatches(dispatches)?;
            self.dispatch(
                &mut encoder,
                &k.radix_scatter,
                &[&scatter[0], &scatter[1]],
                blocks,
            )?;
            self.submit(encoder)?;
            current = other;
            self.statistics.radix_passes = self
                .statistics
                .radix_passes
                .checked_add(1)
                .ok_or_else(overflow)?;
        }
        parameters.shift = 0;
        Ok(current)
    }

    fn certify(&mut self, cells: u32) -> Result<()> {
        let k = self.kernels;
        let groups = [
            self.gpu.bind(
                &k.certify_cells,
                0,
                [
                    (0, self.frame.parameters.as_entire_binding()),
                    (1, self.frame.points.as_entire_binding()),
                    (6, self.frame.segment_end.as_entire_binding()),
                ],
            ),
            self.gpu.bind(
                &k.certify_cells,
                1,
                [
                    (0, self.frame.path_info.as_entire_binding()),
                    (1, self.frame.lengths.as_entire_binding()),
                    (3, self.batch.get(SORTED)?.as_entire_binding()),
                    (4, self.batch.get(CERTIFICATES)?.as_entire_binding()),
                ],
            ),
        ];
        let mut encoder = self.gpu.encoder();
        self.dispatch(
            &mut encoder,
            &k.certify_cells,
            &[&groups[0], &groups[1]],
            u64::from(cells).div_ceil(BLOCK_SIZE),
        )?;
        self.submit(encoder)?;
        Ok(())
    }

    /// Read and validate pair starts: 0, strictly increasing, ending at `cells`.
    fn read_starts(&mut self, cells: u32, pairs: u32, host: &mut Host) -> Result<()> {
        let count = usize::try_from(pairs)
            .ok()
            .and_then(|pairs| pairs.checked_add(1))
            .ok_or_else(overflow)?;
        if count > host.starts.capacity() {
            return Err(Error::Protocol(Protocol::Inventory));
        }
        let starts = self.batch.get(STARTS)?;
        let target = &mut host.starts;
        self.read_records(starts, count, 4, false, |data| {
            target.extend(super::words(data));
            Ok(())
        })?;
        let valid = host.starts.first() == Some(&0)
            && host.starts.last() == Some(&cells)
            && host.starts.len() == count
            && host
                .starts
                .windows(2)
                .all(|pair| pair.first() < pair.last());
        if !valid {
            return Err(Error::Protocol(Protocol::PairStarts));
        }
        Ok(())
    }

    /// Run the serial and cooperative classifiers over the audited schedule.
    fn classify_pairs(
        &mut self,
        parameters: &mut Parameters,
        host: &Host,
        serial: u32,
    ) -> Result<()> {
        let schedule_offset = words(parameters.cells.checked_add(1).ok_or_else(overflow)?, 4)?;
        self.gpu.queue.write_buffer(
            self.batch.get(STARTS)?,
            schedule_offset,
            bytemuck::cast_slice(&host.order),
        );
        self.statistics.uploaded_bytes = self
            .statistics
            .uploaded_bytes
            .checked_add(words(host.order.len(), 4)?)
            .ok_or_else(overflow)?;
        self.write_parameters(parameters)?;
        let k = self.kernels;
        let results = self.batch.get(RESULTS)?;
        let init = [
            self.group0(&k.init_pair_results, false),
            self.gpu
                .bind(&k.init_pair_results, 1, [(11, results.as_entire_binding())]),
        ];
        let mut encoder = self.gpu.encoder();
        encoder.clear_buffer(self.batch.get(AUDIT)?, 0, None);
        self.dispatch(
            &mut encoder,
            &k.init_pair_results,
            &[&init[0], &init[1]],
            u64::from(parameters.pairs).div_ceil(BLOCK_SIZE),
        )?;
        self.submit(encoder)?;
        self.serial_chunks(parameters, host, serial)?;
        self.cooperative_chunks(parameters, host, serial)?;
        self.check_audit(parameters.pairs, serial)
    }

    fn classifier_group(&self, pipeline: &wgpu::ComputePipeline) -> Result<wgpu::BindGroup> {
        Ok(self.gpu.bind(
            pipeline,
            1,
            [
                (0, self.frame.path_info.as_entire_binding()),
                (1, self.frame.lengths.as_entire_binding()),
                (3, self.batch.get(SORTED)?.as_entire_binding()),
                (4, self.batch.get(CERTIFICATES)?.as_entire_binding()),
                (10, self.batch.get(STARTS)?.as_entire_binding()),
                (11, self.batch.get(RESULTS)?.as_entire_binding()),
                (12, self.batch.get(SCRATCH)?.as_entire_binding()),
                (15, self.batch.get(AUDIT)?.as_entire_binding()),
            ],
        ))
    }

    fn size(host: &Host, slot: usize) -> Result<u32> {
        let pair = host
            .order
            .get(slot)
            .and_then(|pair| usize::try_from(*pair).ok())
            .ok_or(Error::Protocol(Protocol::Schedule))?;
        pair_size(&host.starts, pair)
    }

    fn serial_chunks(
        &mut self,
        parameters: &mut Parameters,
        host: &Host,
        serial: u32,
    ) -> Result<()> {
        let k = self.kernels;
        let groups = [
            self.group0(&k.classify_pairs, false),
            self.classifier_group(&k.classify_pairs)?,
        ];
        let mut chunk = 0_u32;
        while chunk < serial {
            let mut end = chunk.checked_add(1).ok_or_else(overflow)?;
            let mut cells = u64::from(Self::size(host, slot(chunk)?)?);
            while end < serial && end.saturating_sub(chunk) < PAIRS_PER_SUBMISSION {
                let next = cells
                    .checked_add(u64::from(Self::size(host, slot(end)?)?))
                    .ok_or_else(overflow)?;
                if next > CELLS_PER_SUBMISSION {
                    break;
                }
                cells = next;
                end = end.checked_add(1).ok_or_else(overflow)?;
            }
            parameters.chunk_start = chunk;
            parameters.chunk_count = end.checked_sub(chunk).ok_or_else(overflow)?;
            self.write_parameters(parameters)?;
            let mut encoder = self.gpu.encoder();
            self.dispatch(
                &mut encoder,
                &k.classify_pairs,
                &[&groups[0], &groups[1]],
                u64::from(parameters.chunk_count).div_ceil(SERIAL_LANES),
            )?;
            self.submit(encoder)?;
            chunk = end;
        }
        Ok(())
    }

    fn cooperative_chunks(
        &mut self,
        parameters: &mut Parameters,
        host: &Host,
        serial: u32,
    ) -> Result<()> {
        let k = self.kernels;
        let wide = [
            self.group0(&k.classify_cooperative, false),
            self.classifier_group(&k.classify_cooperative)?,
        ];
        let narrow = [
            self.group0(&k.classify_narrow, false),
            self.classifier_group(&k.classify_narrow)?,
        ];
        let pairs = parameters.pairs;
        let is_narrow = |cells: u32| u64::from(cells).next_power_of_two() <= NARROW_PADDED;
        let mut chunk = serial;
        while chunk < pairs {
            let first = Self::size(host, slot(chunk)?)?;
            let lanes = is_narrow(first);
            let mut work = cooperative_work(first)?;
            let mut end = chunk.checked_add(1).ok_or_else(overflow)?;
            while end < pairs && end.saturating_sub(chunk) < PAIRS_PER_SUBMISSION {
                let cells = Self::size(host, slot(end)?)?;
                let next = work
                    .checked_add(cooperative_work(cells)?)
                    .ok_or_else(overflow)?;
                if next > WORK_TARGET || is_narrow(cells) != lanes {
                    break;
                }
                work = next;
                end = end.checked_add(1).ok_or_else(overflow)?;
            }
            self.statistics.maximum_submitted_work =
                self.statistics.maximum_submitted_work.max(work);
            parameters.chunk_start = chunk;
            parameters.chunk_count = end.checked_sub(chunk).ok_or_else(overflow)?;
            self.write_parameters(parameters)?;
            let (pipeline, groups) = if lanes {
                (&k.classify_narrow, &narrow)
            } else {
                (&k.classify_cooperative, &wide)
            };
            let mut encoder = self.gpu.encoder();
            self.dispatch(
                &mut encoder,
                pipeline,
                &[&groups[0], &groups[1]],
                u64::from(parameters.chunk_count),
            )?;
            let elapsed = self.submit(encoder)?;
            self.statistics.cooperative_submissions = self
                .statistics
                .cooperative_submissions
                .checked_add(1)
                .ok_or_else(overflow)?;
            self.statistics.maximum_cooperative_wait_seconds = self
                .statistics
                .maximum_cooperative_wait_seconds
                .max(elapsed.as_secs_f64());
            chunk = end;
        }
        Ok(())
    }

    /// Check classifier error bits and per-bucket execution counts.
    fn check_audit(&mut self, pairs: u32, serial: u32) -> Result<()> {
        let mut audit = [0_u32; 64];
        let mut filled = 0_usize;
        let buffer = self.batch.get(AUDIT)?;
        let count = usize::try_from(AUDIT_WORDS).map_err(|_| overflow())?;
        self.read_records(buffer, count, 4, false, |data| {
            for word in super::words(data) {
                let slot = audit
                    .get_mut(filled)
                    .ok_or(Error::Protocol(Protocol::Readback))?;
                *slot = word;
                filled = filled.checked_add(1).ok_or_else(overflow)?;
            }
            Ok(())
        })?;
        let word = |index: usize| audit.get(index).copied().unwrap_or(u32::MAX);
        if filled != count || word(CLAIM_ERRORS) != 0 {
            return Err(Error::Protocol(Protocol::Claims {
                errors: word(CLAIM_ERRORS),
            }));
        }
        let cooperative = pairs.checked_sub(serial).ok_or_else(overflow)?;
        if (word(COOPERATIVE_CLAIMS), word(SERIAL_CLAIMS)) != (cooperative, serial) {
            return Err(Error::Protocol(Protocol::Execution {
                serial: word(SERIAL_CLAIMS),
                cooperative: word(COOPERATIVE_CLAIMS),
            }));
        }
        if word(34) != 0 {
            return Err(Error::Protocol(Protocol::LabelRange { failures: word(34) }));
        }
        for (slot, value) in self.statistics.rounds.iter_mut().zip(audit) {
            *slot = slot.checked_add(u64::from(value)).ok_or_else(overflow)?;
        }
        let add = |total: &mut u64, index| -> Result<()> {
            *total = total
                .checked_add(u64::from(word(index)))
                .ok_or_else(overflow)?;
            Ok(())
        };
        add(&mut self.statistics.rounds_unsettled, ROUND_BINS)?;
        add(&mut self.statistics.audit_failures, 33)?;
        add(&mut self.statistics.label_range_failures, 34)?;
        Ok(())
    }

    /// Fallback counts, their scan and the complete fallback copy. Returns the
    /// fallback cell total after checking every status word.
    fn compact(&mut self, parameters: &Parameters) -> Result<u32> {
        let k = self.kernels;
        let offsets = self.batch.get(OFFSETS)?;
        let results = self.batch.get(RESULTS)?;
        let counts = [
            self.group0(&k.fallback_counts, true),
            self.gpu.bind(
                &k.fallback_counts,
                1,
                [
                    (11, results.as_entire_binding()),
                    (13, offsets.as_entire_binding()),
                    (15, self.batch.get(AUDIT)?.as_entire_binding()),
                ],
            ),
        ];
        let copy = [
            self.group0(&k.fallback_copy, false),
            self.gpu.bind(
                &k.fallback_copy,
                1,
                [
                    (3, self.batch.get(SORTED)?.as_entire_binding()),
                    (9, self.batch.get(MARKS)?.as_entire_binding()),
                    (10, self.batch.get(STARTS)?.as_entire_binding()),
                    (11, results.as_entire_binding()),
                    (13, offsets.as_entire_binding()),
                    (14, self.batch.output()?.as_entire_binding()),
                ],
            ),
        ];
        let next = parameters.pairs.checked_add(1).ok_or_else(overflow)?;
        let mut encoder = self.gpu.encoder();
        self.dispatch(
            &mut encoder,
            &k.fallback_counts,
            &[&counts[0], &counts[1]],
            u64::from(next).div_ceil(BLOCK_SIZE),
        )?;
        let scan = self.batch.get(SCAN)?;
        let dispatches = self.scan.encode(
            &self.gpu,
            &mut encoder,
            ((offsets, 0, next), scan),
            &mut 0_u32,
        )?;
        self.statistics.work.add_dispatches(dispatches)?;
        self.dispatch(
            &mut encoder,
            &k.fallback_copy,
            &[&copy[0], &copy[1]],
            u64::from(parameters.cells).div_ceil(BLOCK_SIZE),
        )?;
        self.submit(encoder)?;
        let ([fallback], status) = self.read_small::<1>(offsets, u64::from(parameters.pairs))?;
        if status.iter().any(|word| *word != 0) {
            return Err(Error::Protocol(Protocol::Status { words: status }));
        }
        if fallback > parameters.cells {
            return Err(Error::Protocol(Protocol::FallbackLength));
        }
        Ok(fallback)
    }

    /// Decode every pair result directly from staging into typed decisions.
    fn decode(&mut self, host: &mut Host, pairs: u32) -> Result<()> {
        let count = usize::try_from(pairs).map_err(|_| overflow())?;
        if count > host.decisions.capacity() {
            return Err(Error::Protocol(Protocol::Inventory));
        }
        let results = self.batch.get(RESULTS)?;
        let decisions = &mut host.decisions;
        self.read_records(results, count, resources::RECORD_BYTES, true, |data| {
            let mut fields = super::words(data);
            while let Some(source) = fields.next() {
                let mut field = || fields.next().ok_or(Error::Protocol(Protocol::Readback));
                let (partner, word, cells) = (field()?, field()?, field()?);
                let route = route(word, decisions.len())?;
                decisions.push(PairDecision {
                    source,
                    partner,
                    cells,
                    route,
                });
            }
            Ok(())
        })
    }

    fn read_fallback(&mut self, host: &mut Host, cells: u32) -> Result<()> {
        let count = usize::try_from(cells).map_err(|_| overflow())?;
        if count > host.fallback.capacity() {
            return Err(Error::Protocol(Protocol::Inventory));
        }
        let output = self.batch.output()?;
        let fallback = &mut host.fallback;
        self.read_records(output, count, resources::RECORD_BYTES, true, |data| {
            for record in data.as_chunks::<16>().0 {
                fallback.push(bytemuck::pod_read_unaligned(record));
            }
            Ok(())
        })
    }

    /// Validate identities, counts and every fallback run (B1) before any
    /// decision is emitted, then record route statistics.
    fn validate(&mut self, resident: &Resident, host: &Host, fallback_cells: u32) -> Result<()> {
        let paths = self.base.path_count;
        let mut previous: Option<(u32, u32)> = None;
        let mut fallback_total = 0_u32;
        let mut offset = 0_usize;
        if host.decisions.len().checked_add(1) != Some(host.starts.len()) {
            return Err(Error::Protocol(Protocol::PairCount {
                pairs: u32::try_from(host.decisions.len()).unwrap_or(u32::MAX),
                cells: resident.cells,
            }));
        }
        for (pair, decision) in host.decisions.iter().enumerate() {
            let identity = (decision.source, decision.partner);
            let source = usize::try_from(decision.source).map_err(|_| overflow())?;
            let valid = resident.sources.contains(&source)
                && decision.partner > decision.source
                && decision.partner < paths
                && previous.is_none_or(|previous| previous < identity)
                && Some(decision.cells) == pair_size(&host.starts, pair).ok()
                && decision.route.valid_for(decision.cells, self.base.admitted);
            if !valid {
                return Err(Error::Protocol(Protocol::Pair { pair }));
            }
            previous = Some(identity);
            self.statistics.largest_pair_cells = self
                .statistics
                .largest_pair_cells
                .max(u64::from(decision.cells));
            match decision.route {
                Route::Contact => self.statistics.certain_contacts.add(decision.cells)?,
                Route::NoContact(Negative::Gate) => {
                    self.statistics
                        .certain_no_contact_gate
                        .add(decision.cells)?;
                }
                Route::NoContact(Negative::Empty) => {
                    self.statistics
                        .certain_no_contact_empty
                        .add(decision.cells)?;
                }
                Route::NoContact(Negative::Length) => {
                    self.statistics
                        .certain_no_contact_length
                        .add(decision.cells)?;
                }
                Route::Fallback(reason) => {
                    let length = usize::try_from(decision.cells).map_err(|_| overflow())?;
                    let end = offset.checked_add(length).ok_or_else(overflow)?;
                    let run = host
                        .fallback
                        .get(offset..end)
                        .ok_or(Error::Protocol(Protocol::FallbackLength))?;
                    self.validate_run(run, identity, pair)?;
                    offset = end;
                    fallback_total = fallback_total
                        .checked_add(decision.cells)
                        .ok_or_else(overflow)?;
                    self.statistics.fallback.add(decision.cells)?;
                    self.statistics
                        .fallback_reasons
                        .slot(reason)
                        .add(decision.cells)?;
                    self.statistics.largest_fallback_pair_cells = self
                        .statistics
                        .largest_fallback_pair_cells
                        .max(u64::from(decision.cells));
                }
            }
        }
        if fallback_total != fallback_cells || offset != host.fallback.len() {
            return Err(Error::Protocol(Protocol::FallbackLength));
        }
        Ok(())
    }

    /// One fallback run: same pair identity, valid segment indices and strictly
    /// increasing (a, b), so the CPU participation bound sees the exact
    /// row-major run.
    fn validate_run(&self, run: &[CandidateCell], identity: (u32, u32), pair: usize) -> Result<()> {
        let count = |path: u32| -> Option<u32> {
            let index = usize::try_from(path).ok()?;
            let start = *self.segment_offsets.get(index)?;
            let end = *self.segment_offsets.get(index.checked_add(1)?)?;
            end.checked_sub(start)
        };
        let (Some(rows), Some(columns)) = (count(identity.0), count(identity.1)) else {
            return Err(Error::Protocol(Protocol::Fallback { pair }));
        };
        let mut previous: Option<(u32, u32)> = None;
        for cell in run {
            let key = (cell.source_segment, cell.partner_segment);
            let valid = (cell.source, cell.partner) == identity
                && cell.source_segment < rows
                && cell.partner_segment < columns
                && previous.is_none_or(|previous| previous < key);
            if !valid {
                return Err(Error::Protocol(Protocol::Fallback { pair }));
            }
            previous = Some(key);
        }
        Ok(())
    }
}

fn slot(index: u32) -> Result<usize> {
    usize::try_from(index).map_err(|_| overflow())
}
fn pair_size(starts: &[u32], pair: usize) -> Result<u32> {
    let start = starts
        .get(pair)
        .ok_or(Error::Protocol(Protocol::Schedule))?;
    let end = starts
        .get(pair.checked_add(1).ok_or_else(overflow)?)
        .ok_or(Error::Protocol(Protocol::Schedule))?;
    end.checked_sub(*start)
        .ok_or(Error::Protocol(Protocol::PairStarts))
}

fn check_sort_status(status: [u32; STATUS_WORDS]) -> Result<()> {
    let [overflowed, cells, ..] = status;
    if overflowed != 0 {
        return Err(Error::Protocol(Protocol::ScanOverflow));
    }
    if cells & 8 != 0 {
        return Err(Error::Protocol(Protocol::DuplicateCell));
    }
    if cells != 0 || status.iter().skip(2).any(|word| *word != 0) {
        return Err(Error::Protocol(Protocol::InvalidCandidate {
            status: cells,
        }));
    }
    Ok(())
}

/// Deterministic bucket schedule: serial pairs (small or oversized) by
/// decreasing size, then cooperative pairs by decreasing size. Audits that
/// every pair index appears exactly once. Returns the serial count.
fn schedule(host: &mut Host, pairs: u32) -> Result<u32> {
    let count = usize::try_from(pairs).map_err(|_| overflow())?;
    if count > host.order.capacity() || count > host.marks.capacity() {
        return Err(Error::Protocol(Protocol::Inventory));
    }
    let starts = &host.starts;
    let size = |pair: &u32| -> u32 {
        usize::try_from(*pair)
            .ok()
            .and_then(|pair| pair_size(starts, pair).ok())
            .unwrap_or(u32::MAX)
    };
    let is_serial = |pair: &u32| {
        let cells = size(pair);
        cells <= SMALL_PAIR || cells > PAIR_CAP
    };
    host.order.extend((0..pairs).filter(|pair| is_serial(pair)));
    let serial = host.order.len();
    host.order
        .extend((0..pairs).filter(|pair| !is_serial(pair)));
    let (small, cooperative) = host
        .order
        .split_at_mut_checked(serial)
        .ok_or(Error::Protocol(Protocol::Schedule))?;
    small.sort_unstable_by_key(|pair| std::cmp::Reverse(size(pair)));
    cooperative.sort_unstable_by_key(|pair| std::cmp::Reverse(size(pair)));
    host.marks.resize(count, 0);
    let mut complete = host.order.len() == count;
    for pair in &host.order {
        match usize::try_from(*pair)
            .ok()
            .and_then(|pair| host.marks.get_mut(pair))
        {
            Some(mark) if *mark == 0 => *mark = 1,
            _ => complete = false,
        }
    }
    if !complete {
        return Err(Error::Protocol(Protocol::Schedule));
    }
    u32::try_from(serial).map_err(|_| overflow())
}

impl Run<'_> {
    /// Schedule-derived bucket statistics (oversized, serial, cooperative sizes).
    pub(super) fn bucket_statistics(&mut self, host: &Host, serial: u32) -> Result<()> {
        for (index, pair) in host.order.iter().enumerate() {
            let cells = usize::try_from(*pair)
                .ok()
                .and_then(|pair| pair_size(&host.starts, pair).ok())
                .ok_or(Error::Protocol(Protocol::Schedule))?;
            let counter = if u32::try_from(index).map_err(|_| overflow())? >= serial {
                let bucket = u64::from(cells).next_power_of_two().trailing_zeros();
                let slot = self
                    .statistics
                    .cooperative_buckets
                    .get_mut(usize::try_from(bucket).map_err(|_| overflow())?)
                    .ok_or(Error::Protocol(Protocol::Schedule))?;
                *slot = slot.checked_add(1).ok_or_else(overflow)?;
                &mut self.statistics.cooperative_pairs
            } else if cells > PAIR_CAP {
                &mut self.statistics.oversized_pairs
            } else {
                &mut self.statistics.serial_pairs
            };
            *counter = counter.checked_add(1).ok_or_else(overflow)?;
        }
        Ok(())
    }
}
