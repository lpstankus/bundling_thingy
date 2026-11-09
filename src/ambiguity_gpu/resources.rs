//! Joint device inventory of one GPU ambiguity frame.
//!
//! Every S1 and S2 buffer comes from one `Pool` under the configured combined
//! byte limit. A buffer is described by a `Spec`; the planner sums the same
//! specs that allocation consumes, so the planned and allocated sizes cannot
//! drift. Sizes round up to 16 bytes with a 16-byte minimum and are checked
//! against the adapter's buffer and binding limits before creation.
//!
//! Stored-record rule: a buffer whose `records` is nonzero materializes
//! candidate-cell or pair-result identities. Its capacity is reserved on the
//! shared stored-record ledger for as long as it exists (resident S1 output,
//! which later holds the fallback copy, sorted cells, pair results and readback
//! staging that carries such records). Certificates, keys, permutations,
//! offsets, schedules, labels and interval scratch hold no separate identity
//! record: they consume device bytes only. Renderer-owned input geometry is
//! borrowed, never allocated or destroyed here, and is reported separately.
//! Driver, pipeline and allocator overhead is outside this byte limit.
use super::{Error, Limit, Result, overflow};
use crate::weak_frechet::Budget;
use std::mem::size_of;

/// Bytes of one resident candidate cell or pair result.
pub(super) const RECORD_BYTES: u64 = 16;
/// Upper bound of one readback staging buffer, a multiple of 12 and 16.
pub(super) const MAXIMUM_STAGING_BYTES: u64 = 1_048_560;
/// Smallest staging buffer that holds a batch's classifier audit words.
const MINIMUM_STAGING_BYTES: u64 = 288;
const MINIMUM_BUFFER_BYTES: u64 = 16;
/// Scratch regions of 4-byte words per batch cell used by the S2 classifiers.
const CONTACT_SCRATCH_REGIONS: u64 = 19;
/// Classifier audit words before the per-pair claims.
pub(super) const AUDIT_WORDS: u64 = 64;
/// Radix digit bins (6-bit digits) and elements per radix block.
pub(super) const RADIX_BINS: u32 = 64;
pub(super) const RADIX_BLOCK: u32 = 1024;
const SCAN_BLOCK: u32 = 512;
const SCAN_ALIGNMENT: u64 = 256;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Usage {
    Storage,
    Uniform,
    Read,
}
impl Usage {
    const fn flags(self) -> wgpu::BufferUsages {
        match self {
            Self::Storage => wgpu::BufferUsages::STORAGE
                .union(wgpu::BufferUsages::COPY_SRC)
                .union(wgpu::BufferUsages::COPY_DST),
            Self::Uniform => wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST),
            Self::Read => wgpu::BufferUsages::COPY_DST.union(wgpu::BufferUsages::MAP_READ),
        }
    }
}

/// One planned buffer.
#[derive(Clone, Copy)]
pub(super) struct Spec {
    pub label: &'static str,
    pub bytes: u64,
    pub usage: Usage,
    /// Stored-record slots this buffer materializes while it exists.
    pub records: u64,
}
const fn storage(label: &'static str, bytes: u64) -> Spec {
    Spec {
        label,
        bytes,
        usage: Usage::Storage,
        records: 0,
    }
}
fn record_storage(label: &'static str, records: u32) -> Result<Spec> {
    Ok(Spec {
        label,
        bytes: words(records, RECORD_BYTES)?,
        usage: Usage::Storage,
        records: u64::from(records),
    })
}

/// A buffer owned by the frame. Dropping it destroys the device storage.
pub(super) struct Owned {
    buffer: wgpu::Buffer,
    bytes: u64,
    records: u64,
}
impl Drop for Owned {
    fn drop(&mut self) {
        self.buffer.destroy();
    }
}
impl Owned {
    pub(super) const fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    pub(super) const fn records(&self) -> u64 {
        self.records
    }
}

/// Planned demand of a group of specs.
#[derive(Clone, Copy, Default)]
pub(super) struct Demand {
    pub bytes: u64,
    pub records: u64,
}
impl Demand {
    pub(super) fn plus(self, other: Self) -> Result<Self> {
        Ok(Self {
            bytes: self.bytes.checked_add(other.bytes).ok_or_else(overflow)?,
            records: self
                .records
                .checked_add(other.records)
                .ok_or_else(overflow)?,
        })
    }
}

/// Counted device allocations of one frame against the combined byte limit.
pub(super) struct Pool {
    device: wgpu::Device,
    limit: u64,
    allocated: u64,
    peak: u64,
    records: u64,
    peak_records: u64,
}

/// A record reservation rolls back until buffer ownership has been installed.
struct PendingRecords<'a> {
    budget: &'a mut Budget,
    records: usize,
}
impl<'a> PendingRecords<'a> {
    fn reserve(budget: &'a mut Budget, records: usize) -> Result<Self> {
        budget.reserve_records(records).map_err(Error::Workspace)?;
        Ok(Self { budget, records })
    }
    fn commit(mut self) {
        self.records = 0;
    }
}
impl Drop for PendingRecords<'_> {
    fn drop(&mut self) {
        let _ = self.budget.release_records(self.records);
    }
}

impl Pool {
    pub(super) const fn new(device: wgpu::Device, limit: u64) -> Self {
        Self {
            device,
            limit,
            allocated: 0,
            peak: 0,
            records: 0,
            peak_records: 0,
        }
    }
    pub(super) const fn allocated(&self) -> u64 {
        self.allocated
    }
    pub(super) const fn peak(&self) -> u64 {
        self.peak
    }
    pub(super) const fn records(&self) -> u64 {
        self.records
    }
    pub(super) const fn peak_records(&self) -> u64 {
        self.peak_records
    }
    /// Allocated size of `spec`, checked against the adapter limits.
    pub(super) fn size(&self, spec: &Spec) -> Result<u64> {
        let size = spec
            .bytes
            .max(MINIMUM_BUFFER_BYTES)
            .checked_next_multiple_of(MINIMUM_BUFFER_BYTES)
            .ok_or_else(overflow)?;
        let limits = self.device.limits();
        let allowed = match spec.usage {
            Usage::Storage => limits
                .max_buffer_size
                .min(u64::from(limits.max_storage_buffer_binding_size)),
            Usage::Uniform => limits
                .max_buffer_size
                .min(u64::from(limits.max_uniform_buffer_binding_size)),
            Usage::Read => limits.max_buffer_size,
        };
        if size > allowed {
            return Err(Error::Limit {
                limit: Limit::AdapterBuffer(spec.label),
                demand: size,
                available: allowed,
            });
        }
        Ok(size)
    }
    /// Allocated bytes and records of `specs`.
    pub(super) fn demand(&self, specs: &[Spec]) -> Result<Demand> {
        let mut demand = Demand::default();
        for spec in specs {
            demand = demand.plus(Demand {
                bytes: self.size(spec)?,
                records: spec.records,
            })?;
        }
        Ok(demand)
    }
    /// Device bytes still available after `releasable` live bytes are freed.
    pub(super) const fn available(&self, releasable: u64) -> u64 {
        self.limit
            .saturating_sub(self.allocated.saturating_sub(releasable))
    }
    /// Reserve the record slots of `spec` on `budget`, then create it.
    pub(super) fn allocate(&mut self, budget: &mut Budget, spec: &Spec) -> Result<Owned> {
        let size = self.size(spec)?;
        let total = self.allocated.checked_add(size).ok_or_else(overflow)?;
        if total > self.limit {
            return Err(Error::Limit {
                limit: Limit::DeviceMemory(spec.label),
                demand: total,
                available: self.limit,
            });
        }
        let records = usize::try_from(spec.records).map_err(|_| overflow())?;
        let record_total = self
            .records
            .checked_add(spec.records)
            .ok_or_else(overflow)?;
        let reservation = PendingRecords::reserve(budget, records)?;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(spec.label),
            size,
            usage: spec.usage.flags(),
            mapped_at_creation: false,
        });
        self.allocated = total;
        self.peak = self.peak.max(total);
        self.records = record_total;
        self.peak_records = self.peak_records.max(record_total);
        reservation.commit();
        Ok(Owned {
            buffer,
            bytes: size,
            records: spec.records,
        })
    }
    /// Destroy `owned`, then release its bytes and record slots.
    pub(super) fn free(&mut self, budget: &mut Budget, owned: Owned) -> Result<()> {
        let (bytes, records) = (owned.bytes, owned.records);
        drop(owned);
        self.allocated = self
            .allocated
            .checked_sub(bytes)
            .ok_or(Error::Protocol(super::Protocol::Inventory))?;
        self.records = self
            .records
            .checked_sub(records)
            .ok_or(Error::Protocol(super::Protocol::Inventory))?;
        budget
            .release_records(usize::try_from(records).map_err(|_| overflow())?)
            .map_err(Error::Workspace)
    }
}

pub(super) fn words(count: impl TryInto<u64>, stride: u64) -> Result<u64> {
    count
        .try_into()
        .map_err(|_| overflow())?
        .checked_mul(stride)
        .ok_or_else(overflow)
}
fn plus_one(value: u32) -> Result<u32> {
    value.checked_add(1).ok_or_else(overflow)
}

pub(super) fn scan_region(blocks: u32) -> Result<u64> {
    words(blocks, 4)?
        .checked_next_multiple_of(SCAN_ALIGNMENT)
        .ok_or_else(overflow)
}
/// Scratch bytes for every recursive level of a scan over `count` values.
pub(super) fn scan_scratch(count: u32) -> Result<u64> {
    let mut total = 0_u64;
    let mut current = count;
    loop {
        let blocks = current.div_ceil(SCAN_BLOCK);
        total = total
            .checked_add(scan_region(blocks)?)
            .ok_or_else(overflow)?;
        if blocks <= 1 {
            return Ok(total);
        }
        current = blocks;
    }
}

/// Frame sizes known when the frame begins.
#[derive(Clone, Copy)]
pub(super) struct Shape {
    pub samples: u32,
    pub paths: u32,
    pub reservations: u32,
    /// Upper bound of positive segments used by S1 before they are counted.
    pub segment_capacity: u32,
    /// Actual positive segments of the CPU preparation.
    pub segments: u32,
    pub cells: u32,
    pub uniform_stride: u64,
    pub s1_parameters: u64,
    pub s2_parameters: u64,
}

pub(super) const FIXED: usize = 20;
/// Geometry, grid, uniform and S2 frame tables, in allocation order.
pub(super) fn fixed(shape: &Shape) -> Result<[Spec; FIXED]> {
    let paths = plus_one(shape.paths)?;
    let setup_scratch = scan_scratch(plus_one(shape.samples)?)?
        .max(scan_scratch(plus_one(shape.cells)?)?)
        .max(scan_scratch(plus_one(shape.segment_capacity)?)?);
    let summary_words = paths
        .checked_mul(2)
        .and_then(|value| value.checked_add(8))
        .ok_or_else(overflow)?;
    let uniform = |label, bytes| Spec {
        label,
        bytes,
        usage: Usage::Uniform,
        records: 0,
    };
    Ok([
        uniform("ambiguity candidate parameters", shape.s1_parameters),
        uniform(
            "ambiguity scan uniforms",
            words(super::UNIFORM_SLOTS, shape.uniform_stride)?,
        ),
        Spec {
            label: "ambiguity summary",
            bytes: words(summary_words, 4)?,
            usage: Usage::Read,
            records: 0,
        },
        storage("ambiguity path order", words(shape.paths, 4)?),
        storage("ambiguity path ordinals", words(shape.reservations, 4)?),
        storage(
            "ambiguity positive point scan",
            words(plus_one(shape.samples)?, 4)?,
        ),
        storage("ambiguity segment offsets", words(paths, 4)?),
        storage("ambiguity path footprint offsets", words(paths, 4)?),
        storage("ambiguity candidate status", super::STATUS_BYTES),
        storage(
            "ambiguity segments",
            words(shape.segment_capacity, super::SEGMENT_BYTES)?,
        ),
        storage(
            "ambiguity segment footprint offsets",
            words(plus_one(shape.segment_capacity)?, 4)?,
        ),
        storage("ambiguity cell starts", words(plus_one(shape.cells)?, 4)?),
        storage("ambiguity cell cursors", words(shape.cells, 4)?),
        storage("ambiguity setup scan sums", setup_scratch),
        storage("ambiguity count output placeholder", RECORD_BYTES),
        uniform("ambiguity contact parameters", shape.s2_parameters),
        storage("ambiguity contact status", super::STATUS_BYTES),
        storage(
            "ambiguity contact path metadata",
            words(shape.paths, words(size_of::<[u32; 4]>(), 1)?)?,
        ),
        storage(
            "ambiguity contact segment lengths",
            words(shape.segments, words(size_of::<[f32; 2]>(), 1)?)?,
        ),
        storage("ambiguity contact segment ends", words(shape.segments, 4)?),
    ])
}

/// Footprint entries, cell entries, item weights and their scan scratch.
pub(super) fn footprint(footprints: u32) -> Result<[Spec; 4]> {
    Ok([
        storage(
            "ambiguity footprint",
            words(footprints, super::FOOTPRINT_BYTES)?,
        ),
        storage("ambiguity cell entries", words(footprints, 4)?),
        storage("ambiguity item offsets", words(plus_one(footprints)?, 4)?),
        storage(
            "ambiguity footprint scan sums",
            scan_scratch(plus_one(footprints)?)?,
        ),
    ])
}

/// Count work-item capacity: per-item counts and their scan scratch.
pub(super) fn items(items: u32) -> Result<[Spec; 2]> {
    let count = items.checked_add(1).ok_or_else(|| Error::Limit {
        limit: Limit::CountOverflow,
        demand: u64::from(items),
        available: u64::from(u32::MAX.saturating_sub(1)),
    })?;
    Ok([
        storage("ambiguity item counts", words(count, 4)?),
        storage("ambiguity item scan sums", scan_scratch(count)?),
    ])
}

pub(super) const BATCH: usize = 15;
/// Resident candidates and S2 workspace for a batch of at most `cells` cells and
/// pairs. The S1 output later holds the fallback copy: certification only reads
/// the sorted copy, and the fallback copy is written after the last read of the
/// output, so both share one record buffer.
pub(super) fn batch(cells: u32) -> Result<[Spec; BATCH]> {
    let maximum_cells = u64::from(u32::MAX)
        .checked_div(CONTACT_SCRATCH_REGIONS)
        .ok_or_else(overflow)?;
    if u64::from(cells) > maximum_cells {
        return Err(Error::Limit {
            limit: Limit::CountOverflow,
            demand: u64::from(cells),
            available: maximum_cells,
        });
    }
    let blocks = cells.div_ceil(RADIX_BLOCK).max(1);
    let histogram = blocks.checked_mul(RADIX_BINS).ok_or_else(overflow)?;
    let starts = cells
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(overflow)?;
    let scratch_words = u64::from(cells)
        .checked_mul(CONTACT_SCRATCH_REGIONS)
        .filter(|value| u32::try_from(*value).is_ok())
        .ok_or_else(overflow)?;
    Ok([
        record_storage("ambiguity candidates and fallback cells", cells)?,
        storage("ambiguity contact keys a", words(cells, 8)?),
        storage("ambiguity contact keys b", words(cells, 8)?),
        storage("ambiguity contact permutation a", words(cells, 4)?),
        storage("ambiguity contact permutation b", words(cells, 4)?),
        storage(
            "ambiguity contact histogram",
            words(plus_one(histogram)?, 4)?,
        ),
        storage("ambiguity contact pair marks", words(plus_one(cells)?, 4)?),
        storage(
            "ambiguity contact scan sums",
            scan_scratch(histogram)?.max(scan_scratch(plus_one(cells)?)?),
        ),
        record_storage("ambiguity contact sorted cells", cells)?,
        storage(
            "ambiguity contact certificates",
            words(cells, super::contact::CERTIFICATE_BYTES)?,
        ),
        storage(
            "ambiguity contact pair starts and schedule",
            words(starts, 4)?,
        ),
        record_storage("ambiguity contact pair results", cells)?,
        storage("ambiguity contact scratch", words(scratch_words, 4)?),
        storage(
            "ambiguity contact audit and claims",
            words(
                u64::from(cells)
                    .checked_add(AUDIT_WORDS)
                    .ok_or_else(overflow)?,
                4,
            )?,
        ),
        storage(
            "ambiguity contact fallback offsets",
            words(plus_one(cells)?, 4)?,
        ),
    ])
}

/// Readback staging for `bytes`, bounded by the fixed cap. Staging that holds
/// candidate or pair records charges one record slot per record it can hold.
pub(super) fn staging(bytes: u64, records: bool) -> Result<Spec> {
    let bytes = bytes
        .clamp(MINIMUM_STAGING_BYTES, MAXIMUM_STAGING_BYTES)
        .checked_next_multiple_of(MINIMUM_BUFFER_BYTES)
        .ok_or_else(overflow)?;
    Ok(Spec {
        label: "ambiguity readback staging",
        bytes,
        usage: Usage::Read,
        records: if records { bytes / RECORD_BYTES } else { 0 },
    })
}
/// Largest staging a batch of `cells` cells needs.
pub(super) fn batch_staging(cells: u32) -> Result<Spec> {
    staging(words(cells, RECORD_BYTES)?, true)
}

/// Owned buffers addressed by fixed slot indices.
pub(super) struct Slots<const N: usize>([Option<Owned>; N]);
impl<const N: usize> Slots<N> {
    /// Allocate every spec in order. On failure the buffers already created
    /// are destroyed and released before the error returns.
    pub(super) fn allocate(
        pool: &mut Pool,
        budget: &mut Budget,
        specs: &[Spec; N],
    ) -> Result<Self> {
        let mut slots = Self(std::array::from_fn(|_| None));
        for (slot, spec) in slots.0.iter_mut().zip(specs) {
            match pool.allocate(budget, spec) {
                Ok(owned) => *slot = Some(owned),
                Err(error) => {
                    let _ = slots.free(pool, budget);
                    return Err(error);
                }
            }
        }
        Ok(slots)
    }
    pub(super) fn get(&self, index: usize) -> Result<&wgpu::Buffer> {
        self.0
            .get(index)
            .and_then(Option::as_ref)
            .map(Owned::buffer)
            .ok_or(Error::Protocol(super::Protocol::Inventory))
    }
    pub(super) fn bytes(&self) -> u64 {
        self.0.iter().flatten().map(|owned| owned.bytes).sum()
    }
    pub(super) fn records(&self) -> u64 {
        self.0.iter().flatten().map(Owned::records).sum()
    }
    /// Destroy every buffer, then release its bytes and record slots.
    pub(super) fn free(self, pool: &mut Pool, budget: &mut Budget) -> Result<()> {
        let mut result = Ok(());
        for owned in self.0.into_iter().flatten() {
            let freed = pool.free(budget, owned);
            result = result.and(freed);
        }
        result
    }
}
