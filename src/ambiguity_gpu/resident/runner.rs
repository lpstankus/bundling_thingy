//! Dispatch runner of the gpu-f32 kernels.
//!
//! Every kernel shares one bind group layout: a dynamic-offset uniform register
//! block, two read-only and five read-write storage bindings, and the status
//! words. Dispatches are encoded into one command buffer until their uniform
//! slots or the work target of a submission are used up; each submission is
//! waited for, so work per submission stays bounded. Buffers come from one
//! counted `Store` under the combined byte limit.
use super::super::{
    Error, Limit, POLL_TIMEOUT, Protocol, Result, Work, overflow,
    resources::{Owned, Pool, Spec, Usage},
};
use super::Fault;
use std::{num::NonZeroU64, sync::mpsc, time::Instant};

pub(super) const WORKGROUP: u64 = 256;
pub(super) const STATUS_WORDS: usize = 32;
const STATUS_BYTES: u64 = 128;
const REGISTER_BYTES: u64 = 80;
/// Bytes of the register ring; one slot per dispatch of a submission.
const RING_BYTES: u64 = 1 << 16;
const DUMMY_BYTES: u64 = 256;
const STAGING_BYTES: u64 = 1 << 20;
/// Work units one dispatch, and one submission, may hold.
///
/// A unit is an innermost-loop iteration or a straight-line block touching at
/// most 32 elements. Uniform-bound dispatches are sliced here. Callers supply
/// and fit Total bounds; excessive invocation or Total work is refused.
pub const WORK_TARGET: u64 = 1 << 31;
/// Work units one invocation may run.
pub const INVOCATION_TARGET: u64 = 1 << 18;
const SCAN_BLOCK: u64 = 512;
pub(super) const RADIX_BLOCK: u64 = 1024;
const STORAGE_BINDINGS: u32 = 7;

/// Scalar registers of one dispatch (the WGSL `P`).
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(super) struct P {
    pub n: u32,
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub d: u32,
    pub e: u32,
    pub f: u32,
    pub g: u32,
    pub h: u32,
    pub i: u32,
    pub j: u32,
    pub k: u32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
    /// First workgroup of a dispatch slice; set by the runner.
    pub o: u32,
    pub pad: [u32; 3],
}
impl P {
    /// Registers of a kernel that takes only the element count.
    pub(super) fn count(n: u64) -> Result<Self> {
        Ok(Self {
            n: word(n)?,
            ..Self::default()
        })
    }
}

pub(super) fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(overflow)
}
pub(super) fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or_else(overflow)
}
/// A device index or count; values beyond u32 are a count overflow.
pub(super) fn word(value: u64) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::Limit {
        limit: Limit::CountOverflow,
        demand: value,
        available: u64::from(u32::MAX),
    })
}
/// Bits needed to represent values below `count`.
pub(super) const fn bits(count: u64) -> u32 {
    u64::BITS.saturating_sub(count.saturating_sub(1).leading_zeros())
}
/// Scratch words of every recursive level of a scan over `count` values.
pub(super) fn scan_words(count: u64) -> Result<u64> {
    let mut total = 0;
    let mut current = count;
    loop {
        let blocks = current.div_ceil(SCAN_BLOCK);
        total = add(total, blocks)?;
        if blocks <= 1 {
            return add(total, 1);
        }
        current = blocks;
    }
}
/// Consecutive `(first, count)` spans of at most `size` elements covering
/// `first..end` in increasing order. Callers pass a positive size; zero is
/// treated as one. `next + count <= end`, so the saturating add is exact.
pub(super) fn spans(first: u64, end: u64, size: u64) -> impl Iterator<Item = (u64, u64)> {
    let size = size.max(1);
    let mut next = first;
    std::iter::from_fn(move || {
        let count = end.saturating_sub(next).min(size);
        if count == 0 {
            return None;
        }
        let span = (next, count);
        next = next.saturating_add(count);
        Some(span)
    })
}
/// Histogram words of a radix sort over `count` records.
pub(super) fn histogram_words(count: u64) -> Result<u64> {
    mul(count.div_ceil(RADIX_BLOCK).max(1), 256)
}

/// Byte ledger of host metadata sized by the input or by device counts:
/// readback vectors, membership and ID tables, batch and chunk plans, pair
/// listings. It is separate from the device byte limit.
#[derive(Debug, Clone, Copy)]
pub(super) struct Host {
    limit: u64,
    used: u64,
    peak: u64,
}
impl Host {
    pub(super) const fn new(limit: u64) -> Self {
        Self {
            limit,
            used: 0,
            peak: 0,
        }
    }
    /// Count `bytes` of host metadata about to be allocated.
    pub(super) fn reserve(&mut self, bytes: u64) -> Result<()> {
        let total = add(self.used, bytes)?;
        if total > self.limit {
            return Err(Error::Limit {
                limit: Limit::HostMetadata,
                demand: total,
                available: self.limit,
            });
        }
        self.used = total;
        self.peak = self.peak.max(total);
        Ok(())
    }
    pub(super) const fn used(&self) -> u64 {
        self.used
    }
    pub(super) const fn peak(&self) -> u64 {
        self.peak
    }
    /// Forget everything reserved after `used` was read (its owners dropped).
    pub(super) const fn restore(&mut self, used: u64) {
        self.used = used;
    }
    /// A vector with room for `count` elements, counted on the ledger.
    pub(super) fn vector<T>(&mut self, count: u64) -> Result<Vec<T>> {
        let size = u64::try_from(std::mem::size_of::<T>()).map_err(|_| overflow())?;
        self.reserve(mul(count, size)?)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(usize::try_from(count).map_err(|_| overflow())?)
            .map_err(|_| Error::Fault(Fault::HostAllocation))?;
        Ok(values)
    }
}

/// Counted buffers under the combined byte limit. Buffers are released in
/// reverse order down to a mark; dropping the store destroys what remains.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Keep {
    /// Released with the scope that allocated it.
    Scope,
    /// Handed to the enclosing scope when its own scope ends.
    Promoted,
    /// Released only with the store.
    Session,
}

pub(super) struct Store {
    pool: Pool,
    /// Each buffer with how long it lives.
    owned: Vec<(Owned, Keep)>,
}
impl Store {
    pub(super) const fn new(device: wgpu::Device, limit: u64) -> Self {
        Self {
            pool: Pool::new(device, limit),
            owned: Vec::new(),
        }
    }
    fn allocate(&mut self, spec: &Spec) -> Result<wgpu::Buffer> {
        self.owned
            .try_reserve(1)
            .map_err(|_| Error::Fault(Fault::HostAllocation))?;
        let owned = self.pool.allocate(spec)?;
        let buffer = owned.buffer().clone();
        self.owned.push((owned, Keep::Scope));
        Ok(buffer)
    }
    /// Zero-initialized storage of `count` 4-byte words.
    pub(super) fn words(&mut self, label: &'static str, count: u64) -> Result<wgpu::Buffer> {
        self.allocate(&Spec {
            label,
            bytes: mul(count, 4)?,
            usage: Usage::Storage,
        })
    }
    pub(super) const fn mark(&self) -> usize {
        self.owned.len()
    }
    /// Let `buffer` outlive the innermost scope; its enclosing scope owns it.
    pub(super) fn promote(&mut self, buffer: &wgpu::Buffer) {
        self.keep(buffer, Keep::Promoted);
    }
    /// Let `buffer` live as long as the store.
    pub(super) fn retain(&mut self, buffer: &wgpu::Buffer) {
        self.keep(buffer, Keep::Session);
    }
    fn keep(&mut self, buffer: &wgpu::Buffer, keep: Keep) {
        for (owned, kept) in &mut self.owned {
            if owned.buffer() == buffer {
                *kept = keep;
            }
        }
    }
    /// Destroy and release every unpromoted buffer allocated since `mark`.
    pub(super) fn release(&mut self, mark: usize) -> Result<()> {
        let mut result = Ok(());
        let mut kept = Vec::new();
        while self.owned.len() > mark {
            match self.owned.pop() {
                Some((owned, Keep::Promoted)) => kept.push((owned, Keep::Scope)),
                Some((owned, Keep::Session)) => kept.push((owned, Keep::Session)),
                Some((owned, Keep::Scope)) => {
                    let freed = self.pool.free(owned);
                    result = result.and(freed);
                }
                None => {}
            }
        }
        kept.reverse();
        self.owned.append(&mut kept);
        result
    }
    pub(super) const fn allocated(&self) -> u64 {
        self.pool.allocated()
    }
    pub(super) const fn peak(&self) -> u64 {
        self.pool.peak()
    }
    /// Bytes still available under the limit.
    pub(super) const fn available(&self) -> u64 {
        self.pool.available()
    }
}

/// A kernel entry point with its worst-case work units per invocation; zero
/// when the bound depends on registers or data and the caller must supply it.
pub(super) type Entry = (&'static str, u64);

pub(super) struct Module {
    label: &'static str,
    pipelines: Vec<(Entry, wgpu::ComputePipeline)>,
}
impl Module {
    fn pipeline(&self, entry: &str) -> Result<(u64, &wgpu::ComputePipeline)> {
        self.pipelines
            .iter()
            .find(|((name, _), _)| *name == entry)
            .map(|((_, units), pipeline)| (*units, pipeline))
            .ok_or(Error::Fault(Fault::Kernel(self.label)))
    }
}

/// Worst-case work of one dispatch.
#[derive(Clone, Copy)]
pub(super) enum Bound {
    /// The kernel's fixed units per invocation.
    Fixed,
    /// Units per invocation proven by the caller from registers or counts.
    Each(u64),
    /// Units per invocation and a tighter total proven by the caller.
    Total { each: u64, total: u64 },
}

/// Elements of one dispatch slice at `units` per element under the target,
/// leaving room for the idle padding of the last workgroup.
pub(super) fn slice_size(units: u64) -> u64 {
    WORK_TARGET
        .saturating_sub(WORKGROUP)
        .checked_div(units.saturating_add(1))
        .unwrap_or(1)
        .max(1)
}
/// Largest step budget each of `invocations` invocations can have under the
/// target when it spends `extra` units besides its steps.
pub(super) fn step_budget(invocations: u64, extra: u64) -> u64 {
    WORK_TARGET
        .saturating_sub(WORKGROUP)
        .checked_div(invocations.max(1))
        .unwrap_or(1)
        .saturating_sub(extra.saturating_add(1))
        .max(1)
}

pub(super) type Bind<'a> = (u32, &'a wgpu::Buffer);

pub(super) struct Runner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    uniforms: wgpu::Buffer,
    status: wgpu::Buffer,
    staging: wgpu::Buffer,
    dummies: Vec<wgpu::Buffer>,
    stride: u64,
    slots: u64,
    slot: u64,
    pending: u64,
    encoder: Option<wgpu::CommandEncoder>,
    pub(super) work: Work,
    pub(super) readback_bytes: u64,
    pub(super) host: Host,
}

impl Runner {
    pub(super) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        store: &mut Store,
        host: Host,
    ) -> Result<Self> {
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: NonZeroU64::new(REGISTER_BYTES),
            },
            count: None,
        }];
        entries.extend((1..=8).map(|binding| storage(binding, binding <= 2)));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ambiguity f32"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ambiguity f32"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let stride = REGISTER_BYTES
            .checked_next_multiple_of(u64::from(
                device.limits().min_uniform_buffer_offset_alignment,
            ))
            .ok_or_else(overflow)?;
        let uniforms = store.allocate(&Spec {
            label: "ambiguity f32 registers",
            bytes: RING_BYTES,
            usage: Usage::Uniform,
        })?;
        let status = store.words("ambiguity f32 status", STATUS_BYTES / 4)?;
        let staging = store.allocate(&Spec {
            label: "ambiguity f32 staging",
            bytes: STAGING_BYTES,
            usage: Usage::Read,
        })?;
        let mut dummies = Vec::new();
        for _ in 0..STORAGE_BINDINGS {
            dummies.push(store.words("ambiguity f32 unbound", DUMMY_BYTES / 4)?);
        }
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            layout,
            pipeline_layout,
            uniforms,
            status,
            staging,
            dummies,
            stride,
            slots: RING_BYTES.checked_div(stride).unwrap_or(1),
            slot: 0,
            pending: 0,
            encoder: None,
            work: Work::default(),
            readback_bytes: 0,
            host,
        })
    }

    /// Compile each `(label, source, entries)` after the shared prelude, in
    /// order, into one table with one pipeline per entry. Entry names must be
    /// unique across all sources.
    pub(super) fn kernels(
        &self,
        label: &'static str,
        sources: &[(&'static str, &str, &[Entry])],
    ) -> Result<Module> {
        let mut pipelines: Vec<(Entry, wgpu::ComputePipeline)> = Vec::new();
        for (module_label, source, entries) in sources {
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(module_label),
                    source: wgpu::ShaderSource::Wgsl(
                        format!(
                            "{}\n{source}",
                            include_str!("../../shaders/ambiguity_f32_prelude.wgsl")
                        )
                        .into(),
                    ),
                });
            for entry in *entries {
                let repeated = pipelines.iter().any(|((name, _), _)| *name == entry.0);
                if repeated {
                    return Err(Error::Fault(Fault::Schedule(
                        "kernel entry name repeats across modules",
                    )));
                }
                let pipeline =
                    self.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some(entry.0),
                            layout: Some(&self.pipeline_layout),
                            module: &module,
                            entry_point: Some(entry.0),
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                            cache: None,
                        });
                pipelines.push((*entry, pipeline));
            }
        }
        Ok(Module { label, pipelines })
    }

    /// Per-buffer storage capacity, including the allocator's 16-byte padding.
    pub(super) fn storage_buffer_bytes(&self) -> u64 {
        let limits = self.device.limits();
        limits
            .max_buffer_size
            .min(u64::from(limits.max_storage_buffer_binding_size))
            & !15
    }

    fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        self.encoder.get_or_insert_with(|| {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
        })
    }

    /// Encode `workgroups` workgroups of `entry`. Unlisted bindings receive
    /// placeholders the kernel must not touch.
    ///
    /// A dispatch with a uniform per-invocation bound is split into slices of
    /// whole workgroups under the work target, each a single row of
    /// workgroups, so the slices partition the invocation range exactly; the
    /// kernel sees its slice through the `o` register. A dispatch with a
    /// caller-proven total is encoded whole and refused when that total
    /// exceeds the target. Pending work is submitted first when a submission
    /// would exceed its target.
    pub(super) fn run_groups(
        &mut self,
        module: &Module,
        entry: &str,
        binds: &[Bind<'_>],
        registers: P,
        workgroups: u64,
        bound: Bound,
    ) -> Result<()> {
        let (fixed, pipeline) = module.pipeline(entry)?;
        let (each, total) = match bound {
            Bound::Fixed => (fixed, None),
            Bound::Each(each) => (each, None),
            Bound::Total { each, total } => (each, Some(total)),
        };
        if each == 0 || each > INVOCATION_TARGET {
            return Err(Error::Fault(Fault::Schedule(
                "kernel invocation work is unbounded or above the invocation target",
            )));
        }
        let maximum = u64::from(self.device.limits().max_compute_workgroups_per_dimension);
        if let Some(total) = total {
            let rows = workgroups.div_ceil(maximum);
            if rows > maximum {
                return Err(Error::Limit {
                    limit: Limit::Dispatch,
                    demand: workgroups,
                    available: maximum.saturating_mul(maximum),
                });
            }
            let shape = (workgroups.div_ceil(rows.max(1)), rows);
            // Include every padding lane in the actual two-dimensional shape.
            let work = add(total, mul(mul(shape.0, shape.1)?, WORKGROUP)?)?;
            if work > WORK_TARGET {
                return Err(Error::Fault(Fault::Schedule(
                    "dispatch work exceeds the per-dispatch target",
                )));
            }
            let registers = P { o: 0, ..registers };
            return self.encode(pipeline, entry, binds, registers, shape, work);
        }
        // Every lane of a slice is charged the full bound.
        let lane = mul(WORKGROUP, each)?;
        let slice = WORK_TARGET.checked_div(lane).unwrap_or(1).clamp(1, maximum);
        for (first, count) in spans(0, workgroups, slice) {
            let sliced = P {
                o: word(first)?,
                ..registers
            };
            self.encode(
                pipeline,
                entry,
                binds,
                sliced,
                (count, 1),
                mul(count, lane)?,
            )?;
        }
        Ok(())
    }

    /// Encode one dispatch of `columns` x `rows` workgroups charged `work`.
    fn encode(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        entry: &str,
        binds: &[Bind<'_>],
        registers: P,
        (columns, rows): (u64, u64),
        work: u64,
    ) -> Result<()> {
        if columns == 0 || rows == 0 {
            return Ok(());
        }
        if self.slot >= self.slots || add(self.pending, work)? > WORK_TARGET {
            self.flush()?;
        }
        tracing::trace!(entry, columns, rows, work, "gpu-f32 dispatch");
        let offset = mul(self.slot, self.stride)?;
        self.queue
            .write_buffer(&self.uniforms, offset, bytemuck::bytes_of(&registers));
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &self.uniforms,
                offset: 0,
                size: NonZeroU64::new(REGISTER_BYTES),
            }),
        }];
        for (binding, dummy) in (1..=STORAGE_BINDINGS).zip(&self.dummies) {
            let buffer = binds
                .iter()
                .find(|(index, _)| *index == binding)
                .map_or(dummy, |(_, buffer)| *buffer);
            entries.push(wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: 8,
            resource: self.status.as_entire_binding(),
        });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(entry),
            layout: &self.layout,
            entries: &entries,
        });
        let dynamic = [word(offset)?];
        let (columns, rows) = (word(columns)?, word(rows)?);
        let encoder = self.encoder();
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(entry),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &dynamic);
            pass.dispatch_workgroups(columns, rows, 1);
        }
        self.slot = add(self.slot, 1)?;
        self.pending = add(self.pending, work)?;
        self.work.add_dispatch()
    }

    /// Encode one invocation per element of `invocations`.
    pub(super) fn run(
        &mut self,
        module: &Module,
        entry: &str,
        binds: &[Bind<'_>],
        registers: P,
        invocations: u64,
        bound: Bound,
    ) -> Result<()> {
        self.run_groups(
            module,
            entry,
            binds,
            registers,
            invocations.div_ceil(WORKGROUP),
            bound,
        )
    }

    pub(super) fn clear(&mut self, buffer: &wgpu::Buffer) {
        self.encoder().clear_buffer(buffer, 0, None);
    }
    /// Zero `count` words of `buffer` from word `first`, in dispatch order.
    pub(super) fn clear_words(
        &mut self,
        buffer: &wgpu::Buffer,
        first: u64,
        count: u64,
    ) -> Result<()> {
        let (offset, size) = (mul(first, 4)?, mul(count, 4)?);
        self.encoder().clear_buffer(buffer, offset, Some(size));
        Ok(())
    }
    pub(super) fn copy(&mut self, source: &wgpu::Buffer, target: &wgpu::Buffer, bytes: u64) {
        self.encoder()
            .copy_buffer_to_buffer(source, 0, target, 0, bytes);
    }
    pub(super) fn write(&self, buffer: &wgpu::Buffer, bytes: &[u8]) {
        self.queue.write_buffer(buffer, 0, bytes);
    }
    /// Zero the counters in dispatch order, preserving the sticky error word.
    pub(super) fn reset(&mut self) {
        let status = self.status.clone();
        self.encoder()
            .clear_buffer(&status, 4, Some(STATUS_BYTES - 4));
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

    /// Drop everything encoded but not submitted (after a failure).
    pub(super) fn abandon(&mut self) {
        self.encoder = None;
        self.slot = 0;
        self.pending = 0;
    }

    /// Submit everything encoded so far and wait for it.
    pub(super) fn flush(&mut self) -> Result<()> {
        self.slot = 0;
        self.pending = 0;
        let Some(encoder) = self.encoder.take() else {
            return Ok(());
        };
        let started = Instant::now();
        self.queue.submit([encoder.finish()]);
        self.wait()?;
        tracing::trace!(
            seconds = started.elapsed().as_secs_f64(),
            "gpu-f32 submission"
        );
        self.work.record_submission(started.elapsed())
    }

    /// Append `bytes` of `source` to `values`, whose capacity is reserved.
    fn read_chunk(
        &mut self,
        source: &wgpu::Buffer,
        (offset, bytes): (u64, u64),
        values: &mut Vec<u32>,
    ) -> Result<()> {
        let staging = self.staging.clone();
        self.encoder()
            .copy_buffer_to_buffer(source, offset, &staging, 0, bytes);
        self.slot = 0;
        self.pending = 0;
        let encoder = self
            .encoder
            .take()
            .ok_or(Error::Protocol(Protocol::Inventory))?;
        let started = Instant::now();
        self.queue.submit([encoder.finish()]);
        let slice = staging.slice(..bytes);
        let (send, receive) = mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
        self.wait()?;
        receive
            .try_recv()
            .map_err(|error| Error::Map(error.to_string()))?
            .map_err(|error| Error::Map(error.to_string()))?;
        let mapped = super::super::Mapped(&staging);
        let view = slice.get_mapped_range();
        let before = values.len();
        values.extend(super::super::words(&view));
        let read = values.len().saturating_sub(before);
        drop(view);
        drop(mapped);
        tracing::trace!(
            seconds = started.elapsed().as_secs_f64(),
            "gpu-f32 submission with readback"
        );
        self.work.record_submission(started.elapsed())?;
        self.readback_bytes = add(self.readback_bytes, bytes)?;
        if u64::try_from(read).ok() != Some(bytes / 4) {
            return Err(Error::Protocol(Protocol::Readback));
        }
        Ok(())
    }

    /// Submit pending work and read `count` words of `source` from word `first`.
    pub(super) fn read(
        &mut self,
        source: &wgpu::Buffer,
        first: u64,
        count: u64,
    ) -> Result<Vec<u32>> {
        let mut values = self.host.vector(count)?;
        for (done, chunk) in spans(0, count, STAGING_BYTES / 4) {
            let offset = mul(add(first, done)?, 4)?;
            self.read_chunk(source, (offset, mul(chunk, 4)?), &mut values)?;
        }
        Ok(values)
    }

    /// Submit pending work and return the status words; a set error word fails
    /// with the stage name before any dependent kernel is encoded.
    pub(super) fn status(&mut self, stage: &'static str) -> Result<[u32; STATUS_WORDS]> {
        let status = self.status.clone();
        // A short-lived fixed-size read: it leaves the host ledger as it was.
        let used = self.host.used();
        let values = self.read(&status, 0, STATUS_BYTES / 4);
        self.host.restore(used);
        let values = values?;
        let values: [u32; STATUS_WORDS] = values
            .try_into()
            .map_err(|_| Error::Protocol(Protocol::Readback))?;
        let [bits, ..] = values;
        if bits != 0 {
            return Err(Error::Fault(Fault::Status { stage, bits }));
        }
        Ok(values)
    }

    /// Exclusive in-place scan of `count` words of `values` from word `first`.
    pub(super) fn scan(
        &mut self,
        kernels: &Module,
        values: &wgpu::Buffer,
        (first, count): (u64, u64),
        scratch: &wgpu::Buffer,
    ) -> Result<()> {
        self.scan_level(kernels, Some(values), (first, count), 0, scratch)
    }
    fn scan_level(
        &mut self,
        kernels: &Module,
        values: Option<&wgpu::Buffer>,
        (first, count): (u64, u64),
        sums: u64,
        scratch: &wgpu::Buffer,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let blocks = count.div_ceil(SCAN_BLOCK);
        let registers = P {
            n: word(count)?,
            a: word(first)?,
            b: word(sums)?,
            e: u32::from(values.is_none()),
            ..P::default()
        };
        let mut binds = vec![(6, scratch)];
        if let Some(values) = values {
            binds.push((5, values));
        }
        self.run_groups(
            kernels,
            "scan_blocks",
            &binds,
            registers,
            blocks,
            Bound::Fixed,
        )?;
        if blocks > 1 {
            self.scan_level(kernels, None, (sums, blocks), add(sums, blocks)?, scratch)?;
            self.run_groups(kernels, "scan_add", &binds, registers, blocks, Bound::Fixed)?;
        }
        Ok(())
    }

    /// Stable sort of `count` three-word records in `a` by the low `low_bits`
    /// of key word 0 and then the low `high_bits` of key word 1. Returns
    /// whether the result is in `b`.
    pub(super) fn sort(
        &mut self,
        kernels: &Module,
        (a, b): (&wgpu::Buffer, &wgpu::Buffer),
        (histogram, scratch): (&wgpu::Buffer, &wgpu::Buffer),
        count: u64,
        (low_bits, high_bits): (u32, u32),
    ) -> Result<bool> {
        let blocks = count.div_ceil(RADIX_BLOCK);
        let mut flipped = false;
        for (key, key_bits) in [(0, low_bits), (1, high_bits)] {
            let mut shift = 0;
            while shift < key_bits.min(32) {
                let registers = P {
                    n: word(count)?,
                    b: word(blocks)?,
                    c: key,
                    d: shift,
                    e: u32::from(flipped),
                    ..P::default()
                };
                let binds = [(3, a), (4, b), (5, histogram)];
                self.run(
                    kernels,
                    "radix_count",
                    &binds,
                    registers,
                    blocks,
                    Bound::Fixed,
                )?;
                self.scan(kernels, histogram, (0, mul(blocks, 256)?), scratch)?;
                self.run(
                    kernels,
                    "radix_scatter",
                    &binds,
                    registers,
                    blocks,
                    Bound::Fixed,
                )?;
                flipped = !flipped;
                shift = shift.saturating_add(8);
            }
        }
        Ok(flipped)
    }
}
