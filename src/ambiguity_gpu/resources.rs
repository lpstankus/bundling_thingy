//! Counted device allocations of the gpu-f32 backend.
//!
//! Every buffer comes from one `Pool` under the configured combined byte
//! limit. A buffer is described by a `Spec`. Sizes round up to 16 bytes with a
//! 16-byte minimum and are checked against the adapter's buffer and binding
//! limits before creation. Renderer-owned input geometry is borrowed, never
//! allocated or destroyed here. Driver, pipeline and allocator overhead is
//! outside this byte limit.
use super::{Error, Limit, Result, overflow};

const MINIMUM_BUFFER_BYTES: u64 = 16;

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
}

/// A counted buffer. Dropping it destroys the device storage.
pub(super) struct Owned {
    buffer: wgpu::Buffer,
    bytes: u64,
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
}

/// Counted device allocations against the combined byte limit.
pub(super) struct Pool {
    device: wgpu::Device,
    limit: u64,
    allocated: u64,
    peak: u64,
}

impl Pool {
    pub(super) const fn new(device: wgpu::Device, limit: u64) -> Self {
        Self {
            device,
            limit,
            allocated: 0,
            peak: 0,
        }
    }
    pub(super) const fn allocated(&self) -> u64 {
        self.allocated
    }
    pub(super) const fn peak(&self) -> u64 {
        self.peak
    }
    /// Allocated size of `spec`, checked against the adapter limits.
    fn size(&self, spec: &Spec) -> Result<u64> {
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
    /// Device bytes still available under the limit.
    pub(super) const fn available(&self) -> u64 {
        self.limit.saturating_sub(self.allocated)
    }
    /// Create `spec` if it fits the adapter and the combined limit.
    pub(super) fn allocate(&mut self, spec: &Spec) -> Result<Owned> {
        let size = self.size(spec)?;
        let total = self.allocated.checked_add(size).ok_or_else(overflow)?;
        if total > self.limit {
            return Err(Error::Limit {
                limit: Limit::DeviceMemory(spec.label),
                demand: total,
                available: self.limit,
            });
        }
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(spec.label),
            size,
            usage: spec.usage.flags(),
            mapped_at_creation: false,
        });
        self.allocated = total;
        self.peak = self.peak.max(total);
        Ok(Owned {
            buffer,
            bytes: size,
        })
    }
    /// Destroy `owned`, then release its bytes.
    pub(super) fn free(&mut self, owned: Owned) -> Result<()> {
        let bytes = owned.bytes;
        drop(owned);
        self.allocated = self
            .allocated
            .checked_sub(bytes)
            .ok_or(Error::Protocol(super::Protocol::Inventory))?;
        Ok(())
    }
}
