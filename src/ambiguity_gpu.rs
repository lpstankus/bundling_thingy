//! Support of the GPU-resident f32 ambiguity backend (`gpu-f32`).
//!
//! Error and limit types, borrowed device geometry, submission accounting,
//! device error scopes, mapped staging and the counted device allocations of
//! `resources`.
use std::{fmt, time::Duration};

pub mod resident;
mod resources;

const POLL_TIMEOUT: Duration = Duration::from_secs(60);

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
    /// Contact pairs one gpu-f32 frame may log.
    ContactPairs,
    /// Host metadata of the gpu-f32 backend.
    HostMetadata,
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
    Protocol(Protocol),
    /// A failure of the GPU-resident f32 backend.
    Fault(resident::Fault),
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
            Self::Fault(fault) => fault.fmt(f),
        }
    }
}
impl std::error::Error for Error {}
type Result<T> = std::result::Result<T, Error>;

const fn overflow() -> Error {
    Error::SizeOverflow
}

/// Current geometry already resident on the device, borrowed from the renderer.
/// The backend validates it on the device and it must not be mutated while a
/// frame is evaluated.
#[derive(Clone)]
pub struct Geometry {
    pub points: wgpu::Buffer,
    pub paths: wgpu::Buffer,
    pub samples: u32,
    pub reservations: u32,
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
const fn nearest_f32(value: f64) -> f32 {
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
