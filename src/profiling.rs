use color_eyre::eyre::Result;
use std::io::{self, IsTerminal};
use tracing_subscriber::fmt::{format::FmtSpan, time::Uptime};

pub fn init() -> Result<()> {
    // Keep stage timings enabled on every run, independent of RUST_LOG.
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_span_events(FmtSpan::CLOSE)
        .with_timer(Uptime::default())
        .with_target(false)
        .with_ansi(io::stderr().is_terminal())
        .with_writer(io::stderr)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;
    Ok(())
}

/// Pass timestamps measure GPU execution. Tracing spans include CPU work and waits.
pub struct GpuTimings {
    stages: &'static [&'static str],
    timestamps: Option<Timestamps>,
}

struct Timestamps {
    queries: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    readback: wgpu::Buffer,
    count: u32,
    period: f32,
}

impl GpuTimings {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        stages: &'static [&'static str],
    ) -> Result<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return Ok(Self {
                stages,
                timestamps: None,
            });
        }

        let count = u32::try_from(stages.len())?.saturating_mul(2);
        let bytes = u64::from(count).saturating_mul(u64::from(wgpu::QUERY_SIZE));
        let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("pipeline timings"),
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        let resolved = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resolved pipeline timings"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pipeline timing readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Ok(Self {
            stages,
            timestamps: Some(Timestamps {
                queries,
                resolved,
                readback,
                count,
                period: queue.get_timestamp_period(),
            }),
        })
    }

    pub const fn compute_pass(&self, stage: u32) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let Some(timestamps) = &self.timestamps else {
            return None;
        };
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &timestamps.queries,
            beginning_of_pass_write_index: Some(stage.saturating_mul(2)),
            end_of_pass_write_index: Some(stage.saturating_mul(2).saturating_add(1)),
        })
    }

    pub const fn render_pass(&self) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let Some(timestamps) = &self.timestamps else {
            return None;
        };
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &timestamps.queries,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        })
    }

    /// Resolve and map with the existing submission, without another GPU wait.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let Some(timestamps) = &self.timestamps else {
            Self::log_unavailable(self.stages);
            return;
        };
        encoder.resolve_query_set(
            &timestamps.queries,
            0..timestamps.count,
            &timestamps.resolved,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &timestamps.resolved,
            0,
            &timestamps.readback,
            0,
            timestamps.readback.size(),
        );

        let readback = timestamps.readback.clone();
        let stages = self.stages;
        let period = timestamps.period;
        encoder.map_buffer_on_submit(
            &timestamps.readback,
            wgpu::MapMode::Read,
            ..,
            move |result| {
                if let Err(error) = result {
                    tracing::warn!(
                        %error,
                        "GPU timestamp readback failed. GPU durations default to zero."
                    );
                    Self::log_unavailable(stages);
                    return;
                }
                {
                    let data = readback.get_mapped_range(..);
                    for (timestamps, stage) in data.as_chunks::<16>().0.iter().zip(stages) {
                        let [start, end] = bytemuck::pod_read_unaligned::<[u64; 2]>(timestamps);
                        let elapsed = std::time::Duration::from_nanos(end.wrapping_sub(start))
                            .mul_f64(f64::from(period));
                        tracing::info!(
                            stage,
                            gpu_ms = elapsed.as_secs_f64() * 1000.0,
                            "GPU timing"
                        );
                    }
                }
                readback.unmap();
            },
        );
    }

    fn log_unavailable(stages: &[&str]) {
        for stage in stages {
            tracing::info!(stage, gpu_ms = 0.0, "GPU timing");
        }
    }
}
