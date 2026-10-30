use crate::{graph::Graph, profiling::GpuTimings, shader};
use color_eyre::eyre::{OptionExt, Result, ensure};
use wgpu::util::DeviceExt;

const TIMING_STAGES: [&str; 3] = [
    "graph.count",
    "graph.emit_and_density",
    "graph.deduplicate_endpoints",
];
pub const PATH_DRAW_OFFSET: u64 = 16;
pub const ENDPOINT_DRAW_OFFSET: u64 = 32;

const MIB: u64 = 1024 * 1024;
const WORKGROUP_SIZE: u32 = 8;
const STATE_BYTES: u64 = 48;
const STATISTICS_BYTES: u64 = 16;

pub struct Resampler {
    layout: wgpu::BindGroupLayout,
    count: wgpu::ComputePipeline,
    emit: wgpu::ComputePipeline,
    deduplicate: Option<wgpu::ComputePipeline>,
    budget: u64,
    output: Option<OutputBuffers>,
}

struct OutputBuffers {
    points: wgpu::Buffer,
    segments: wgpu::Buffer,
    capacity: u32,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Parameters {
    minimum: [f32; 2],
    scale: f32,
    path_count: u32,
    translation: [f32; 2],
    capacity: u32,
    padding: u32,
}

pub struct Geometry {
    pub points: wgpu::Buffer,
    pub segments: wgpu::Buffer,
    pub endpoints: wgpu::Buffer,
    pub commands: wgpu::Buffer,
}

pub struct PendingGraph {
    geometry: Geometry,
    statistics: shader::Readback<[u32; 4]>,
    capacity: u32,
}

impl Resampler {
    pub fn new(
        device: &wgpu::Device,
        budget_mib: u64,
        deduplicate_endpoints: bool,
    ) -> Result<Self> {
        let budget = budget_mib
            .checked_mul(MIB)
            .ok_or_eyre("graph memory budget is too large")?;
        ensure!(budget > 0, "graph memory budget must be nonzero");
        let entries = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if binding == 4 || binding == 9 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage {
                        read_only: binding == 1,
                    }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GPU resampling"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GPU resampling"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = shader::module(
            device,
            "GPU resampling",
            include_str!("shaders/resampling.wgsl"),
        );
        let constants = [("block_size", f64::from(WORKGROUP_SIZE))];
        let [count, emit] = ["count_paths", "emit_paths"].map(|entry| {
            shader::compute(device, &module, entry, Some(&pipeline_layout), &constants)
        });
        let deduplicate = if deduplicate_endpoints {
            let module =
                shader::module(device, "endpoints", include_str!("shaders/endpoints.wgsl"));
            Some(shader::compute(
                device,
                &module,
                "deduplicate",
                Some(&pipeline_layout),
                &constants,
            ))
        } else {
            None
        };
        Ok(Self {
            layout,
            count,
            emit,
            deduplicate,
            budget,
            output: None,
        })
    }

    pub const fn timing_stages(&self) -> &'static [&'static str] {
        if self.deduplicate.is_some() {
            &TIMING_STAGES
        } else {
            TIMING_STAGES.split_at(2).0
        }
    }

    #[tracing::instrument(name = "graph.prepare_gpu", skip_all, err)]
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        graph: &Graph,
        counts: &wgpu::Buffer,
        map: &wgpu::Buffer,
        timings: &GpuTimings,
    ) -> Result<PendingGraph> {
        let allocation = Allocation::new(device, graph, self.budget)?;
        let capacity = allocation.parameters.capacity;

        if self
            .output
            .as_ref()
            .is_some_and(|output| output.capacity != capacity)
        {
            // Release the old allocation before reserving another graph's capacity.
            self.output = None;
        }
        let output = self.output.get_or_insert_with(|| OutputBuffers {
            points: shader::storage(
                device,
                "GPU resampled points",
                u64::from(capacity).saturating_mul(8),
            ),
            segments: shader::storage(
                device,
                "GPU segment indices",
                u64::from(capacity).saturating_mul(4),
            ),
            capacity,
        });
        let (bindings, geometry, source) = bind_graph(
            device,
            &self.layout,
            graph,
            counts,
            output,
            &allocation,
            map,
        );
        for ((pipeline, label), stage) in [&self.count, &self.emit]
            .into_iter()
            .chain(self.deduplicate.iter())
            .zip(self.timing_stages())
            .zip(0..)
        {
            if stage == 2 {
                // Control-point reads are finished. Reuse their allocation as the hash table.
                encoder.clear_buffer(&source, 0, None);
                encoder.clear_buffer(&geometry.commands, 8, Some(4));
                encoder.clear_buffer(
                    &geometry.commands,
                    ENDPOINT_DRAW_OFFSET.saturating_add(4),
                    Some(4),
                );
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: timings.compute_pass(stage),
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(allocation.workgroups[0], allocation.workgroups[1], 1);
        }
        timings.resolve(encoder);

        let statistics = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU graph statistics"),
            size: STATISTICS_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&geometry.commands, 0, &statistics, 0, STATISTICS_BYTES);
        let statistics = shader::readback(encoder, &statistics, |result| {
            result.map(bytemuck::pod_read_unaligned::<[u32; 4]>)
        });
        tracing::info!(
            capacity,
            reserved_bytes = allocation.reserved_bytes,
            budget_bytes = self.budget,
            "GPU graph buffers"
        );
        Ok(PendingGraph {
            geometry,
            statistics,
            capacity,
        })
    }
}

impl PendingGraph {
    /// Read only the counters after the submission's existing GPU wait.
    pub fn finish(self) -> Result<(Geometry, u32)> {
        let [points, segments, endpoints, overflow] = self.statistics.recv()??;
        ensure!(
            overflow != 2,
            "GPU resampling count and write passes disagree"
        );
        ensure!(
            overflow == 0,
            "GPU resampling exceeds the {}-sample buffer capacity; increase --max-graph-memory-mib or reduce the graph or render dimensions. Device buffer limits also apply.",
            self.capacity
        );
        ensure!(segments > 0, "resampled graph has no drawable edges");
        tracing::info!(points, segments, endpoints, "GPU resampled graph");
        Ok((self.geometry, points))
    }
}

#[derive(Clone, Copy)]
struct Allocation {
    parameters: Parameters,
    path_bytes: u64,
    endpoint_bytes: u64,
    workgroups: [u32; 2],
    reserved_bytes: u64,
}

impl Allocation {
    fn new(device: &wgpu::Device, graph: &Graph, budget: u64) -> Result<Self> {
        let normalization = graph.normalization()?;
        let path_count = u32::try_from(graph.starts().len().saturating_sub(1))?;
        let point_bytes = u64::try_from(std::mem::size_of_val(graph.points()))?;
        let start_bytes = u64::try_from(std::mem::size_of_val(graph.starts()))?;
        let path_bytes = u64::from(path_count).saturating_mul(16);
        let endpoint_bytes = u64::from(path_count).saturating_mul(8);
        let fixed_bytes = point_bytes
            .saturating_add(start_bytes)
            .saturating_add(path_bytes)
            .saturating_add(endpoint_bytes)
            .saturating_add(u64::try_from(std::mem::size_of::<Parameters>())?)
            .saturating_add(shader::MAP_UNIFORM_BYTES)
            .saturating_add(STATE_BYTES)
            .saturating_add(STATISTICS_BYTES);
        let available = budget
            .checked_sub(fixed_bytes)
            .ok_or_eyre("control points and path metadata exceed --max-graph-memory-mib")?;
        let limits = device.limits();
        let binding_limit = limits
            .max_buffer_size
            .min(u64::from(limits.max_storage_buffer_binding_size));
        for (name, bytes) in [
            ("control points", point_bytes),
            ("path starts", start_bytes),
            ("path metadata", path_bytes),
            ("endpoints", endpoint_bytes),
        ] {
            ensure!(bytes <= binding_limit, "{name} exceed GPU buffer limits");
        }
        // Every sample needs an eight-byte point and at most one four-byte segment index.
        let capacity = u32::try_from(available.saturating_div(12).min(binding_limit / 8))?;
        ensure!(
            capacity >= 3,
            "graph memory budget has no room for resampled geometry"
        );
        let groups = path_count.div_ceil(WORKGROUP_SIZE);
        let y = groups.div_ceil(limits.max_compute_workgroups_per_dimension);
        ensure!(
            y <= limits.max_compute_workgroups_per_dimension,
            "paths exceed GPU dispatch limits"
        );
        let x = groups.div_ceil(y);

        Ok(Self {
            parameters: Parameters {
                minimum: normalization.minimum.to_array(),
                scale: normalization.scale,
                path_count,
                translation: normalization.translation.to_array(),
                capacity,
                padding: 0,
            },
            path_bytes,
            endpoint_bytes,
            workgroups: [x, y],
            reserved_bytes: fixed_bytes.saturating_add(u64::from(capacity).saturating_mul(12)),
        })
    }
}

fn bind_graph(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    graph: &Graph,
    counts: &wgpu::Buffer,
    output: &OutputBuffers,
    allocation: &Allocation,
    map: &wgpu::Buffer,
) -> (wgpu::BindGroup, Geometry, wgpu::Buffer) {
    let source = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("graph control points"),
        contents: bytemuck::cast_slice(graph.points()),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let starts = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("graph path starts"),
        contents: bytemuck::cast_slice(graph.starts()),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let paths = shader::storage(device, "GPU path reservations", allocation.path_bytes);
    let endpoints = shader::storage(device, "GPU endpoint indices", allocation.endpoint_bytes);
    let commands = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GPU graph counts and draw commands"),
        size: STATE_BYTES,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let parameters = shader::uniform(
        device,
        "GPU graph normalization and capacity",
        bytemuck::bytes_of(&allocation.parameters),
    );
    let bindings = shader::bindings(
        device,
        layout,
        [
            (0, source.as_entire_binding()),
            (1, starts.as_entire_binding()),
            (2, paths.as_entire_binding()),
            (3, commands.as_entire_binding()),
            (4, parameters.as_entire_binding()),
            (5, output.points.as_entire_binding()),
            (6, output.segments.as_entire_binding()),
            (7, endpoints.as_entire_binding()),
            (8, counts.as_entire_binding()),
            (9, map.as_entire_binding()),
        ],
    );
    (
        bindings,
        Geometry {
            points: output.points.clone(),
            segments: output.segments.clone(),
            endpoints,
            commands,
        },
        source,
    )
}
