use crate::{palette::ColorBy, profiling::GpuTimings, resampling::Geometry, shader};
use color_eyre::eyre::{Result, ensure};

const WORKGROUP_SIZE: u32 = 128;
const TIMING_STAGES: [&str; 7] = [
    "graph.grid_count",
    "graph.grid_reserve",
    "graph.grid_scatter",
    "graph.grid_dispatch",
    "graph.grid_summarize",
    "graph.neighbor_mean",
    "graph.neighbor_mean_sparse",
];
// Output points, indexed positions, and at most one query task per sample.
pub const EXTRA_SAMPLE_BYTES: u64 = 32;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct GridParameters {
    size: [u32; 2],
    radius: u32,
    dispatch_limit: u32,
}

impl GridParameters {
    fn new(size: [u32; 2], radius: u32) -> Self {
        Self {
            size: size.map(|dimension| dimension.div_ceil(radius)),
            radius,
            dispatch_limit: 0,
        }
    }

    fn cells(&self) -> u64 {
        let [width, height] = self.size;
        u64::from(width).saturating_mul(u64::from(height))
    }

    fn buffer_bytes(&self) -> u64 {
        self.cells().saturating_mul(8).saturating_add(8)
    }

    fn summary_bytes(&self) -> u64 {
        self.cells().saturating_mul(24)
    }
}

struct Scratch {
    points: wgpu::Buffer,
    positions: wgpu::Buffer,
    tasks: wgpu::Buffer,
}

struct GridBuffers {
    parameters: wgpu::Buffer,
    cells: wgpu::Buffer,
    summaries: wgpu::Buffer,
    cell_count: u32,
    radius: u32,
}

impl GridBuffers {
    fn new(device: &wgpu::Device, size: [u32; 2], radius: u32) -> Result<Self> {
        let mut parameters = GridParameters::new(size, radius);
        let bytes = parameters.buffer_bytes();
        let summary_bytes = parameters.summary_bytes();
        let limits = device.limits();
        parameters.dispatch_limit = limits.max_compute_workgroups_per_dimension;
        ensure!(
            bytes.max(summary_bytes) <= limits.max_buffer_size
                && bytes.max(summary_bytes) <= u64::from(limits.max_storage_buffer_binding_size),
            "neighbor grid exceeds GPU buffer limits; increase --neighbor-radius or reduce render dimensions"
        );
        Ok(Self {
            parameters: shader::uniform(
                device,
                "neighbor grid dimensions and radius",
                bytemuck::bytes_of(&parameters),
            ),
            cells: shader::storage(device, "neighbor grid cells", bytes),
            summaries: shader::storage(device, "neighbor cell summaries", summary_bytes),
            cell_count: u32::try_from(parameters.cells())?,
            radius,
        })
    }
}

pub struct NeighborMean {
    pipelines: [wgpu::ComputePipeline; 7],
    dispatch: wgpu::Buffer,
    grid: GridBuffers,
    map: wgpu::Buffer,
    timings: GpuTimings,
    scratch: Option<Scratch>,
}

impl NeighborMean {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        radius: u32,
        size: [u32; 2],
        color_by: ColorBy,
        map: &wgpu::Buffer,
    ) -> Result<Self> {
        let grid = GridBuffers::new(device, size, radius)?;
        let entries = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if binding == 2 || binding == 9 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage {
                        read_only: binding == 0 || binding == 4,
                    }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("neighbor grid and mean"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("neighbor grid and mean"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = shader::module(
            device,
            "neighbor mean",
            &format!(
                "{}\n{}",
                include_str!("shaders/spatial_grid.wgsl"),
                include_str!("shaders/neighbor_mean.wgsl"),
            ),
        );
        let constants = [
            ("block_size", f64::from(WORKGROUP_SIZE)),
            (
                "accumulate_density",
                f64::from(u8::from(color_by == ColorBy::Density)),
            ),
        ];
        Ok(Self {
            pipelines: [
                "count_cells",
                "reserve_cells",
                "scatter_points",
                "prepare_dispatch",
                "summarize_cells",
                "neighbor_mean",
                "neighbor_mean_sparse",
            ]
            .map(|entry| {
                // The argument writer uses its own small layout. Indirect passes never bind it.
                let layout = (entry != "prepare_dispatch").then_some(&pipeline_layout);
                shader::compute(device, &module, entry, layout, &constants)
            }),
            dispatch: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("neighbor indirect arguments"),
                size: 36,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            }),
            grid,
            map: map.clone(),
            timings: GpuTimings::new(device, queue, &TIMING_STAGES)?,
            scratch: None,
        })
    }

    pub fn grid_bytes(size: [u32; 2], radius: u32) -> u64 {
        // Include the grid uniform and three indirect dispatches.
        let parameters = GridParameters::new(size, radius);
        parameters
            .buffer_bytes()
            .saturating_add(parameters.summary_bytes())
            .saturating_add(52)
    }

    pub const fn radius(&self) -> u32 {
        self.grid.radius
    }

    pub fn resize(&mut self, device: &wgpu::Device, size: [u32; 2]) -> Result<()> {
        self.grid = GridBuffers::new(device, size, self.grid.radius)?;
        self.reset();
        Ok(())
    }

    pub fn reset(&mut self) {
        // The resampler can reuse the first input allocation for a different graph.
        self.scratch = None;
    }

    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        geometry: &mut Geometry,
        samples: u32,
        counts: &wgpu::Buffer,
    ) -> Result<()> {
        let sample_groups = workgroups(device, samples.div_ceil(WORKGROUP_SIZE))?;
        let cell_groups = workgroups(device, self.grid.cell_count.div_ceil(WORKGROUP_SIZE))?;
        let tasks = samples.min(
            self.grid
                .cell_count
                .saturating_add(samples.div_ceil(WORKGROUP_SIZE)),
        );
        // Bound the indirect dispatch without reading the task count back to the CPU.
        workgroups(device, tasks)?;
        let bytes = u64::from(samples) * 12;
        let mut scratch = self.scratch.take().unwrap_or_else(|| Scratch {
            points: shader::storage(device, "neighbor mean points", bytes),
            positions: shader::storage(device, "neighbor indexed positions", bytes),
            tasks: shader::storage(device, "neighbor query tasks", u64::from(tasks) * 8),
        });
        let input = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &geometry.points,
            offset: 0,
            size: wgpu::BufferSize::new(bytes),
        });
        let bindings = shader::bindings(
            device,
            &self.pipelines[0].get_bind_group_layout(0),
            [
                (0, input),
                (1, scratch.points.as_entire_binding()),
                (2, self.grid.parameters.as_entire_binding()),
                (3, counts.as_entire_binding()),
                (4, geometry.paths.as_entire_binding()),
                (5, self.grid.cells.as_entire_binding()),
                (6, scratch.positions.as_entire_binding()),
                (7, scratch.tasks.as_entire_binding()),
                (8, self.grid.summaries.as_entire_binding()),
                (9, self.map.as_entire_binding()),
            ],
        );
        let dispatch_grid = shader::bindings(
            device,
            &self.pipelines[3].get_bind_group_layout(0),
            [
                (2, self.grid.parameters.as_entire_binding()),
                (5, self.grid.cells.as_entire_binding()),
            ],
        );
        let dispatch_binding = shader::bindings(
            device,
            &self.pipelines[3].get_bind_group_layout(1),
            [(0, self.dispatch.as_entire_binding())],
        );
        encoder.clear_buffer(&self.grid.cells, 0, None);
        for ((pipeline, label), stage) in self.pipelines.iter().zip(TIMING_STAGES).zip(0..) {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: self.timings.compute_pass(stage),
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(
                0,
                if stage == 3 {
                    &dispatch_grid
                } else {
                    &bindings
                },
                &[],
            );
            match stage {
                0 | 2 => pass.dispatch_workgroups(sample_groups[0], sample_groups[1], 1),
                1 => pass.dispatch_workgroups(cell_groups[0], cell_groups[1], 1),
                3 => {
                    pass.set_bind_group(1, &dispatch_binding, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                4 => pass.dispatch_workgroups_indirect(&self.dispatch, 0),
                5 => pass.dispatch_workgroups_indirect(&self.dispatch, 12),
                _ => pass.dispatch_workgroups_indirect(&self.dispatch, 24),
            }
        }
        self.timings.resolve(encoder);
        std::mem::swap(&mut geometry.points, &mut scratch.points);
        self.scratch = Some(scratch);
        Ok(())
    }
}

fn workgroups(device: &wgpu::Device, groups: u32) -> Result<[u32; 2]> {
    let limit = device.limits().max_compute_workgroups_per_dimension;
    let y = groups.div_ceil(limit);
    ensure!(y <= limit, "neighbor grid exceeds GPU dispatch limits");
    Ok([groups.div_ceil(y), y])
}
