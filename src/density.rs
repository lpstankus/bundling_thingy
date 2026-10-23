use crate::{graph::ResampledGraph, profiling::GpuTimings};
use color_eyre::eyre::{Result, ensure};
use wgpu::util::DeviceExt;

pub const TIMING_STAGES: [&str; 6] = [
    "density.count",
    "density.resolve",
    "density.blur_horizontal",
    "density.blur_vertical",
    "density.maximum",
    "density.normalize",
];

pub struct Density {
    counts: wgpu::Buffer,
    maximum: wgpu::Buffer,
    count_pipeline: wgpu::ComputePipeline,
    stages: [Stage; 5],
    pub texture: wgpu::Texture,
}

pub struct DensityInput {
    points: wgpu::Buffer,
    dimensions: [u32; 2],
    workgroups: [u32; 2],
}

pub struct DensityEncoder<'a> {
    device: &'a wgpu::Device,
    commands: &'a mut wgpu::CommandEncoder,
    timings: &'a GpuTimings,
    next_query: u32,
}

struct Stage {
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
}

impl Density {
    #[tracing::instrument(name = "density.init", skip_all, fields(width = width, height = height), err)]
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Result<Self> {
        let limits = device.limits();
        ensure!(
            width > 0 && height > 0,
            "density map dimensions must be nonzero"
        );
        ensure!(
            width <= limits.max_texture_dimension_2d && height <= limits.max_texture_dimension_2d,
            "density map exceeds GPU texture limits"
        );
        let bytes = u64::from(width)
            .saturating_mul(u64::from(height))
            .saturating_mul(4);
        ensure!(
            bytes <= u64::from(limits.max_storage_buffer_binding_size),
            "density map exceeds GPU storage buffer limit"
        );

        let counts = create_buffer(device, "density counts", bytes);
        let image_a = create_buffer(device, "density image A", bytes);
        let image_b = create_buffer(device, "density image B", bytes);
        let maximum = create_buffer(device, "density maximum", 4);
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("normalized density"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let density_source = include_str!("shaders/density.wgsl");
        let normalization_source = include_str!("shaders/normalization.wgsl");
        let count_pipeline = build_pipeline(device, density_source, "count_points", size);
        let stages = [
            Stage::new(
                device,
                build_pipeline(device, density_source, "resolve_counts", size),
                &[buffer_entry(1, &counts), buffer_entry(2, &image_a)],
            ),
            Stage::new(
                device,
                build_gaussian_pipeline(device, size, 0.0),
                &[buffer_entry(0, &image_a), buffer_entry(1, &image_b)],
            ),
            Stage::new(
                device,
                build_gaussian_pipeline(device, size, 1.0),
                &[buffer_entry(0, &image_b), buffer_entry(1, &image_a)],
            ),
            Stage::new(
                device,
                build_pipeline(device, normalization_source, "find_maximum", size),
                &[buffer_entry(0, &image_a), buffer_entry(1, &maximum)],
            ),
            Stage::new(
                device,
                build_pipeline(device, normalization_source, "normalize", size),
                &[
                    buffer_entry(0, &image_a),
                    buffer_entry(1, &maximum),
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                ],
            ),
        ];
        Ok(Self {
            counts,
            maximum,
            count_pipeline,
            stages,
            texture,
        })
    }

    pub fn encode(&self, mut encoder: DensityEncoder<'_>, input: &DensityInput) -> Result<()> {
        let size = self.texture.size();
        ensure!(
            input.dimensions == [size.width, size.height],
            "resampled graph dimensions differ from density map"
        );
        encoder.commands.clear_buffer(&self.counts, 0, None);
        encoder.commands.clear_buffer(&self.maximum, 0, None);
        let stage = Stage::new(
            encoder.device,
            self.count_pipeline.clone(),
            &[
                buffer_entry(0, &input.points),
                buffer_entry(1, &self.counts),
            ],
        );
        encoder.dispatch(&stage, input.workgroups);

        let workgroups = [size.width.div_ceil(32), size.height.div_ceil(4)];
        for stage in &self.stages {
            encoder.dispatch(stage, workgroups);
        }
        encoder.timings.resolve(encoder.commands);
        Ok(())
    }
}

impl DensityInput {
    pub fn new(device: &wgpu::Device, graph: &ResampledGraph) -> Result<Self> {
        let count = u32::try_from(graph.points().len())?;
        let groups = count.div_ceil(256);
        let limit = device.limits().max_compute_workgroups_per_dimension;
        let x = groups.min(limit);
        let y = groups.div_ceil(x);
        ensure!(y <= limit, "density samples exceed GPU dispatch limit");
        let contents = bytemuck::cast_slice(graph.points());
        ensure!(
            u64::try_from(contents.len())?
                <= u64::from(device.limits().max_storage_buffer_binding_size),
            "resampled points exceed GPU storage buffer limit"
        );
        Ok(Self {
            points: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("resampled graph points"),
                contents,
                usage: wgpu::BufferUsages::STORAGE,
            }),
            dimensions: graph.dimensions(),
            workgroups: [x, y],
        })
    }

    pub const fn points(&self) -> &wgpu::Buffer {
        &self.points
    }
}

impl Stage {
    fn new(
        device: &wgpu::Device,
        pipeline: wgpu::ComputePipeline,
        entries: &[wgpu::BindGroupEntry<'_>],
    ) -> Self {
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("density"),
            layout: &pipeline.get_bind_group_layout(0),
            entries,
        });
        Self { pipeline, bindings }
    }
}

impl<'a> DensityEncoder<'a> {
    pub const fn new(
        device: &'a wgpu::Device,
        commands: &'a mut wgpu::CommandEncoder,
        timings: &'a GpuTimings,
    ) -> Self {
        Self {
            device,
            commands,
            timings,
            next_query: 0,
        }
    }

    fn dispatch(&mut self, stage: &Stage, [x, y]: [u32; 2]) {
        let timestamp_writes = self.timings.compute_pass(self.next_query);
        self.next_query = self.next_query.saturating_add(1);
        let mut pass = self
            .commands
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("density"),
                timestamp_writes,
            });
        pass.set_pipeline(&stage.pipeline);
        pass.set_bind_group(0, &stage.bindings, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }
}

fn create_buffer(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn build_pipeline(
    device: &wgpu::Device,
    source: &str,
    entry: &str,
    size: wgpu::Extent3d,
) -> wgpu::ComputePipeline {
    create_pipeline(device, source, entry, size, &[])
}

fn build_gaussian_pipeline(
    device: &wgpu::Device,
    size: wgpu::Extent3d,
    axis: f64,
) -> wgpu::ComputePipeline {
    create_pipeline(
        device,
        include_str!("shaders/blur.wgsl"),
        "blur",
        size,
        &[("axis", axis)],
    )
}

fn create_pipeline(
    device: &wgpu::Device,
    source: &str,
    entry: &str,
    size: wgpu::Extent3d,
    constants: &[(&str, f64)],
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(entry),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let mut overrides = vec![
        ("map_width", f64::from(size.width)),
        ("map_height", f64::from(size.height)),
    ];
    overrides.extend_from_slice(constants);
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module: &shader,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &overrides,
            ..Default::default()
        },
        cache: None,
    })
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
