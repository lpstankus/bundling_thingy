use crate::graph::ResampledGraph;
use color_eyre::eyre::{Result, ensure};
use wgpu::util::DeviceExt;

#[cfg(test)]
mod tests;

pub struct Density {
    counts: wgpu::Buffer,
    maximum: wgpu::Buffer,
    count_pipeline: wgpu::ComputePipeline,
    stages: [Stage; 3],
    pub texture: wgpu::Texture,
}

pub struct DensityInput {
    points: wgpu::Buffer,
    dimensions: [u32; 2],
    workgroups: [u32; 2],
}

struct Stage {
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
}

impl Density {
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
        let buffer = |label, bytes| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let counts = buffer("density counts", bytes);
        let horizontal = buffer("horizontal density blur", bytes);
        let blurred = buffer("blurred density", bytes);
        let maximum = buffer("density maximum", 4);
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
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("density"),
            source: wgpu::ShaderSource::Wgsl(include_str!("density.wgsl").into()),
        });
        let count_pipeline = pipeline(device, &shader, "count_points", size);
        let stages = [
            Stage::new(
                device,
                pipeline(device, &shader, "blur_x", size),
                &[buffer_entry(1, &counts), buffer_entry(2, &horizontal)],
            ),
            Stage::new(
                device,
                pipeline(device, &shader, "blur_y", size),
                &[
                    buffer_entry(2, &horizontal),
                    buffer_entry(3, &blurred),
                    buffer_entry(4, &maximum),
                ],
            ),
            Stage::new(
                device,
                pipeline(device, &shader, "normalize", size),
                &[
                    buffer_entry(3, &blurred),
                    buffer_entry(4, &maximum),
                    wgpu::BindGroupEntry {
                        binding: 5,
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

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: Option<&DensityInput>,
    ) -> Result<()> {
        let size = self.texture.size();
        if let Some(input) = input {
            ensure!(
                input.dimensions == [size.width, size.height],
                "resampled graph dimensions differ from density map"
            );
        }
        encoder.clear_buffer(&self.counts, 0, None);
        encoder.clear_buffer(&self.maximum, 0, None);
        if let Some(input) = input {
            let stage = Stage::new(
                device,
                self.count_pipeline.clone(),
                &[
                    buffer_entry(0, &input.points),
                    buffer_entry(1, &self.counts),
                ],
            );
            let [x, y] = input.workgroups;
            stage.encode(encoder, x, y);
        }

        for stage in &self.stages {
            stage.encode(encoder, size.width.div_ceil(16), size.height.div_ceil(16));
        }
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

    fn encode(&self, encoder: &mut wgpu::CommandEncoder, x: u32, y: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("density"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bindings, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }
}

fn pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    entry: &str,
    size: wgpu::Extent3d,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module: shader,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[
                ("map_width", f64::from(size.width)),
                ("map_height", f64::from(size.height)),
            ],
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
