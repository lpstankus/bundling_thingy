use crate::{profiling::GpuTimings, shader};
use color_eyre::eyre::{Result, ensure};
use wgpu::TextureViewDescriptor;

pub const TIMING_STAGES: [&str; 3] = [
    "density.blur_horizontal",
    "density.blur_vertical_and_maximum",
    "density.normalize",
];

pub struct Density {
    counts: wgpu::Buffer,
    maximum: wgpu::Buffer,
    stages: [Stage; 3],
    blur_directions: [wgpu::Buffer; 2],
    pub texture: wgpu::Texture,
}

struct Stage {
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    workgroups: [u32; 2],
}

impl Density {
    pub fn new(device: &wgpu::Device, size: [u32; 2], map: &wgpu::Buffer) -> Result<Self> {
        let blur = shader::module(device, "blur", include_str!("shaders/blur.wgsl"));
        let normalization = shader::module(
            device,
            "normalization",
            include_str!("shaders/normalization.wgsl"),
        );
        let blur = shader::compute(device, &blur, "blur", None, &[]);
        let blur_directions = [[1_u32, 0, 0, 0], [0, 1, 0, 0]].map(|direction| {
            shader::uniform(device, "blur direction", bytemuck::cast_slice(&direction))
        });
        let pipelines = [
            blur.clone(),
            blur,
            shader::compute(device, &normalization, "normalize", None, &[]),
        ];
        Self::with_pipelines(device, size, pipelines, blur_directions, map)
    }

    pub fn resize(
        &self,
        device: &wgpu::Device,
        size: [u32; 2],
        map: &wgpu::Buffer,
    ) -> Result<Self> {
        let pipelines = self.stages.each_ref().map(|stage| stage.pipeline.clone());
        Self::with_pipelines(device, size, pipelines, self.blur_directions.clone(), map)
    }

    #[tracing::instrument(name = "density.allocate", skip_all, fields(width = size[0], height = size[1]), err)]
    fn with_pipelines(
        device: &wgpu::Device,
        size: [u32; 2],
        pipelines: [wgpu::ComputePipeline; 3],
        blur_directions: [wgpu::Buffer; 2],
        map: &wgpu::Buffer,
    ) -> Result<Self> {
        let [width, height] = size;
        let [horizontal, vertical, normalize] = pipelines;
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

        let counts = shader::storage(device, "density counts", bytes);
        let image = shader::storage(device, "horizontal density", bytes);
        let maximum = shader::storage(device, "density maximum", 4);
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
        let view = texture.create_view(&TextureViewDescriptor::default());

        let blur_groups = [width.div_ceil(32), height.div_ceil(32)];
        let [horizontal, vertical] = [
            (horizontal, &counts, &image, &blur_directions[0]),
            (vertical, &image, &counts, &blur_directions[1]),
        ]
        .map(|(pipeline, input, output, direction)| {
            Stage::new(
                device,
                pipeline,
                blur_groups,
                [
                    (0, input.as_entire_binding()),
                    (1, output.as_entire_binding()),
                    (2, direction.as_entire_binding()),
                    (3, maximum.as_entire_binding()),
                    (9, map.as_entire_binding()),
                ],
            )
        });
        let stages = [
            horizontal,
            vertical,
            Stage::new(
                device,
                normalize,
                [width.div_ceil(32), height.div_ceil(4)],
                [
                    (0, counts.as_entire_binding()),
                    (1, maximum.as_entire_binding()),
                    (2, wgpu::BindingResource::TextureView(&view)),
                    (9, map.as_entire_binding()),
                ],
            ),
        ];
        Ok(Self {
            counts,
            maximum,
            stages,
            blur_directions,
            texture,
        })
    }

    /// Integer counts until encode reuses the buffer for the vertical f32 output.
    pub const fn counts(&self) -> &wgpu::Buffer {
        &self.counts
    }

    pub fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.counts, 0, None);
        encoder.clear_buffer(&self.maximum, 0, None);
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, timings: &GpuTimings) {
        for (stage, query) in self.stages.iter().zip(0..) {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("density"),
                timestamp_writes: timings.compute_pass(query),
            });
            pass.set_pipeline(&stage.pipeline);
            pass.set_bind_group(0, &stage.bindings, &[]);
            pass.dispatch_workgroups(stage.workgroups[0], stage.workgroups[1], 1);
        }
        timings.resolve(encoder);
    }
}

impl Stage {
    fn new<const N: usize>(
        device: &wgpu::Device,
        pipeline: wgpu::ComputePipeline,
        workgroups: [u32; 2],
        inputs: [(u32, wgpu::BindingResource<'_>); N],
    ) -> Self {
        let layout = pipeline.get_bind_group_layout(0);
        let bindings = shader::bindings(device, &layout, inputs);
        Self {
            pipeline,
            bindings,
            workgroups,
        }
    }
}
