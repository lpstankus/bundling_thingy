use crate::{Options, palette::ColorBy, resampling::Geometry, shader};
use color_eyre::eyre::{Result, ensure};

const WORKGROUP_SIZE: u32 = 128;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Parameters {
    window: u32,
    strength: f32,
    padding: [u32; 2],
}

/// Moves each sample toward the mean of its neighbors along the same path.
pub struct Smoothing {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    parameters: wgpu::Buffer,
    map: wgpu::Buffer,
}

impl Smoothing {
    pub fn from_options(device: &wgpu::Device, opts: &Options, map: &wgpu::Buffer) -> Option<Self> {
        opts.smoothing_window.map(|window| {
            Self::new(
                device,
                window,
                opts.smoothing,
                opts.color_by == ColorBy::Density,
                map,
            )
        })
    }

    fn new(
        device: &wgpu::Device,
        window: u32,
        strength: f32,
        accumulate_density: bool,
        map: &wgpu::Buffer,
    ) -> Self {
        let entries = [0, 1, 3, 4, 9, 10].map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if binding >= 9 {
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
            label: Some("path smoothing"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("path smoothing"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = shader::module(
            device,
            "path smoothing",
            include_str!("shaders/smoothing.wgsl"),
        );
        let constants = [
            ("block_size", f64::from(WORKGROUP_SIZE)),
            (
                "accumulate_density",
                f64::from(u8::from(accumulate_density)),
            ),
        ];
        Self {
            layout,
            pipeline: shader::compute(
                device,
                &module,
                "smooth_paths",
                Some(&pipeline_layout),
                &constants,
            ),
            parameters: shader::uniform(
                device,
                "path smoothing window and strength",
                bytemuck::bytes_of(&Parameters {
                    window,
                    strength,
                    padding: [0; 2],
                }),
            ),
            map: map.clone(),
        }
    }

    /// Reads `input` and writes smoothed samples into `geometry.points`.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        geometry: &Geometry,
        samples: u32,
        counts: &wgpu::Buffer,
    ) -> Result<()> {
        let limit = device.limits().max_compute_workgroups_per_dimension;
        let groups = samples.div_ceil(WORKGROUP_SIZE);
        let y = groups.div_ceil(limit);
        ensure!(y <= limit, "path smoothing exceeds GPU dispatch limits");
        let bytes = wgpu::BufferSize::new(u64::from(samples) * 12);
        let bindings = shader::bindings(
            device,
            &self.layout,
            [
                (
                    0,
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: input,
                        offset: 0,
                        size: bytes,
                    }),
                ),
                (1, geometry.points.as_entire_binding()),
                (3, counts.as_entire_binding()),
                (4, geometry.paths.as_entire_binding()),
                (9, self.map.as_entire_binding()),
                (10, self.parameters.as_entire_binding()),
            ],
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("graph.smooth_paths"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(groups.div_ceil(y), y, 1);
        Ok(())
    }
}
