use crate::{palette::ColorBy, profiling::GpuTimings, resampling::Geometry, shader};
use color_eyre::eyre::{Result, ensure};

const WORKGROUP_SIZE: u32 = 128;

pub struct NeighborMean {
    pipeline: wgpu::ComputePipeline,
    radius: wgpu::Buffer,
    map: wgpu::Buffer,
    timings: GpuTimings,
    scratch: Option<wgpu::Buffer>,
}

impl NeighborMean {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        radius: u32,
        color_by: ColorBy,
        map: &wgpu::Buffer,
    ) -> Result<Self> {
        let module = shader::module(
            device,
            "neighbor mean",
            include_str!("shaders/neighbor_mean.wgsl"),
        );
        let constants = [
            ("block_size", f64::from(WORKGROUP_SIZE)),
            (
                "accumulate_density",
                f64::from(u8::from(color_by == ColorBy::Density)),
            ),
        ];
        Ok(Self {
            pipeline: shader::compute(device, &module, "neighbor_mean", None, &constants),
            radius: shader::uniform(
                device,
                "neighbor radius in texels",
                bytemuck::cast_slice(&[radius, 0, 0, 0]),
            ),
            map: map.clone(),
            timings: GpuTimings::new(device, queue, &["graph.neighbor_mean"])?,
            scratch: None,
        })
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
        let groups = samples.div_ceil(WORKGROUP_SIZE);
        let limit = device.limits().max_compute_workgroups_per_dimension;
        let y = groups.div_ceil(limit);
        ensure!(
            y <= limit,
            "samples exceed neighbor mean GPU dispatch limits"
        );
        let bytes = u64::from(samples) * 12;
        let output = self
            .scratch
            .take()
            .unwrap_or_else(|| shader::storage(device, "neighbor mean points", bytes));
        let input = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &geometry.points,
            offset: 0,
            size: wgpu::BufferSize::new(bytes),
        });
        let bindings = shader::bindings(
            device,
            &self.pipeline.get_bind_group_layout(0),
            [
                (0, input),
                (1, output.as_entire_binding()),
                (2, self.radius.as_entire_binding()),
                (3, counts.as_entire_binding()),
                (4, geometry.paths.as_entire_binding()),
                (9, self.map.as_entire_binding()),
            ],
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("neighbor mean"),
                timestamp_writes: self.timings.compute_pass(0),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(groups.div_ceil(y), y, 1);
        }
        self.timings.resolve(encoder);
        self.scratch = Some(std::mem::replace(&mut geometry.points, output));
        Ok(())
    }
}
