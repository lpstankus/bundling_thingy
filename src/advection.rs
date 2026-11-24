use crate::{
    density::{Density, DensityInput, create_pipeline},
    profiling::GpuTimings,
};

pub struct Advection {
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    workgroups: [u32; 2],
}

impl Advection {
    pub fn new(
        device: &wgpu::Device,
        density: &Density,
        input: &DensityInput,
        segments: &wgpu::Buffer,
        step_size: f64,
    ) -> Self {
        let pipeline = create_pipeline(
            device,
            include_str!("shaders/advection.wgsl"),
            "advect",
            density.texture.size(),
            &[("step_size", step_size)],
        );
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point advection"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.points().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: segments.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: density.values.as_entire_binding(),
                },
            ],
        });
        Self {
            pipeline,
            bindings,
            workgroups: input.workgroups(),
        }
    }

    pub fn encode(&self, commands: &mut wgpu::CommandEncoder, timings: Option<&GpuTimings>) {
        let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("point advection"),
            timestamp_writes: timings.and_then(|timings| timings.compute_pass(0)),
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bindings, &[]);
        let [x, y] = self.workgroups;
        pass.dispatch_workgroups(x, y, 1);
    }
}
