use crate::{
    density::{buffer_entry, create_pipeline},
    profiling::GpuTimings,
    resampling::Geometry,
};

#[derive(Clone, Copy, Debug)]
pub enum PathSmoothing {
    Enabled,
    Disabled,
}

pub struct Smoothing {
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    output: wgpu::Buffer,
    input: wgpu::Buffer,
    workgroups: [u32; 2],
}

impl Smoothing {
    pub fn new(
        device: &wgpu::Device,
        graph: &Geometry,
        size: wgpu::Extent3d,
        spacing: f32,
    ) -> Self {
        let pipeline = create_pipeline(
            device,
            include_str!("shaders/smoothing.wgsl"),
            "smooth_paths",
            size,
            &[("spacing", f64::from(spacing))],
        );
        let input = graph.input.points().clone();
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smoothed path points"),
            size: input.size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("path smoothing"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                buffer_entry(0, &input),
                buffer_entry(1, &graph.ranges),
                buffer_entry(2, &output),
            ],
        });
        Self {
            pipeline,
            bindings,
            output,
            input,
            workgroups: graph.input.workgroups(),
        }
    }

    pub fn encode(&self, commands: &mut wgpu::CommandEncoder, timings: Option<&GpuTimings>) {
        {
            let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("path smoothing"),
                timestamp_writes: timings.and_then(|timings| timings.compute_pass(1)),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bindings, &[]);
            let [x, y] = self.workgroups;
            pass.dispatch_workgroups(x, y, 1);
        }
        commands.copy_buffer_to_buffer(&self.output, 0, &self.input, 0, self.input.size());
    }
}
