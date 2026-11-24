use crate::{
    density::{DensityInput, buffer_entry, create_pipeline},
    graph::ResampledGraph,
};
use color_eyre::eyre::{OptionExt, Result, ensure};
use wgpu::util::DeviceExt;

pub struct Geometry {
    pub input: DensityInput,
    pub segments: wgpu::Buffer,
    pub ranges: wgpu::Buffer,
    pub edge_count: u32,
    pub segment_count: u32,
}

pub struct Resampling {
    count: wgpu::ComputePipeline,
    sample: wgpu::ComputePipeline,
    size: wgpu::Extent3d,
}

impl Geometry {
    pub fn new(device: &wgpu::Device, graph: &ResampledGraph) -> Result<Self> {
        let input = DensityInput::new(device, graph)?;
        let segments = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("graph segments"),
            contents: bytemuck::cast_slice(graph.segments()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        Self::from_parts(device, input, segments, graph.ranges())
    }

    fn from_parts(
        device: &wgpu::Device,
        input: DensityInput,
        segments: wgpu::Buffer,
        ranges: &[[u32; 2]],
    ) -> Result<Self> {
        let edge_count = u32::try_from(ranges.len())?;
        let &[_start, end] = ranges.last().ok_or_eyre("graph has no edges")?;
        let segment_count = end
            .checked_sub(edge_count)
            .ok_or_eyre("invalid edge ranges")?;
        let ranges = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("graph edge ranges"),
            contents: bytemuck::cast_slice(ranges),
            usage: wgpu::BufferUsages::STORAGE,
        });
        Ok(Self {
            input,
            segments,
            ranges,
            edge_count,
            segment_count,
        })
    }
}

impl Resampling {
    pub fn new(device: &wgpu::Device, size: wgpu::Extent3d, spacing: f32) -> Self {
        let source = include_str!("shaders/resampling.wgsl");
        let constants = &[("spacing", f64::from(spacing))];
        Self {
            count: create_pipeline(device, source, "count_samples", size, constants),
            sample: create_pipeline(device, source, "resample", size, constants),
            size,
        }
    }

    pub fn count(
        &self,
        device: &wgpu::Device,
        commands: &mut wgpu::CommandEncoder,
        graph: &Geometry,
    ) -> wgpu::Buffer {
        let bytes = u64::from(graph.edge_count).saturating_mul(4);
        let counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resampled edge counts"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("edge count readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self::dispatch(
            device,
            commands,
            &self.count,
            &[
                buffer_entry(0, graph.input.points()),
                buffer_entry(1, &graph.ranges),
                buffer_entry(2, &counts),
            ],
            graph.edge_count,
        );
        commands.copy_buffer_to_buffer(&counts, 0, &readback, 0, bytes);
        readback
    }

    pub fn sample(
        &self,
        device: &wgpu::Device,
        commands: &mut wgpu::CommandEncoder,
        graph: &Geometry,
        counts: &[u8],
    ) -> Result<Geometry> {
        ensure!(
            counts.len() == usize::try_from(graph.edge_count)?.saturating_mul(4),
            "edge count readback has an invalid size"
        );
        // Like the released CUBu code, only counts cross to the CPU for an offset sum.
        let mut ranges = Vec::with_capacity(usize::try_from(graph.edge_count)?);
        let mut end = 0u32;
        for &bytes in counts.as_chunks::<4>().0 {
            let count = u32::from_ne_bytes(bytes);
            ensure!(count >= 2, "resampling produced an invalid edge count");
            let start = end;
            end = end
                .checked_add(count)
                .ok_or_eyre("too many resampled points")?;
            ranges.push([start, end]);
        }
        let input = DensityInput::allocate(device, end, [self.size.width, self.size.height])?;
        let segments = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resampled graph segments"),
            size: u64::from(end.saturating_sub(graph.edge_count)).saturating_mul(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let output = Geometry::from_parts(device, input, segments, &ranges)?;
        Self::dispatch(
            device,
            commands,
            &self.sample,
            &[
                buffer_entry(0, graph.input.points()),
                buffer_entry(1, &graph.ranges),
                buffer_entry(3, &output.ranges),
                buffer_entry(4, output.input.points()),
                buffer_entry(5, &output.segments),
            ],
            graph.edge_count,
        );
        Ok(output)
    }

    fn dispatch(
        device: &wgpu::Device,
        commands: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        entries: &[wgpu::BindGroupEntry<'_>],
        edges: u32,
    ) {
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("edge resampling"),
            layout: &pipeline.get_bind_group_layout(0),
            entries,
        });
        let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("edge resampling"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        let groups = edges.div_ceil(64);
        let x = groups.min(device.limits().max_compute_workgroups_per_dimension);
        pass.dispatch_workgroups(x, groups.div_ceil(x), 1);
    }
}
