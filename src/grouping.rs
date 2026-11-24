use crate::{
    palette::ColorBy,
    profiling::GpuTimings,
    resampling::{Geometry, Statistics},
    shader,
};
use color_eyre::eyre::{OptionExt, Result, ensure};
use wgpu::util::DeviceExt;

const BLOCK: u32 = 128;
const CHUNK_GROUPS: u32 = 512;
const CHUNK_WORK: u64 = 24_000_000_000;
const COMMAND_BATCH: u32 = 128;
const PARAMETER_BYTES: u64 = 32;
const VERTEX_BYTES: u64 = 20;
const POINT_BYTES: u64 = 12;
const STATE_BYTES: u64 = 20;
// Two vertex buffers, resolved samples, index and membership buffers, and aligned chunk parameters.
pub const SAMPLE_RESERVE_BYTES: u64 = 2 * VERTEX_BYTES + POINT_BYTES + 10 * 4 + 5;
pub const FIXED_RESERVE_BYTES: u64 = 64;
const STAGES: [&str; 11] = [
    "group.index",
    "group.average",
    "group.index_moved",
    "group.choose_partner",
    "group.compact_vertices",
    "group.resolve_samples",
    "group.index_segments",
    "group.count_paths",
    "group.build_memberships",
    "group.summarize_paths",
    "group.finish_iteration",
];

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Parameters {
    attraction: u32,
    merge: f32,
    samples: u32,
    segments: u32,
    base: u32,
    cell_mode: u32,
    padding: [u32; 2],
}

pub struct Grouping {
    initialize: wgpu::ComputePipeline,
    index_vertices: wgpu::ComputePipeline,
    average: wgpu::ComputePipeline,
    choose_partner: wgpu::ComputePipeline,
    compact: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    index_segments: wgpu::ComputePipeline,
    count_paths: wgpu::ComputePipeline,
    build_memberships: wgpu::ComputePipeline,
    summarize: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
    timings: GpuTimings,
    map: wgpu::Buffer,
    attraction: u32,
    merge: f32,
}

pub struct GroupedGraph {
    // Original samples retain their edge IDs. Each sample maps to one shared vertex.
    original: wgpu::Buffer,
    current: wgpu::Buffer,
    moved: wgpu::Buffer,
    sample_vertex: wgpu::Buffer,
    heads: wgpu::Buffer,
    links: wgpu::Buffer,
    partner: wgpu::Buffer,
    remap: wgpu::Buffer,
    state: wgpu::Buffer,
    pub segment_paths: wgpu::Buffer,
    pub path_counts: wgpu::Buffer,
    // Membership nodes are original segment occurrences, so their edge IDs stay available.
    membership_heads: wgpu::Buffer,
    membership_next: wgpu::Buffer,
    pub resolved: wgpu::Buffer,
    parameters: ParameterBuffer,
    merge_parameters: ParameterBuffer,
    path_parameters: ParameterBuffer,
    readback: wgpu::Buffer,
    workgroups: [u32; 2],
    active: u32,
    samples: u32,
    segments: u32,
}

struct Dispatch<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
}

struct ParameterBuffer {
    buffer: wgpu::Buffer,
    stride: u64,
    count: u32,
    chunk_samples: u32,
}

impl ParameterBuffer {
    fn binding(&self, index: u32) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.stride.saturating_mul(u64::from(index)),
            size: wgpu::BufferSize::new(PARAMETER_BYTES),
        })
    }
}

impl Grouping {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        attraction: u32,
        merge: f32,
        color_by: ColorBy,
        map: &wgpu::Buffer,
    ) -> Result<Self> {
        ensure!(attraction > 0, "attraction radius must be positive");
        ensure!(
            merge.is_finite() && merge >= 0.0 && f64::from(merge) <= f64::from(attraction),
            "merge radius must be finite and between zero and the attraction radius"
        );
        let module = shader::module(
            device,
            "shared paths",
            include_str!("shaders/grouping.wgsl"),
        );
        let constants = [("block_size", f64::from(BLOCK))];
        let compact_constants = [
            ("block_size", f64::from(BLOCK)),
            (
                "accumulate_density",
                f64::from(u8::from(color_by == ColorBy::Density)),
            ),
        ];
        let compute = |entry| shader::compute(device, &module, entry, None, &constants);
        Ok(Self {
            initialize: compute("initialize"),
            index_vertices: compute("index_vertices"),
            average: compute("average"),
            choose_partner: compute("choose_partner"),
            compact: shader::compute(
                device,
                &module,
                "compact_vertices",
                None,
                &compact_constants,
            ),
            resolve: compute("resolve_samples"),
            index_segments: compute("index_segments"),
            count_paths: compute("count_paths"),
            build_memberships: compute("build_memberships"),
            summarize: compute("summarize_paths"),
            finish: shader::compute(device, &module, "finish_iteration", None, &[]),
            timings: GpuTimings::new(device, queue, &STAGES)?,
            map: map.clone(),
            attraction,
            merge,
        })
    }

    pub fn prepare(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        geometry: &Geometry,
        statistics: Statistics,
    ) -> Result<GroupedGraph> {
        let Statistics { samples, segments } = statistics;
        ensure!(samples > 0 && segments > 0, "cannot group an empty graph");
        let buckets = samples
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or_eyre("too many samples for the grouping hash table")?;
        let sample_bytes = u64::from(samples);
        let segment_bytes = u64::from(segments).saturating_mul(4);
        let limits = device.limits();
        let limit = limits
            .max_buffer_size
            .min(u64::from(limits.max_storage_buffer_binding_size));
        for bytes in [
            sample_bytes.saturating_mul(VERTEX_BYTES),
            sample_bytes.saturating_mul(POINT_BYTES),
            u64::from(buckets).saturating_mul(4),
        ] {
            ensure!(bytes <= limit, "grouping buffer exceeds GPU limits");
        }
        let groups = samples.div_ceil(BLOCK);
        let y = groups.div_ceil(limits.max_compute_workgroups_per_dimension);
        ensure!(
            y <= limits.max_compute_workgroups_per_dimension,
            "samples exceed grouping GPU dispatch limits"
        );
        let chunk_samples =
            Self::chunk_samples(samples, limits.max_compute_workgroups_per_dimension)?;
        let short_search_samples = limits
            .max_compute_workgroups_per_dimension
            .min(CHUNK_GROUPS)
            .saturating_mul(BLOCK);
        let parameters = self.parameters(device, statistics, chunk_samples, 0)?;
        let merge_parameters = self.parameters(device, statistics, short_search_samples, 1)?;
        let path_parameters = self.parameters(device, statistics, short_search_samples, 0)?;
        let state = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grouping counts"),
            contents: bytemuck::cast_slice(&[samples, 0, segments, 0, 1]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let buffer = |label, bytes| shader::storage(device, label, bytes);
        let graph = GroupedGraph {
            original: geometry.points.clone(),
            current: buffer("group vertices", sample_bytes.saturating_mul(VERTEX_BYTES)),
            moved: buffer(
                "moved group vertices",
                sample_bytes.saturating_mul(VERTEX_BYTES),
            ),
            sample_vertex: buffer("sample to vertex", sample_bytes.saturating_mul(4)),
            heads: buffer("group hash heads", u64::from(buckets).saturating_mul(4)),
            links: buffer("group hash links", sample_bytes.saturating_mul(4)),
            partner: buffer("nearest group partners", sample_bytes.saturating_mul(4)),
            remap: buffer("group vertex remap", sample_bytes.saturating_mul(4)),
            state,
            segment_paths: buffer("edge path references", segment_bytes),
            path_counts: buffer("distinct edges per path", segment_bytes),
            membership_heads: buffer("path membership heads", segment_bytes),
            membership_next: buffer("path membership links", segment_bytes),
            resolved: buffer(
                "grouped sample positions",
                sample_bytes.saturating_mul(POINT_BYTES),
            ),
            parameters,
            merge_parameters,
            path_parameters,
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("grouping counts readback"),
                size: STATE_BYTES,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            workgroups: [groups.div_ceil(y), y],
            active: samples,
            samples,
            segments,
        };
        let bindings = shader::bindings(
            device,
            &self.initialize.get_bind_group_layout(0),
            [
                (0, graph.original.as_entire_binding()),
                (1, geometry.paths.as_entire_binding()),
                (2, graph.current.as_entire_binding()),
                (4, graph.sample_vertex.as_entire_binding()),
                (12, graph.segment_paths.as_entire_binding()),
                (13, graph.path_counts.as_entire_binding()),
                (15, graph.parameters.binding(0)),
                (17, graph.membership_heads.as_entire_binding()),
                (18, graph.membership_next.as_entire_binding()),
            ],
        );
        Self::pass(encoder, &self.initialize, &bindings, graph.workgroups, None);
        Ok(graph)
    }

    fn chunk_samples(samples: u32, limit: u32) -> Result<u32> {
        // Dense buckets increase the work done by each invocation.
        let group_work = u64::from(samples).saturating_mul(u64::from(BLOCK));
        let affordable = CHUNK_WORK
            .checked_div(group_work)
            .ok_or_eyre("empty graph")?;
        let groups = u32::try_from(affordable.min(u64::from(CHUNK_GROUPS)))?
            .max(1)
            .min(limit);
        let samples = groups.saturating_mul(BLOCK);
        ensure!(samples > 0, "GPU supports no compute workgroups");
        Ok(samples)
    }

    fn parameters(
        &self,
        device: &wgpu::Device,
        statistics: Statistics,
        chunk_samples: u32,
        cell_mode: u32,
    ) -> Result<ParameterBuffer> {
        let count = statistics.samples.div_ceil(chunk_samples);
        let alignment = u64::from(device.limits().min_uniform_buffer_offset_alignment).max(1);
        let stride = PARAMETER_BYTES
            .div_ceil(alignment)
            .saturating_mul(alignment);
        let bytes = stride.saturating_mul(u64::from(count));
        ensure!(
            bytes <= device.limits().max_buffer_size,
            "grouping parameters exceed GPU limits"
        );
        let mut contents = Vec::with_capacity(usize::try_from(bytes)?);
        for chunk in 0..count {
            contents.extend_from_slice(bytemuck::bytes_of(&Parameters {
                attraction: self.attraction,
                merge: self.merge,
                samples: statistics.samples,
                segments: statistics.segments,
                base: chunk.saturating_mul(chunk_samples),
                cell_mode,
                padding: [0; 2],
            }));
            contents.resize(
                usize::try_from(stride.saturating_mul(u64::from(chunk.saturating_add(1))))?,
                0,
            );
        }
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grouping parameters"),
            contents: &contents,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        Ok(ParameterBuffer {
            buffer,
            stride,
            count,
            chunk_samples,
        })
    }

    fn pass(
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        bindings: &wgpu::BindGroup,
        workgroups: [u32; 2],
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shared paths"),
            timestamp_writes,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bindings, &[]);
        pass.dispatch_workgroups(workgroups[0], workgroups[1], 1);
    }

    fn bindings<const N: usize>(
        device: &wgpu::Device,
        pipeline: &wgpu::ComputePipeline,
        entries: [(u32, wgpu::BindingResource<'_>); N],
    ) -> wgpu::BindGroup {
        shader::bindings(device, &pipeline.get_bind_group_layout(0), entries)
    }

    fn chunks(
        &self,
        dispatch: &Dispatch<'_>,
        pipeline: &wgpu::ComputePipeline,
        count: u32,
        stage: u32,
        parameters: &ParameterBuffer,
        bindings: impl Fn(&ParameterBuffer, u32) -> wgpu::BindGroup,
    ) -> Result<()> {
        let chunks = count.div_ceil(parameters.chunk_samples);
        ensure!(
            chunks > 0 && chunks <= parameters.count,
            "invalid grouping chunk count"
        );
        let mut commands = Vec::with_capacity(usize::try_from(chunks.min(COMMAND_BATCH))?);
        for index in 0..chunks {
            let base = index.saturating_mul(parameters.chunk_samples);
            let remaining = count.saturating_sub(base).min(parameters.chunk_samples);
            // A stage spans separate submissions, so its timestamps include any gaps between chunks.
            let mut timestamps = self.timings.compute_pass(stage);
            if let Some(writes) = &mut timestamps {
                if index > 0 {
                    writes.beginning_of_pass_write_index = None;
                }
                if index.saturating_add(1) < chunks {
                    writes.end_of_pass_write_index = None;
                }
                if writes.beginning_of_pass_write_index.is_none()
                    && writes.end_of_pass_write_index.is_none()
                {
                    timestamps = None;
                }
            }
            let mut encoder = dispatch
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            Self::pass(
                &mut encoder,
                pipeline,
                &bindings(parameters, index),
                [remaining.div_ceil(BLOCK), 1],
                timestamps,
            );
            commands.push(encoder.finish());
            if commands.len() == usize::try_from(COMMAND_BATCH)? {
                for command in std::mem::take(&mut commands) {
                    dispatch.queue.submit([command]);
                }
            }
        }
        for command in commands {
            dispatch.queue.submit([command]);
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "The passes share buffers and must run in this order."
    )]
    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mut encoder: wgpu::CommandEncoder,
        geometry: &Geometry,
        graph: &GroupedGraph,
        density: &wgpu::Buffer,
    ) -> Result<(wgpu::CommandEncoder, shader::Readback<[u32; 5]>)> {
        encoder.clear_buffer(&graph.state, 4, Some(16));
        encoder.clear_buffer(&graph.heads, 0, None);
        encoder.clear_buffer(&graph.path_counts, 0, None);
        encoder.clear_buffer(&graph.membership_heads, 0, None);
        let parameters = &graph.parameters;
        let merge_parameters = &graph.merge_parameters;
        let dispatch = Dispatch { device, queue };

        let index = |vertices: &wgpu::Buffer, parameters: &ParameterBuffer| {
            Self::bindings(
                device,
                &self.index_vertices,
                [
                    (2, vertices.as_entire_binding()),
                    (5, graph.heads.as_entire_binding()),
                    (6, graph.links.as_entire_binding()),
                    (9, self.map.as_entire_binding()),
                    (10, graph.state.as_entire_binding()),
                    (15, parameters.binding(0)),
                ],
            )
        };
        let first_index = index(&graph.current, parameters);
        Self::pass(
            &mut encoder,
            &self.index_vertices,
            &first_index,
            graph.workgroups,
            self.timings.compute_pass(0),
        );
        queue.submit([encoder.finish()]);
        self.chunks(
            &dispatch,
            &self.average,
            graph.active,
            1,
            &graph.parameters,
            |parameters, chunk| {
                Self::bindings(
                    device,
                    &self.average,
                    [
                        (2, graph.current.as_entire_binding()),
                        (3, graph.moved.as_entire_binding()),
                        (5, graph.heads.as_entire_binding()),
                        (6, graph.links.as_entire_binding()),
                        (9, self.map.as_entire_binding()),
                        (10, graph.state.as_entire_binding()),
                        (15, parameters.binding(chunk)),
                    ],
                )
            },
        )?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.clear_buffer(&graph.heads, 0, None);
        let moved_index = index(&graph.moved, merge_parameters);
        Self::pass(
            &mut encoder,
            &self.index_vertices,
            &moved_index,
            graph.workgroups,
            self.timings.compute_pass(2),
        );
        queue.submit([encoder.finish()]);
        self.chunks(
            &dispatch,
            &self.choose_partner,
            graph.active,
            3,
            &graph.merge_parameters,
            |parameters, chunk| {
                Self::bindings(
                    device,
                    &self.choose_partner,
                    [
                        (2, graph.moved.as_entire_binding()),
                        (5, graph.heads.as_entire_binding()),
                        (6, graph.links.as_entire_binding()),
                        (7, graph.partner.as_entire_binding()),
                        (9, self.map.as_entire_binding()),
                        (10, graph.state.as_entire_binding()),
                        (15, parameters.binding(chunk)),
                    ],
                )
            },
        )?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let compact = Self::bindings(
            device,
            &self.compact,
            [
                (2, graph.moved.as_entire_binding()),
                (3, graph.current.as_entire_binding()),
                (7, graph.partner.as_entire_binding()),
                (8, graph.remap.as_entire_binding()),
                (9, self.map.as_entire_binding()),
                (10, graph.state.as_entire_binding()),
                (14, density.as_entire_binding()),
                (15, parameters.binding(0)),
            ],
        );
        Self::pass(
            &mut encoder,
            &self.compact,
            &compact,
            graph.workgroups,
            self.timings.compute_pass(4),
        );
        let resolve = Self::bindings(
            device,
            &self.resolve,
            [
                (0, graph.original.as_entire_binding()),
                (2, graph.current.as_entire_binding()),
                (4, graph.sample_vertex.as_entire_binding()),
                (8, graph.remap.as_entire_binding()),
                (15, parameters.binding(0)),
                (16, graph.resolved.as_entire_binding()),
            ],
        );
        Self::pass(
            &mut encoder,
            &self.resolve,
            &resolve,
            graph.workgroups,
            self.timings.compute_pass(5),
        );
        encoder.clear_buffer(&graph.heads, 0, None);
        let index_segments = Self::bindings(
            device,
            &self.index_segments,
            [
                (4, graph.sample_vertex.as_entire_binding()),
                (5, graph.heads.as_entire_binding()),
                (6, graph.links.as_entire_binding()),
                (11, geometry.segments.as_entire_binding()),
                (12, graph.segment_paths.as_entire_binding()),
                (15, parameters.binding(0)),
            ],
        );
        Self::pass(
            &mut encoder,
            &self.index_segments,
            &index_segments,
            graph.workgroups,
            self.timings.compute_pass(6),
        );
        queue.submit([encoder.finish()]);
        self.chunks(
            &dispatch,
            &self.count_paths,
            graph.segments,
            7,
            &graph.path_parameters,
            |parameters, chunk| {
                Self::bindings(
                    device,
                    &self.count_paths,
                    [
                        (0, graph.original.as_entire_binding()),
                        (4, graph.sample_vertex.as_entire_binding()),
                        (5, graph.heads.as_entire_binding()),
                        (6, graph.links.as_entire_binding()),
                        (11, geometry.segments.as_entire_binding()),
                        (12, graph.segment_paths.as_entire_binding()),
                        (15, parameters.binding(chunk)),
                        (18, graph.membership_next.as_entire_binding()),
                    ],
                )
            },
        )?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let memberships = Self::bindings(
            device,
            &self.build_memberships,
            [
                (12, graph.segment_paths.as_entire_binding()),
                (13, graph.path_counts.as_entire_binding()),
                (15, parameters.binding(0)),
                (17, graph.membership_heads.as_entire_binding()),
                (18, graph.membership_next.as_entire_binding()),
            ],
        );
        Self::pass(
            &mut encoder,
            &self.build_memberships,
            &memberships,
            graph.workgroups,
            self.timings.compute_pass(8),
        );
        let summarize = Self::bindings(
            device,
            &self.summarize,
            [
                (10, graph.state.as_entire_binding()),
                (12, graph.segment_paths.as_entire_binding()),
                (13, graph.path_counts.as_entire_binding()),
                (15, parameters.binding(0)),
            ],
        );
        Self::pass(
            &mut encoder,
            &self.summarize,
            &summarize,
            graph.workgroups,
            self.timings.compute_pass(9),
        );
        let finish = Self::bindings(
            device,
            &self.finish,
            [(10, graph.state.as_entire_binding())],
        );
        Self::pass(
            &mut encoder,
            &self.finish,
            &finish,
            [1, 1],
            self.timings.compute_pass(10),
        );
        self.timings.resolve(&mut encoder);
        encoder.copy_buffer_to_buffer(&graph.state, 0, &graph.readback, 0, STATE_BYTES);
        let stats = shader::readback(&encoder, &graph.readback, |result| {
            result.map(bytemuck::pod_read_unaligned::<[u32; 5]>)
        });
        Ok((encoder, stats))
    }
}

impl GroupedGraph {
    pub const fn state(&self) -> &wgpu::Buffer {
        &self.state
    }

    pub fn set_active(&mut self, active: u32) -> Result<()> {
        ensure!(
            active > 0 && active <= self.samples,
            "invalid grouped vertex count"
        );
        self.active = active;
        Ok(())
    }
}
