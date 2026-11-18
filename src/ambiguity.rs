use crate::{
    Options, gpu::SIZE, neighbor_mean::workgroups, profiling::GpuTimings, resampling::Geometry,
    shader,
};
use color_eyre::eyre::{OptionExt, Result, ensure};
use serde_json::{Value, json};
use std::time::{Duration, Instant};

/// Arc-length resampled points per original route for weak Frechet distances.
const ORIGINAL_POINTS: u32 = 32;
const WORKGROUP_SIZE: u32 = 128;
const STATS_BYTES: u64 = 32;
/// Pair set slot: one directed route pair key.
const SLOT_BYTES: u64 = 4;
/// Half the export size, checked below.
const PIXELS_PER_CLIP: f32 = 800.0;
const _: () = assert!(SIZE == 1600);
/// Near length rows, one per concurrent contact workgroup, each 8 bytes per route.
const SCRATCH_BYTES: u64 = 128 << 20;
const MAX_ROWS: u64 = 1024;
/// Scoring workgroups per submission, each accumulating its partial sum over all submissions.
const SCORE_GROUPS: u32 = 1 << 14;
/// Pair set slots each scoring lane scans.
const SCORE_ROUNDS: u32 = 32;
/// Submissions grow while faster and shrink while slower than these, far below driver watchdogs.
const FAST_SUBMISSION: Duration = Duration::from_millis(100);
const SLOW_SUBMISSION: Duration = Duration::from_millis(400);
/// Byte offset of the submission's first workgroup in the parameters uniform.
const BASE_OFFSET: u64 = 20;
/// Grid entries reserved per sample for segment anchors.
const ANCHORS_PER_SAMPLE: u64 = 2;
const CONTACTS: u32 = 3;
const SCORE: u32 = 4;
const ENTRIES: [&str; 6] = [
    "count_anchors",
    "reserve_cells",
    "scatter_anchors",
    "accumulate_contacts",
    "score_contacts",
    "reduce_partials",
];
const TIMING_STAGES: [&str; 6] = [
    "ambiguity.grid_count",
    "ambiguity.grid_reserve",
    "ambiguity.grid_scatter",
    "ambiguity.contacts",
    "ambiguity.score",
    "ambiguity.reduce",
];

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Parameters {
    radius: f32,
    length: f32,
    tolerance: f32,
    pixels_per_clip: f32,
    capacity: u32,
    base: u32,
    padding: [u32; 2],
}

pub struct Measurement {
    pub contact_pairs: u32,
    pub incompatible_pairs: u32,
    pub score: f64,
}

struct GraphBuffers {
    originals: wgpu::Buffer,
    positions: wgpu::Buffer,
    tasks: wgpu::Buffer,
    scratch: wgpu::Buffer,
    rows: u32,
    path_count: u32,
}

pub struct Ambiguity {
    originals: wgpu::ComputePipeline,
    pipelines: [wgpu::ComputePipeline; 6],
    settings: Parameters,
    parameters: wgpu::Buffer,
    grid: wgpu::Buffer,
    cells: wgpu::Buffer,
    cell_count: u32,
    table: wgpu::Buffer,
    stats: wgpu::Buffer,
    map: wgpu::Buffer,
    timings: GpuTimings,
    graph: Option<GraphBuffers>,
}

impl Ambiguity {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        opts: &Options,
        map: &wgpu::Buffer,
    ) -> Result<Option<Self>> {
        if !opts.evaluate_ambiguity {
            return Ok(None);
        }
        let capacity = opts.ambiguity_pair_capacity;
        ensure!(
            u64::from(capacity).saturating_mul(SLOT_BYTES) <= binding_limit(device),
            "--ambiguity-pair-capacity exceeds GPU buffer limits"
        );
        // Near length rows add a ninth storage buffer to the default eight.
        ensure!(
            device.limits().max_storage_buffers_per_shader_stage >= 9,
            "ambiguity evaluation needs 9 storage buffers per shader stage"
        );
        let radius = opts.ambiguity_contact_radius;
        // Smaller radii multiply the query pieces and larger ones the anchors each piece scans.
        ensure!(
            (0.25..=16.0).contains(&radius),
            "--ambiguity-contact-radius must be between 0.25 and 16 pixels"
        );
        // Cells also set the anchor spacing, so queries reach the radius plus half a cell.
        let cell_size = (1u32..16)
            .find(|size| f64::from(*size) >= f64::from(radius))
            .unwrap_or(16)
            .saturating_add(1);
        let size = SIZE.div_ceil(cell_size);
        let cell_count = size.saturating_mul(size);
        let (originals, pipelines) = Self::pipelines(device);
        let settings = Parameters {
            radius,
            length: opts.ambiguity_contact_length,
            tolerance: opts.ambiguity_route_tolerance,
            pixels_per_clip: PIXELS_PER_CLIP,
            capacity,
            base: 0,
            padding: [0; 2],
        };
        Ok(Some(Self {
            originals,
            pipelines,
            settings,
            parameters: shader::uniform(
                device,
                "ambiguity parameters",
                bytemuck::bytes_of(&settings),
            ),
            grid: shader::uniform(
                device,
                "ambiguity grid dimensions and cell size",
                bytemuck::cast_slice(&[size, size, cell_size, 0]),
            ),
            cells: shader::storage(
                device,
                "ambiguity grid cells",
                u64::from(cell_count).saturating_mul(8).saturating_add(8),
            ),
            cell_count,
            table: shader::storage(
                device,
                "ambiguity pair table",
                u64::from(capacity).saturating_mul(SLOT_BYTES),
            ),
            // Totals followed by one partial sum per scoring workgroup.
            stats: shader::storage(
                device,
                "ambiguity totals",
                u64::from(SCORE_GROUPS)
                    .saturating_mul(4)
                    .saturating_add(STATS_BYTES),
            ),
            map: map.clone(),
            timings: GpuTimings::new(device, queue, &TIMING_STAGES)?,
            graph: None,
        }))
    }

    fn pipelines(device: &wgpu::Device) -> (wgpu::ComputePipeline, [wgpu::ComputePipeline; 6]) {
        let entries =
            [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12].map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if matches!(binding, 2 | 9 | 12) {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: matches!(binding, 0 | 4),
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ambiguity"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ambiguity"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = shader::module(
            device,
            "ambiguity",
            &format!(
                "const K: u32 = {ORIGINAL_POINTS}u;\nconst SCORE_ROUNDS: u32 = {SCORE_ROUNDS}u;\n{}\n{}",
                include_str!("shaders/spatial_grid.wgsl"),
                include_str!("shaders/ambiguity.wgsl"),
            ),
        );
        let constants = [("block_size", f64::from(WORKGROUP_SIZE))];
        let pipeline =
            |entry| shader::compute(device, &module, entry, Some(&pipeline_layout), &constants);
        (pipeline("resample_originals"), ENTRIES.map(pipeline))
    }

    /// Resample the uploaded iteration-0 routes once. These originals never change afterwards.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        geometry: &Geometry,
        samples: u32,
    ) -> Result<()> {
        let path_count = u32::try_from(geometry.paths.size() / 12)?;
        ensure!(
            path_count <= 1 << 16,
            "ambiguity evaluation supports at most 65536 routes"
        );
        let sample_groups = samples.div_ceil(WORKGROUP_SIZE);
        let row_bytes = u64::from(path_count.max(1)).saturating_mul(8);
        let rows = SCRATCH_BYTES.div_ceil(row_bytes).clamp(1, MAX_ROWS);
        let scratch_bytes = rows.saturating_mul(row_bytes);
        ensure!(
            scratch_bytes <= binding_limit(device),
            "ambiguity near length rows for {path_count} routes exceed GPU buffer limits"
        );
        let anchors = u64::from(samples).saturating_mul(ANCHORS_PER_SAMPLE);
        ensure!(
            anchors.saturating_mul(12) <= binding_limit(device),
            "ambiguity grid anchors for {samples} samples exceed GPU buffer limits"
        );
        self.graph = Some(GraphBuffers {
            originals: shader::storage(
                device,
                "ambiguity original routes",
                u64::from(path_count).saturating_mul(u64::from(ORIGINAL_POINTS) * 8),
            ),
            positions: shader::storage(
                device,
                "ambiguity grid anchors",
                anchors.saturating_mul(12),
            ),
            // Only the shared reserve_cells writes tasks, one per started block of a nonempty cell.
            tasks: shader::storage(
                device,
                "ambiguity grid tasks",
                anchors
                    .div_ceil(u64::from(WORKGROUP_SIZE))
                    .saturating_add(u64::from(self.cell_count))
                    .saturating_mul(8),
            ),
            scratch: shader::storage(device, "ambiguity near lengths", scratch_bytes),
            rows: u32::try_from(rows)?,
            path_count,
        });
        let bindings = self.bindings(device, geometry, samples)?;
        let groups = workgroups(device, sample_groups)?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ambiguity.originals"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.originals);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(groups[0], groups[1], 1);
        Ok(())
    }

    fn bindings(
        &self,
        device: &wgpu::Device,
        geometry: &Geometry,
        samples: u32,
    ) -> Result<wgpu::BindGroup> {
        let graph = self.graph()?;
        // The live points buffer can be larger than the active samples.
        let points = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &geometry.points,
            offset: 0,
            size: wgpu::BufferSize::new(u64::from(samples) * 12),
        });
        Ok(shader::bindings(
            device,
            &self.originals.get_bind_group_layout(0),
            [
                (0, points),
                (1, graph.originals.as_entire_binding()),
                (2, self.grid.as_entire_binding()),
                (3, self.table.as_entire_binding()),
                (4, geometry.paths.as_entire_binding()),
                (5, self.cells.as_entire_binding()),
                (6, graph.positions.as_entire_binding()),
                (7, graph.tasks.as_entire_binding()),
                (9, self.map.as_entire_binding()),
                (10, self.stats.as_entire_binding()),
                (11, graph.scratch.as_entire_binding()),
                (12, self.parameters.as_entire_binding()),
            ],
        ))
    }

    /// Measure the current geometry. Only the final totals return to the CPU.
    pub fn submit(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        geometry: &Geometry,
        samples: u32,
    ) -> Result<shader::Readback<[u32; 8]>> {
        let bindings = self.bindings(device, geometry, samples)?;
        let graph = self.graph()?;
        let sample_groups = workgroups(device, samples.div_ceil(WORKGROUP_SIZE))?;
        let cell_groups = workgroups(device, self.cell_count.div_ceil(WORKGROUP_SIZE))?;
        let new_encoder =
            || device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut encoder = new_encoder();
        encoder.clear_buffer(&self.cells, 0, None);
        encoder.clear_buffer(&self.table, 0, None);
        encoder.clear_buffer(&self.stats, 0, None);
        // A failed frame can leave rows unzeroed.
        encoder.clear_buffer(&graph.scratch, 0, None);
        for (stage, groups) in (0..).zip([sample_groups, cell_groups, sample_groups]) {
            self.dispatch(&mut encoder, &bindings, stage, [true, true], groups)?;
        }
        queue.submit([encoder.finish()]);
        let (routes, rows) = (graph.path_count, graph.rows);
        self.run_timed(device, queue, &bindings, CONTACTS, routes, rows)?;
        let slots = self
            .settings
            .capacity
            .div_ceil(WORKGROUP_SIZE * SCORE_ROUNDS);
        self.run_timed(device, queue, &bindings, SCORE, slots, SCORE_GROUPS)?;
        let mut encoder = new_encoder();
        self.dispatch(&mut encoder, &bindings, SCORE + 1, [true, true], [1, 1])?;
        self.timings.resolve(&mut encoder);
        // A fresh zeroed buffer cannot report a previous frame if the GPU drops a submission.
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ambiguity totals readback"),
            size: STATS_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&self.stats, 0, &readback, 0, STATS_BYTES);
        let totals = shader::readback(&encoder, &readback, |result| {
            result.map(bytemuck::pod_read_unaligned::<[u32; 8]>)
        });
        queue.submit([encoder.finish()]);
        Ok(totals)
    }

    fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &wgpu::BindGroup,
        stage: u32,
        [first, last]: [bool; 2],
        [x, y]: [u32; 2],
    ) -> Result<()> {
        let (pipeline, label) = (self.pipelines.iter().zip(TIMING_STAGES))
            .nth(usize::try_from(stage)?)
            .ok_or_eyre("unknown ambiguity stage")?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: self.timings.compute_part(stage, first, last),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bindings, &[]);
        pass.dispatch_workgroups(x, y, 1);
        Ok(())
    }

    /// Runs `total` workgroups of a stage in submissions sized by the previous submission's time.
    fn run_timed(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bindings: &wgpu::BindGroup,
        stage: u32,
        total: u32,
        limit: u32,
    ) -> Result<()> {
        // One workgroup's time does not fall with the chunk, so smaller chunks only idle the GPU.
        let floor = limit.min(64);
        let (mut base, mut chunk, mut submissions) = (0, floor, 0u32);
        let mut slowest = Duration::ZERO;
        while base < total {
            let count = chunk.min(total.saturating_sub(base));
            queue.write_buffer(&self.parameters, BASE_OFFSET, bytemuck::bytes_of(&base));
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let parts = [base == 0, base.saturating_add(count) == total];
            self.dispatch(&mut encoder, bindings, stage, parts, [count, 1])?;
            let start = Instant::now();
            queue.submit([encoder.finish()]);
            device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(60)),
            })?;
            let elapsed = start.elapsed();
            slowest = slowest.max(elapsed);
            if elapsed < FAST_SUBMISSION {
                chunk = chunk.saturating_mul(2).min(limit);
            } else if elapsed > SLOW_SUBMISSION {
                chunk = (chunk / 2).max(floor);
            }
            base = base.saturating_add(count);
            submissions = submissions.saturating_add(1);
        }
        tracing::info!(
            stage = TIMING_STAGES.get(usize::try_from(stage)?),
            submissions,
            slowest_ms = slowest.as_secs_f64() * 1e3,
            "ambiguity submissions"
        );
        Ok(())
    }

    fn graph(&self) -> Result<&GraphBuffers> {
        self.graph
            .as_ref()
            .ok_or_eyre("ambiguity originals are not prepared")
    }

    pub fn finish(&self, totals: [u32; 8]) -> Result<Measurement> {
        let [
            anchors,
            occupied,
            contact_pairs,
            incompatible_pairs,
            status,
            sum_squares,
            routes,
            scored,
        ] = totals;
        ensure!(
            status & 8 != 0,
            "ambiguity evaluation did not complete; the GPU may have reset a long submission"
        );
        ensure!(
            status & 1 == 0,
            "ambiguity pair set is full ({occupied} of {} slots); raise --ambiguity-pair-capacity",
            self.settings.capacity
        );
        ensure!(
            status & 4 == 0,
            "bundled geometry needs {anchors} ambiguity grid anchors, more than the reserved {} per sample",
            ANCHORS_PER_SAMPLE
        );
        ensure!(
            status & 2 == 0,
            "ambiguity weak Frechet relaxation did not converge"
        );
        // Status bits alone cannot show that every submission ran.
        ensure!(
            routes == self.graph()?.path_count && scored == self.settings.capacity,
            "ambiguity evaluation is partial ({routes} routes, {scored} slots); the GPU may have dropped a submission"
        );
        tracing::info!(
            anchors,
            occupied,
            capacity = self.settings.capacity,
            "ambiguity pair set"
        );
        let sum_squares = f64::from(f32::from_bits(sum_squares));
        let score = if contact_pairs == 0 {
            0.0
        } else {
            (sum_squares / f64::from(contact_pairs)).sqrt() / f64::from(self.settings.tolerance)
        };
        Ok(Measurement {
            contact_pairs,
            incompatible_pairs,
            score,
        })
    }
}

fn binding_limit(device: &wgpu::Device) -> u64 {
    let limits = device.limits();
    limits
        .max_buffer_size
        .min(limits.max_storage_buffer_binding_size.into())
}

pub fn write(opts: &Options, images: &[Value]) -> Result<()> {
    let report = json!({
        "schema_version": 1, "metric_version": 1, "metric": "ambiguity",
        "definition": "Contacts are route pairs where each route has at least contact_length_pixels of its current length within contact_radius_pixels of the other. d is the continuous weak Frechet distance between both original routes, minimized over orientation and computed on arc-length resampled originals. Score is sqrt(mean over contacts of max(d - T, 0)^2) / T with T = route_tolerance_pixels, or 0 without contacts.",
        "interpretation": "Lower means fewer and less severe false connections: drawn-together routes whose original routes differ by more than T. Contact length is the total near length per pair, not one connected stretch. Values can exceed 1. GPU ordering can cause run-to-run variation.",
        "input": opts.input.display().to_string(),
        "parameters": {
            "route_tolerance_pixels": opts.ambiguity_route_tolerance,
            "contact_radius_pixels": opts.ambiguity_contact_radius,
            "contact_length_pixels": opts.ambiguity_contact_length,
            "pair_capacity": opts.ambiguity_pair_capacity,
            "original_points_per_route": ORIGINAL_POINTS,
            "width": SIZE, "height": SIZE, "pixels_per_clip_unit": PIXELS_PER_CLIP
        },
        "bundling": {"neighbor_radius": opts.neighbor_radius,
            "iterations": if opts.neighbor_radius.is_some() { opts.iterations } else { 0 }},
        "images": images
    });
    crate::report::publish(&opts.out, "ambiguity.json", &report)
}
