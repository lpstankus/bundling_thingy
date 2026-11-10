//! Per-frame sustained contact of the gpu-f32 backend: host scheduling of the
//! frame tables, the broad phase, and the batched component and interval-union
//! kernels. The host reads counts to size buffers and plan whole-route batches;
//! every geometric decision is made on the device.
use super::super::{Error, Geometry, Limit, Protocol, Result};
use super::{
    Config, Core, Fault, FrameStatistics, Originals, Stage,
    originals::ROUTE_LIMIT,
    runner::{Bind, Bound, Host, P, add, bits, histogram_words, mul, scan_words, slice_size, word},
    status_word,
};

/// Work items per dispatch; each examines at most 64 partner entries.
const ITEM_SLICE: u64 = 1 << 18;
/// Chain entries one cell may walk per invocation of a resumable kernel.
const WALK_STEPS: u32 = 128;
/// Rounds after which an unfinished chain walk fails (a resource limit).
const WALK_ROUNDS: u64 = 1 << 16;
/// Label rounds between convergence checks.
const ROUNDS_PER_CHECK: u64 = 4;
/// Label rounds after which an unconverged batch fails. A resource limit, not
/// a convergence bound.
const ROUND_LIMIT: u64 = 1 << 20;
/// Device bytes per batch cell: records, chains, links, covered ends, labels
/// with sums and up to four bucket heads, marks and scan sums.
const CELL_BYTES: u64 = 104;
const MAXIMUM_BATCH_CELLS: u64 = 1 << 26;
const MAXIMUM_GRID: u32 = 2048;
const S_PENDING: usize = 1;
const S_CHANGED: usize = 2;
const S_A: usize = 3;
const S_B: usize = 4;
const S_C: usize = 5;

/// Frame tables that live while a frame is evaluated.
struct Tables {
    segments: wgpu::Buffer,
    routes: wgpu::Buffer,
    segment_count: u64,
    segment_bits: u32,
}

struct Broad {
    grid: wgpu::Buffer,
    footprint: wgpu::Buffer,
    entries: wgpu::Buffer,
    weights: wgpu::Buffer,
    registers: P,
    /// First work item of each route, then the total.
    items: Vec<u32>,
    /// Free-space cells whose lower route is each route.
    cells: Vec<u32>,
    /// Word of the per-route cell counts; the record cursor follows them.
    counts: u64,
    cursor: u64,
}

/// Whole-route source range with its cell count.
struct Batch {
    first: u32,
    end: u32,
    cells: u64,
}

#[derive(Clone, Copy)]
struct Capacity {
    cells: u64,
    limit: Limit,
}

/// Positive-segment marks and per-route offsets, both scanned; returns the
/// segment count.
fn positive(
    core: &mut Core,
    originals: &Originals,
    geometry: &Geometry,
    points: &wgpu::Buffer,
    (marks, offsets, scratch): (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer),
) -> Result<u64> {
    let samples = u64::from(geometry.samples);
    let route_count = u64::from(originals.route_count);
    core.runner.reset();
    core.run(
        Stage::Frame,
        "frame_mark",
        &[
            (1, points),
            (2, &geometry.paths),
            (4, marks),
            (7, &originals.ordinals),
        ],
        P {
            n: geometry.samples,
            a: geometry.reservations,
            ..P::default()
        },
        add(samples, 1)?,
    )?;
    // Later kernels index through the validated ownership; stop on a fault.
    core.runner.status("frame points")?;
    core.scan(marks, (0, add(samples, 1)?), scratch)?;
    core.run_bounded(
        Stage::Frame,
        "frame_counts",
        &[
            (1, points),
            (2, &geometry.paths),
            (3, &originals.routes),
            (4, marks),
            (5, offsets),
        ],
        P {
            n: originals.route_count,
            ..P::default()
        },
        add(route_count, 1)?,
        // Each route checks its own points; together they are the samples.
        Bound::Total {
            each: u64::from(ROUTE_LIMIT),
            total: add(samples, add(route_count, 1)?)?,
        },
    )?;
    core.runner.status("frame layout")?;
    core.total(offsets, route_count, scratch)
}

fn tables(
    core: &mut Core,
    originals: &Originals,
    geometry: &Geometry,
    points: &wgpu::Buffer,
    config: &Config,
    statistics: &mut FrameStatistics,
) -> Result<Tables> {
    let samples = u64::from(geometry.samples);
    let route_count = u64::from(originals.route_count);
    core.scoped(|core| {
        let scratch = core
            .store
            .words("ambiguity f32 scan sums", scan_words(add(samples, 1)?)?)?;
        let marks = core
            .store
            .words("ambiguity f32 positive marks", add(samples, 1)?)?;
        let offsets = core
            .store
            .words("ambiguity f32 segment offsets", add(route_count, 1)?)?;
        let buffers = (&marks, &offsets, &scratch);
        let segment_count = positive(core, originals, geometry, points, buffers)?;
        let segments = core.store.words(
            "ambiguity f32 positive segments",
            mul(segment_count.max(1), 6)?,
        )?;
        let routes = core
            .store
            .words("ambiguity f32 frame routes", mul(route_count, 4)?)?;
        core.store.promote(&segments);
        core.store.promote(&routes);
        core.run(
            Stage::Frame,
            "frame_segments",
            &[
                (1, points),
                (3, &originals.routes),
                (4, &marks),
                (5, &offsets),
                (6, &segments),
                (7, &originals.ordinals),
            ],
            P {
                n: geometry.samples,
                ..P::default()
            },
            samples,
        )?;
        let routes_n = P {
            n: originals.route_count,
            x: config.minimum_contact,
            ..P::default()
        };
        core.runner.reset();
        core.run(
            Stage::Frame,
            "frame_routes",
            &[
                (1, points),
                (3, &originals.routes),
                (5, &offsets),
                (6, &routes),
            ],
            routes_n,
            route_count,
        )?;
        let lengths = [(4, &segments), (5, &routes)];
        // Each route adds its own positive segments.
        let short = Bound::Total {
            each: u64::from(ROUTE_LIMIT),
            total: add(segment_count, route_count)?,
        };
        core.run_bounded(
            Stage::Frame,
            "frame_short",
            &lengths,
            routes_n,
            route_count,
            short,
        )?;
        let status = core.runner.status("frame tables")?;
        statistics.collapsed_paths = status_word(&status, S_A);
        statistics.short_paths = status_word(&status, S_B);
        statistics.positive_segments = segment_count;
        statistics.validated_points = samples;
        Ok(Tables {
            segments,
            routes,
            segment_count,
            segment_bits: bits(status_word(&status, S_C)),
        })
    })
}

/// Grid cells per axis: the most cells of at least the radius, up to the cap.
fn grid_size(radius: f32) -> u32 {
    let mut size = MAXIMUM_GRID;
    while size > 1 && 4.0 / f64::from(size) < f64::from(radius) {
        size /= 2;
    }
    size
}

/// Grid registers: cells per axis, inflated half radius, cell scale, radius.
fn grid_registers(originals: &Originals, config: &Config) -> P {
    let size = grid_size(config.radius);
    #[expect(
        clippy::cast_precision_loss,
        clippy::as_conversions,
        reason = "A power of two up to 2048 is exact in f32."
    )]
    let scale = size as f32 / 4.0;
    P {
        h: size,
        i: originals.route_count,
        x: (config.radius * 0.5).mul_add(1.001, 9.536_743e-7),
        y: scale,
        z: config.radius,
        ..P::default()
    }
}

/// Footprint entries, cell lists and work items of the frame's segments.
fn footprints(
    core: &mut Core,
    originals: &Originals,
    tables: &Tables,
    registers: P,
    statistics: &mut FrameStatistics,
) -> Result<Broad> {
    let route_count = u64::from(originals.route_count);
    let segment_count = tables.segment_count;
    let cells = mul(u64::from(registers.h), u64::from(registers.h))?;
    let counts = add(mul(cells, 2)?, 1)?;
    let cursor = add(counts, route_count)?;
    let store = &mut core.store;
    let grid = store.words("ambiguity f32 grid", add(cursor, 1)?)?;
    let offsets = store.words("ambiguity f32 footprint offsets", add(segment_count, 1)?)?;
    let scratch = store.words(
        "ambiguity f32 scan sums",
        scan_words(add(segment_count.max(cells), 1)?)?,
    )?;
    core.runner.reset();
    core.run(
        Stage::Broad,
        "foot_count",
        &[(1, &tables.segments), (6, &offsets)],
        P {
            n: word(segment_count)?,
            ..registers
        },
        add(segment_count, 1)?,
    )?;
    let total = core.count("segment footprint count", &offsets, segment_count, &scratch)?;
    let store = &mut core.store;
    let footprint = store.words("ambiguity f32 footprint", mul(total.max(1), 2)?)?;
    let entries = store.words("ambiguity f32 cell entries", total.max(1))?;
    let weights = store.words("ambiguity f32 item offsets", add(total, 1)?)?;
    let wide = store.words(
        "ambiguity f32 footprint scan sums",
        scan_words(add(total, 1)?)?,
    )?;
    let footprint_n = P {
        n: word(total)?,
        a: word(segment_count)?,
        ..registers
    };
    let emitted = [
        (1, &tables.segments),
        (3, &grid),
        (4, &footprint),
        (6, &offsets),
    ];
    core.run(Stage::Broad, "foot_emit", &emitted, footprint_n, total)?;
    core.scan(&grid, (0, add(cells, 1)?), &scratch)?;
    let filled = [(3, &grid), (4, &footprint), (5, &entries)];
    core.run(Stage::Broad, "cell_fill", &filled, footprint_n, total)?;
    let weighed = [(3, &grid), (4, &footprint), (6, &weights)];
    let padded = add(total, 1)?;
    core.run(Stage::Broad, "item_weights", &weighed, footprint_n, padded)?;
    // The total is validated before any offset is used.
    core.count("work-item count", &weights, total, &wide)?;
    let route_items = core
        .store
        .words("ambiguity f32 route items", add(route_count, 1)?)?;
    core.run(
        Stage::Broad,
        "route_items",
        &[
            (2, &tables.routes),
            (4, &offsets),
            (5, &weights),
            (6, &route_items),
        ],
        P {
            n: originals.route_count,
            a: word(segment_count)?,
            ..registers
        },
        add(route_count, 1)?,
    )?;
    let items = core.runner.read(&route_items, 0, add(route_count, 1)?)?;
    core.runner.status("work items")?;
    statistics.grid_cells_per_axis = registers.h;
    statistics.footprint_entries = total;
    Ok(Broad {
        grid,
        footprint,
        entries,
        weights,
        registers: P {
            b: word(total)?,
            ..registers
        },
        items,
        cells: Vec::new(),
        counts,
        cursor,
    })
}

impl Broad {
    /// Dispatch work items `[first, end)` in bounded slices.
    fn items(
        &self,
        core: &mut Core,
        tables: &Tables,
        entry: &str,
        (first, end): (u64, u64),
        registers: P,
        output: Option<&wgpu::Buffer>,
    ) -> Result<()> {
        let mut binds = vec![
            (1, &tables.segments),
            (2, &tables.routes),
            (3, &self.grid),
            (4, &self.footprint),
            (5, &self.entries),
            (6, &self.weights),
        ];
        binds.extend(output.map(|output| (7, output)));
        let mut item = first;
        while item < end {
            let slice = end.saturating_sub(item).min(ITEM_SLICE);
            core.run(
                Stage::Broad,
                entry,
                &binds,
                P {
                    n: word(end)?,
                    g: word(item)?,
                    ..registers
                },
                slice,
            )?;
            item = add(item, slice)?;
        }
        Ok(())
    }

    /// Count the free-space cells of every lower route.
    fn count(
        mut self,
        core: &mut Core,
        tables: &Tables,
        route_count: u64,
        statistics: &mut FrameStatistics,
    ) -> Result<Self> {
        let total = u64::from(self.items.last().copied().unwrap_or(0));
        core.runner.reset();
        self.items(
            core,
            tables,
            "count_cells",
            (0, total),
            self.registers,
            None,
        )?;
        core.runner.status("free-space cell count")?;
        statistics.work_items = total;
        self.cells = core.runner.read(&self.grid, self.counts, route_count)?;
        Ok(self)
    }
}

/// Whole-route batches in increasing source order under the cell capacity and
/// the source bits left in the 64-bit key.
fn plan(
    host: &mut Host,
    cells: &[u32],
    capacity: Capacity,
    source_bits: u32,
) -> Result<Vec<Batch>> {
    let span = 1_u64.checked_shl(source_bits).unwrap_or(u64::MAX);
    let mut batches = host.vector(u64::try_from(cells.len()).map_err(|_| Error::SizeOverflow)?)?;
    let mut current: Option<Batch> = None;
    for (route, count) in (0_u32..).zip(cells) {
        let count = u64::from(*count);
        if count > capacity.cells {
            return Err(Error::SourceCapacity {
                source: route,
                limit: capacity.limit,
                demand: count,
                available: capacity.cells,
            });
        }
        let next = route.checked_add(1).ok_or(Error::SizeOverflow)?;
        match &mut current {
            Some(batch)
                if add(batch.cells, count)? <= capacity.cells
                    && u64::from(route.saturating_sub(batch.first)) < span =>
            {
                batch.cells = add(batch.cells, count)?;
                batch.end = next;
            }
            _ => {
                batches.extend(current.take().filter(|batch| batch.cells > 0));
                current = Some(Batch {
                    first: route,
                    end: next,
                    cells: count,
                });
            }
        }
    }
    batches.extend(current.filter(|batch| batch.cells > 0));
    Ok(batches)
}

struct Workspace {
    records: wgpu::Buffer,
    chains: wgpu::Buffer,
    links: wgpu::Buffer,
    /// Covered interval end per cell and side during measurement.
    covered: wgpu::Buffer,
    /// Labels, component sums and bucket heads.
    state: wgpu::Buffer,
    marks: wgpu::Buffer,
    scratch: wgpu::Buffer,
}

/// Word counts shared by capacity planning and workspace allocation.
fn workspace_words(cells: u64) -> Result<[u64; 7]> {
    Ok([
        mul(cells, 8)?,
        mul(cells, 2)?,
        mul(cells, 2)?,
        mul(cells, 2)?,
        add(mul(cells, 5)?, mul(buckets(cells), 2)?)?,
        add(cells, 1)?,
        scan_words(add(cells, 1)?)?,
    ])
}

/// Largest padded storage binding during classification and pair logging.
fn largest_buffer_bytes(cells: u64) -> Result<u64> {
    let histogram = histogram_words(cells)?;
    let largest = workspace_words(cells)?
        .into_iter()
        .chain([
            mul(cells, 3)?,
            histogram,
            scan_words(add(cells, 1)?.max(histogram))?,
        ])
        .max()
        .unwrap_or(0);
    mul(largest, 4)?
        .max(16)
        .checked_next_multiple_of(16)
        .ok_or(Error::SizeOverflow)
}

fn batch_capacity(available: u64, storage: u64) -> Result<Capacity> {
    let global = available
        .checked_div(CELL_BYTES)
        .unwrap_or(0)
        .saturating_sub(4096)
        .min(MAXIMUM_BATCH_CELLS);
    // Every allocation grows monotonically, including power-of-two bucket
    // jumps. Find the largest cell count whose entire bindings still fit.
    let (mut low, mut high) = (0, global);
    while low < high {
        let middle = add(low, high.saturating_sub(low).div_ceil(2))?;
        if largest_buffer_bytes(middle)? <= storage {
            low = middle;
        } else {
            high = middle.saturating_sub(1);
        }
    }
    Ok(Capacity {
        cells: low,
        limit: if low < global {
            Limit::AdapterBuffer("ambiguity f32 contact workspace")
        } else {
            Limit::DeviceMemory("ambiguity f32 contact batch")
        },
    })
}

impl Workspace {
    fn new(core: &mut Core, cells: u64) -> Result<Self> {
        let [records, chains, links, covered, state, marks, scratch] = workspace_words(cells)?;
        let store = &mut core.store;
        Ok(Self {
            records: store.words("ambiguity f32 cell records", records)?,
            chains: store.words("ambiguity f32 cell chains", chains)?,
            links: store.words("ambiguity f32 cell links", links)?,
            covered: store.words("ambiguity f32 covered ends", covered)?,
            state: store.words("ambiguity f32 labels, sums and buckets", state)?,
            marks: store.words("ambiguity f32 component marks", marks)?,
            scratch: store.words("ambiguity f32 cell scan sums", scratch)?,
        })
    }
}

/// Hash buckets per side for `cells` cells: a power of two of at least the
/// cell count.
fn buckets(cells: u64) -> u64 {
    cells.max(1).next_power_of_two()
}

/// Where contact pairs are appended.
#[derive(Clone, Copy)]
pub(super) struct Log<'a> {
    pub buffer: &'a wgpu::Buffer,
    pub capacity: u64,
}

/// Evaluate every route pair of the frame for sustained contact and append the
/// contacting pairs to `log` in increasing (lower ordinal, higher ordinal)
/// order. Returns the number of contacts.
pub(super) fn contacts(
    core: &mut Core,
    originals: &Originals,
    (geometry, points): (&Geometry, &wgpu::Buffer),
    config: &Config,
    log: Log<'_>,
    statistics: &mut FrameStatistics,
) -> Result<u64> {
    core.scoped(|core| {
        let tables = tables(core, originals, geometry, points, config, statistics)?;
        if tables.segment_count == 0 {
            return Ok(0);
        }
        let registers = grid_registers(originals, config);
        let broad = footprints(core, originals, &tables, registers, statistics)?.count(
            core,
            &tables,
            u64::from(originals.route_count),
            statistics,
        )?;
        tracing::info!(
            segments = tables.segment_count,
            footprints = statistics.footprint_entries,
            items = statistics.work_items,
            "gpu-f32 free-space cells counted"
        );
        let partner_bits = bits(u64::from(originals.route_count));
        let used = partner_bits.saturating_add(tables.segment_bits.saturating_mul(2));
        if used > 63 {
            return Err(Error::KeyWidth {
                required: used.saturating_add(1),
            });
        }
        let capacity = batch_capacity(core.store.available(), core.runner.storage_buffer_bytes())?;
        let batches = plan(
            &mut core.runner.host,
            &broad.cells,
            capacity,
            64_u32.saturating_sub(used),
        )?;
        let largest = batches.iter().map(|batch| batch.cells).max().unwrap_or(0);
        statistics.batch_capacity_cells = capacity.cells;
        statistics.largest_batch_cells = largest;
        statistics.batches = u64::try_from(batches.len()).map_err(|_| Error::SizeOverflow)?;
        if largest == 0 {
            return Ok(0);
        }
        let workspace = Workspace::new(core, largest)?;
        let mut logged = 0;
        for batch in &batches {
            let run = Run {
                tables: &tables,
                broad: &broad,
                workspace: &workspace,
                cells: batch.cells,
                keys: P {
                    n: word(batch.cells)?,
                    h: word(buckets(batch.cells))?,
                    d: partner_bits,
                    e: tables.segment_bits,
                    f: batch.first,
                    ..P::default()
                },
            };
            let found = run.classify(core, batch, config, (log, logged), statistics)?;
            tracing::info!(
                first = batch.first,
                end = batch.end,
                cells = batch.cells,
                contacts = found,
                "gpu-f32 contact batch"
            );
            logged = add(logged, found)?;
        }
        Ok(logged)
    })
}

/// One batch of cells.
struct Run<'a> {
    tables: &'a Tables,
    broad: &'a Broad,
    workspace: &'a Workspace,
    cells: u64,
    /// Cell count and key layout registers.
    keys: P,
}

impl Run<'_> {
    /// Write the batch's cell records and check them against the count pass.
    fn fill(&self, core: &mut Core, batch: &Batch) -> Result<()> {
        let item = |route: u32| {
            usize::try_from(route)
                .ok()
                .and_then(|index| self.broad.items.get(index))
                .copied()
                .map(u64::from)
                .ok_or(Error::Protocol(Protocol::Inventory))
        };
        core.runner.reset();
        core.runner.clear(&self.workspace.state);
        core.runner
            .clear_words(&self.broad.grid, self.broad.cursor, 1)?;
        self.broad.items(
            core,
            self.tables,
            "fill_cells",
            (item(batch.first)?, item(batch.end)?),
            P {
                c: self.keys.n,
                d: self.keys.d,
                e: self.keys.e,
                f: self.keys.f,
                ..self.broad.registers
            },
            Some(&self.workspace.records),
        )?;
        core.runner.status("free-space cell fill")?;
        if core.word(&self.broad.grid, self.broad.cursor)? != self.cells {
            return Err(Error::Fault(Fault::Schedule(
                "free-space cell count and fill disagree",
            )));
        }
        Ok(())
    }

    /// Run a resumable chain-walk kernel over every cell, in slices under the
    /// work target, until no cell has a walk left.
    fn walk(
        &self,
        core: &mut Core,
        (entry, stage): (&str, &'static str),
        binds: &[Bind<'_>],
        registers: P,
    ) -> Result<()> {
        let units = add(u64::from(WALK_STEPS), 4)?;
        let slice = slice_size(units);
        let mut round = 0;
        loop {
            core.runner.reset();
            let mut first = 0;
            while first < self.cells {
                let end = add(first, slice)?.min(self.cells);
                core.run_bounded(
                    Stage::Contact,
                    entry,
                    binds,
                    P {
                        a: word(first)?,
                        b: word(end)?,
                        c: WALK_STEPS,
                        g: u32::from(round == 0),
                        ..registers
                    },
                    end.saturating_sub(first),
                    Bound::Each(units),
                )?;
                first = end;
            }
            let status = core.runner.status(stage)?;
            if status_word(&status, S_PENDING) == 0 {
                return Ok(());
            }
            round = add(round, 1)?;
            if round > WALK_ROUNDS {
                return Err(Error::Fault(Fault::Rounds {
                    stage,
                    limit: WALK_ROUNDS,
                }));
            }
        }
    }

    /// Chain the cells into their buckets and link joined neighbors. A join
    /// without its neighbor cell fails the status check.
    fn link(&self, core: &mut Core) -> Result<()> {
        let workspace = self.workspace;
        let chained = [
            (1, &workspace.records),
            (3, &workspace.state),
            (4, &workspace.chains),
        ];
        core.run(Stage::Contact, "chain", &chained, self.keys, self.cells)?;
        self.walk(
            core,
            ("links", "free-space cell joins"),
            &[
                (1, &workspace.records),
                (2, &self.tables.routes),
                (3, &workspace.state),
                (4, &workspace.chains),
                (5, &workspace.links),
            ],
            self.keys,
        )
    }

    /// Propagate labels until a complete round changes nothing.
    fn components(&self, core: &mut Core, statistics: &mut FrameStatistics) -> Result<()> {
        let binds = [(3, &self.workspace.state), (5, &self.workspace.links)];
        let mut rounds = 0;
        loop {
            core.runner.reset();
            for _ in 0..ROUNDS_PER_CHECK {
                core.run(Stage::Contact, "label_round", &binds, self.keys, self.cells)?;
            }
            rounds = add(rounds, ROUNDS_PER_CHECK)?;
            let status = core.runner.status("contact components")?;
            if status_word(&status, S_CHANGED) == 0 {
                break;
            }
            if rounds >= ROUND_LIMIT {
                return Err(Error::Fault(Fault::Rounds {
                    stage: "contact components",
                    limit: ROUND_LIMIT,
                }));
            }
        }
        statistics.label_rounds = add(statistics.label_rounds, rounds)?;
        statistics.largest_batch_label_rounds = statistics.largest_batch_label_rounds.max(rounds);
        Ok(())
    }

    /// Measure every component on both routes and mark the sustained ones.
    /// Returns their count.
    fn measure(&self, core: &mut Core, minimum: f32) -> Result<u64> {
        let workspace = self.workspace;
        // The links are final; their buffer now holds the walk cursors.
        self.walk(
            core,
            ("measure", "contact lengths"),
            &[
                (1, &workspace.records),
                (2, &self.tables.routes),
                (3, &workspace.state),
                (4, &workspace.chains),
                (5, &workspace.links),
                (6, &workspace.covered),
                (7, &self.tables.segments),
            ],
            P {
                x: minimum,
                ..self.keys
            },
        )?;
        let marked = [(3, &workspace.state), (6, &workspace.marks)];
        let padded = add(self.cells, 1)?;
        core.run(Stage::Contact, "sustained", &marked, self.keys, padded)?;
        core.total(&workspace.marks, self.cells, &workspace.scratch)
    }

    /// Deduplicate the sustained components' route pairs and append them to
    /// the log in order. Returns the number of contacts.
    fn log(&self, core: &mut Core, components: u64, (log, logged): (Log<'_>, u64)) -> Result<u64> {
        if components == 0 {
            return Ok(0);
        }
        let workspace = self.workspace;
        core.scoped(|core| {
            let store = &mut core.store;
            let first = store.words("ambiguity f32 component pairs", mul(components, 3)?)?;
            let second = store.words("ambiguity f32 component pairs", mul(components, 3)?)?;
            let histogram = store.words(
                "ambiguity f32 component pair histogram",
                histogram_words(components)?,
            )?;
            let scratch = store.words(
                "ambiguity f32 component pair scan sums",
                scan_words(add(components, 1)?.max(histogram_words(components)?))?,
            )?;
            let unique = store.words("ambiguity f32 component pair marks", add(components, 1)?)?;
            core.run(
                Stage::Contact,
                "sustained_pairs",
                &[(1, &workspace.records), (6, &workspace.marks), (7, &first)],
                self.keys,
                self.cells,
            )?;
            let route_bits = self.keys.d;
            let (sorted, _) = core.sort(
                (&first, &second),
                (&histogram, &scratch),
                components,
                (route_bits, route_bits),
            )?;
            let count = P {
                n: word(components)?,
                a: word(logged)?,
                ..P::default()
            };
            let keyed = [(4, sorted), (5, &unique)];
            let padded = add(components, 1)?;
            core.run(
                Stage::Originals,
                "key_marks",
                &keyed,
                P { a: 0, ..count },
                padded,
            )?;
            let contacts = core.total(&unique, components, &scratch)?;
            let total = add(logged, contacts)?;
            if total > log.capacity {
                return Err(Error::Limit {
                    limit: Limit::ContactPairs,
                    demand: total,
                    available: log.capacity,
                });
            }
            let written = [(4, sorted), (5, &unique), (7, log.buffer)];
            core.run(Stage::Contact, "pair_out", &written, count, components)?;
            Ok(contacts)
        })
    }

    /// Fill, chain, link, label, measure and log one batch.
    fn classify(
        &self,
        core: &mut Core,
        batch: &Batch,
        config: &Config,
        log: (Log<'_>, u64),
        statistics: &mut FrameStatistics,
    ) -> Result<u64> {
        self.fill(core, batch)?;
        self.link(core)?;
        self.components(core, statistics)?;
        let components = self.measure(core, config.minimum_contact)?;
        core.runner.status("contact lengths")?;
        let contacts = self.log(core, components, log)?;
        statistics.cells = add(statistics.cells, self.cells)?;
        statistics.sustained_components = add(statistics.sustained_components, components)?;
        Ok(contacts)
    }
}
