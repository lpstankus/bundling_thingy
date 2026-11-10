//! Whole-route distance assessments of the gpu-f32 backend: host scheduling of
//! the bounded task, frontier-search, relaxation and refinement dispatches. No
//! distance, bound or flag is computed on the host.
use super::super::{Error, Protocol, Result, overflow};
use super::{
    Core, Fault, Originals, Stage,
    runner::{Bind, Bound, P, WORK_TARGET, add, mul, scan_words, slice_size, word},
    status_word,
};

/// Assessment pairs of one chunk.
pub(super) const PAIRS: u64 = 1 << 16;
/// Fast-branch tasks per dispatch; each is bounded by the route limit.
const TASK_SLICE: u64 = 1 << 14;
/// Largest accepted search-cell limit: at it, the label units of one chunk's
/// 131,072 tasks still fit the u32 prefix scan.
pub const SEARCH_CELLS_LIMIT: u64 = 1 << 22;
/// Relaxation tiles per dispatch (up to 32 x 32 cells each).
const TILE_SLICE: u64 = 1 << 18;
/// Sweep pairs of one tile invocation; a tile stops earlier at its own
/// fixpoint.
const SWEEPS: u32 = 1;
/// Device bytes per label cell: the label and two boundary weights.
const CELL_BYTES: u64 = 12;
/// Complete relaxation rounds between convergence checks; only the last one
/// is counted.
const ROUNDS_PER_CHECK: u64 = 8;
/// Label cells the search chunks aim for when the byte limit allows.
const LABEL_TARGET: u64 = 1 << 27;
/// Complete rounds after which an unconverged chunk fails. A resource limit,
/// not a convergence bound: no proof says every admitted search fits it.
const ROUND_LIMIT: u64 = 1 << 16;
const S_PENDING: usize = 1;
const S_CHANGED: usize = 2;
const S_A: usize = 3;
const S_B: usize = 4;
const S_D: usize = 6;
/// Bounded frontier tiers: table slots (a power of two) and heap entries. A
/// search may see three quarters of its table. The search is resumable: one
/// invocation pops at most `FRONTIER_STEPS` entries.
const FRONTIER_STEPS: u32 = 512;
/// Work units of one pop: five table lookups of at most 64 probes, four pushes
/// and one pop through at most 13 heap levels, four boundary weights.
const POP_UNITS: u64 = 448;
const FRONTIER_TIERS: [(u32, u32); 2] = [(1 << 10, 1 << 10), (1 << 12, 1 << 12)];
const STAGE: Stage = Stage::Distance;

/// Buffers of the assessment stage, sized once for the frozen originals.
pub(super) struct Assessor {
    /// Assessment pairs: 16 words each.
    pub pairs: wgpu::Buffer,
    tasks: wgpu::Buffer,
    grid: wgpu::Buffer,
    labels: wgpu::Buffer,
    scratch: wgpu::Buffer,
    /// Requested largest cell count of one search, enforced exactly.
    limit: u64,
    /// Work units one fast-branch task can take on the longest routes.
    task_units: u64,
    /// Label units (256 cells) of one chunk bucket; at least one search.
    bucket: u64,
    /// Words of the label buffer.
    label_words: u64,
    pub statistics: Statistics,
}

/// Device work of the assessments, for reports.
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct Statistics {
    pub assessed_pairs: u64,
    /// Cell searches finished by the bounded frontier kernel.
    pub frontier_searches: u64,
    /// Cell searches finished by tiled relaxation.
    pub cell_searches: u64,
    pub cell_search_chunks: u64,
    pub relaxation_rounds: u64,
    pub largest_search_cells_limit: u64,
}

/// One chunk of searches whose labels share the label buffer.
struct Chunk {
    index: u32,
    searches: (u32, u32),
    tiles: (u32, u32),
}

type Binds<'a> = [Bind<'a>; 7];

impl Assessor {
    /// `longest` is the largest canonical point count; `limit` caps the cells of
    /// one search (a search beyond it fails explicitly when it occurs).
    pub(super) fn new(core: &mut Core, longest: u64, limit: u64) -> Result<Self> {
        if limit == 0 || limit > SEARCH_CELLS_LIMIT {
            return Err(Error::Fault(Fault::Range(
                "search cells must be within 1..=4194304",
            )));
        }
        let segments = longest.saturating_sub(1).max(1);
        // Buffer capacity only: admission compares with `limit` itself.
        let cells = mul(
            mul(segments, segments)?.min(limit).max(1024).div_ceil(256),
            256,
        )?;
        let tasks = mul(PAIRS, 2)?;
        // Scanned unit offsets of every task must fit u32.
        word(mul(tasks, cells.div_ceil(256))?)?;
        // Larger buckets mean fewer chunks; a quarter of the free bytes at most.
        let affordable = core
            .store
            .available()
            .checked_div(mul(CELL_BYTES, 8)?)
            .unwrap_or(0);
        let bucket = cells.max(affordable.min(LABEL_TARGET / 2)).div_ceil(256);
        // Two buckets of labels and of both boundary weights.
        let label_cells = mul(mul(bucket, 256)?, 6)?;
        let table = mul(add(tasks, 1)?, 4)?;
        let grid_words = add(table, mul(add(tasks, 2)?, 2)?)?;
        let label_words = add(label_cells, mul(add(mul(bucket, 35)?, add(tasks, 1)?)?, 2)?)?;
        let store = &mut core.store;
        Ok(Self {
            pairs: store.words("ambiguity f32 assessment pairs", mul(PAIRS, 16)?)?,
            tasks: store.words("ambiguity f32 distance tasks", mul(tasks, 8)?)?,
            grid: store.words("ambiguity f32 search tables", grid_words)?,
            // A chunk's local span stays below two buckets; boundary weights
            // and two flags per tile follow.
            labels: store.words("ambiguity f32 search labels", label_words)?,
            scratch: store.words(
                "ambiguity f32 search scan sums",
                scan_words(add(tasks, 1)?)?,
            )?,
            limit,
            // Point, chain, synchronized and staircase branches each walk at
            // most the points of both curves.
            task_units: add(mul(longest, 6)?, 8)?,
            bucket,
            label_words,
            statistics: Statistics {
                largest_search_cells_limit: limit,
                ..Statistics::default()
            },
        })
    }

    const fn binds<'a>(&'a self, originals: &'a Originals) -> Binds<'a> {
        [
            (1, &originals.canon),
            (2, &originals.words),
            (3, &originals.routes),
            (4, &self.pairs),
            (5, &self.tasks),
            (6, &self.grid),
            (7, &self.labels),
        ]
    }

    /// Complete the `count` pairs written to `pairs` (representative ordinals in
    /// words 0 and 1): orientation, bounds and flags in words 12..16.
    pub(super) fn assess(
        &mut self,
        core: &mut Core,
        originals: &Originals,
        count: u64,
        (route, endpoint): (f32, f32),
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let pairs = P {
            n: word(count)?,
            x: route,
            y: endpoint,
            ..P::default()
        };
        let tasks = mul(count, 2)?;
        core.run(STAGE, "assess_begin", &self.binds(originals), pairs, count)?;
        self.statistics.assessed_pairs = add(self.statistics.assessed_pairs, count)?;
        // One simplified orientation, both, both full curves, then the chosen
        // full orientation.
        for _ in 0..4 {
            core.runner.reset();
            let mut first = 0;
            while first < tasks {
                let slice = tasks
                    .saturating_sub(first)
                    .min(TASK_SLICE)
                    .min(slice_size(self.task_units));
                core.run_bounded(
                    STAGE,
                    "task_fast",
                    &self.binds(originals),
                    P {
                        n: word(tasks)?,
                        g: word(first)?,
                        c: word(self.limit)?,
                        k: originals.knots,
                        ..P::default()
                    },
                    slice,
                    Bound::Each(self.task_units),
                )?;
                first = add(first, slice)?;
            }
            let status = core.runner.status("route distance tasks")?;
            let mut searches = status_word(&status, S_A);
            if searches > 0 {
                searches = self.frontier(core, originals, tasks, searches)?;
            }
            if searches > 0 {
                self.search(core, originals, tasks, searches)?;
            }
            core.runner.reset();
            core.run(STAGE, "assess_step", &self.binds(originals), pairs, count)?;
            let status = core.runner.status("route distance refinement")?;
            if status_word(&status, S_PENDING) == 0 {
                return Ok(());
            }
        }
        Err(Error::Fault(Fault::Rounds {
            stage: "route distance refinement",
            limit: 4,
        }))
    }

    /// Bounded frontier searches of the pending tasks, tier by tier, as many
    /// at once as the label buffer holds. Returns how many searches are still
    /// pending for the tiled relaxation.
    fn frontier(
        &mut self,
        core: &mut Core,
        originals: &Originals,
        tasks: u64,
        mut pending: u64,
    ) -> Result<u64> {
        let table = add(tasks, 1)?;
        for (seen, heap) in FRONTIER_TIERS {
            if pending == 0 {
                break;
            }
            let slot_words = add(mul(add(u64::from(seen), u64::from(heap))?, 2)?, 4)?;
            let slots = self.label_words.checked_div(slot_words).unwrap_or(0);
            if slots == 0 {
                continue;
            }
            let marks = P {
                n: word(tasks)?,
                ..P::default()
            };
            core.run(STAGE, "grid_marks", &self.binds(originals), marks, table)?;
            core.scan(&self.grid, (0, table), &self.scratch)?;
            let tier = P {
                a: seen,
                b: seen.saturating_mul(3) / 4,
                c: heap,
                d: 32_u32.saturating_sub(seen.trailing_zeros()),
                j: FRONTIER_STEPS,
                k: originals.knots,
                ..marks
            };
            // Every settled cell is popped once and pushes at most four entries.
            let limit = add(
                mul(u64::from(seen), 8)?.div_ceil(u64::from(FRONTIER_STEPS)),
                2,
            )?;
            // One invocation: the table clear, then at most the step budget of
            // pops. Only the group's searches are active in a dispatch.
            let units = add(
                mul(u64::from(FRONTIER_STEPS), POP_UNITS)?,
                add(u64::from(seen), 64)?,
            )?;
            // Room for every idle task invocation and the workgroup padding.
            let idle = mul(add(tasks, 256)?, 2)?;
            let active = WORK_TARGET
                .saturating_sub(idle)
                .checked_div(units)
                .unwrap_or(1)
                .max(1);
            let mut finished = 0;
            let mut first = 0;
            while first < pending {
                let group = pending.saturating_sub(first).min(slots).min(active);
                let bound = Bound::Total {
                    each: units,
                    total: add(mul(group, units)?, tasks)?,
                };
                let mut round = 0;
                loop {
                    core.runner.reset();
                    core.run_bounded(
                        STAGE,
                        "grid_frontier",
                        &self.binds(originals),
                        P {
                            e: word(first)?,
                            g: u32::from(round == 0),
                            h: word(group)?,
                            ..tier
                        },
                        tasks,
                        bound,
                    )?;
                    let status = core.runner.status("route distance frontier searches")?;
                    finished = add(finished, status_word(&status, S_D))?;
                    if status_word(&status, S_PENDING) == 0 {
                        break;
                    }
                    round = add(round, 1)?;
                    if round > limit {
                        return Err(Error::Fault(Fault::Rounds {
                            stage: "route distance frontier searches",
                            limit,
                        }));
                    }
                }
                first = add(first, group)?;
            }
            self.statistics.frontier_searches = add(self.statistics.frontier_searches, finished)?;
            pending = pending.saturating_sub(finished);
        }
        Ok(pending)
    }

    /// List the pending searches and group them into label chunks.
    fn chunks(
        &self,
        core: &mut Core,
        originals: &Originals,
        (tasks, searches): (u64, u64),
        base: P,
    ) -> Result<Vec<Chunk>> {
        let table = add(tasks, 1)?;
        let binds = self.binds(originals);
        let task_n = P {
            n: word(tasks)?,
            d: word(searches)?,
            ..base
        };
        core.run(STAGE, "grid_marks", &binds, task_n, table)?;
        core.scan(&self.grid, (0, table), &self.scratch)?;
        core.run(STAGE, "grid_list", &binds, task_n, table)?;
        let scanned = add(searches, 1)?;
        core.scan(&self.grid, (u64::from(base.b), scanned), &self.scratch)?;
        core.scan(&self.grid, (u64::from(base.c), scanned), &self.scratch)?;
        core.runner.reset();
        let search_n = P {
            n: word(searches)?,
            ..base
        };
        core.run(STAGE, "grid_chunks", &binds, search_n, searches)?;
        let status = core.runner.status("route distance search chunks")?;
        let count = add(status_word(&status, S_B), 1)?;
        let heads = core
            .runner
            .read(&self.grid, u64::from(base.e), mul(count, 2)?)?;
        let total_tiles = core.word(&self.grid, add(u64::from(base.c), searches)?)?;
        let end = [word(searches)?, word(total_tiles)?];
        let mut heads = heads.as_chunks::<2>().0.iter().copied().peekable();
        let mut chunks = core.runner.host.vector(count)?;
        let mut index = 0_u32;
        while let Some([first_search, first_tile]) = heads.next() {
            let [end_search, end_tile] = heads.peek().copied().unwrap_or(end);
            if end_search <= first_search || end_tile <= first_tile {
                return Err(Error::Fault(Fault::Schedule(
                    "route distance search chunks",
                )));
            }
            chunks.push(Chunk {
                index,
                searches: (first_search, end_search),
                tiles: (first_tile, end_tile),
            });
            index = index.checked_add(1).ok_or_else(overflow)?;
        }
        if u64::try_from(chunks.len()).ok() != Some(count) {
            return Err(Error::Protocol(Protocol::Readback));
        }
        Ok(chunks)
    }

    /// Dispatch `entry` over every tile of `chunk` in bounded slices.
    fn tiles(
        &self,
        core: &mut Core,
        originals: &Originals,
        entry: &str,
        chunk: &Chunk,
        registers: P,
    ) -> Result<()> {
        let (first, end) = (u64::from(chunk.tiles.0), u64::from(chunk.tiles.1));
        let mut tile = first;
        while tile < end {
            let slice = end.saturating_sub(tile).min(TILE_SLICE);
            core.run(
                STAGE,
                entry,
                &self.binds(originals),
                P {
                    g: word(tile)?,
                    ..registers
                },
                slice,
            )?;
            tile = add(tile, slice)?;
        }
        Ok(())
    }

    /// Run every pending cell search to its fixpoint, chunk by chunk.
    fn search(
        &mut self,
        core: &mut Core,
        originals: &Originals,
        tasks: u64,
        searches: u64,
    ) -> Result<()> {
        let table = add(tasks, 1)?;
        let base = P {
            a: word(table)?,
            b: word(mul(table, 2)?)?,
            c: word(mul(table, 3)?)?,
            d: word(self.bucket)?,
            e: word(mul(table, 4)?)?,
            k: originals.knots,
            ..P::default()
        };
        let chunks = self.chunks(core, originals, (tasks, searches), base)?;
        for chunk in &chunks {
            let registers = P {
                n: chunk.tiles.1,
                f: chunk.index,
                h: chunk.searches.0,
                i: chunk.searches.1,
                j: SWEEPS,
                ..base
            };
            let search_count = u64::from(chunk.searches.1.saturating_sub(chunk.searches.0));
            let tile_count = chunk.tiles.1.saturating_sub(chunk.tiles.0);
            self.tiles(core, originals, "grid_init", chunk, registers)?;
            let mut rounds = 0;
            loop {
                // Complete rounds over every tile of the chunk. After each
                // round the tiles to sweep next are those marked during it.
                // The status is reset before the last round, so a zero count
                // means one complete round changed nothing anywhere.
                for round in 0..ROUNDS_PER_CHECK {
                    if round.checked_add(1) == Some(ROUNDS_PER_CHECK) {
                        core.runner.reset();
                    }
                    self.tiles(core, originals, "grid_relax", chunk, registers)?;
                    core.run(
                        STAGE,
                        "grid_roll",
                        &self.binds(originals),
                        P {
                            n: tile_count,
                            ..registers
                        },
                        u64::from(tile_count),
                    )?;
                }
                rounds = add(rounds, ROUNDS_PER_CHECK)?;
                let status = core.runner.status("route distance relaxation")?;
                if status_word(&status, S_CHANGED) == 0 {
                    break;
                }
                if rounds >= ROUND_LIMIT {
                    return Err(Error::Fault(Fault::Rounds {
                        stage: "route distance relaxation",
                        limit: ROUND_LIMIT,
                    }));
                }
            }
            self.statistics.relaxation_rounds = add(self.statistics.relaxation_rounds, rounds)?;
            core.run(
                STAGE,
                "grid_finish",
                &self.binds(originals),
                registers,
                search_count,
            )?;
        }
        // A search that ended without a finite label fails here, before the
        // refinement that would consume it is encoded.
        core.runner.status("route distance searches")?;
        self.statistics.cell_searches = add(self.statistics.cell_searches, searches)?;
        self.statistics.cell_search_chunks = add(
            self.statistics.cell_search_chunks,
            u64::try_from(chunks.len()).map_err(|_| overflow())?,
        )?;
        Ok(())
    }
}
