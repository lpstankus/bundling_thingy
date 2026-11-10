//! Frozen originals of the gpu-f32 backend: host scheduling of the device
//! kernels that copy the baseline points, validate the layout, canonicalize,
//! order, simplify and infer complete-link cohorts. The host reads only counts
//! and the final membership table.
use super::super::{Error, Geometry, Result};
use super::{
    Config, Core, Fault, Preparation, Stage,
    distance::{Assessor, PAIRS},
    runner::{Bound, P, add, bits, histogram_words, mul, scan_words, step_budget, word},
    status_word,
};

/// Largest stored point count of one route; bounds every per-route loop.
pub(super) const ROUTE_LIMIT: u32 = 1 << 15;
/// Earlier unique routes tested per endpoint dispatch.
const PARTNERS: u64 = 1024;
/// Simplifier steps of one invocation.
const SIMPLIFY_STEPS: u64 = 4096;
/// List entries or routes visited by one first-fit invocation.
const FIT_STEPS: u32 = 1 << 16;
const S_PENDING: usize = 1;
const S_A: usize = 3;
const S_B: usize = 4;
const S_C: usize = 5;
const STAGE: Stage = Stage::Originals;

/// Device-resident frozen originals.
pub(super) struct Originals {
    /// Baseline copy of the resident points taken before any mutation.
    pub frozen: wgpu::Buffer,
    /// Ordinal + 1 by reservation ID.
    pub ordinals: wgpu::Buffer,
    pub routes: wgpu::Buffer,
    pub canon: wgpu::Buffer,
    /// Canonical order, unique representatives and simplified knots.
    pub words: wgpu::Buffer,
    pub route_count: u32,
    /// Word offset of the knots in `words`.
    pub knots: u32,
    pub preparation: Preparation,
    /// (reservation ID, cohort) in canonical order.
    pub memberships: Vec<[u32; 2]>,
    /// Eligible reservation IDs in increasing order (ordinal order).
    pub ids: Vec<u32>,
}

fn count(n: u64) -> Result<P> {
    Ok(P {
        n: word(n)?,
        ..P::default()
    })
}

struct Canonical {
    routes: wgpu::Buffer,
    canon: wgpu::Buffer,
    owners: wgpu::Buffer,
    route_count: u64,
    points: u64,
    longest: u64,
    closed: u64,
    constants: u64,
}

/// Eligible routes with their validated ranges; returns the route records and
/// the keep marks of the deduplicated points, scanned.
fn layout(
    core: &mut Core,
    geometry: &Geometry,
    (frozen, ordinals): (&wgpu::Buffer, &wgpu::Buffer),
    (marks, scratch): (&wgpu::Buffer, &wgpu::Buffer),
) -> Result<(wgpu::Buffer, u64, u64)> {
    let samples = u64::from(geometry.samples);
    let reservations = u64::from(geometry.reservations);
    core.run(
        STAGE,
        "route_mark",
        &[(2, &geometry.paths), (4, marks)],
        count(reservations)?,
        add(reservations, 1)?,
    )?;
    let route_count = core.total(marks, reservations, scratch)?;
    if route_count == 0 {
        return Err(Error::Parameters(
            "gpu-f32 ambiguity needs an eligible route",
        ));
    }
    let routes = core
        .store
        .words("ambiguity f32 routes", mul(route_count, 16)?)?;
    core.store.promote(&routes);
    core.runner.reset();
    core.run_bounded(
        STAGE,
        "route_init",
        &[
            (1, frozen),
            (2, &geometry.paths),
            (3, &routes),
            (4, marks),
            (7, ordinals),
        ],
        P {
            n: geometry.reservations,
            b: geometry.samples,
            c: ROUTE_LIMIT,
            ..P::default()
        },
        reservations,
        // Each route checks its own points; together they are the active points.
        Bound::Total {
            each: u64::from(ROUTE_LIMIT),
            total: add(samples, reservations)?,
        },
    )?;
    // Dependent kernels index through these ranges; stop here on a fault.
    core.runner.status("original route layout")?;
    tracing::debug!(route_count, "gpu-f32 route layout validated");
    core.run(
        STAGE,
        "point_mark",
        &[(1, frozen), (2, &geometry.paths), (4, marks), (7, ordinals)],
        P {
            n: geometry.samples,
            a: geometry.reservations,
            ..P::default()
        },
        add(samples, 1)?,
    )?;
    core.runner.status("original points")?;
    let distinct = core.total(marks, samples, scratch)?;
    tracing::debug!(distinct, "gpu-f32 points deduplicated");
    Ok((routes, route_count, distinct))
}

/// Eligible routes, layout validation and canonical points.
fn canonicalize(
    core: &mut Core,
    geometry: &Geometry,
    frozen: &wgpu::Buffer,
    ordinals: &wgpu::Buffer,
) -> Result<Canonical> {
    let samples = u64::from(geometry.samples);
    let span = add(samples.max(u64::from(geometry.reservations)), 1)?;
    core.scoped(|core| {
        let scratch = core
            .store
            .words("ambiguity f32 scan sums", scan_words(span)?)?;
        let marks = core.store.words("ambiguity f32 point marks", span)?;
        let (routes, route_count, distinct) =
            layout(core, geometry, (frozen, ordinals), (&marks, &scratch))?;
        let deduplicated = core
            .store
            .words("ambiguity f32 deduplicated points", mul(distinct, 3)?)?;
        core.run(
            STAGE,
            "dedup_write",
            &[(1, frozen), (4, &marks), (5, &deduplicated), (7, ordinals)],
            count(samples)?,
            samples,
        )?;
        let routes_n = count(route_count)?;
        let per_route = [(3, &routes), (4, &marks)];
        core.run(STAGE, "route_dedup", &per_route, routes_n, route_count)?;
        let kept = core
            .store
            .words("ambiguity f32 vertex marks", add(distinct, 1)?)?;
        core.run(
            STAGE,
            "collinear_mark",
            &[(3, &routes), (5, &deduplicated), (6, &kept)],
            count(distinct)?,
            add(distinct, 1)?,
        )?;
        let points = core.total(&kept, distinct, &scratch)?;
        tracing::debug!(points, "gpu-f32 collinear vertices removed");
        let unoriented = core
            .store
            .words("ambiguity f32 unoriented points", mul(points, 3)?)?;
        core.run(
            STAGE,
            "canon_tmp",
            &[(5, &deduplicated), (6, &kept), (7, &unoriented)],
            count(distinct)?,
            distinct,
        )?;
        let ranges = [(3, &routes), (6, &kept)];
        core.run(STAGE, "route_canon", &ranges, routes_n, route_count)?;
        core.runner.reset();
        let oriented = [(3, &routes), (7, &unoriented)];
        // Each route compares its own canonical points from both ends.
        let orient = Bound::Total {
            each: u64::from(ROUTE_LIMIT),
            total: add(points, route_count)?,
        };
        core.run_bounded(
            STAGE,
            "route_orient",
            &oriented,
            routes_n,
            route_count,
            orient,
        )?;
        let status = core.runner.status("canonical orientation")?;
        let canon = core
            .store
            .words("ambiguity f32 canonical points", mul(points, 2)?)?;
        let owners = core.store.words("ambiguity f32 canonical owners", points)?;
        core.store.promote(&canon);
        core.store.promote(&owners);
        core.run(
            STAGE,
            "canon_write",
            &[(3, &routes), (4, &canon), (5, &owners), (7, &unoriented)],
            count(points)?,
            points,
        )?;
        Ok(Canonical {
            routes,
            canon,
            owners,
            route_count,
            points,
            longest: status_word(&status, S_C),
            closed: status_word(&status, S_A),
            constants: status_word(&status, S_B),
        })
    })
}

/// Sort workspace of the naming passes.
struct Naming {
    records: [wgpu::Buffer; 2],
    sorter: [wgpu::Buffer; 2],
    marks: wgpu::Buffer,
}
impl Naming {
    /// Sort `count` records, then mark and scan the key changes. Returns the
    /// sorted records.
    fn rank(
        &self,
        core: &mut Core,
        n: u64,
        key_bits: (u32, u32),
        high_only: bool,
    ) -> Result<&wgpu::Buffer> {
        let [first, second] = &self.records;
        let [histogram, scratch] = &self.sorter;
        let (sorted, _) = core.sort((first, second), (histogram, scratch), n, key_bits)?;
        core.run(
            STAGE,
            "key_marks",
            &[(4, sorted), (5, &self.marks)],
            P {
                a: u32::from(high_only),
                ..count(n)?
            },
            add(n, 1)?,
        )?;
        core.scan(&self.marks, (0, add(n, 1)?), scratch)?;
        Ok(sorted)
    }
}

/// Block names of every canonical point's block, level by level; returns the
/// buffer whose word at each route's first point names the whole route.
fn names(
    core: &mut Core,
    canonical: &Canonical,
    naming: &Naming,
    name_bits: u32,
) -> Result<wgpu::Buffer> {
    let points = canonical.points;
    let [first, _] = &naming.records;
    let mut current = core.store.words("ambiguity f32 block names", points)?;
    let mut next = core.store.words("ambiguity f32 block names", points)?;
    let all = count(points)?;
    let entries = [(4, &canonical.canon), (5, first)];
    core.run(STAGE, "name_entries0", &entries, all, points)?;
    let mut key_bits = (32, 32);
    let mut level = 0_u32;
    loop {
        let sorted = naming.rank(core, points, key_bits, false)?;
        tracing::debug!(level, "gpu-f32 block naming level");
        core.run(
            STAGE,
            "name_write",
            &[(4, &naming.marks), (5, sorted), (6, &current)],
            all,
            points,
        )?;
        if 1_u64.checked_shl(level).unwrap_or(u64::MAX) >= canonical.longest {
            return Ok(current);
        }
        level = level.saturating_add(1);
        core.run(
            STAGE,
            "name_entries",
            &[
                (3, &canonical.routes),
                (4, &current),
                (5, &canonical.owners),
                (6, first),
            ],
            P { a: level, ..all },
            points,
        )?;
        std::mem::swap(&mut current, &mut next);
        key_bits = (name_bits, name_bits);
    }
}

/// Canonical order and identity by block naming; returns the persistent words
/// (order, representatives, knot space) and the unique route count.
fn order(core: &mut Core, canonical: &Canonical, reservations: u64) -> Result<(wgpu::Buffer, u64)> {
    let (points, route_count) = (canonical.points, canonical.route_count);
    core.scoped(|core| {
        let records = |core: &mut Core| {
            core.store
                .words("ambiguity f32 sort records", mul(points, 3)?)
        };
        let naming = Naming {
            records: [records(core)?, records(core)?],
            sorter: [
                core.store
                    .words("ambiguity f32 sort histogram", histogram_words(points)?)?,
                core.store.words(
                    "ambiguity f32 scan sums",
                    scan_words(add(points, 1)?.max(histogram_words(points)?))?,
                )?,
            ],
            marks: core
                .store
                .words("ambiguity f32 key marks", add(points, 1)?)?,
        };
        let name_bits = bits(add(points, 2)?);
        let named = names(core, canonical, &naming, name_bits)?;
        let routes_n = count(route_count)?;
        let [first, _] = &naming.records;
        core.run(
            STAGE,
            "route_entries",
            &[(3, &canonical.routes), (4, &named), (5, first)],
            routes_n,
            route_count,
        )?;
        let sorted = naming.rank(core, route_count, (bits(reservations), name_bits), true)?;
        let unique = core.word(&naming.marks, route_count)?;
        let words = core.store.words(
            "ambiguity f32 order, representatives and knots",
            add(mul(route_count, 2)?, points)?,
        )?;
        core.store.promote(&words);
        let placed = P {
            b: word(route_count)?,
            ..routes_n
        };
        core.run(
            STAGE,
            "route_place",
            &[
                (3, &canonical.routes),
                (4, &naming.marks),
                (5, sorted),
                (6, &words),
            ],
            placed,
            route_count,
        )?;
        let representatives = [(3, &canonical.routes), (6, &words)];
        core.run(STAGE, "route_rep", &representatives, placed, route_count)?;
        Ok((words, unique))
    })
}

/// Simplified knots of every unique route.
fn simplify(core: &mut Core, originals: &Originals, unique: u64, sigma: f32) -> Result<()> {
    core.scoped(|core| {
        let state = core
            .store
            .words("ambiguity f32 simplifier state", mul(unique, 4)?)?;
        // Every route advances at most `steps` vertex examinations per dispatch.
        let steps = step_budget(unique, 1).min(SIMPLIFY_STEPS);
        // A resource limit, not a proven bound: 16 examinations per stored
        // point cover every route measured so far; exceeding it is an error.
        let limit = add(mul(u64::from(ROUTE_LIMIT), 16)?.div_ceil(steps), 8)?;
        for round in 0..limit {
            core.runner.reset();
            core.run_bounded(
                STAGE,
                "simplify_step",
                &[
                    (3, &originals.routes),
                    (4, &originals.canon),
                    (5, &state),
                    (6, &originals.words),
                ],
                P {
                    n: word(unique)?,
                    b: originals.route_count,
                    c: originals.knots,
                    d: word(steps)?,
                    e: u32::from(round == 0),
                    x: sigma,
                    ..P::default()
                },
                unique,
                Bound::Each(add(steps, 1)?),
            )?;
            let status = core.runner.status("original simplification")?;
            if status_word(&status, S_PENDING) == 0 {
                return Ok(());
            }
        }
        Err(Error::Fault(Fault::Rounds {
            stage: "original simplification",
            limit,
        }))
    })
}

/// Endpoint-compatible earlier unique routes of each unique route.
struct Candidates {
    /// Scanned counts by unique route.
    offsets: wgpu::Buffer,
    /// Earlier unique route per entry; the top bit marks incompatibility.
    list: wgpu::Buffer,
    total: u64,
}

fn endpoint_pass(
    core: &mut Core,
    originals: &Originals,
    entry: &str,
    (offsets, list): (&wgpu::Buffer, Option<&wgpu::Buffer>),
    (unique, tolerance): (u64, f32),
) -> Result<()> {
    // Every route tests at most `width` earlier routes per dispatch.
    let width = step_budget(unique, 1).min(PARTNERS);
    let mut first = 0;
    while first < unique {
        let end = add(first, width)?.min(unique);
        let mut binds = vec![
            (3, &originals.routes),
            (4, &originals.canon),
            (5, offsets),
            (6, &originals.words),
        ];
        binds.extend(list.map(|list| (7, list)));
        core.run_bounded(
            STAGE,
            entry,
            &binds,
            P {
                n: word(unique)?,
                a: word(first)?,
                b: word(end)?,
                c: originals.route_count,
                x: tolerance,
                ..P::default()
            },
            unique,
            Bound::Each(add(width, 1)?),
        )?;
        first = end;
    }
    Ok(())
}

/// Enumerate and assess every endpoint-compatible earlier unique route.
fn candidates(
    core: &mut Core,
    originals: &Originals,
    assessor: &mut Assessor,
    scratch: &wgpu::Buffer,
    (unique, config): (u64, &Config),
) -> Result<Candidates> {
    // Unique indices share a word with the incompatibility bit.
    if unique >= 1 << 31 {
        return Err(Error::SizeOverflow);
    }
    let offsets = core
        .store
        .words("ambiguity f32 endpoint candidate offsets", add(unique, 1)?)?;
    let scale = (unique, config.endpoint_tolerance);
    endpoint_pass(core, originals, "endpoint_count", (&offsets, None), scale)?;
    let total = core.count("endpoint candidate count", &offsets, unique, scratch)?;
    tracing::info!(candidates = total, "gpu-f32 endpoint candidates counted");
    let list = core
        .store
        .words("ambiguity f32 endpoint candidates", total.max(1))?;
    endpoint_pass(
        core,
        originals,
        "endpoint_fill",
        (&offsets, Some(&list)),
        scale,
    )?;
    let mut first = 0;
    while first < total {
        let chunk = total.saturating_sub(first).min(PAIRS);
        let registers = P {
            n: word(chunk)?,
            a: word(first)?,
            b: word(unique)?,
            c: originals.route_count,
            ..P::default()
        };
        let pairs = assessor.pairs.clone();
        core.run(
            STAGE,
            "cohort_pairs",
            &[
                (4, &list),
                (5, &offsets),
                (6, &originals.words),
                (7, &pairs),
            ],
            registers,
            chunk,
        )?;
        assessor.assess(
            core,
            originals,
            chunk,
            (config.route_tolerance, config.endpoint_tolerance),
        )?;
        let collected = [(4, &list), (7, &pairs)];
        core.run(STAGE, "cohort_collect", &collected, registers, chunk)?;
        core.runner.status("cohort assessments")?;
        first = add(first, chunk)?;
    }
    Ok(Candidates {
        offsets,
        list,
        total,
    })
}

/// Bounded first-fit invocations until every unique route is assigned.
fn first_fit(
    core: &mut Core,
    candidates: &Candidates,
    state: &wgpu::Buffer,
    unique: u64,
) -> Result<()> {
    let visits = add(mul(candidates.total, 3)?, mul(unique, 3)?)?;
    let limit = add(
        visits.checked_div(u64::from(FIT_STEPS)).unwrap_or(visits),
        2,
    )?;
    let mut round = 0;
    loop {
        core.runner.reset();
        core.run_once(
            STAGE,
            "first_fit",
            &[(4, &candidates.list), (5, &candidates.offsets), (6, state)],
            P {
                n: word(unique)?,
                c: FIT_STEPS,
                e: u32::from(round == 0),
                ..P::default()
            },
            // Each visit or transition is one step; transitions are at most
            // three per charged visit.
            Bound::Each(mul(u64::from(FIT_STEPS), 4)?),
        )?;
        let status = core.runner.status("cohort assignment")?;
        if status_word(&status, S_A) == unique {
            return Ok(());
        }
        round = add(round, 1)?;
        if round > limit {
            return Err(Error::Fault(Fault::Rounds {
                stage: "cohort assignment",
                limit,
            }));
        }
    }
}

/// First-fit complete-link cohorts over the unique routes, then memberships.
fn cohorts(
    core: &mut Core,
    originals: &mut Originals,
    assessor: &mut Assessor,
    unique: u64,
    config: &Config,
) -> Result<()> {
    let route_count = u64::from(originals.route_count);
    core.scoped(|core| {
        let scratch = core
            .store
            .words("ambiguity f32 scan sums", scan_words(add(unique, 1)?)?)?;
        let candidates = candidates(core, originals, assessor, &scratch, (unique, config))?;
        let state = core
            .store
            .words("ambiguity f32 cohort state", add(mul(unique, 3)?, 4)?)?;
        first_fit(core, &candidates, &state, unique)?;
        let flags = core
            .store
            .words("ambiguity f32 cohort marks", add(unique, 1)?)?;
        core.run(
            STAGE,
            "cohort_flags",
            &[(6, &state), (7, &flags)],
            count(unique)?,
            add(unique, 1)?,
        )?;
        let cohort_count = core.total(&flags, unique, &scratch)?;
        let routes_n = count(route_count)?;
        core.run(
            STAGE,
            "cohort_write",
            &[(3, &originals.routes), (6, &state), (7, &flags)],
            routes_n,
            route_count,
        )?;
        let table = core
            .store
            .words("ambiguity f32 memberships", mul(route_count, 2)?)?;
        core.run(
            STAGE,
            "membership",
            &[(3, &originals.routes), (6, &originals.words), (7, &table)],
            routes_n,
            route_count,
        )?;
        let words = core.runner.read(&table, 0, mul(route_count, 2)?)?;
        let mut memberships = core.runner.host.vector(route_count)?;
        memberships.extend(words.as_chunks::<2>().0.iter().copied());
        let mut ids = core.runner.host.vector(route_count)?;
        ids.extend(memberships.iter().map(|[id, _]: &[u32; 2]| *id));
        ids.sort_unstable();
        originals.memberships = memberships;
        originals.ids = ids;
        originals.preparation.cohorts = cohort_count;
        originals.preparation.endpoint_candidate_pairs = candidates.total;
        Ok(())
    })
}

/// Freeze the baseline points and prepare every original on the device.
pub(super) fn prepare(
    core: &mut Core,
    geometry: &Geometry,
    config: &Config,
) -> Result<(Originals, Assessor)> {
    let samples = u64::from(geometry.samples);
    let reservations = u64::from(geometry.reservations);
    if geometry.points.size() < mul(samples, 12)? || geometry.paths.size() < mul(reservations, 12)?
    {
        return Err(Error::Parameters(
            "gpu-f32 ambiguity geometry buffers are smaller than their counts",
        ));
    }
    let frozen = core
        .store
        .words("ambiguity f32 frozen baseline points", mul(samples, 3)?)?;
    // An actual device copy: later mutation of the renderer's points cannot
    // reach the originals.
    core.runner
        .copy(&geometry.points, &frozen, mul(samples, 12)?);
    let ordinals = core
        .store
        .words("ambiguity f32 route ordinals", reservations.max(1))?;
    // The canonical owners are only needed for naming.
    let (canonical, words, unique) = core.scoped(|core| {
        let canonical = canonicalize(core, geometry, &frozen, &ordinals)?;
        tracing::info!(
            points = canonical.points,
            longest = canonical.longest,
            "gpu-f32 originals canonical"
        );
        let (words, unique) = order(core, &canonical, reservations)?;
        for buffer in [&canonical.routes, &canonical.canon, &words] {
            core.store.promote(buffer);
        }
        Ok((canonical, words, unique))
    })?;
    tracing::info!(
        routes = canonical.route_count,
        canonical_points = canonical.points,
        unique,
        "gpu-f32 originals ordered"
    );
    let mut assessor = Assessor::new(core, canonical.longest, config.search_cells)?;
    let mut originals = Originals {
        frozen,
        ordinals,
        routes: canonical.routes,
        canon: canonical.canon,
        words,
        route_count: word(canonical.route_count)?,
        knots: word(mul(canonical.route_count, 2)?)?,
        preparation: Preparation {
            eligible_routes: canonical.route_count,
            skipped_reservations: reservations.saturating_sub(canonical.route_count),
            canonical_points: canonical.points,
            longest_canonical_route: canonical.longest,
            closed_stored_start_routes: canonical.closed,
            original_constants: canonical.constants,
            unique_canonical_routes: unique,
            ..Preparation::default()
        },
        memberships: Vec::new(),
        ids: Vec::new(),
    };
    simplify(core, &originals, unique, config.sigma)?;
    tracing::info!("gpu-f32 originals simplified");
    cohorts(core, &mut originals, &mut assessor, unique, config)?;
    // The membership and ID tables outlive the scope that read them.
    core.runner
        .host
        .reserve(mul(u64::from(originals.route_count), 12)?)?;
    tracing::info!(
        cohorts = originals.preparation.cohorts,
        endpoint_candidates = originals.preparation.endpoint_candidate_pairs,
        "gpu-f32 cohorts inferred"
    );
    Ok((originals, assessor))
}
