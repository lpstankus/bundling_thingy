//! Frozen originals: host scheduling of the device kernels that copy the
//! baseline points, validate the layout, canonicalize, order and simplify. The
//! host reads only counts.
use super::{
    Config, Core, Error, Geometry, Key, Labels, Preparation, Ranking, Result,
    distance::Assessor,
    runner::{Bound, P, add, bits, mul, scan_words, step_budget, word},
    status_word,
};

/// Largest stored point count of one route; bounds every per-route loop.
pub(super) const ROUTE_LIMIT: u32 = 1 << 15;
/// Words of one device route record (the WGSL `Route`).
const ROUTE_WORDS: u64 = 12;
/// Simplifier steps of one invocation.
const SIMPLIFY_STEPS: u64 = 4096;
const S_PENDING: usize = 1;
const S_A: usize = 3;
const S_B: usize = 4;
const S_C: usize = 5;

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
        "route_mark",
        &[(2, &geometry.paths), (4, marks)],
        P::count(reservations)?,
        add(reservations, 1)?,
    )?;
    let route_count = core.total(marks, reservations, scratch)?;
    if route_count == 0 {
        return Err(Error::Parameters("ambiguity needs an eligible route"));
    }
    let routes = core.words("ambiguity f32 routes", mul(route_count, ROUTE_WORDS)?)?;
    core.store.promote(&routes);
    core.runner.reset();
    core.run_bounded(
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
    tracing::debug!(route_count, "ambiguity route layout validated");
    core.run(
        "point_mark",
        &[(1, frozen), (2, &geometry.paths), (4, marks), (7, ordinals)],
        P {
            n: geometry.samples,
            a: geometry.reservations,
            b: 1,
            ..P::default()
        },
        add(samples, 1)?,
    )?;
    core.runner.status("original points")?;
    let distinct = core.total(marks, samples, scratch)?;
    tracing::debug!(distinct, "ambiguity points deduplicated");
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
        let scratch = core.words("ambiguity f32 scan sums", scan_words(span)?)?;
        let marks = core.words("ambiguity f32 point marks", span)?;
        let (routes, route_count, distinct) =
            layout(core, geometry, (frozen, ordinals), (&marks, &scratch))?;
        let deduplicated = core.words("ambiguity f32 deduplicated points", mul(distinct, 3)?)?;
        core.run(
            "dedup_write",
            &[(1, frozen), (4, &marks), (5, &deduplicated), (7, ordinals)],
            P::count(samples)?,
            samples,
        )?;
        let routes_n = P::count(route_count)?;
        let per_route = [(3, &routes), (4, &marks)];
        core.run("route_dedup", &per_route, routes_n, route_count)?;
        let kept = core.words("ambiguity f32 vertex marks", add(distinct, 1)?)?;
        core.run(
            "collinear_mark",
            &[(3, &routes), (5, &deduplicated), (6, &kept)],
            P::count(distinct)?,
            add(distinct, 1)?,
        )?;
        let points = core.total(&kept, distinct, &scratch)?;
        tracing::debug!(points, "ambiguity collinear vertices removed");
        let unoriented = core.words("ambiguity f32 unoriented points", mul(points, 3)?)?;
        core.run(
            "canon_tmp",
            &[(5, &deduplicated), (6, &kept), (7, &unoriented)],
            P::count(distinct)?,
            distinct,
        )?;
        let ranges = [(3, &routes), (6, &kept)];
        core.run("route_canon", &ranges, routes_n, route_count)?;
        core.runner.reset();
        let oriented = [(3, &routes), (7, &unoriented)];
        // Each route compares its own canonical points from both ends.
        let orient = Bound::Total {
            each: u64::from(ROUTE_LIMIT),
            total: add(points, route_count)?,
        };
        core.run_bounded("route_orient", &oriented, routes_n, route_count, orient)?;
        let status = core.runner.status("canonical orientation")?;
        let canon = core.words("ambiguity f32 canonical points", mul(points, 2)?)?;
        let owners = core.words("ambiguity f32 canonical owners", points)?;
        core.store.promote(&canon);
        core.store.promote(&owners);
        core.run(
            "canon_write",
            &[(3, &routes), (4, &canon), (5, &owners), (7, &unoriented)],
            P::count(points)?,
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

/// Block names of every canonical point's block, level by level; returns the
/// buffer whose word at each route's first point names the whole route.
fn names(
    core: &mut Core,
    canonical: &Canonical,
    naming: &Ranking,
    name_bits: u32,
) -> Result<wgpu::Buffer> {
    let points = canonical.points;
    let [first, _] = &naming.records;
    let mut current = core.words("ambiguity f32 block names", points)?;
    let mut next = core.words("ambiguity f32 block names", points)?;
    let all = P::count(points)?;
    let entries = [(4, &canonical.canon), (5, first)];
    core.run("name_entries0", &entries, all, points)?;
    let mut key_bits = (32, 32);
    let mut level = 0_u32;
    loop {
        let sorted = naming.rank(core, points, key_bits, Key::Both)?;
        tracing::debug!(level, "ambiguity block naming level");
        core.run(
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
        let labels = Labels {
            records: "ambiguity f32 sort records",
            histogram: "ambiguity f32 sort histogram",
            scratch: "ambiguity f32 scan sums",
            marks: "ambiguity f32 key marks",
        };
        let naming = Ranking::new(core, points, &labels)?;
        let name_bits = bits(add(points, 2)?);
        let named = names(core, canonical, &naming, name_bits)?;
        let routes_n = P::count(route_count)?;
        let [first, _] = &naming.records;
        core.run(
            "route_entries",
            &[(3, &canonical.routes), (4, &named), (5, first)],
            routes_n,
            route_count,
        )?;
        let route_keys = (bits(reservations), name_bits);
        let sorted = naming.rank(core, route_count, route_keys, Key::High)?;
        let unique = core.word(&naming.marks, route_count)?;
        let words = core.words(
            "ambiguity f32 order, representatives and knots",
            add(mul(route_count, 2)?, points)?,
        )?;
        core.store.promote(&words);
        let placed = P {
            b: word(route_count)?,
            ..routes_n
        };
        core.run(
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
        core.run("route_rep", &representatives, placed, route_count)?;
        Ok((words, unique))
    })
}

/// Simplified knots of every unique route.
fn simplify(core: &mut Core, originals: &Originals, unique: u64, sigma: f32) -> Result<()> {
    core.scoped(|core| {
        let state = core.words("ambiguity f32 simplifier state", mul(unique, 4)?)?;
        // Every route advances at most `steps` vertex examinations per dispatch.
        let steps = step_budget(unique, 1).min(SIMPLIFY_STEPS);
        // A resource limit, not a proven bound: 16 examinations per stored
        // point cover every route measured so far; exceeding it is an error.
        let limit = add(mul(u64::from(ROUTE_LIMIT), 16)?.div_ceil(steps), 8)?;
        let rounds = ("original simplification", limit, limit);
        let step = |core: &mut Core, round| {
            core.runner.reset();
            core.run_bounded(
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
            )
        };
        core.settle(rounds, step, |status| status_word(status, S_PENDING) == 0)?;
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
            "ambiguity geometry buffers are smaller than their counts",
        ));
    }
    let frozen = core.words("ambiguity f32 frozen baseline points", mul(samples, 3)?)?;
    // An actual device copy: later mutation of the renderer's points cannot
    // reach the originals.
    core.runner
        .copy(&geometry.points, &frozen, mul(samples, 12)?);
    let ordinals = core.words("ambiguity f32 route ordinals", reservations.max(1))?;
    // The canonical owners are only needed for naming.
    let (canonical, words, unique) = core.scoped(|core| {
        let canonical = canonicalize(core, geometry, &frozen, &ordinals)?;
        tracing::info!(
            points = canonical.points,
            longest = canonical.longest,
            "ambiguity originals canonical"
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
        "ambiguity originals ordered"
    );
    let assessor = Assessor::new(core, canonical.longest, config.search_cells)?;
    let originals = Originals {
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
    };
    simplify(core, &originals, unique, config.sigma)?;
    tracing::info!("ambiguity originals simplified");
    Ok((originals, assessor))
}
