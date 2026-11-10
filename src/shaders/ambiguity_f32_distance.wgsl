// Whole-route original distance of the gpu-f32 ambiguity backend: continuous
// endpoint-constrained weak Frechet distance (f32 point estimates), coherent
// endpoint-aware orientation, and simplification bounds with full-curve
// refinement.
//
// A pair holds the later canonical route first, as cohort inference and scoring
// both do. Each orientation is one task over simplified knots or full canonical
// points. The fast branches are the reference's: point curve, single segments,
// chain, synchronized vertices and staircase. Other tasks need the minimax cell
// search. A bounded frontier search settles cells in increasing bottleneck
// order and stops at the final cell, in resumable steps. Tasks beyond its
// capacity run asynchronous label relaxation over tiles: labels are bit
// patterns of nonnegative squared distances lowered only by atomicMin, so every
// label stays the bottleneck of an actual path and the final cell's label at
// the fixpoint is the f32 minimax of the f32 boundary weights. Relaxation ends
// when a complete round changes nothing.

@group(0) @binding(1) var<storage, read> cn: array<u32>;
@group(0) @binding(2) var<storage, read> pw: array<u32>;
@group(0) @binding(3) var<storage, read_write> routes: array<Route>;
@group(0) @binding(4) var<storage, read_write> ap: array<u32>;
@group(0) @binding(5) var<storage, read_write> tk: array<u32>;
@group(0) @binding(6) var<storage, read_write> gw: array<u32>;
@group(0) @binding(7) var<storage, read_write> lb: array<atomic<u32>>;

const TILE: u32 = 32u;
const SIMPLIFIED: u32 = 1u;
const FULL: u32 = 2u;
const PENDING: u32 = 0u;
const DONE: u32 = 1u;
const GRID: u32 = 2u;
// Pair words.
const A_STAGE: u32 = 2u;
// Endpoint distance of the orientation not assessed yet.
const A_OTHER: u32 = 3u;
const STAGE_BOTH: u32 = 0u;
const STAGE_FULL: u32 = 1u;
const STAGE_CHOSEN: u32 = 2u;
const STAGE_FIRST: u32 = 3u;
const A_OUT: u32 = 12u;
const OUT_REVERSED: u32 = 1u;
const OUT_GENERAL: u32 = 2u;
const OUT_ENDPOINT: u32 = 4u;
const OUT_DONE: u32 = 8u;

fn canon(index: u32) -> vec2<f32> {
    return vec2<f32>(f(cn[2u * index]), f(cn[2u * index + 1u]));
}

// p.n pairs. Starts the simplified orientation with the smaller endpoint
// distance (forward on a tie); the other is assessed only if it could win.
@compute @workgroup_size(WG)
fn assess_begin(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let t = index_of(id, groups);
    if t >= p.n {
        return;
    }
    let pair = 16u * t;
    for (var k = 2u; k < 16u; k += 1u) {
        ap[pair + k] = 0u;
    }
    let a = routes[ap[pair]];
    let b = routes[ap[pair + 1u]];
    let a0 = canon(a.canon_start);
    let a1 = canon(a.canon_start + a.canon_count - 1u);
    let b0 = canon(b.canon_start);
    let b1 = canon(b.canon_start + b.canon_count - 1u);
    let forward = sqrt(max(sq(a0, b0), sq(a1, b1)));
    let reverse = sqrt(max(sq(a0, b1), sq(a1, b0)));
    let first = select(0u, 1u, reverse < forward);
    tk[8u * (2u * t + first)] = SIMPLIFIED;
    tk[8u * (2u * t + first) + 1u] = PENDING;
    tk[8u * (2u * t + 1u - first)] = 0u;
    ap[pair + A_STAGE] = STAGE_FIRST;
    ap[pair + A_OTHER] = bitcast<u32>(max(forward, reverse));
    ap[pair + A_OUT + 3u] = select(0u, OUT_REVERSED, first == 1u);
}

struct Curve { base: u32, last: u32, knots: u32, reversed: bool }
fn curve_of(route: Route, kind: u32, reversed: bool) -> Curve {
    if kind == SIMPLIFIED {
        return Curve(route.canon_start, route.knot_count - 1u, p.k + route.canon_start, reversed);
    }
    return Curve(route.canon_start, route.canon_count - 1u, NONE, reversed);
}
fn at(c: Curve, k: u32) -> vec2<f32> {
    var index = select(k, c.last - k, c.reversed);
    if c.knots != NONE {
        index = pw[c.knots + index];
    }
    let word = 2u * (c.base + index);
    return vec2<f32>(f(cn[word]), f(cn[word + 1u]));
}
struct Task { a: Curve, b: Curve, m: u32, n: u32 }
fn task_of(index: u32) -> Task {
    let pair = 16u * (index / 2u);
    let kind = tk[8u * index];
    let a = curve_of(routes[ap[pair]], kind, false);
    let b = curve_of(routes[ap[pair + 1u]], kind, (index & 1u) == 1u);
    return Task(a, b, a.last, b.last);
}

// The reference's fast branches for tasks [p.g, p.n). p.c: largest admitted
// cell count of one search; p.k: knot offset. Loops are bounded by the route
// limit. status: S_A tasks that need the cell search.
@compute @workgroup_size(WG)
fn task_fast(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = p.g + index_of(id, groups);
    if index >= p.n || tk[8u * index] == 0u || tk[8u * index + 1u] != PENDING {
        return;
    }
    let t = task_of(index);
    let endpoints = max(sq(at(t.a, 0u), at(t.b, 0u)), sq(at(t.a, t.m), at(t.b, t.n)));
    var cost = endpoints;
    var state = DONE;
    if t.m == 0u {
        cost = 0.0;
        for (var k = 0u; k <= t.n; k += 1u) {
            cost = max(cost, sq(at(t.a, 0u), at(t.b, k)));
        }
    } else if t.n == 0u {
        cost = 0.0;
        for (var k = 0u; k <= t.m; k += 1u) {
            cost = max(cost, sq(at(t.b, 0u), at(t.a, k)));
        }
    } else if t.m == 1u && t.n > 1u {
        for (var k = 1u; k < t.n; k += 1u) {
            cost = max(cost, boundary(at(t.b, k), at(t.a, 0u), at(t.a, 1u)));
        }
    } else if t.n == 1u && t.m > 1u {
        for (var k = 1u; k < t.m; k += 1u) {
            cost = max(cost, boundary(at(t.a, k), at(t.b, 0u), at(t.b, 1u)));
        }
    } else if t.m > 1u && t.n > 1u {
        var certified = false;
        if t.m == t.n {
            var upper = 0.0;
            for (var k = 0u; k <= t.m; k += 1u) {
                upper = max(upper, sq(at(t.a, k), at(t.b, k)));
            }
            certified = upper <= endpoints;
        }
        if !certified {
            // Each staircase step crosses one shared cell boundary.
            certified = true;
            var row = 0u;
            var column = 0u;
            while row + 1u < t.m || column + 1u < t.n {
                let advance_row = row + 1u < t.m
                    && (column + 1u == t.n || (row + 1u) * t.n <= (column + 1u) * t.m);
                var weight: f32;
                if advance_row {
                    weight = boundary(at(t.a, row + 1u), at(t.b, column), at(t.b, column + 1u));
                    row += 1u;
                } else {
                    weight = boundary(at(t.b, column + 1u), at(t.a, row), at(t.a, row + 1u));
                    column += 1u;
                }
                if weight > endpoints {
                    certified = false;
                    break;
                }
            }
        }
        if !certified {
            // p.c is the requested limit itself, not a padded capacity.
            if t.m > p.c / t.n {
                fail(E_GRID);
                return;
            }
            state = GRID;
            atomicAdd(&status[S_A], 1u);
        }
    }
    let task = 8u * index;
    tk[task + 1u] = state;
    tk[task + 2u] = bitcast<u32>(cost);
    tk[task + 3u] = bitcast<u32>(endpoints);
    tk[task + 4u] = t.m;
    tk[task + 5u] = t.n;
}

// ---- bounded frontier search. One invocation settles cells in increasing
// bottleneck order, exactly as the reference does, and stops at the final cell;
// most searches settle a narrow band of the grid. Each search owns a slot of lb:
// an open-addressing table of seen cells (cell + 1 with the settled bit, best
// label) and a binary heap of (label, cell). A search that would exceed either
// stays pending for a larger tier or the tiled relaxation below, which has no
// such limit. Work per pop is bounded: table lookups probe at most PROBE slots
// (a longer probe also leaves the tier) and the heap depth is at most 13. p.a: table slots (a power of two, with p.d = 32 - log2 of it);
// p.b: most seen cells; p.c: heap entries.
const SETTLED: u32 = 0x80000000u;

// Table slot of `cell`: its own or the empty one where it belongs. NONE when
// neither is found within PROBE slots; the search then leaves this tier.
const PROBE: u32 = 64u;
fn seen_find(base: u32, cell: u32) -> u32 {
    var slot = (cell * 0x9e3779b1u) >> p.d;
    for (var probe = 0u; probe < PROBE; probe += 1u) {
        let key = atomicLoad(&lb[base + 2u * slot]) & ~SETTLED;
        if key == 0u || key == cell + 1u {
            return slot;
        }
        slot = (slot + 1u) & (p.a - 1u);
    }
    return NONE;
}
// Heap order: smaller label first; on equal labels the later cell first, which
// carries a plateau toward the final cell as the reference's order does.
fn precedes_entry(label: u32, cell: u32, other_label: u32, other_cell: u32) -> bool {
    return label < other_label || (label == other_label && cell > other_cell);
}
fn heap_push(base: u32, length: u32, label: u32, cell: u32) {
    let heap = base + 2u * p.a;
    var index = length;
    while index > 0u {
        let parent = (index - 1u) / 2u;
        let other_label = atomicLoad(&lb[heap + 2u * parent]);
        let other_cell = atomicLoad(&lb[heap + 2u * parent + 1u]);
        if !precedes_entry(label, cell, other_label, other_cell) {
            break;
        }
        atomicStore(&lb[heap + 2u * index], other_label);
        atomicStore(&lb[heap + 2u * index + 1u], other_cell);
        index = parent;
    }
    atomicStore(&lb[heap + 2u * index], label);
    atomicStore(&lb[heap + 2u * index + 1u], cell);
}
// Removes the root of a heap of `length` entries (length > 0).
fn heap_pop(base: u32, length: u32) {
    let heap = base + 2u * p.a;
    let last = length - 1u;
    let label = atomicLoad(&lb[heap + 2u * last]);
    let cell = atomicLoad(&lb[heap + 2u * last + 1u]);
    var index = 0u;
    loop {
        var child = 2u * index + 1u;
        if child >= last {
            break;
        }
        var least_label = atomicLoad(&lb[heap + 2u * child]);
        var least_cell = atomicLoad(&lb[heap + 2u * child + 1u]);
        if child + 1u < last {
            let right_label = atomicLoad(&lb[heap + 2u * child + 2u]);
            let right_cell = atomicLoad(&lb[heap + 2u * child + 3u]);
            if precedes_entry(right_label, right_cell, least_label, least_cell) {
                least_label = right_label;
                least_cell = right_cell;
                child += 1u;
            }
        }
        if !precedes_entry(least_label, least_cell, label, cell) {
            break;
        }
        atomicStore(&lb[heap + 2u * index], least_label);
        atomicStore(&lb[heap + 2u * index + 1u], least_cell);
        index = child;
    }
    atomicStore(&lb[heap + 2u * index], label);
    atomicStore(&lb[heap + 2u * index + 1u], cell);
}

// Pending searches with rank [p.e, p.e + p.h) among the p.n tasks (gw: scanned
// search marks); slot = rank - p.e. Resumable: one invocation pops at most p.j
// entries and keeps its state (1 running, heap length, seen cells) in the four
// words before its table; p.g = 1 starts. status: S_D searches finished here,
// S_PENDING searches to continue.
@compute @workgroup_size(WG)
fn grid_frontier(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = index_of(id, groups);
    if index >= p.n || gw[index + 1u] == gw[index] || gw[index] < p.e || gw[index] - p.e >= p.h {
        return;
    }
    let t = task_of(index);
    let state = (gw[index] - p.e) * (4u + 2u * (p.a + p.c));
    let base = state + 4u;
    let last = t.m * t.n - 1u;
    var length = 1u;
    var seen = 1u;
    if p.g == 1u {
        for (var slot = 0u; slot < p.a; slot += 1u) {
            atomicStore(&lb[base + 2u * slot], 0u);
        }
        let start = tk[8u * index + 3u];
        let origin = seen_find(base, 0u);
        atomicStore(&lb[base + 2u * origin], 1u);
        atomicStore(&lb[base + 2u * origin + 1u], start);
        heap_push(base, 0u, start, 0u);
    } else {
        if atomicLoad(&lb[state]) != 1u {
            return;
        }
        length = atomicLoad(&lb[state + 1u]);
        seen = atomicLoad(&lb[state + 2u]);
    }
    var result = NONE;
    var full = false;
    for (var step = 0u; step < p.j && length > 0u && !full; step += 1u) {
        let heap = base + 2u * p.a;
        let label = atomicLoad(&lb[heap]);
        let cell = atomicLoad(&lb[heap + 1u]);
        heap_pop(base, length);
        length -= 1u;
        let found = seen_find(base, cell);
        if found == NONE {
            full = true;
            break;
        }
        let entry = base + 2u * found;
        if (atomicLoad(&lb[entry]) & SETTLED) != 0u || atomicLoad(&lb[entry + 1u]) != label {
            continue;
        }
        atomicStore(&lb[entry], (cell + 1u) | SETTLED);
        if cell == last {
            result = label;
            break;
        }
        let row = cell / t.n;
        let column = cell % t.n;
        for (var side = 0u; side < 4u; side += 1u) {
            var other = cell;
            var weight: f32;
            if side == 0u {
                if row == 0u { continue; }
                other = cell - t.n;
                weight = boundary(at(t.a, row), at(t.b, column), at(t.b, column + 1u));
            } else if side == 1u {
                if row + 1u >= t.m { continue; }
                other = cell + t.n;
                weight = boundary(at(t.a, row + 1u), at(t.b, column), at(t.b, column + 1u));
            } else if side == 2u {
                if column == 0u { continue; }
                other = cell - 1u;
                weight = boundary(at(t.b, column), at(t.a, row), at(t.a, row + 1u));
            } else {
                if column + 1u >= t.n { continue; }
                other = cell + 1u;
                weight = boundary(at(t.b, column + 1u), at(t.a, row), at(t.a, row + 1u));
            }
            let candidate = max(label, bitcast<u32>(weight));
            let place = seen_find(base, other);
            if place == NONE {
                full = true;
                break;
            }
            let slot = base + 2u * place;
            let key = atomicLoad(&lb[slot]);
            if key == 0u {
                if seen >= p.b {
                    full = true;
                    break;
                }
                seen += 1u;
                atomicStore(&lb[slot], other + 1u);
            } else if (key & SETTLED) != 0u || atomicLoad(&lb[slot + 1u]) <= candidate {
                continue;
            }
            if length >= p.c {
                full = true;
                break;
            }
            atomicStore(&lb[slot + 1u], candidate);
            heap_push(base, length, candidate, other);
            length += 1u;
        }
    }
    if result != NONE {
        tk[8u * index + 2u] = result;
        tk[8u * index + 1u] = DONE;
        atomicAdd(&status[S_D], 1u);
        atomicStore(&lb[state], 2u);
    } else if full {
        // Left pending for a larger tier or the tiled relaxation.
        atomicStore(&lb[state], 2u);
    } else if length == 0u {
        // The final cell is always reachable.
        fail(E_INTERNAL);
        atomicStore(&lb[state], 2u);
    } else {
        atomicStore(&lb[state], 1u);
        atomicStore(&lb[state + 1u], length);
        atomicStore(&lb[state + 2u], seen);
        atomicAdd(&status[S_PENDING], 1u);
    }
}

// gw: 1 at tasks that need the cell search (plus zero at p.n).
@compute @workgroup_size(WG)
fn grid_marks(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = index_of(id, groups);
    if index > p.n {
        return;
    }
    gw[index] = select(0u, 1u, index < p.n && tk[8u * index] != 0u && tk[8u * index + 1u] == GRID);
}

fn tiles_of(m: u32, n: u32) -> u32 {
    return ((m + TILE - 1u) / TILE) * ((n + TILE - 1u) / TILE);
}

// gw: scanned marks at 0; task list at p.a; label units (256 cells) at p.b;
// tile counts at p.c (each followed by a zero written here for the search
// count p.d).
@compute @workgroup_size(WG)
fn grid_list(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = index_of(id, groups);
    if index > p.n {
        return;
    }
    if index == p.n {
        gw[p.b + p.d] = 0u;
        gw[p.c + p.d] = 0u;
        return;
    }
    if gw[index + 1u] == gw[index] {
        return;
    }
    let g = gw[index];
    let m = tk[8u * index + 4u];
    let n = tk[8u * index + 5u];
    gw[p.a + g] = index;
    gw[p.b + g] = (m * n + 255u) / 256u;
    gw[p.c + g] = tiles_of(m, n);
}

// Searches whose scanned unit offsets share floor(offset / p.d) form one chunk.
// No search exceeds p.d units, so chunks are contiguous and a chunk's local
// span stays below 2 * p.d units. gw: scanned units at p.b, scanned
// tiles at p.c, (first search, first tile) per chunk at p.e. status: S_B last chunk.
@compute @workgroup_size(WG)
fn grid_chunks(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let g = index_of(id, groups);
    if g >= p.n {
        return;
    }
    let chunk = gw[p.b + g] / p.d;
    if g == 0u || gw[p.b + g - 1u] / p.d != chunk {
        gw[p.e + 2u * chunk] = g;
        gw[p.e + 2u * chunk + 1u] = gw[p.c + g];
    }
    atomicMax(&status[S_B], chunk);
}

// `local` is the tile's index within its search, `tile` within the chunk.
struct Tile { task: u32, base: u32, m: u32, n: u32, rows: vec2<u32>, columns: vec2<u32>, local: u32, tile: u32 }
// lb holds two buckets of labels, then both boundary weights of every cell
// (toward the next row, toward the next column), then per-tile flags of the
// chunk: whether the tile is swept this round, and whether it must be swept in
// the next one because it or a neighboring tile changed.
fn weights_of(cell: u32) -> u32 {
    return 2u * p.d * 256u + 2u * cell;
}
fn active_flag(tile: u32) -> u32 {
    return 6u * p.d * 256u + tile;
}
fn next_flag(tile: u32) -> u32 {
    // A chunk of fewer than 512 p.d cells has fewer than 35 p.d tiles plus one
    // per search.
    return 6u * p.d * 256u + 35u * p.d + p.a + tile;
}
// Work item p.g + invocation of chunk p.f, among searches [p.h, p.i).
fn tile_of(item: u32) -> Tile {
    var low = p.h;
    var high = p.i;
    while high - low > 1u {
        let middle = low + (high - low) / 2u;
        if gw[p.c + middle] <= item {
            low = middle;
        } else {
            high = middle;
        }
    }
    let task = gw[p.a + low];
    let m = tk[8u * task + 4u];
    let n = tk[8u * task + 5u];
    let local = item - gw[p.c + low];
    let across = (n + TILE - 1u) / TILE;
    let row = (local / across) * TILE;
    let column = (local % across) * TILE;
    return Tile(
        task,
        (gw[p.b + low] - p.f * p.d) * 256u,
        m,
        n,
        vec2<u32>(row, min(row + TILE, m)),
        vec2<u32>(column, min(column + TILE, n)),
        local,
        item - gw[p.c + p.h],
    );
}

// Labels and boundary weights of one tile. Unreached cells start at infinity;
// the first cell starts at the endpoint level. Each weight is computed once.
@compute @workgroup_size(WG)
fn grid_init(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let item = p.g + index_of(id, groups);
    if item >= p.n {
        return;
    }
    let t = tile_of(item);
    let curves = task_of(t.task);
    for (var row = t.rows.x; row < t.rows.y; row += 1u) {
        let a0 = at(curves.a, row);
        let a1 = at(curves.a, row + 1u);
        for (var column = t.columns.x; column < t.columns.y; column += 1u) {
            let cell = t.base + row * t.n + column;
            let b0 = at(curves.b, column);
            let b1 = at(curves.b, column + 1u);
            atomicStore(&lb[cell], INF_BITS);
            atomicStore(&lb[weights_of(cell)], bitcast<u32>(boundary(a1, b0, b1)));
            atomicStore(&lb[weights_of(cell) + 1u], bitcast<u32>(boundary(b1, a0, a1)));
        }
    }
    if t.local == 0u {
        atomicStore(&lb[t.base], tk[8u * t.task + 3u]);
    }
    atomicStore(&lb[active_flag(t.tile)], 1u);
    atomicStore(&lb[next_flag(t.tile)], 0u);
}

// Lowers one cell's label through its four neighbors; returns whether it fell.
// Labels at or above the final cell's label cannot improve it and are skipped.
fn relax_cell(t: Tile, row: u32, column: u32, bound: u32) -> bool {
    let cell = t.base + row * t.n + column;
    let old = atomicLoad(&lb[cell]);
    var best = old;
    if row > 0u {
        let other = cell - t.n;
        let label = atomicLoad(&lb[other]);
        if label < best {
            best = min(best, max(label, atomicLoad(&lb[weights_of(other)])));
        }
    }
    if row + 1u < t.m {
        let label = atomicLoad(&lb[cell + t.n]);
        if label < best {
            best = min(best, max(label, atomicLoad(&lb[weights_of(cell)])));
        }
    }
    if column > 0u {
        let other = cell - 1u;
        let label = atomicLoad(&lb[other]);
        if label < best {
            best = min(best, max(label, atomicLoad(&lb[weights_of(other) + 1u])));
        }
    }
    if column + 1u < t.n {
        let label = atomicLoad(&lb[cell + 1u]);
        if label < best {
            best = min(best, max(label, atomicLoad(&lb[weights_of(cell) + 1u])));
        }
    }
    let last = row + 1u == t.m && column + 1u == t.n;
    if best < old && (best < bound || last) {
        atomicMin(&lb[cell], best);
        return true;
    }
    return false;
}

// Alternating raster sweeps of one tile until it stops changing, at most p.j
// pairs. A tile is swept only while it or a neighboring tile changed in the
// previous round: its labels depend on nothing else, so a tile left out is at
// its fixpoint. status: S_CHANGED counts tiles that lowered a label this round.
@compute @workgroup_size(WG)
fn grid_relax(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let item = p.g + index_of(id, groups);
    if item >= p.n {
        return;
    }
    let t = tile_of(item);
    if atomicLoad(&lb[active_flag(t.tile)]) == 0u {
        return;
    }
    var any = false;
    for (var sweep = 0u; sweep < p.j; sweep += 1u) {
        var changed = false;
        let bound = atomicLoad(&lb[t.base + t.m * t.n - 1u]);
        for (var row = t.rows.x; row < t.rows.y; row += 1u) {
            for (var column = t.columns.x; column < t.columns.y; column += 1u) {
                changed = relax_cell(t, row, column, bound) || changed;
            }
        }
        for (var row = t.rows.y; row > t.rows.x; row -= 1u) {
            for (var column = t.columns.y; column > t.columns.x; column -= 1u) {
                changed = relax_cell(t, row - 1u, column - 1u, bound) || changed;
            }
        }
        if !changed {
            break;
        }
        any = true;
    }
    if !any {
        return;
    }
    atomicAdd(&status[S_CHANGED], 1u);
    let across = (t.n + TILE - 1u) / TILE;
    atomicStore(&lb[next_flag(t.tile)], 1u);
    if t.columns.x > 0u {
        atomicStore(&lb[next_flag(t.tile - 1u)], 1u);
    }
    if t.columns.y < t.n {
        atomicStore(&lb[next_flag(t.tile + 1u)], 1u);
    }
    if t.rows.x > 0u {
        atomicStore(&lb[next_flag(t.tile - across)], 1u);
    }
    if t.rows.y < t.m {
        atomicStore(&lb[next_flag(t.tile + across)], 1u);
    }
}

// After a complete round over the p.n tiles of the chunk: the tiles to sweep
// next are those marked during it.
@compute @workgroup_size(WG)
fn grid_roll(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let tile = index_of(id, groups);
    if tile >= p.n {
        return;
    }
    atomicStore(&lb[active_flag(tile)], atomicLoad(&lb[next_flag(tile)]));
    atomicStore(&lb[next_flag(tile)], 0u);
}

// Searches [p.h, p.i) of chunk p.f after their fixpoint.
@compute @workgroup_size(WG)
fn grid_finish(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let g = p.h + index_of(id, groups);
    if g >= p.i {
        return;
    }
    let task = gw[p.a + g];
    let cells = tk[8u * task + 4u] * tk[8u * task + 5u];
    let cost = atomicLoad(&lb[(gw[p.b + g] - p.f * p.d) * 256u + cells - 1u]);
    if cost == INF_BITS {
        fail(E_INTERNAL);
    }
    tk[8u * task + 2u] = cost;
    tk[8u * task + 1u] = DONE;
}

struct Bounds { lower: f32, upper: f32, endpoints: f32, full: bool }
fn key(route: f32, endpoints: f32) -> f32 {
    return max(route / p.x, endpoints / p.y);
}
// Whether `first` certainly orders before `second` (ties favor forward).
fn precedes(first: Bounds, second: Bounds, forward: bool) -> bool {
    let a = key(first.upper, first.endpoints);
    let b = key(second.lower, second.endpoints);
    if a != b {
        return a < b;
    }
    if first.upper != second.lower {
        return first.upper < second.lower;
    }
    return first.endpoints < second.endpoints || (first.endpoints == second.endpoints && forward);
}
fn task_bounds(pair: u32, orientation: u32, epsilon: f32) -> Bounds {
    let task = 8u * (2u * pair + orientation);
    let route = sqrt(f(tk[task + 2u]));
    let endpoints = sqrt(f(tk[task + 3u]));
    if tk[task] == FULL {
        return Bounds(route, route, endpoints, true);
    }
    return Bounds(max(max(endpoints, route - epsilon), 0.0), max(endpoints, route + epsilon), endpoints, false);
}
fn request(pair: u32, orientation: u32, kind: u32) {
    let task = 8u * (2u * pair + orientation);
    tk[task] = kind;
    tk[task + 1u] = PENDING;
}

// Advances one pair after its tasks finished. p.x: route tolerance; p.y:
// endpoint tolerance. First stage: the endpoint-nearer simplified orientation
// alone; it is chosen outright when its upper key is below the other's endpoint
// key, which no route distance of the other can undercut. Otherwise both
// simplified bounds are compared; an undecidable orientation requests both
// full curves. A chosen orientation whose bounds straddle the route tolerance
// is refined on its full curves. status[S_PENDING] counts pairs that requested
// tasks.
@compute @workgroup_size(WG)
fn assess_step(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let t = index_of(id, groups);
    if t >= p.n {
        return;
    }
    let pair = 16u * t;
    if (ap[pair + A_OUT + 3u] & OUT_DONE) != 0u {
        return;
    }
    let stage = ap[pair + A_STAGE];
    let epsilon = routes[ap[pair]].error + routes[ap[pair + 1u]].error;
    var reversed = (ap[pair + A_OUT + 3u] & OUT_REVERSED) != 0u;
    var chosen: Bounds;
    if stage == STAGE_FIRST {
        let first = select(0u, 1u, reversed);
        chosen = task_bounds(t, first, epsilon);
        if !(key(chosen.upper, chosen.endpoints) < f(ap[pair + A_OTHER]) / p.y) {
            request(t, 1u - first, SIMPLIFIED);
            ap[pair + A_STAGE] = STAGE_BOTH;
            atomicAdd(&status[S_PENDING], 1u);
            return;
        }
    } else if stage == STAGE_CHOSEN {
        chosen = task_bounds(t, select(0u, 1u, reversed), epsilon);
    } else {
        let forward = task_bounds(t, 0u, epsilon);
        let reverse = task_bounds(t, 1u, epsilon);
        if stage == STAGE_BOTH && !precedes(forward, reverse, true) && !precedes(reverse, forward, false) {
            request(t, 0u, FULL);
            request(t, 1u, FULL);
            ap[pair + A_STAGE] = STAGE_FULL;
            atomicAdd(&status[S_PENDING], 1u);
            return;
        }
        reversed = !precedes(forward, reverse, true);
        if reversed {
            chosen = reverse;
        } else {
            chosen = forward;
        }
    }
    if stage != STAGE_CHOSEN && chosen.lower <= p.x && chosen.upper > p.x {
        let orientation = select(0u, 1u, reversed);
        request(t, orientation, FULL);
        tk[8u * (2u * t + 1u - orientation)] = 0u;
        ap[pair + A_OUT + 3u] = select(0u, OUT_REVERSED, reversed);
        ap[pair + A_STAGE] = STAGE_CHOSEN;
        atomicAdd(&status[S_PENDING], 1u);
        return;
    }
    var flags = OUT_DONE | select(0u, OUT_REVERSED, reversed);
    if chosen.lower > p.x {
        flags |= OUT_GENERAL;
    }
    if chosen.endpoints > p.y {
        flags |= OUT_ENDPOINT;
    }
    ap[pair + A_OUT] = bitcast<u32>(chosen.lower);
    ap[pair + A_OUT + 1u] = bitcast<u32>(chosen.upper);
    ap[pair + A_OUT + 2u] = bitcast<u32>(chosen.endpoints);
    ap[pair + A_OUT + 3u] = flags;
}
