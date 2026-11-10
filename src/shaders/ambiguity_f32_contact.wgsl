// Sustained contact of the gpu-f32 backend over one batch of free-space cells.
//
// Each cell is chained into two hash buckets, by (route pair, row) and by
// (route pair, column), with lock-free exchanges; chain order is arbitrary and
// reaches no result. A cell joins its forward row and column neighbors
// (periodic on closed routes) when the shared boundary point is within the
// radius. The cell kernel guarantees that such a neighbor exists, so a join
// without one is a protocol failure. Components are the fixpoint of atomic
// minimum-label propagation with pointer jumping: labels only fall and always
// name a cell of the same component, so a complete round without change leaves
// one label per component.
//
// A component's length on one route is the interval-union length of its cells
// per segment. Each cell adds the part of its interval not covered by the
// same-component cells of its segment that start earlier (ties by the other
// index), which sums to the union. Contributions are f32 values in units of the
// minimum length (p.x), capped at one and truncated to multiples of 2^-32; their
// 64-bit integer sum does not depend on order. A pair is in contact when one
// component reaches one unit on both routes.
//
// Chain walks are resumable: neighbor lookups and length measurement keep a
// cursor per cell and side and advance at most p.c chain entries per
// invocation, over cells [p.a, p.b); p.g = 1 starts. status[S_PENDING] counts
// cells with a walk left.
//
// `at` holds labels at 0, the source-side sums at n (low) and 2n (high), the
// partner-side sums at 3n and 4n, row bucket heads at 5n and column bucket
// heads at 5n + p.h (p.h buckets each, a power of two; heads and chain links
// store index + 1). p.d: partner bits. p.e: segment bits. p.f: first source
// route of the batch.

@group(0) @binding(1) var<storage, read> recs: array<u32>;
@group(0) @binding(2) var<storage, read> rt: array<u32>;
@group(0) @binding(3) var<storage, read_write> at: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> ca: array<u32>;
@group(0) @binding(5) var<storage, read_write> cb: array<u32>;
@group(0) @binding(6) var<storage, read_write> cc: array<u32>;
@group(0) @binding(7) var<storage, read_write> cd: array<u32>;

struct Key { low: u32, high: u32 }
fn shift(key: Key, bits: u32) -> Key {
    if bits == 0u {
        return key;
    }
    if bits >= 32u {
        return Key(key.high >> (bits - 32u), 0u);
    }
    return Key((key.low >> bits) | (key.high << (32u - bits)), key.high >> bits);
}
fn mask(bits: u32) -> u32 {
    if bits >= 32u {
        return NONE;
    }
    return (1u << bits) - 1u;
}
struct Cell { pair: Key, row: u32, column: u32 }
fn cell_of(index: u32) -> Cell {
    let key = Key(recs[8u * index], recs[8u * index + 1u]);
    return Cell(shift(key, 2u * p.e), shift(key, p.e).low & mask(p.e), key.low & mask(p.e));
}
fn source_of(cell: Cell) -> u32 {
    return shift(cell.pair, p.d).low + p.f;
}
fn partner_of(cell: Cell) -> u32 {
    return cell.pair.low & mask(p.d);
}
fn same_pair(a: Key, b: Key) -> bool {
    return a.low == b.low && a.high == b.high;
}
// Bucket of (route pair, segment); side 0 rows, side 1 columns.
fn bucket(pair: Key, segment: u32, side: u32) -> u32 {
    var h = pair.low * 0x9e3779b1u;
    h = (h ^ (h >> 15u)) + pair.high * 0x85ebca77u;
    h = (h ^ (h >> 13u)) + segment * 0xc2b2ae3du;
    h = (h ^ (h >> 16u)) * 0x27d4eb2fu;
    h ^= h >> 15u;
    return 5u * p.n + side * p.h + (h & (p.h - 1u));
}

// ca: chain links (row, column) per cell. Labels start as each cell's index.
@compute @workgroup_size(WG)
fn chain(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = index_of(id, groups);
    if i >= p.n {
        return;
    }
    atomicStore(&at[i], i);
    let cell = cell_of(i);
    ca[2u * i] = atomicExchange(&at[bucket(cell.pair, cell.row, 0u)], i + 1u);
    ca[2u * i + 1u] = atomicExchange(&at[bucket(cell.pair, cell.column, 1u)], i + 1u);
}

// Forward neighbor index on a route of `count` segments, or NONE at an open end.
fn forward(index: u32, count: u32, closed: bool) -> u32 {
    if index + 1u < count {
        return index + 1u;
    }
    return select(NONE, 0u, closed);
}
// A link word still being searched holds this bit and the next chain entry.
const SEARCH: u32 = 0x80000000u;
// Row and column of the neighbor cell joined on `side`, or NONE.
fn neighbor(cell: Cell, flags: u32, side: u32) -> vec2<u32> {
    if (flags & (1u << side)) == 0u {
        return vec2<u32>(NONE);
    }
    if side == 0u {
        let source = source_of(cell);
        let row = forward(cell.row, rt[4u * source + 1u], rt[4u * source + 2u] == 1u);
        return vec2<u32>(row, select(cell.column, NONE, row == NONE));
    }
    let partner = partner_of(cell);
    let column = forward(cell.column, rt[4u * partner + 1u], rt[4u * partner + 2u] == 1u);
    return vec2<u32>(select(cell.row, NONE, column == NONE), column);
}

// ca: chain links. cb: forward row and column neighbor of each cell (NONE
// without a join; SEARCH and a cursor while its row chain is being walked).
@compute @workgroup_size(WG)
fn links(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = p.a + index_of(id, groups);
    if i >= p.b {
        return;
    }
    let cell = cell_of(i);
    let flags = recs[8u * i + 6u];
    var budget = p.c;
    var pending = false;
    for (var side = 0u; side < 2u; side += 1u) {
        let wanted = neighbor(cell, flags, side);
        var word = cb[2u * i + side];
        if p.g == 1u {
            word = NONE;
            if wanted.x != NONE {
                word = SEARCH | atomicLoad(&at[bucket(cell.pair, wanted.x, 0u)]);
            }
        }
        if word == NONE || (word & SEARCH) == 0u {
            cb[2u * i + side] = word;
            continue;
        }
        var next = word & ~SEARCH;
        var found = NONE;
        while next != 0u && budget > 0u {
            budget -= 1u;
            let j = next - 1u;
            let other = cell_of(j);
            if other.row == wanted.x && other.column == wanted.y && same_pair(other.pair, cell.pair) {
                found = j;
                break;
            }
            next = ca[2u * j];
        }
        if found != NONE {
            cb[2u * i + side] = found;
        } else if next == 0u {
            // The joined neighbor cell must exist.
            fail(E_NEIGHBOR);
            cb[2u * i + side] = NONE;
        } else {
            cb[2u * i + side] = SEARCH | next;
            pending = true;
        }
    }
    if pending {
        atomicAdd(&status[S_PENDING], 1u);
    }
}

fn lower(index: u32, value: u32) -> bool {
    if atomicLoad(&at[index]) > value {
        atomicMin(&at[index], value);
        return true;
    }
    return false;
}

// One propagation round. cb: links. status[S_CHANGED] counts cells that lowered
// a label.
@compute @workgroup_size(WG)
fn label_round(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = index_of(id, groups);
    if i >= p.n {
        return;
    }
    var changed = false;
    for (var side = 0u; side < 2u; side += 1u) {
        let j = cb[2u * i + side];
        if j == NONE {
            continue;
        }
        let mine = atomicLoad(&at[i]);
        let other = atomicLoad(&at[j]);
        let least = min(mine, other);
        changed = lower(max(mine, other), least) || changed;
        changed = lower(i, least) || changed;
        changed = lower(j, least) || changed;
    }
    let label = atomicLoad(&at[i]);
    changed = lower(i, atomicLoad(&at[label])) || changed;
    if changed {
        atomicAdd(&status[S_CHANGED], 1u);
    }
}

// Adds `value` units (capped at one, truncated to 2^-32) to the 64-bit sum at
// (low, low + n).
fn accumulate(low: u32, value: f32) {
    if value >= 1.0 {
        atomicAdd(&at[low + p.n], 1u);
        return;
    }
    let fraction = u32(value * 4294967296.0);
    if fraction == 0u {
        return;
    }
    let before = atomicAdd(&at[low], fraction);
    if before + fraction < before {
        atomicAdd(&at[low + p.n], 1u);
    }
}

// Each cell's uncovered interval length on both routes, added to its
// component. ca: chain links. cb: walk cursor per cell and side (next chain
// entry + 1; zero when that side is done). cc: covered end so far. cd: segment
// records. Labels are final.
@compute @workgroup_size(WG)
fn measure(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = p.a + index_of(id, groups);
    if i >= p.b {
        return;
    }
    let cell = cell_of(i);
    let label = atomicLoad(&at[i]);
    var budget = p.c;
    var pending = false;
    for (var side = 0u; side < 2u; side += 1u) {
        let segment = select(cell.row, cell.column, side == 1u);
        var next = cb[2u * i + side];
        var covered = f(cc[2u * i + side]);
        if p.g == 1u {
            next = atomicLoad(&at[bucket(cell.pair, segment, side)]);
            covered = 0.0;
        }
        if next == 0u {
            continue;
        }
        let tie = select(cell.column, cell.row, side == 1u);
        let low = f(normal_zero(recs[8u * i + 2u + 2u * side]));
        while next != 0u && budget > 0u {
            budget -= 1u;
            let j = next - 1u;
            next = ca[2u * j + side];
            if j == i || atomicLoad(&at[j]) != label {
                continue;
            }
            let other = cell_of(j);
            if select(other.row, other.column, side == 1u) != segment || !same_pair(other.pair, cell.pair) {
                continue;
            }
            let other_low = f(normal_zero(recs[8u * j + 2u + 2u * side]));
            let other_tie = select(other.column, other.row, side == 1u);
            if other_low < low || (other_low == low && other_tie < tie) {
                covered = max(covered, f(recs[8u * j + 3u + 2u * side]));
            }
        }
        cb[2u * i + side] = next;
        cc[2u * i + side] = bitcast<u32>(covered);
        if next != 0u {
            pending = true;
            continue;
        }
        let high = f(recs[8u * i + 3u + 2u * side]);
        let route = select(source_of(cell), partner_of(cell), side == 1u);
        let slot = 6u * (rt[4u * route] + segment);
        let length = norm(vec2<f32>(f(cd[slot + 2u]), f(cd[slot + 3u])) - vec2<f32>(f(cd[slot]), f(cd[slot + 1u])));
        accumulate((1u + 2u * side) * p.n + label, max(high - max(low, covered), 0.0) * length / p.x);
    }
    if pending {
        atomicAdd(&status[S_PENDING], 1u);
    }
}

// cc: 1 at the root cell of every component that reaches the minimum on both
// routes (plus zero at p.n).
@compute @workgroup_size(WG)
fn sustained(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = index_of(id, groups);
    if i > p.n {
        return;
    }
    var mark = 0u;
    if i < p.n && atomicLoad(&at[i]) == i
        && atomicLoad(&at[2u * p.n + i]) >= 1u && atomicLoad(&at[4u * p.n + i]) >= 1u {
        mark = 1u;
    }
    cc[i] = mark;
}

// cc: scanned marks. cd: sort records keyed by (source ordinal, partner
// ordinal), one per sustained component.
@compute @workgroup_size(WG)
fn sustained_pairs(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = index_of(id, groups);
    if i >= p.n || cc[i + 1u] == cc[i] {
        return;
    }
    let cell = cell_of(i);
    cd[3u * cc[i]] = partner_of(cell);
    cd[3u * cc[i] + 1u] = source_of(cell);
    cd[3u * cc[i] + 2u] = i;
}

// ca: sorted component records. cb: scanned key marks. cd: contact pair log
// (source ordinal, partner ordinal), appended at p.a in pair order.
@compute @workgroup_size(WG)
fn pair_out(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let s = index_of(id, groups);
    if s >= p.n || cb[s + 1u] == cb[s] {
        return;
    }
    cd[2u * (p.a + cb[s])] = ca[3u * s + 1u];
    cd[2u * (p.a + cb[s]) + 1u] = ca[3u * s];
}
