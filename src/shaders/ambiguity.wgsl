// Rust prepends `const K: u32`, the original points kept per route, and `SCORE_ROUNDS`.
const CELLS: u32 = K - 1u;
// Fixed-point contact length units per export pixel.
const UNITS: f32 = 256.0;
const MAX_PROBES: u32 = 1024u;
const MAX_SWEEPS: u32 = 64u;
const TABLE_FULL: u32 = 1u;
const SWEEP_LIMIT: u32 = 2u;
const ANCHOR_OVERFLOW: u32 = 4u;
const COMPLETE: u32 = 8u;

struct Parameters {
    radius: f32,
    length: f32,
    tolerance: f32,
    pixels_per_clip: f32,
    capacity: u32,
    // First workgroup of the current submission.
    base: u32,
}

// Near length a route has to one other route, and the last query piece that credited it.
struct Near {
    length: atomic<u32>,
    stamp: atomic<u32>,
}

struct Stats {
    anchors: u32,
    occupied: atomic<u32>,
    contacts: atomic<u32>,
    incompatible: atomic<u32>,
    status: atomic<u32>,
    sum_squares: f32,
    // Finished work, so Rust can reject a frame with a dropped submission.
    routes: atomic<u32>,
    scored: atomic<u32>,
    // One squared-excess sum per scoring workgroup.
    partials: array<f32>,
}

override block_size: u32;

@group(0) @binding(0) var<storage, read> input_points: array<Point>;
@group(0) @binding(1) var<storage, read_write> originals: array<vec2<f32>>;
// Hash set of directed pairs (route << 16 | other) where route reaches the contact length.
// Zero marks an empty slot.
@group(0) @binding(3) var<storage, read_write> table: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> paths: array<Path>;
@group(0) @binding(10) var<storage, read_write> stats: Stats;
// One row of near lengths to every route per workgroup of a contact submission.
@group(0) @binding(11) var<storage, read_write> scratch: array<Near>;
@group(0) @binding(12) var<uniform> parameters: Parameters;

var<workgroup> partial_sums: array<f32, block_size>;
var<workgroup> stop: u32;
var<workgroup> pending: array<u32, block_size * 2u>;
var<workgroup> pending_count: atomic<u32>;
var<workgroup> batch: u32;

fn invocation(id: vec3<u32>, groups: vec3<u32>) -> u32 {
    return id.y * groups.x * block_size + id.x;
}

fn pixel(index: u32) -> vec2<f32> {
    return point_position(input_points[index]) * parameters.pixels_per_clip;
}

// Paths are contiguous, so equal neighbouring IDs share a route.
fn same_route(index: u32, next: u32) -> bool {
    return next < arrayLength(&input_points) && input_points[index].edge_id == input_points[next].edge_id;
}

// Parameter of the closest segment point. Degenerate segments use their start.
fn closest(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>) -> f32 {
    let t = dot(point - start, end - start) / dot(end - start, end - start);
    return select(min(t, 1.0), 0.0, !(t > 0.0));
}

fn segment_distance(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>) -> f32 {
    let t = closest(point, start, end);
    // Clamped ends use the vertex itself so both adjacent segments agree exactly.
    if t == 0.0 {
        return distance(point, start);
    }
    if t == 1.0 {
        return distance(point, end);
    }
    return distance(point, mix(start, end, t));
}

fn route_distance(point: vec2<f32>, segment: u32) -> f32 {
    return segment_distance(point, pixel(segment), pixel(segment + 1u));
}

@compute @workgroup_size(block_size)
fn resample_originals(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    // The invocation at each route's first sample walks the whole route.
    let first = invocation(id, groups);
    if first >= arrayLength(&input_points) || (first > 0u && same_route(first - 1u, first)) {
        return;
    }
    let route = input_points[first].edge_id;
    var last = first;
    var total = 0.0;
    while same_route(last, last + 1u) {
        total += distance(pixel(last), pixel(last + 1u));
        last += 1u;
    }
    var segment = first;
    var walked = 0.0;
    for (var k = 0u; k < K; k += 1u) {
        let target_length = total * f32(k) / f32(CELLS);
        var length = distance(pixel(segment), pixel(segment + 1u));
        while segment + 1u < last && walked + length < target_length {
            walked += length;
            segment += 1u;
            length = distance(pixel(segment), pixel(segment + 1u));
        }
        let t = select(0.0, clamp((target_length - walked) / length, 0.0, 1.0), length > 0.0);
        originals[route * K + k] = mix(pixel(segment), pixel(segment + 1u), t);
    }
    originals[route * K] = pixel(first);
    originals[route * K + CELLS] = pixel(last);
}

// Segments are binned as anchors at most one cell apart, so long segments do not widen queries.
fn anchor_count(segment: u32) -> u32 {
    return max(u32(ceil(distance(pixel(segment), pixel(segment + 1u)) / f32(grid.radius))), 1u);
}

fn anchor_cell(segment: u32, anchor: u32, count: u32) -> u32 {
    let t = (f32(anchor) + 0.5) / f32(count);
    return cell_index(mix(point_position(input_points[segment]), point_position(input_points[segment + 1u]), t));
}

@compute @workgroup_size(block_size)
fn count_anchors(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let segment = invocation(id, groups);
    if !same_route(segment, segment + 1u) {
        return;
    }
    let count = anchor_count(segment);
    for (var anchor = 0u; anchor < count; anchor += 1u) {
        atomicAdd(&spatial.cells[anchor_cell(segment, anchor, count)].end, 1u);
    }
}

// Each entry stores its anchor number in x and its segment start in index.
@compute @workgroup_size(block_size)
fn scatter_anchors(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let segment = invocation(id, groups);
    if !same_route(segment, segment + 1u) {
        return;
    }
    let count = anchor_count(segment);
    for (var anchor = 0u; anchor < count; anchor += 1u) {
        let slot = atomicAdd(&spatial.cells[anchor_cell(segment, anchor, count)].end, 1u);
        if slot < arrayLength(&positions) {
            positions[slot] = IndexedPoint(f32(anchor), 0.0, segment);
        }
    }
}

fn hash(key: u32) -> u32 {
    var h = key ^ (key >> 16u);
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    return h ^ (h >> 16u);
}

// Inserts a directed pair. Each route inserts its pairs once per frame, so keys never repeat.
fn insert_pair(key: u32) {
    var slot = hash(key) % parameters.capacity;
    for (var probe = 0u; probe < MAX_PROBES; probe += 1u) {
        let exchange = atomicCompareExchangeWeak(&table[slot], 0u, key);
        if exchange.exchanged {
            atomicAdd(&stats.occupied, 1u);
            return;
        }
        // A zero old value is a spurious weak failure, so retry this slot.
        if exchange.old_value != 0u {
            slot = (slot + 1u) % parameters.capacity;
        }
    }
    atomicOr(&stats.status, TABLE_FULL);
}

fn contains_pair(key: u32) -> bool {
    var slot = hash(key) % parameters.capacity;
    for (var probe = 0u; probe < MAX_PROBES; probe += 1u) {
        let found = atomicLoad(&table[slot]);
        if found == key || found == 0u {
            return found == key;
        }
        slot = (slot + 1u) % parameters.capacity;
    }
    return false;
}

// Credits one query piece to the other route once. The workgroup queries one piece at a
// time, so the first lane to swap in the piece's stamp adds its length.
fn add_contact(row: u32, other: u32, units: u32, stamp: u32) {
    let near = row * arrayLength(&paths) + other;
    if atomicExchange(&scratch[near].stamp, stamp) != stamp {
        atomicAdd(&scratch[near].length, units);
    }
}

// The workgroup's lanes share the anchors near a query piece's midpoint and credit every
// other route with a segment within the radius.
fn query_contacts(route: u32, row: u32, position: vec2<f32>, units: u32, stamp: u32, lane: u32) {
    let point = position * parameters.pixels_per_clip;
    // A segment point within the radius lies within half an anchor spacing of an anchor.
    let reach = parameters.radius + f32(grid.radius) * 0.5;
    let texel_reach = vec2<f32>(reach / parameters.pixels_per_clip * 0.5) * vec2<f32>(map.size - vec2<u32>(1u)) + 0.5;
    let texels = texel_position(position);
    let first = grid_cell(texels - texel_reach);
    let last = grid_cell(texels + texel_reach);
    for (var y = first.y; y <= last.y; y += 1u) {
        for (var x = first.x; x <= last.x; x += 1u) {
            let cell = y * grid.size.x + x;
            let end = atomicLoad(&spatial.cells[cell].end);
            for (var slot = spatial.cells[cell].start + lane; slot < end; slot += block_size) {
                let segment = positions[slot].index;
                let other = input_points[segment].edge_id;
                if other != route && route_distance(point, segment) <= parameters.radius {
                    add_contact(row, other, units, stamp);
                }
            }
        }
    }
}

fn fixed(length: f32) -> u32 {
    return u32(round(length * UNITS));
}

// One workgroup per route walks its segments in query pieces no longer than the radius,
// so each piece credits at most about the radius of length to each other route.
@compute @workgroup_size(block_size)
fn accumulate_contacts(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let route = parameters.base + group.x;
    let row = group.x;
    if route >= arrayLength(&paths) {
        return;
    }
    // Scatter dropped anchors that did not fit, so cell ranges may point past the buffer.
    if lane == 0u && atomicLoad(&spatial.cursor) > arrayLength(&positions) {
        atomicOr(&stats.status, ANCHOR_OVERFLOW);
    }
    let path = paths[route];
    var stamp = 0u;
    var walked = 0.0;
    for (var segment = path.point_offset; segment + 1u < path.point_offset + path.count; segment += 1u) {
        let start = point_position(input_points[segment]);
        let end = point_position(input_points[segment + 1u]);
        let length = distance(start, end) * parameters.pixels_per_clip;
        let pieces = max(u32(ceil(length / parameters.radius)), 1u);
        for (var piece = 0u; piece < pieces; piece += 1u) {
            // Rounding cumulative arc length keeps the error from growing with the piece count.
            let units = fixed(walked + length * f32(piece + 1u) / f32(pieces)) - fixed(walked + length * f32(piece) / f32(pieces));
            if units == 0u {
                continue;
            }
            // Finish the previous piece's stamps before the next piece starts.
            storageBarrier();
            if lane == 0u {
                stop = atomicLoad(&stats.status) & (TABLE_FULL | ANCHOR_OVERFLOW);
            }
            workgroupBarrier();
            // Either flag already fails this frame, so stop early.
            if workgroupUniformLoad(&stop) != 0u {
                return;
            }
            stamp += 1u;
            query_contacts(route, row, mix(start, end, (f32(piece) + 0.5) / f32(pieces)), units, stamp, lane);
        }
        walked += length;
    }
    storageBarrier();
    // Keep the routes this route reaches and zero the row for the next submission.
    let count = arrayLength(&paths);
    for (var other = lane; other < count; other += block_size) {
        let near = row * count + other;
        if atomicLoad(&scratch[near].stamp) != 0u {
            if f32(atomicLoad(&scratch[near].length)) / UNITS >= parameters.length {
                insert_pair((route << 16u) | other);
            }
            atomicStore(&scratch[near].length, 0u);
            atomicStore(&scratch[near].stamp, 0u);
        }
    }
    if lane == 0u {
        atomicAdd(&stats.routes, 1u);
    }
}

fn original(route: u32, index: u32, reversed: bool) -> vec2<f32> {
    return originals[route * K + select(index, CELLS - index, reversed)];
}

// Continuous weak Frechet distance as a bottleneck path over free-space cells, capped at bound.
fn weak_frechet(first: u32, second: u32, reversed: bool, bound: f32) -> f32 {
    let lower = endpoint_bound(first, second, reversed);
    if lower >= bound {
        return bound;
    }
    // Costs at or above the bound cannot lower the result, so they are never explored.
    var cost: array<f32, CELLS * CELLS>;
    for (var cell = 0u; cell < CELLS * CELLS; cell += 1u) {
        cost[cell] = bound;
    }
    cost[0] = lower;
    var changed = true;
    for (var sweep = 0u; changed; sweep += 1u) {
        if sweep == MAX_SWEEPS {
            atomicOr(&stats.status, SWEEP_LIMIT);
            break;
        }
        changed = false;
        for (var step = 0u; step < CELLS * CELLS * 2u; step += 1u) {
            // Alternate forward and backward sweeps; each cell relaxes from its two upstream neighbours.
            let backward = step >= CELLS * CELLS;
            let cell = select(step, CELLS * CELLS * 2u - 1u - step, backward);
            let a = cell / CELLS;
            let b = cell % CELLS;
            var best = cost[cell];
            if backward && a + 1u < CELLS {
                let edge = segment_distance(original(first, a + 1u, false), original(second, b, reversed), original(second, b + 1u, reversed));
                best = min(best, max(cost[cell + CELLS], edge));
            }
            if backward && b + 1u < CELLS {
                let edge = segment_distance(original(second, b + 1u, reversed), original(first, a, false), original(first, a + 1u, false));
                best = min(best, max(cost[cell + 1u], edge));
            }
            if !backward && a > 0u {
                let edge = segment_distance(original(first, a, false), original(second, b, reversed), original(second, b + 1u, reversed));
                best = min(best, max(cost[cell - CELLS], edge));
            }
            if !backward && b > 0u {
                let edge = segment_distance(original(second, b, reversed), original(first, a, false), original(first, a + 1u, false));
                best = min(best, max(cost[cell - 1u], edge));
            }
            if best < cost[cell] {
                cost[cell] = best;
                changed = true;
            }
        }
    }
    return cost[CELLS * CELLS - 1u];
}

fn endpoint_bound(first: u32, second: u32, reversed: bool) -> f32 {
    let starts = distance(original(first, 0u, false), original(second, 0u, reversed));
    return max(starts, distance(original(first, CELLS, false), original(second, CELLS, reversed)));
}

fn workgroup_sum(lane: u32, value: f32) -> f32 {
    partial_sums[lane] = value;
    workgroupBarrier();
    for (var stride = block_size / 2u; stride > 0u; stride /= 2u) {
        if lane < stride {
            partial_sums[lane] += partial_sums[lane + stride];
        }
        workgroupBarrier();
    }
    return partial_sums[0];
}

// Synchronous walk over the resampled points, an upper bound on the weak Frechet distance.
fn synchronous_bound(first: u32, second: u32, reversed: bool) -> f32 {
    var bound = 0.0;
    for (var k = 0u; k < K; k += 1u) {
        bound = max(bound, distance(original(first, k, false), original(second, k, reversed)));
    }
    return bound;
}

// Squared excess of a contact. Pairs within the tolerance under a synchronous walk have none.
fn score_pair(key: u32) -> f32 {
    let first = key >> 16u;
    let second = key & 0xffffu;
    var separation = min(synchronous_bound(first, second, false), synchronous_bound(first, second, true));
    if separation <= parameters.tolerance {
        return 0.0;
    }
    // The orientation with the smaller endpoint bound usually decides the minimum.
    let reversed = endpoint_bound(first, second, true) < endpoint_bound(first, second, false);
    let likely = weak_frechet(first, second, reversed, separation);
    separation = weak_frechet(first, second, !reversed, likely);
    if separation > parameters.tolerance {
        atomicAdd(&stats.incompatible, 1u);
    }
    let excess = max(separation - parameters.tolerance, 0.0);
    return excess * excess;
}

// Each workgroup scans SCORE_ROUNDS slots per lane and queues contacts, pairs present in both
// directions, so every lane scores one when a batch runs even in a sparse set.
@compute @workgroup_size(block_size)
fn score_contacts(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let first_slot = (parameters.base + group.x) * block_size * SCORE_ROUNDS;
    if lane == 0u && first_slot < parameters.capacity {
        atomicAdd(&stats.scored, min(block_size * SCORE_ROUNDS, parameters.capacity - first_slot));
    }
    var sum = 0.0;
    for (var round = 0u; round <= SCORE_ROUNDS; round += 1u) {
        let slot = first_slot + round * block_size + lane;
        if round < SCORE_ROUNDS && slot < parameters.capacity {
            let key = atomicLoad(&table[slot]);
            let first = key >> 16u;
            let second = key & 0xffffu;
            if first < second && contains_pair((second << 16u) | first) {
                atomicAdd(&stats.contacts, 1u);
                pending[atomicAdd(&pending_count, 1u)] = key;
            }
        }
        // Score full batches, then the rest after the last round. Fewer than a batch remain
        // before each round adds at most one key per lane.
        loop {
            workgroupBarrier();
            if lane == 0u {
                batch = atomicLoad(&pending_count);
            }
            let count = workgroupUniformLoad(&batch);
            if count < select(block_size, 1u, round == SCORE_ROUNDS) {
                break;
            }
            let taken = min(count, block_size);
            if lane < taken {
                sum += score_pair(pending[count - 1u - lane]);
            }
            workgroupBarrier();
            if lane == 0u {
                atomicStore(&pending_count, count - taken);
            }
        }
    }
    sum = workgroup_sum(lane, sum);
    if lane == 0u {
        stats.partials[group.x] += sum;
    }
}

@compute @workgroup_size(block_size)
fn reduce_partials(@builtin(local_invocation_index) lane: u32) {
    var sum = 0.0;
    for (var index = lane; index < arrayLength(&stats.partials); index += block_size) {
        sum += stats.partials[index];
    }
    sum = workgroup_sum(lane, sum);
    if lane == 0u {
        stats.anchors = atomicLoad(&spatial.cursor);
        if stats.anchors > arrayLength(&positions) {
            atomicOr(&stats.status, ANCHOR_OVERFLOW);
        }
        stats.sum_squares = sum;
        atomicOr(&stats.status, COMPLETE);
    }
}
