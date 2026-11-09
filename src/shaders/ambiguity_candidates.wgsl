// Conservative ambiguity contact candidates over current GPU geometry.
//
// Floating-point work is limited to one subtraction or addition per expanded
// bound. Results are widened by two order-key steps, and tiny inputs/results are
// moved outside the subnormal range before and after that operation, so flushed
// subnormals, ignored zero signs and one-ulp rounding in any mode stay inside.
// Positive-segment tests, grid cells and candidate filters use exact integers.

struct Parameters {
    point_count: u32,
    path_count: u32,
    reservation_count: u32,
    grid_size: u32,
    cell_shift: i32,
    half_radius: f32,
    segment_capacity: u32,
    footprint_start: u32,
    footprint_count: u32,
    item_start: u32,
    item_count: u32,
    capacity: u32,
    padding_0: u32,
    padding_1: u32,
}

// Coordinates are read as bits; no floating-point load or comparison is needed.
struct RawPoint {
    x: u32,
    y: u32,
    edge_id: u32,
}

struct Segment {
    path: u32,
    index: u32,
    low: vec2<u32>,
    high: vec2<u32>,
}

// One (segment, covered grid cell) pair; segments are listed in segment order.
struct Footprint {
    segment: u32,
    cell: u32,
}

struct Candidate {
    source: u32,
    partner: u32,
    source_segment: u32,
    partner_segment: u32,
}

const block_size: u32 = 256u;
// Partner entries examined by one candidate work item.
const item_entries: u32 = 64u;
const scan_block: u32 = 512u;
const invalid_ordinal: u32 = 0xffffffffu;
const scan_overflow: u32 = 0u;
const count_mismatch: u32 = 1u;
const invalid_geometry: u32 = 2u;
const capacity_exceeded: u32 = 3u;
// Order keys of -2^-126, +2^-126, -2^-125 and +2^-125.
const negative_normal: u32 = 0x7f7fffffu;
const positive_normal: u32 = 0x80800000u;
const negative_tiny: u32 = 0x7effffffu;
const positive_tiny: u32 = 0x81000000u;

@group(0) @binding(0) var<uniform> parameters: Parameters;
@group(0) @binding(1) var<storage, read> points: array<RawPoint>;
@group(0) @binding(2) var<storage, read> paths: array<Path>;
@group(0) @binding(3) var<storage, read> order: array<u32>;
@group(0) @binding(4) var<storage, read> ordinals: array<u32>;
@group(0) @binding(5) var<storage, read_write> point_scan: array<u32>;
@group(0) @binding(6) var<storage, read_write> path_base: array<u32>;
@group(0) @binding(7) var<storage, read_write> status: array<atomic<u32>, 4>;

@group(1) @binding(0) var<storage, read_write> segments: array<Segment>;
@group(1) @binding(1) var<storage, read_write> cell_counts: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> cell_cursor: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> entries: array<u32>;
@group(1) @binding(4) var<storage, read_write> item_counts: array<u32>;
@group(1) @binding(5) var<storage, read_write> output: array<Candidate>;
@group(1) @binding(9) var<storage, read_write> footprint_offsets: array<u32>;
@group(1) @binding(10) var<storage, read_write> footprint: array<Footprint>;
@group(1) @binding(11) var<storage, read_write> path_footprints: array<u32>;
@group(1) @binding(12) var<storage, read_write> weights: array<u32>;

fn invocation(id: vec3<u32>, groups: vec3<u32>) -> u32 {
    return id.y * groups.x * block_size + id.x;
}

// Monotone total order on non-NaN f32 bit patterns with -0 equal to +0.
fn bits_key(bits: u32) -> u32 {
    if (bits & 0x7fffffffu) == 0u {
        return 0x80000000u;
    }
    return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn order_key(value: f32) -> u32 {
    return bits_key(bitcast<u32>(value));
}

fn key_value(key: u32) -> f32 {
    return bitcast<f32>(select(~key, key & 0x7fffffffu, key >= 0x80000000u));
}

fn is_tiny(key: u32, negative: u32, positive: u32) -> bool {
    return key > negative && key < positive;
}

fn lower_bound(key: u32) -> u32 {
    let input = select(key, negative_normal, is_tiny(key, negative_normal, positive_normal));
    let result = order_key(key_value(input) - parameters.half_radius);
    let widened = result - 2u;
    return select(widened, min(widened, negative_tiny), is_tiny(result, negative_tiny, positive_tiny));
}

fn upper_bound(key: u32) -> u32 {
    let input = select(key, positive_normal, is_tiny(key, negative_normal, positive_normal));
    let result = order_key(key_value(input) + parameters.half_radius);
    let widened = result + 2u;
    return select(widened, max(widened, positive_tiny), is_tiny(result, negative_tiny, positive_tiny));
}

// Exact clamp(floor(x / 2^cell_shift) + grid_size / 2, 0, grid_size - 1) from an
// order key. Infinity and NaN-pattern keys from widening saturate.
fn cell_coordinate(key: u32) -> u32 {
    let negative = key < 0x80000000u;
    let bits = select(key, ~key, negative) & 0x7fffffffu;
    let exponent = bits >> 23u;
    let mantissa = (bits & 0x7fffffu) | select(0u, 0x800000u, exponent != 0u);
    let shift = i32(max(exponent, 1u)) - 150 - parameters.cell_shift;
    let last = parameters.grid_size - 1u;
    if exponent == 255u || (shift >= 0 && mantissa != 0u) {
        return select(last, 0u, negative);
    }
    let right = u32(min(-shift, 32));
    var floor_magnitude = 0u;
    var ceiling_magnitude = select(0u, 1u, mantissa != 0u);
    if right < 32u {
        floor_magnitude = mantissa >> right;
        ceiling_magnitude = (mantissa + ((1u << right) - 1u)) >> right;
    }
    let half = i32(parameters.grid_size / 2u);
    let cell = select(half + i32(floor_magnitude), half - i32(ceiling_magnitude), negative);
    return u32(clamp(cell, 0, i32(last)));
}

fn cell_range(segment: Segment) -> vec4<u32> {
    return vec4<u32>(
        cell_coordinate(segment.low.x),
        cell_coordinate(segment.low.y),
        cell_coordinate(segment.high.x),
        cell_coordinate(segment.high.y),
    );
}

fn same_coordinate(first: u32, second: u32) -> bool {
    return first == second || ((first | second) & 0x7fffffffu) == 0u;
}

fn checked_add(a: u32, b: u32) -> u32 {
    let sum = a + b;
    if sum < a {
        atomicOr(&status[scan_overflow], 1u);
    }
    return sum;
}

// One extra invocation writes zero so the exclusive scan also yields the total.
@compute @workgroup_size(block_size)
fn mark_points(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = invocation(id, groups);
    if index > parameters.point_count {
        return;
    }
    if index == parameters.point_count {
        point_scan[index] = 0u;
        return;
    }
    let point = points[index];
    if point.edge_id >= parameters.reservation_count {
        atomicOr(&status[invalid_geometry], 1u);
        point_scan[index] = 0u;
        return;
    }
    let path = paths[point.edge_id];
    let inside = index >= path.point_offset && index - path.point_offset < path.count;
    if !inside || ordinals[point.edge_id] == invalid_ordinal {
        atomicOr(&status[invalid_geometry], 1u);
    }
    var positive = false;
    if inside && index > path.point_offset {
        let previous = points[index - 1u];
        positive = !(same_coordinate(point.x, previous.x) && same_coordinate(point.y, previous.y));
    }
    point_scan[index] = select(0u, 1u, positive);
}

@compute @workgroup_size(block_size)
fn count_segments(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let ordinal = invocation(id, groups);
    if ordinal > parameters.path_count {
        return;
    }
    if ordinal == parameters.path_count {
        path_base[ordinal] = 0u;
        return;
    }
    let path = paths[order[ordinal]];
    let end = path.point_offset + path.count;
    if end > parameters.point_count || end < path.point_offset {
        atomicOr(&status[invalid_geometry], 1u);
        path_base[ordinal] = 0u;
        return;
    }
    path_base[ordinal] = point_scan[end] - point_scan[path.point_offset];
}

// Segments are ordered by path ordinal, then positive-segment index. A positive
// end point differs from its predecessor, whose value equals the retained start.
@compute @workgroup_size(block_size)
fn emit_segments(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = invocation(id, groups);
    if index >= parameters.point_count || point_scan[index + 1u] == point_scan[index] {
        return;
    }
    let end = points[index];
    let start = points[index - 1u];
    let path = paths[end.edge_id];
    let ordinal = ordinals[end.edge_id];
    if ordinal == invalid_ordinal {
        return;
    }
    let logical = point_scan[index] - point_scan[path.point_offset];
    let low_x = min(bits_key(start.x), bits_key(end.x));
    let low_y = min(bits_key(start.y), bits_key(end.y));
    let high_x = max(bits_key(start.x), bits_key(end.x));
    let high_y = max(bits_key(start.y), bits_key(end.y));
    segments[path_base[ordinal] + logical] = Segment(
        ordinal,
        logical,
        vec2<u32>(lower_bound(low_x), lower_bound(low_y)),
        vec2<u32>(upper_bound(high_x), upper_bound(high_y)),
    );
}

fn cell_area(range: vec4<u32>) -> u32 {
    return (range.z - range.x + 1u) * (range.w - range.y + 1u);
}

// Largest index k in [0, count) with prefix[k] <= value, for a strictly
// increasing exclusive prefix with prefix[count] > value.
fn search_footprints(value: u32, count: u32) -> u32 {
    var low = 0u;
    var high = count;
    while high - low > 1u {
        let middle = low + (high - low) / 2u;
        if footprint_offsets[middle] <= value {
            low = middle;
        } else {
            high = middle;
        }
    }
    return low;
}

fn search_weights(value: u32, count: u32) -> u32 {
    var low = 0u;
    var high = count;
    while high - low > 1u {
        let middle = low + (high - low) / 2u;
        if weights[middle] <= value {
            low = middle;
        } else {
            high = middle;
        }
    }
    return low;
}

// Cell counts per segment, zero past the last segment, so one scan over the
// fixed capacity yields offsets and the total.
@compute @workgroup_size(block_size)
fn count_footprints(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = invocation(id, groups);
    if index > parameters.segment_capacity {
        return;
    }
    var area = 0u;
    if index < path_base[parameters.path_count] {
        area = cell_area(cell_range(segments[index]));
    }
    footprint_offsets[index] = area;
}

@compute @workgroup_size(block_size)
fn gather_footprints(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let ordinal = invocation(id, groups);
    if ordinal > parameters.path_count {
        return;
    }
    path_footprints[ordinal] = footprint_offsets[path_base[ordinal]];
}

// One invocation per covered cell: a bounded search instead of a cell loop.
@compute @workgroup_size(block_size)
fn emit_footprint(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = invocation(id, groups);
    let segments_total = path_base[parameters.path_count];
    if index >= footprint_offsets[segments_total] {
        return;
    }
    let segment = search_footprints(index, segments_total);
    let range = cell_range(segments[segment]);
    let local = index - footprint_offsets[segment];
    let width = range.z - range.x + 1u;
    let cell = (range.y + local / width) * parameters.grid_size + range.x + local % width;
    footprint[index] = Footprint(segment, cell);
    atomicAdd(&cell_counts[cell], 1u);
}

@compute @workgroup_size(block_size)
fn fill_cells(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = invocation(id, groups);
    if index >= footprint_offsets[path_base[parameters.path_count]] {
        return;
    }
    let entry = footprint[index];
    let start = atomicLoad(&cell_counts[entry.cell]);
    let slot = atomicAdd(&cell_cursor[entry.cell], 1u);
    if slot >= atomicLoad(&cell_counts[entry.cell + 1u]) - start {
        atomicOr(&status[count_mismatch], 1u);
        return;
    }
    entries[start + slot] = entry.segment;
}

// Work items per batch footprint: one per `item_entries` partners in its cell.
@compute @workgroup_size(block_size)
fn count_items(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = invocation(id, groups);
    if index > parameters.footprint_count {
        return;
    }
    var items = 0u;
    if index < parameters.footprint_count {
        let cell = footprint[parameters.footprint_start + index].cell;
        let occupancy = atomicLoad(&cell_counts[cell + 1u]) - atomicLoad(&cell_counts[cell]);
        items = occupancy / item_entries + min(occupancy % item_entries, 1u);
    }
    weights[index] = items;
}

// Each partner pair is emitted once: by its lower-ordinal source, in the low
// corner of their cell-range intersection, by the item holding the partner's
// entry. Expanded boxes overlap only when those ranges intersect.
fn visit(item: u32, write: bool, offset: u32, expected: u32) -> u32 {
    let local = search_weights(item, parameters.footprint_count);
    let chunk = item - weights[local];
    let entry = footprint[parameters.footprint_start + local];
    let source = segments[entry.segment];
    let range = cell_range(source);
    let x = entry.cell % parameters.grid_size;
    let y = entry.cell / parameters.grid_size;
    let first = atomicLoad(&cell_counts[entry.cell]) + chunk * item_entries;
    let end = min(first + item_entries, atomicLoad(&cell_counts[entry.cell + 1u]));
    var found = 0u;
    for (var index = first; index < end; index += 1u) {
        let partner = segments[entries[index]];
        let overlaps = all(source.low <= partner.high) && all(partner.low <= source.high);
        if partner.path <= source.path || !overlaps {
            continue;
        }
        let other = cell_range(partner);
        if max(range.x, other.x) != x || max(range.y, other.y) != y {
            continue;
        }
        if write {
            if found >= expected {
                atomicOr(&status[count_mismatch], 1u);
                return found;
            }
            output[offset + found] = Candidate(source.path, partner.path, source.index, partner.index);
        }
        found += 1u;
    }
    return found;
}

// Items are dispatched in bounded chunks starting at `item_start`. The extra
// item `item_count` writes zero so the scan also yields the batch total.
@compute @workgroup_size(block_size)
fn count_candidates(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let item = parameters.item_start + invocation(id, groups);
    if item > parameters.item_count {
        return;
    }
    var count = 0u;
    if item < parameters.item_count {
        count = visit(item, false, 0u, 0u);
    }
    item_counts[item] = count;
}

@compute @workgroup_size(block_size)
fn fill_candidates(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let item = parameters.item_start + invocation(id, groups);
    if item >= parameters.item_count {
        return;
    }
    let offset = item_counts[item];
    let expected = item_counts[item + 1u] - offset;
    if offset > parameters.capacity || expected > parameters.capacity - offset {
        atomicOr(&status[capacity_exceeded], 1u);
        return;
    }
    if visit(item, true, offset, expected) != expected {
        atomicOr(&status[count_mismatch], 1u);
    }
}

// Exclusive scan of `values[0..count]` in blocks; block totals go to `sums`.
struct ScanParameters {
    count: u32,
    padding_0: u32,
    padding_1: u32,
    padding_2: u32,
}

@group(1) @binding(6) var<uniform> scan_parameters: ScanParameters;
@group(1) @binding(7) var<storage, read_write> values: array<u32>;
@group(1) @binding(8) var<storage, read_write> sums: array<u32>;

var<workgroup> partial: array<u32, block_size>;

@compute @workgroup_size(block_size)
fn scan_blocks(
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let block = group.y * groups.x + group.x;
    let count = scan_parameters.count;
    let blocks = count / scan_block + min(count % scan_block, 1u);
    if block >= blocks {
        return;
    }
    let first = block * scan_block + 2u * local;
    let a = select(0u, values[min(first, count - 1u)], first < count);
    let b = select(0u, values[min(first + 1u, count - 1u)], first + 1u < count);
    var total = checked_add(a, b);
    partial[local] = total;
    workgroupBarrier();
    for (var step = 1u; step < block_size; step <<= 1u) {
        let addend = select(0u, partial[local - min(step, local)], local >= step);
        workgroupBarrier();
        total = checked_add(total, addend);
        partial[local] = total;
        workgroupBarrier();
    }
    let exclusive = total - a - b;
    if first < count {
        values[first] = exclusive;
    }
    if first + 1u < count {
        values[first + 1u] = exclusive + a;
    }
    if local == block_size - 1u {
        sums[block] = total;
    }
}

@compute @workgroup_size(block_size)
fn scan_add(
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let block = group.y * groups.x + group.x;
    let count = scan_parameters.count;
    let blocks = count / scan_block + min(count % scan_block, 1u);
    if block >= blocks {
        return;
    }
    let first = block * scan_block + 2u * local;
    let offset = sums[block];
    for (var index = first; index < min(first + 2u, scan_parameters.count); index += 1u) {
        values[index] = checked_add(values[index], offset);
    }
}
