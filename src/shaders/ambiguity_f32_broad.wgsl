// Broad phase and cell geometry of the gpu-f32 contact stage.
//
// Each positive segment's box, widened by an inflated half radius (p.x) and a
// margin for f32 error, covers a range of uniform grid cells; two segments
// within the contact radius share a cell. Cell coordinates are a monotone f32
// function, so widened boxes that overlap have intersecting ranges, and a pair
// of segments is examined exactly once: by the lower route, in the low corner
// of the range intersection.
//
// One invocation decides a segment pair completely. Endpoint distance tests
// seed both parameter intervals and define boundary joins. Only a cell without
// an endpoint hit may use a crossing seed, and its segment boxes must overlap.
// A noisy near-collinear crossing therefore cannot stretch an endpoint-seeded
// interval. Strip and disk clips extend the intervals as f32 estimates.
//
// Distances are compared unsquared. Each segment is reduced to a unit
// direction and a length by scaling with its larger coordinate difference, so
// normalization uses a scale of at least 2^-100. Every divisor is normal and
// every quotient is bounded; clip and crossing divisors can reach 2^-126.
// Products are a length times a unit component and cannot overflow. A product below 2^-126
// may flush to zero, an absolute error of 2^-126, never a relative one. A
// segment whose larger coordinate difference is below SMALL is a point.
//
// Work items examine at most 64 partner entries. The grid buffer holds cell
// starts at 0, fill cursors at p.h * p.h + 1, per-route cell counts after them
// and the record cursor last.

@group(0) @binding(1) var<storage, read> sg: array<u32>;
@group(0) @binding(2) var<storage, read> rt: array<u32>;
@group(0) @binding(3) var<storage, read_write> grid: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> ba: array<u32>;
@group(0) @binding(5) var<storage, read_write> bb: array<u32>;
@group(0) @binding(6) var<storage, read_write> bc: array<u32>;
@group(0) @binding(7) var<storage, read_write> bd: array<u32>;

const ITEM: u32 = 64u;

struct Segment { a: vec2<f32>, b: vec2<f32>, route: u32, index: u32 }
fn segment(s: u32) -> Segment {
    let w = 6u * s;
    return Segment(vec2<f32>(f(sg[w]), f(sg[w + 1u])), vec2<f32>(f(sg[w + 2u]), f(sg[w + 3u])), sg[w + 4u], sg[w + 5u]);
}
// Grid cell of one coordinate: p.h cells per axis over [-2, 2), clamped.
fn cell_of(value: f32) -> u32 {
    return u32(clamp(floor((value + 2.0) * p.y), 0.0, f32(p.h - 1u)));
}
// Beyond the half radius each side gets 2^-14 of the segment's largest
// coordinate magnitude. A true predicate puts the segments within the radius
// plus 2^-18 of (both lengths + radius) plus 2^-99, and a length is below
// 2 sqrt(2) times that magnitude, so the two widenings always cover it.
fn range_of(s: Segment) -> vec4<u32> {
    let reach = max(max(abs(s.a.x), abs(s.a.y)), max(abs(s.b.x), abs(s.b.y))) * 6.1035156e-5 + p.x;
    let low = min(s.a, s.b) - vec2<f32>(reach);
    let high = max(s.a, s.b) + vec2<f32>(reach);
    return vec4<u32>(cell_of(low.x), cell_of(low.y), cell_of(high.x), cell_of(high.y));
}
fn fill_base() -> u32 {
    return p.h * p.h + 1u;
}
fn route_counts() -> u32 {
    return 2u * p.h * p.h + 1u;
}

// bc: covered cells per segment (plus zero at p.n).
@compute @workgroup_size(WG)
fn foot_count(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let s = index_of(id, groups);
    if s > p.n {
        return;
    }
    var area = 0u;
    if s < p.n {
        let range = range_of(segment(s));
        area = (range.z - range.x + 1u) * (range.w - range.y + 1u);
    }
    bc[s] = area;
}

// Footprint entry e of p.n: (segment, cell) in segment order. bc: scanned
// counts of p.a segments. ba: entries. Counts each cell's occupancy.
@compute @workgroup_size(WG)
fn foot_emit(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let e = index_of(id, groups);
    if e >= p.n {
        return;
    }
    var low = 0u;
    var high = p.a;
    while high - low > 1u {
        let middle = low + (high - low) / 2u;
        if bc[middle] <= e {
            low = middle;
        } else {
            high = middle;
        }
    }
    let range = range_of(segment(low));
    let local = e - bc[low];
    let width = range.z - range.x + 1u;
    let cell = (range.y + local / width) * p.h + range.x + local % width;
    ba[2u * e] = low;
    ba[2u * e + 1u] = cell;
    atomicAdd(&grid[cell], 1u);
}

// ba: entries. bb: segments by cell (order within a cell is arbitrary and does
// not reach any result).
@compute @workgroup_size(WG)
fn cell_fill(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let e = index_of(id, groups);
    if e >= p.n {
        return;
    }
    let cell = ba[2u * e + 1u];
    let slot = atomicLoad(&grid[cell]) + atomicAdd(&grid[fill_base() + cell], 1u);
    bb[slot] = ba[2u * e];
}

// ba: entries. bc: work items per entry (plus zero at p.n).
@compute @workgroup_size(WG)
fn item_weights(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let e = index_of(id, groups);
    if e > p.n {
        return;
    }
    var items = 0u;
    if e < p.n {
        let cell = ba[2u * e + 1u];
        let occupancy = atomicLoad(&grid[cell + 1u]) - atomicLoad(&grid[cell]);
        items = (occupancy + ITEM - 1u) / ITEM;
    }
    bc[e] = items;
}

// First work item of each route (plus the total at p.n). ba: scanned segment
// footprint counts. bb: scanned item weights. bc: output.
@compute @workgroup_size(WG)
fn route_items(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let o = index_of(id, groups);
    if o > p.n {
        return;
    }
    var first = p.a;
    if o < p.n {
        first = rt[4u * o];
    }
    bc[o] = bb[ba[first]];
}

// ---- continuous cell geometry (derivation: unsquared, normalized distances).
struct Interval { low: f32, high: f32, some: bool }
fn cross2(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}
fn include(result: Interval, low: f32, high: f32) -> Interval {
    if !result.some {
        return Interval(low, high, true);
    }
    return Interval(min(result.low, low), max(result.high, high), true);
}
// Clips `interval` to the parameters where value + t * slope lies in [low, high].
// A zero or subnormal slope is constant over the segment. Roots outside [-2, 2]
// are equivalent for clipping a subset of [0, 1]. Clamp their numerators first
// so every quotient stays finite, even for the smallest normal slope.
fn clip_linear(interval: Interval, value: f32, slope: f32, low: f32, high: f32) -> Interval {
    if abs(slope) < 1.1754944e-38 {
        return Interval(interval.low, interval.high, interval.some && value >= low && value <= high);
    }
    let bound = 2.0 * abs(slope);
    let first = clamp(low - value, -bound, bound) / slope;
    let last = clamp(high - value, -bound, bound) / slope;
    let result = Interval(max(interval.low, min(first, last)), min(interval.high, max(first, last)), interval.some);
    return Interval(result.low, result.high, result.some && result.low <= result.high);
}
// Unit direction and length of a segment; a point when its larger coordinate
// difference is below SMALL. The divisors are that difference (at least
// 2^-100) and a root in [1, sqrt 2].
struct Unit { n: vec2<f32>, len: f32, point: bool }
fn unit_of(s: Segment) -> Unit {
    let d = s.b - s.a;
    let scale = max(abs(d.x), abs(d.y));
    if scale < SMALL {
        return Unit(vec2<f32>(0.0), 0.0, true);
    }
    let u = d / scale;
    let w = sqrt(u.x * u.x + u.y * u.y);
    return Unit(u / w, scale * w, false);
}
// Distance from `q` to segment `s`: to an endpoint beyond its ends, otherwise
// the unsquared perpendicular.
fn point_distance(q: vec2<f32>, s: Segment, u: Unit) -> f32 {
    // Preserve exact shared vertices at radius zero. The projected length and
    // independently normalized segment length need not round identically.
    if all(q == s.a) || all(q == s.b) {
        return 0.0;
    }
    let relative = q - s.a;
    if u.point {
        return norm(relative);
    }
    let along = dot(relative, u.n);
    if along <= 0.0 {
        return norm(relative);
    }
    if along >= u.len {
        return norm(q - s.b);
    }
    return abs(cross2(relative, u.n));
}
// Parameter of the point of segment `s` closest to `q`. The divisor is a
// normal length and the quotient is finite before the clamp.
fn closest(q: vec2<f32>, s: Segment, u: Unit) -> f32 {
    if u.point {
        return 0.0;
    }
    return clamp(dot(q - s.a, u.n) / u.len, 0.0, 1.0);
}
// Extent estimate of the parameters of `source` within the radius of `other`:
// the strip and band of the other segment's line (skipped when that segment is
// a point) and the disks around its endpoints. A point source has no extent of
// its own.
fn extent(source: Segment, su: Unit, other: Segment, ou: Unit) -> Interval {
    let radius = p.z;
    if su.point {
        return Interval(0.0, 1.0, true);
    }
    var result = Interval(0.0, 0.0, false);
    if !ou.point {
        let relative = source.a - other.a;
        let strip = clip_linear(Interval(0.0, 1.0, true), dot(relative, ou.n), su.len * dot(su.n, ou.n), 0.0, ou.len);
        if strip.some {
            let band = clip_linear(strip, cross2(relative, ou.n), su.len * cross2(su.n, ou.n), -radius, radius);
            if band.some {
                result = include(result, band.low, band.high);
            }
        }
    }
    for (var k = 0u; k < 2u; k += 1u) {
        var point = other.a;
        if k == 1u {
            point = other.b;
        }
        let offset = point - source.a;
        let center = dot(offset, su.n);
        let perpendicular = abs(cross2(offset, su.n));
        if perpendicular <= radius {
            let half = sqrt((radius - perpendicular) * (radius + perpendicular));
            let low = max(center - half, 0.0);
            let high = min(center + half, su.len);
            if low <= high {
                result = include(result, low / su.len, high / su.len);
            }
        }
    }
    return result;
}
// Crossing parameter from two signed distances of strictly opposite sign.
fn crossing(near: f32, far: f32) -> f32 {
    let span = near - far;
    if abs(span) < 1.1754944e-38 {
        return 0.5;
    }
    return clamp(near / span, 0.0, 1.0);
}

struct Patch { along_source: Interval, along_partner: Interval, joins: u32 }
// The free-space cell of two segments: both intervals, or neither.
fn patch_of(source: Segment, partner: Segment) -> Patch {
    let radius = p.z;
    let su = unit_of(source);
    let pu = unit_of(partner);
    var along_source = Interval(0.0, 0.0, false);
    var along_partner = Interval(0.0, 0.0, false);
    var joins = 0u;
    // Endpoints within the radius of the other segment: a free point at the
    // endpoint's own parameter and at its closest point on the other segment.
    if point_distance(source.a, partner, pu) <= radius {
        let other = closest(source.a, partner, pu);
        along_source = include(along_source, 0.0, 0.0);
        along_partner = include(along_partner, other, other);
    }
    if point_distance(source.b, partner, pu) <= radius {
        let other = closest(source.b, partner, pu);
        along_source = include(along_source, 1.0, 1.0);
        along_partner = include(along_partner, other, other);
        joins |= 1u;
    }
    if point_distance(partner.a, source, su) <= radius {
        let other = closest(partner.a, source, su);
        along_partner = include(along_partner, 0.0, 0.0);
        along_source = include(along_source, other, other);
    }
    if point_distance(partner.b, source, su) <= radius {
        let other = closest(partner.b, source, su);
        along_partner = include(along_partner, 1.0, 1.0);
        along_source = include(along_source, other, other);
        joins |= 2u;
    }
    // Crossing seeds are unnecessary when endpoint predicates already seed a
    // convex patch. Near-collinear orientation noise must not stretch its hull.
    // Raw segment boxes must overlap for any geometric crossing; this also
    // encloses crossing-only candidates independently of orientation rounding.
    let box_low = max(min(source.a, source.b), min(partner.a, partner.b));
    let box_high = min(max(source.a, source.b), max(partner.a, partner.b));
    let endpoint_hit = along_source.some;
    if !endpoint_hit && !su.point && !pu.point && all(box_low <= box_high) {
        let o1 = cross2(su.n, partner.a - source.a);
        let o2 = cross2(su.n, partner.b - source.a);
        let o3 = cross2(pu.n, source.a - partner.a);
        let o4 = cross2(pu.n, source.b - partner.a);
        if ((o1 > 0.0 && o2 < 0.0) || (o1 < 0.0 && o2 > 0.0)) && ((o3 > 0.0 && o4 < 0.0) || (o3 < 0.0 && o4 > 0.0)) {
            let s = crossing(o3, o4);
            let t = crossing(o1, o2);
            along_source = include(along_source, s, s);
            along_partner = include(along_partner, t, t);
        }
    }
    if !along_source.some {
        return Patch(along_source, along_partner, 0u);
    }
    let wide_source = extent(source, su, partner, pu);
    if wide_source.some {
        if endpoint_hit {
            along_source = include(along_source, wide_source.low, wide_source.high);
        } else {
            // Do not hull two independently rounded estimates of a crossing.
            // At radius zero they are points and must contribute no arc length.
            along_source = wide_source;
        }
    }
    let wide_partner = extent(partner, pu, source, su);
    if wide_partner.some {
        if endpoint_hit {
            along_partner = include(along_partner, wide_partner.low, wide_partner.high);
        } else {
            along_partner = wide_partner;
        }
    }
    return Patch(along_source, along_partner, joins);
}

struct Key { low: u32, high: u32 }
fn push(key: Key, bits: u32, value: u32) -> Key {
    if bits == 0u {
        return key;
    }
    return Key((key.low << bits) | value, (key.high << bits) | (key.low >> (32u - bits)));
}

// Examines one work item. Items [p.g + invocation) below p.n belong to entries
// [p.a, p.b) with scanned weights in bc. Counting adds present cells to the
// source route's count; writing appends records (8 words: key low, key high,
// interval on the source, interval on the partner, join flags, zero) to bd.
// p.c: record capacity. p.d: partner bits. p.e: segment bits. p.f: first source
// route of the batch. p.z: radius.
fn visit(item: u32, write: bool) {
    var low = p.a;
    var high = p.b;
    while high - low > 1u {
        let middle = low + (high - low) / 2u;
        if bc[middle] <= item {
            low = middle;
        } else {
            high = middle;
        }
    }
    let chunk = item - bc[low];
    let cell = ba[2u * low + 1u];
    let source = segment(ba[2u * low]);
    let range = range_of(source);
    let x = cell % p.h;
    let y = cell / p.h;
    let first = atomicLoad(&grid[cell]) + chunk * ITEM;
    let end = min(first + ITEM, atomicLoad(&grid[cell + 1u]));
    var found = 0u;
    for (var index = first; index < end; index += 1u) {
        let partner = segment(bb[index]);
        if partner.route <= source.route {
            continue;
        }
        let other = range_of(partner);
        if max(range.x, other.x) != x || max(range.y, other.y) != y {
            continue;
        }
        let found_cell = patch_of(source, partner);
        if !found_cell.along_source.some {
            continue;
        }
        let along_source = found_cell.along_source;
        let along_partner = found_cell.along_partner;
        found += 1u;
        if !write {
            continue;
        }
        let slot = atomicAdd(&grid[route_counts() + p.i], 1u);
        if slot >= p.c {
            fail(E_CAPACITY);
            return;
        }
        var key = Key(source.route - p.f, 0u);
        key = push(key, p.d, partner.route);
        key = push(key, p.e, source.index);
        key = push(key, p.e, partner.index);
        let w = 8u * slot;
        bd[w] = key.low;
        bd[w + 1u] = key.high;
        bd[w + 2u] = bitcast<u32>(along_source.low);
        bd[w + 3u] = bitcast<u32>(along_source.high);
        bd[w + 4u] = bitcast<u32>(along_partner.low);
        bd[w + 5u] = bitcast<u32>(along_partner.high);
        bd[w + 6u] = found_cell.joins;
        bd[w + 7u] = 0u;
    }
    if !write && found > 0u {
        let before = atomicAdd(&grid[route_counts() + source.route], found);
        if before + found < before {
            fail(E_OVERFLOW);
        }
    }
}

@compute @workgroup_size(WG)
fn count_cells(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let item = p.g + index_of(id, groups);
    if item < p.n {
        visit(item, false);
    }
}

// p.i: routes (the record cursor follows the per-route counts).
@compute @workgroup_size(WG)
fn fill_cells(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let item = p.g + index_of(id, groups);
    if item < p.n {
        visit(item, true);
    }
}
