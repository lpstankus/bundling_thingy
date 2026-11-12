// Per-frame tables of the gpu-f32 ambiguity backend, derived on the device from
// the current resident points: layout validation, positive segments in route
// order and per-route segment ranges.
//
// A positive segment ends at a point whose normalized bits differ from its
// predecessor's. Segment records are six words: start x, start y, end x, end y
// (f32 bits), route ordinal and index within the route. Route records are four
// words: first segment, segment count, closed flag and reservation ID.

@group(0) @binding(1) var<storage, read> points: array<RawPoint>;
@group(0) @binding(2) var<storage, read> paths: array<Reservation>;
@group(0) @binding(3) var<storage, read_write> routes: array<Route>;
@group(0) @binding(4) var<storage, read_write> fa: array<u32>;
@group(0) @binding(5) var<storage, read_write> fb: array<u32>;
@group(0) @binding(6) var<storage, read_write> fc: array<u32>;
@group(0) @binding(7) var<storage, read_write> fd: array<u32>;

// fa: mark per raw point (plus zero at p.n). fd: ordinal + 1 by ID. p.a:
// reservations. Every point must lie inside its eligible reservation and be
// finite and admitted. A point is marked when its normalized bits differ from
// its predecessor's. p.b = 1 also marks each route's first point: the keep
// marks of the originals. p.b = 0 gives the positive segment ends of a frame.
@compute @workgroup_size(WG)
fn point_mark(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = index_of(id, groups);
    if i > p.n {
        return;
    }
    fa[i] = 0u;
    if i == p.n {
        return;
    }
    let point = points[i];
    if point.edge_id >= p.a {
        fail(E_LAYOUT);
        return;
    }
    let path = paths[point.edge_id];
    if i < path.point_offset || i - path.point_offset >= path.count || fd[point.edge_id] == 0u {
        fail(E_LAYOUT);
        return;
    }
    if !(is_finite_bits(point.x) && is_finite_bits(point.y)) {
        fail(E_NONFINITE);
        return;
    }
    if !(admitted_bits(point.x) && admitted_bits(point.y)) {
        fail(E_RANGE);
        return;
    }
    var mark = p.b == 1u && i == path.point_offset;
    if i > path.point_offset {
        let previous = points[i - 1u];
        mark = normal_zero(point.x) != normal_zero(previous.x) || normal_zero(point.y) != normal_zero(previous.y);
    }
    fa[i] = select(0u, 1u, mark);
}

// fa: scanned marks. fb: positive segments per route (plus zero at p.n). The
// reservation must equal the frozen one and own each of its points.
@compute @workgroup_size(WG)
fn frame_counts(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let o = index_of(id, groups);
    if o > p.n {
        return;
    }
    fb[o] = 0u;
    if o == p.n {
        return;
    }
    let route = routes[o];
    let path = paths[route.id];
    if path.point_offset != route.point_offset || path.count != route.count {
        fail(E_LAYOUT);
        return;
    }
    for (var k = route.point_offset; k < route.point_offset + route.count; k += 1u) {
        if points[k].edge_id != route.id {
            fail(E_LAYOUT);
            return;
        }
    }
    fb[o] = fa[route.point_offset + route.count] - fa[route.point_offset];
}

// fa: scanned marks. fb: scanned segment counts. fc: segment records.
// fd: ordinal + 1 by ID.
@compute @workgroup_size(WG)
fn frame_segments(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = index_of(id, groups);
    if i >= p.n || fa[i + 1u] == fa[i] {
        return;
    }
    let point = points[i];
    let previous = points[i - 1u];
    let o = fd[point.edge_id] - 1u;
    let local = fa[i] - fa[routes[o].point_offset];
    let slot = 6u * (fb[o] + local);
    fc[slot] = normal_zero(previous.x);
    fc[slot + 1u] = normal_zero(previous.y);
    fc[slot + 2u] = normal_zero(point.x);
    fc[slot + 3u] = normal_zero(point.y);
    fc[slot + 4u] = o;
    fc[slot + 5u] = local;
}

// fb: scanned segment counts. fc: route records. A route is closed when its
// first and last stored points are equal as values. status: S_A routes without
// a positive segment, S_C largest segment count.
@compute @workgroup_size(WG)
fn frame_routes(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let o = index_of(id, groups);
    if o >= p.n {
        return;
    }
    let route = routes[o];
    let first = points[route.point_offset];
    let last = points[route.point_offset + route.count - 1u];
    let count = fb[o + 1u] - fb[o];
    fc[4u * o] = fb[o];
    fc[4u * o + 1u] = count;
    fc[4u * o + 2u] = select(0u, 1u, normal_zero(first.x) == normal_zero(last.x)
        && normal_zero(first.y) == normal_zero(last.y));
    fc[4u * o + 3u] = route.id;
    if count == 0u {
        atomicAdd(&status[S_A], 1u);
    }
    atomicMax(&status[S_C], count);
}

// fa: segment records. fb: route records. Counts positive routes whose f32
// length sum is below the minimum contact length p.x (diagnostic). status: S_B.
@compute @workgroup_size(WG)
fn frame_short(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let o = index_of(id, groups);
    if o >= p.n {
        return;
    }
    let base = fb[4u * o];
    let count = fb[4u * o + 1u];
    var total = 0.0;
    for (var k = 0u; k < count; k += 1u) {
        let slot = 6u * (base + k);
        total += norm(vec2<f32>(f(fa[slot + 2u]), f(fa[slot + 3u])) - vec2<f32>(f(fa[slot]), f(fa[slot + 1u])));
    }
    if count > 0u && total < p.x {
        atomicAdd(&status[S_B], 1u);
    }
}
