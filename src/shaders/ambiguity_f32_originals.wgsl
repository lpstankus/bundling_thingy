// Frozen original routes of the gpu-f32 ambiguity backend: eligible routes and
// layout validation, canonical geometry, canonical order and identity,
// index-correspondence simplification and first-fit complete-link cohorts.
//
// wa..wd are generic word buffers; each kernel documents its view. All orderings
// and identity tests are exact integer comparisons of normalized bit patterns.
// Collinearity is an exact integer determinant. Floating-point appears only in
// the simplifier deviation and the endpoint predicate, as f32 point estimates.
// Sort records are three words (key low, key high, payload). Loops over one
// route are bounded by the route limit enforced in `route_init`.

@group(0) @binding(1) var<storage, read> points: array<RawPoint>;
@group(0) @binding(2) var<storage, read> paths: array<Reservation>;
@group(0) @binding(3) var<storage, read_write> routes: array<Route>;
@group(0) @binding(4) var<storage, read_write> wa: array<u32>;
@group(0) @binding(5) var<storage, read_write> wb: array<u32>;
@group(0) @binding(6) var<storage, read_write> wc: array<u32>;
@group(0) @binding(7) var<storage, read_write> wd: array<u32>;

const REVERSED: u32 = 1u;
const CLOSED: u32 = 2u;

// wa: 1 at reservations with points (plus zero at p.n).
@compute @workgroup_size(WG)
fn route_mark(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let edge = index_of(id, groups);
    if edge > p.n {
        return;
    }
    wa[edge] = select(0u, 1u, edge < p.n && paths[edge].count > 0u);
}

// wa: scanned marks (route ordinals by increasing ID). wd: ordinal + 1 by ID.
// p.b: active points; p.c: largest admitted point count of one route. Every
// point of a route must carry its ID; with the per-point range check of
// `point_mark` (frame kernels) this proves that the routes tile the active
// points.
@compute @workgroup_size(WG)
fn route_init(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let edge = index_of(id, groups);
    if edge >= p.n || wa[edge + 1u] == wa[edge] {
        return;
    }
    let path = paths[edge];
    let end = path.point_offset + path.count;
    if path.count < 2u || end > p.b || end < path.point_offset {
        fail(E_LAYOUT);
        return;
    }
    if path.count > p.c {
        fail(E_ROUTE);
        return;
    }
    for (var k = path.point_offset; k < end; k += 1u) {
        if points[k].edge_id != edge {
            fail(E_LAYOUT);
            return;
        }
    }
    let o = wa[edge];
    routes[o].id = edge;
    routes[o].point_offset = path.point_offset;
    routes[o].count = path.count;
    routes[o].scratch = 0u;
    wd[edge] = o + 1u;
}

// wa: scanned keep marks. wb: deduplicated points (x, y, ordinal). wd: ordinal table.
@compute @workgroup_size(WG)
fn dedup_write(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = index_of(id, groups);
    if i >= p.n || wa[i + 1u] == wa[i] {
        return;
    }
    let point = points[i];
    let slot = 3u * wa[i];
    wb[slot] = normal_zero(point.x);
    wb[slot + 1u] = normal_zero(point.y);
    wb[slot + 2u] = wd[point.edge_id] - 1u;
}

// wa: scanned keep marks. Stores each route's deduplicated range.
@compute @workgroup_size(WG)
fn route_dedup(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let o = index_of(id, groups);
    if o >= p.n {
        return;
    }
    let start = wa[routes[o].point_offset];
    routes[o].canon_start = start;
    routes[o].canon_count = wa[routes[o].point_offset + routes[o].count] - start;
}

// ---- exact collinearity: the 2x2 determinant sum of six products of admitted
// f32 values (subnormals included), accumulated in a 384-bit two's complement
// integer scaled by 2^298.
const LIMBS: u32 = 12u;
struct Big { w: array<u32, 12> }
fn big_add(acc: ptr<function, Big>, bits_a: u32, bits_b: u32, negate: bool) {
    let exponent_a = (bits_a >> 23u) & 0xffu;
    let exponent_b = (bits_b >> 23u) & 0xffu;
    let ma = (bits_a & 0x7fffffu) | select(0u, 0x800000u, exponent_a > 0u);
    let mb = (bits_b & 0x7fffffu) | select(0u, 0x800000u, exponent_b > 0u);
    if ma == 0u || mb == 0u {
        return;
    }
    // Each value is its integer mantissa times 2^(max(exponent, 1) - 150);
    // admitted exponents are at most 146, so the shift is within [0, 290].
    let shift = max(exponent_a, 1u) + max(exponent_b, 1u) - 2u;
    let al = ma & 0xfffu;
    let ah = ma >> 12u;
    let bl = mb & 0xfffu;
    let bh = mb >> 12u;
    let ll = al * bl;
    let mid = ah * bl + al * bh;
    let hh = ah * bh;
    let first = ll + (mid << 12u);
    let carry_first = select(0u, 1u, first < ll);
    let lo = first + (hh << 24u);
    let carry_second = select(0u, 1u, lo < first);
    let hi = (mid >> 20u) + (hh >> 8u) + carry_first + carry_second;
    var term: Big;
    for (var k = 0u; k < LIMBS; k += 1u) {
        term.w[k] = 0u;
    }
    let limb = shift / 32u;
    let bit = shift % 32u;
    if bit == 0u {
        term.w[limb] = lo;
        term.w[limb + 1u] = hi;
    } else {
        term.w[limb] = lo << bit;
        term.w[limb + 1u] = (lo >> (32u - bit)) | (hi << bit);
        term.w[limb + 2u] = hi >> (32u - bit);
    }
    let negative = (((bits_a ^ bits_b) & 0x80000000u) != 0u) != negate;
    var carry = select(0u, 1u, negative);
    for (var k = 0u; k < LIMBS; k += 1u) {
        let addend = select(term.w[k], ~term.w[k], negative);
        let partial = (*acc).w[k] + addend;
        let total = partial + carry;
        carry = select(0u, 1u, partial < addend || total < partial);
        (*acc).w[k] = total;
    }
}
fn collinear(a: vec2<u32>, b: vec2<u32>, c: vec2<u32>) -> bool {
    var acc: Big;
    for (var k = 0u; k < LIMBS; k += 1u) {
        acc.w[k] = 0u;
    }
    big_add(&acc, a.x, b.y, false);
    big_add(&acc, b.x, c.y, false);
    big_add(&acc, c.x, a.y, false);
    big_add(&acc, a.y, b.x, true);
    big_add(&acc, b.y, c.x, true);
    big_add(&acc, c.y, a.x, true);
    var any = 0u;
    for (var k = 0u; k < LIMBS; k += 1u) {
        any |= acc.w[k];
    }
    return any == 0u;
}
fn between(a: u32, b: u32, c: u32) -> bool {
    let ka = okey(a);
    let kb = okey(b);
    let kc = okey(c);
    return kb >= min(ka, kc) && kb <= max(ka, kc);
}
fn dedup_point(j: u32) -> vec2<u32> {
    return vec2<u32>(wb[3u * j], wb[3u * j + 1u]);
}

// wb: deduplicated points. wc: keep mark (plus zero at n). A vertex is dropped
// exactly when it lies between its neighbors on their exact common line; this
// local rule equals the reference's sequential removal.
@compute @workgroup_size(WG)
fn collinear_mark(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let j = index_of(id, groups);
    if j > p.n {
        return;
    }
    if j == p.n {
        wc[j] = 0u;
        return;
    }
    let route = routes[wb[3u * j + 2u]];
    var keep = j == route.canon_start || j + 1u == route.canon_start + route.canon_count;
    if !keep {
        let a = dedup_point(j - 1u);
        let b = dedup_point(j);
        let c = dedup_point(j + 1u);
        keep = !(between(a.x, b.x, c.x) && between(a.y, b.y, c.y) && collinear(a, b, c));
    }
    wc[j] = select(0u, 1u, keep);
}

// wb: deduplicated points. wc: scanned marks. wd: canonical unoriented points.
@compute @workgroup_size(WG)
fn canon_tmp(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let j = index_of(id, groups);
    if j >= p.n || wc[j + 1u] == wc[j] {
        return;
    }
    let slot = 3u * wc[j];
    wd[slot] = wb[3u * j];
    wd[slot + 1u] = wb[3u * j + 1u];
    wd[slot + 2u] = wb[3u * j + 2u];
}

// wc: scanned marks. Replaces each route's range by its canonical range.
@compute @workgroup_size(WG)
fn route_canon(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let o = index_of(id, groups);
    if o >= p.n {
        return;
    }
    let old_start = routes[o].canon_start;
    let start = wc[old_start];
    routes[o].canon_count = wc[old_start + routes[o].canon_count] - start;
    routes[o].canon_start = start;
}

// wd: canonical unoriented points. Chooses the lexicographically smaller of the
// forward and reversed sequences, and counts closed and constant routes.
// status: S_A closed, S_B constant, S_C largest canonical point count.
@compute @workgroup_size(WG)
fn route_orient(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let o = index_of(id, groups);
    if o >= p.n {
        return;
    }
    let start = routes[o].canon_start;
    let count = routes[o].canon_count;
    if count == 0u {
        fail(E_INTERNAL);
        return;
    }
    var flags = 0u;
    for (var k = 0u; k < count; k += 1u) {
        let back = 3u * (start + count - 1u - k);
        let front = 3u * (start + k);
        let bx = okey(wd[back]);
        let fx = okey(wd[front]);
        let by = okey(wd[back + 1u]);
        let fy = okey(wd[front + 1u]);
        if bx != fx || by != fy {
            if bx < fx || (bx == fx && by < fy) {
                flags = REVERSED;
            }
            break;
        }
    }
    let last = 3u * (start + count - 1u);
    if count > 1u && wd[3u * start] == wd[last] && wd[3u * start + 1u] == wd[last + 1u] {
        flags |= CLOSED;
        atomicAdd(&status[S_A], 1u);
    }
    if count == 1u {
        atomicAdd(&status[S_B], 1u);
    }
    atomicMax(&status[S_C], count);
    routes[o].flags = flags;
}

// wd: canonical unoriented points. wa: canonical oriented points (x, y).
// wb: owning ordinal of each canonical point.
@compute @workgroup_size(WG)
fn canon_write(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = index_of(id, groups);
    if c >= p.n {
        return;
    }
    let o = wd[3u * c + 2u];
    let route = routes[o];
    let k = c - route.canon_start;
    let source = 3u * select(c, route.canon_start + route.canon_count - 1u - k, (route.flags & REVERSED) != 0u);
    wa[2u * c] = wd[source];
    wa[2u * c + 1u] = wd[source + 1u];
    wb[c] = o;
}

// wa: canonical points. wb: sort records keyed by (x, y).
@compute @workgroup_size(WG)
fn name_entries0(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = index_of(id, groups);
    if c >= p.n {
        return;
    }
    wb[3u * c] = okey(wa[2u * c + 1u]);
    wb[3u * c + 1u] = okey(wa[2u * c]);
    wb[3u * c + 2u] = c;
}

// wa: sorted records. wb: 1 where the key differs from its predecessor (plus
// zero at n). p.a = 1 compares high words only.
@compute @workgroup_size(WG)
fn key_marks(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let s = index_of(id, groups);
    if s > p.n {
        return;
    }
    var mark = 0u;
    if s < p.n {
        mark = 1u;
        if s > 0u {
            let same_high = wa[3u * s + 1u] == wa[3u * s - 2u];
            let same_low = wa[3u * s] == wa[3u * s - 3u];
            mark = select(1u, 0u, same_high && (same_low || p.a == 1u));
        }
    }
    wb[s] = mark;
}

// wa: scanned marks. wb: sorted records. wc: names (dense rank from 1).
@compute @workgroup_size(WG)
fn name_write(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let s = index_of(id, groups);
    if s >= p.n {
        return;
    }
    wc[wb[3u * s + 2u]] = wa[s + 1u];
}

// Level p.a >= 1 of block naming: block b of a route joins the names of blocks
// 2b and 2b + 1 of the previous level (zero past the end, so a proper prefix
// orders first). wa: previous names. wb: owners. wc: sort records.
@compute @workgroup_size(WG)
fn name_entries(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = index_of(id, groups);
    if c >= p.n {
        return;
    }
    let route = routes[wb[c]];
    let b = c - route.canon_start;
    let round = (1u << p.a) - 1u;
    let previous = (route.canon_count + (round >> 1u)) >> (p.a - 1u);
    let current = (route.canon_count + round) >> p.a;
    var first = 0u;
    var second = 0u;
    if b < current {
        first = wa[route.canon_start + 2u * b];
        if 2u * b + 1u < previous {
            second = wa[route.canon_start + 2u * b + 1u];
        }
    }
    wc[3u * c] = second;
    wc[3u * c + 1u] = first;
    wc[3u * c + 2u] = c;
}

// wa: final names. wb: sort records keyed by (name, ID).
@compute @workgroup_size(WG)
fn route_entries(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let o = index_of(id, groups);
    if o >= p.n {
        return;
    }
    let name = wa[routes[o].canon_start];
    routes[o].name = name;
    wb[3u * o] = routes[o].id;
    wb[3u * o + 1u] = name;
    wb[3u * o + 2u] = o;
}

// wa: scanned name marks. wb: sorted records. wc: persistent words with the
// canonical order at p.a and the unique representatives at p.b.
@compute @workgroup_size(WG)
fn route_place(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let s = index_of(id, groups);
    if s >= p.n {
        return;
    }
    let o = wb[3u * s + 2u];
    let unique = wa[s + 1u] - 1u;
    routes[o].position = s;
    routes[o].unique = unique;
    wc[p.a + s] = o;
    if wa[s + 1u] != wa[s] {
        wc[p.b + unique] = o;
    }
}

// wc: persistent words (unique representatives at p.b).
@compute @workgroup_size(WG)
fn route_rep(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let o = index_of(id, groups);
    if o >= p.n {
        return;
    }
    routes[o].rep = wc[p.b + routes[o].unique];
    routes[o].knot_count = 0u;
    routes[o].error = 0.0;
}

fn canon_point(index: u32) -> vec2<f32> {
    return vec2<f32>(f(wa[2u * index]), f(wa[2u * index + 1u]));
}

// Resumable greedy simplifier of one unique route. A shortcut from knot `start`
// to `end` is accepted when every skipped vertex lies within sigma (p.x) of the
// chord point at its index fraction; otherwise the span halves. wa: canonical
// points. wb: state (start, end, next, maximum) per unique route; p.e = 1
// initializes. wc: persistent words (representatives at p.b, knots at p.c).
// p.d bounds the steps of one invocation. status[S_PENDING] counts unfinished.
@compute @workgroup_size(WG)
fn simplify_step(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let u = index_of(id, groups);
    if u >= p.n {
        return;
    }
    let o = wc[p.b + u];
    let base = routes[o].canon_start;
    let count = routes[o].canon_count;
    let knots = p.c + base;
    let exact = p.x == 0.0;
    var start = wb[4u * u];
    var end = wb[4u * u + 1u];
    var next = wb[4u * u + 2u];
    var maximum = f(wb[4u * u + 3u]);
    var knot_count = routes[o].knot_count;
    var error = routes[o].error;
    if p.e == 1u {
        start = 0u;
        end = select(count - 1u, min(1u, count - 1u), exact);
        next = 1u;
        maximum = 0.0;
        knot_count = 1u;
        error = 0.0;
        wc[knots] = 0u;
    }
    var budget = p.d;
    while budget > 0u && start + 1u < count {
        budget -= 1u;
        if next < end {
            let a = canon_point(base + start);
            let b = canon_point(base + end);
            let fraction = f32(next - start) / f32(end - start);
            let d = canon_point(base + next) - (a + fraction * (b - a));
            maximum = max(maximum, sqrt(d.x * d.x + d.y * d.y));
            next += 1u;
            if maximum > p.x {
                end = start + (end - start) / 2u;
                next = start + 1u;
                maximum = 0.0;
            }
        } else {
            error = max(error, maximum);
            wc[knots + knot_count] = end;
            knot_count += 1u;
            start = end;
            end = select(count - 1u, min(start + 1u, count - 1u), exact);
            next = start + 1u;
            maximum = 0.0;
        }
    }
    wb[4u * u] = start;
    wb[4u * u + 1u] = end;
    wb[4u * u + 2u] = next;
    wb[4u * u + 3u] = bitcast<u32>(maximum);
    routes[o].knot_count = knot_count;
    routes[o].error = error;
    if start + 1u < count {
        atomicAdd(&status[S_PENDING], 1u);
    }
}

// Either coherent endpoint matching within the endpoint tolerance p.x.
fn endpoints_compatible(a: Route, b: Route) -> bool {
    let first_a = canon_point(a.canon_start);
    let last_a = canon_point(a.canon_start + a.canon_count - 1u);
    let first_b = canon_point(b.canon_start);
    let last_b = canon_point(b.canon_start + b.canon_count - 1u);
    return sqrt(max(sq(first_a, first_b), sq(last_a, last_b))) <= p.x
        || sqrt(max(sq(first_a, last_b), sq(last_a, first_b))) <= p.x;
}

// Visits, for unique route i, the earlier unique routes in [p.a, p.b) whose
// endpoints are compatible, in increasing order. wa: canonical points. wc:
// persistent words (representatives at p.c). p.e = 0 counts them into wb
// (accumulated over the slices). p.e = 1 fills them: wb holds the scanned
// counts, wd receives the earlier unique routes by later unique route, and the
// route's scratch word keeps its cursor between slices.
@compute @workgroup_size(WG)
fn endpoint_visit(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = index_of(id, groups);
    if i >= p.n {
        return;
    }
    let o = wc[p.c + i];
    let route = routes[o];
    let fill = p.e == 1u;
    var cursor = select(0u, route.scratch, fill);
    for (var j = p.a; j < min(p.b, i); j += 1u) {
        if endpoints_compatible(route, routes[wc[p.c + j]]) {
            if fill {
                wd[wb[i] + cursor] = j;
            }
            cursor += 1u;
        }
    }
    if fill {
        routes[o].scratch = cursor;
    } else {
        wb[i] += cursor;
    }
}

// Assessment pairs of list entries [p.a, p.a + p.n): the later unique route
// first, both as representatives. wa: list. wb: scanned counts of p.b unique
// routes. wc: persistent words (representatives at p.c). wd: pairs (16 words).
@compute @workgroup_size(WG)
fn cohort_pairs(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let t = index_of(id, groups);
    if t >= p.n {
        return;
    }
    let entry = p.a + t;
    var low = 0u;
    var high = p.b;
    while high - low > 1u {
        let middle = low + (high - low) / 2u;
        if wb[middle] <= entry {
            low = middle;
        } else {
            high = middle;
        }
    }
    wd[16u * t] = wc[p.c + low];
    wd[16u * t + 1u] = wc[p.c + (wa[entry] & 0x7fffffffu)];
}

// Marks list entries [p.a, p.a + p.n) whose completed assessment is
// incompatible. wa: list. wd: assessed pairs.
@compute @workgroup_size(WG)
fn cohort_collect(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let t = index_of(id, groups);
    if t >= p.n {
        return;
    }
    let flags = wd[16u * t + 15u];
    if (flags & 8u) == 0u {
        fail(E_INTERNAL);
        return;
    }
    if (flags & 6u) != 0u {
        wa[p.a + t] |= 0x80000000u;
    }
}

// Sequential first-fit complete-link assignment in canonical order. A unique
// route joins the earliest cohort all of whose members are compatible with it;
// members absent from its list are incompatible. Each route takes three passes
// over its list (count compatible members by cohort, choose, clear). Route,
// phase, cursor and choice persist at wc[3n..], so one invocation visits at most
// p.c entries or routes. wa: assessed list. wb: scanned counts. wc: cohort of
// each unique route at 0, cohort sizes at n, scratch counts at 2n. p.e = 1
// starts. status[S_A] receives the next route.
@compute @workgroup_size(1)
fn first_fit(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x != 0u || id.y != 0u {
        return;
    }
    let n = p.n;
    var i = wc[3u * n];
    var phase = wc[3u * n + 1u];
    var cursor = wc[3u * n + 2u];
    var best = wc[3u * n + 3u];
    if p.e == 1u {
        i = 0u;
        phase = 0u;
        cursor = 0u;
        best = NONE;
    }
    var budget = p.c;
    while i < n && budget > 0u {
        let entry = wb[i] + cursor;
        if entry < wb[i + 1u] {
            budget -= 1u;
            let compatible = (wa[entry] >> 31u) == 0u;
            let cohort = wc[wa[entry] & 0x7fffffffu];
            if phase == 0u {
                if compatible {
                    wc[2u * n + cohort] += 1u;
                }
            } else if phase == 1u {
                if compatible && wc[2u * n + cohort] == wc[n + cohort] {
                    best = min(best, cohort);
                }
            } else {
                wc[2u * n + cohort] = 0u;
            }
            cursor += 1u;
        } else if phase < 2u {
            phase += 1u;
            cursor = 0u;
        } else {
            budget -= 1u;
            if best == NONE {
                wc[i] = i;
                wc[n + i] = 1u;
            } else {
                wc[i] = best;
                wc[n + best] += 1u;
            }
            i += 1u;
            phase = 0u;
            cursor = 0u;
            best = NONE;
        }
    }
    wc[3u * n] = i;
    wc[3u * n + 1u] = phase;
    wc[3u * n + 2u] = cursor;
    wc[3u * n + 3u] = best;
    atomicStore(&status[S_A], i);
}

// wc: cohort state. wd: 1 at cohort representatives (plus zero at n).
@compute @workgroup_size(WG)
fn cohort_flags(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let u = index_of(id, groups);
    if u > p.n {
        return;
    }
    wd[u] = select(0u, 1u, u < p.n && wc[u] == u);
}

// wc: cohort state. wd: scanned flags (creation-order cohort IDs).
@compute @workgroup_size(WG)
fn cohort_write(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let o = index_of(id, groups);
    if o >= p.n {
        return;
    }
    routes[o].cohort = wd[wc[routes[o].unique]];
}

// wc: persistent words (canonical order at p.a). wd: (ID, cohort) by position.
@compute @workgroup_size(WG)
fn membership(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let s = index_of(id, groups);
    if s >= p.n {
        return;
    }
    let route = routes[wc[p.a + s]];
    wd[2u * s] = route.id;
    wd[2u * s + 1u] = route.cohort;
}
