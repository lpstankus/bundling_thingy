// Scoring of one frame's contact pairs in the gpu-f32 backend.
//
// The pair log holds (lower ordinal, higher ordinal) in increasing order. Score
// records are eight words: both ordinals, flags (1 general, 2 endpoint), the
// squared general excess from the lower and upper route distance, the squared
// endpoint excess, and both cohorts. Same-cohort pairs score zero; every other
// pair is assessed with the representatives of its routes, the later canonical
// route first, exactly as cohort inference does. Baseline evidence is the
// baseline frame's incompatible records in the same order.
//
// Counts are exact integers. Sums are f32, reduced in index order within
// blocks of 256 rows and then across blocks, level by level.

@group(0) @binding(1) var<storage, read> pairs: array<u32>;
@group(0) @binding(2) var<storage, read> evidence: array<u32>;
@group(0) @binding(3) var<storage, read_write> routes: array<Route>;
@group(0) @binding(4) var<storage, read_write> sa: array<u32>;
@group(0) @binding(5) var<storage, read_write> sb: array<u32>;
@group(0) @binding(6) var<storage, read_write> sc: array<u32>;
@group(0) @binding(7) var<storage, read_write> sd: array<atomic<u32>>;

const ROW: u32 = 12u;
const REDUCE: u32 = 256u;

fn before(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
}
// Index of the pair in the p.e baseline evidence records, or NONE.
fn evidence_of(key: vec2<u32>, count: u32) -> u32 {
    var low = 0u;
    var high = count;
    while low < high {
        let middle = low + (high - low) / 2u;
        if before(vec2<u32>(evidence[8u * middle], evidence[8u * middle + 1u]), key) {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }
    if low < count && evidence[8u * low] == key.x && evidence[8u * low + 1u] == key.y {
        return low;
    }
    return NONE;
}

// sa: score records. sb: 1 at pairs that need an assessment (plus zero at p.n).
// A pair with frozen baseline evidence (p.e records) reuses its flags and
// excesses; a same-cohort pair scores zero.
@compute @workgroup_size(WG)
fn score_init(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let k = index_of(id, groups);
    if k > p.n {
        return;
    }
    if k == p.n {
        sb[k] = 0u;
        return;
    }
    let a = pairs[2u * k];
    let b = pairs[2u * k + 1u];
    let first = routes[a].cohort;
    let second = routes[b].cohort;
    let w = 8u * k;
    sa[w] = a;
    sa[w + 1u] = b;
    sa[w + 2u] = 0u;
    sa[w + 3u] = 0u;
    sa[w + 4u] = 0u;
    sa[w + 5u] = 0u;
    sa[w + 6u] = first;
    sa[w + 7u] = second;
    let known = evidence_of(vec2<u32>(a, b), p.e);
    if known != NONE {
        for (var word = 2u; word < 6u; word += 1u) {
            sa[w + word] = evidence[8u * known + word];
        }
    }
    sb[k] = select(0u, 1u, first != second && known == NONE);
}

// Assessment pair at word `slot` of sc: the representatives of both routes,
// the later canonical route first.
fn write_pair(slot: u32, a: Route, b: Route) {
    let later = a.position > b.position;
    sc[slot] = select(b.rep, a.rep, later);
    sc[slot + 1u] = select(a.rep, b.rep, later);
}

// Assessment pairs of needed pairs [p.a, p.a + p.b). sb: scanned marks.
// sc: pairs (16 words).
@compute @workgroup_size(WG)
fn score_pairs(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let k = index_of(id, groups);
    if k >= p.n || sb[k + 1u] == sb[k] || sb[k] < p.a || sb[k] - p.a >= p.b {
        return;
    }
    write_pair(16u * (sb[k] - p.a), routes[sa[8u * k]], routes[sa[8u * k + 1u]]);
}

// Diagnostic: assessment pairs of the p.n logged pairs themselves, whatever
// their cohorts. sc: pairs (16 words).
@compute @workgroup_size(WG)
fn probe_pairs(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let k = index_of(id, groups);
    if k >= p.n {
        return;
    }
    write_pair(16u * k, routes[pairs[2u * k]], routes[pairs[2u * k + 1u]]);
}

fn excess(value: f32, tolerance: f32) -> f32 {
    let over = max(value - tolerance, 0.0);
    return over * over;
}

// Stores the completed assessments of needed pairs [p.a, p.a + p.b). p.x: route
// tolerance. p.y: endpoint tolerance.
@compute @workgroup_size(WG)
fn score_collect(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let k = index_of(id, groups);
    if k >= p.n || sb[k + 1u] == sb[k] || sb[k] < p.a || sb[k] - p.a >= p.b {
        return;
    }
    let slot = 16u * (sb[k] - p.a) + 12u;
    let flags = sc[slot + 3u];
    if (flags & 8u) == 0u {
        fail(E_INTERNAL);
        return;
    }
    let w = 8u * k;
    sa[w + 2u] = (flags >> 1u) & 3u;
    sa[w + 3u] = bitcast<u32>(excess(f(sc[slot]), p.x));
    sa[w + 4u] = bitcast<u32>(excess(f(sc[slot + 1u]), p.x));
    sa[w + 5u] = bitcast<u32>(excess(f(sc[slot + 2u]), p.y));
}

fn in_pairs(key: vec2<u32>, count: u32) -> bool {
    var low = 0u;
    var high = count;
    while low < high {
        let middle = low + (high - low) / 2u;
        if before(vec2<u32>(pairs[2u * middle], pairs[2u * middle + 1u]), key) {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }
    return low < count && pairs[2u * low] == key.x && pairs[2u * low + 1u] == key.y;
}

// Reduction rows: p.n rows of current pairs, then p.e rows of baseline evidence
// (zero in the baseline frame, p.f = 1). Columns 0..2 current sums, 3..5
// introduced, 6..8 resolved. sa: score records. sb: rows. status: S_A/S_B
// current general/endpoint, S_C/S_D introduced, S_E/S_F resolved.
@compute @workgroup_size(WG)
fn score_rows(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let j = index_of(id, groups);
    if j >= p.n + p.e {
        return;
    }
    let out = ROW * j;
    for (var column = 0u; column < ROW; column += 1u) {
        sb[out + column] = 0u;
    }
    if j < p.n {
        let w = 8u * j;
        let flags = sa[w + 2u];
        for (var column = 0u; column < 3u; column += 1u) {
            sb[out + column] = sa[w + 3u + column];
        }
        if (flags & 1u) != 0u {
            atomicAdd(&status[S_A], 1u);
        }
        if (flags & 2u) != 0u {
            atomicAdd(&status[S_B], 1u);
        }
        if flags != 0u && p.f == 0u && evidence_of(vec2<u32>(sa[w], sa[w + 1u]), p.e) == NONE {
            if (flags & 1u) != 0u {
                atomicAdd(&status[S_C], 1u);
                sb[out + 3u] = sa[w + 3u];
                sb[out + 4u] = sa[w + 4u];
            }
            if (flags & 2u) != 0u {
                atomicAdd(&status[S_D], 1u);
                sb[out + 5u] = sa[w + 5u];
            }
        }
        return;
    }
    let w = 8u * (j - p.n);
    if in_pairs(vec2<u32>(evidence[w], evidence[w + 1u]), p.n) {
        return;
    }
    let flags = evidence[w + 2u];
    if (flags & 1u) != 0u {
        atomicAdd(&status[S_E], 1u);
        sb[out + 6u] = evidence[w + 3u];
        sb[out + 7u] = evidence[w + 4u];
    }
    if (flags & 2u) != 0u {
        atomicAdd(&status[S_F], 1u);
        sb[out + 8u] = evidence[w + 5u];
    }
}

// Sums rows [256 b, 256 b + 256) of the p.b rows at row p.a into row p.c + b.
@compute @workgroup_size(WG)
fn reduce(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let block = index_of(id, groups);
    if block >= p.n {
        return;
    }
    var sums: array<f32, 12>;
    for (var column = 0u; column < ROW; column += 1u) {
        sums[column] = 0.0;
    }
    let end = min(p.b, (block + 1u) * REDUCE);
    for (var row = block * REDUCE; row < end; row += 1u) {
        for (var column = 0u; column < ROW; column += 1u) {
            sums[column] += f(sb[ROW * (p.a + row) + column]);
        }
    }
    for (var column = 0u; column < ROW; column += 1u) {
        sb[ROW * (p.c + block) + column] = bitcast<u32>(sums[column]);
    }
}

// sa: score records. sb: 1 at incompatible records (p.a = 0) or at cross-cohort
// records (p.a = 1), plus zero at p.n.
@compute @workgroup_size(WG)
fn record_marks(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let k = index_of(id, groups);
    if k > p.n {
        return;
    }
    var mark = false;
    if k < p.n {
        if p.a == 0u {
            mark = sa[8u * k + 2u] != 0u;
        } else {
            mark = sa[8u * k + 6u] != sa[8u * k + 7u];
        }
    }
    sb[k] = select(0u, 1u, mark);
}

// sb: scanned incompatible marks. sc: baseline evidence records.
@compute @workgroup_size(WG)
fn evidence_write(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let k = index_of(id, groups);
    if k >= p.n || sb[k + 1u] == sb[k] {
        return;
    }
    for (var word = 0u; word < 8u; word += 1u) {
        sc[8u * sb[k] + word] = sa[8u * k + word];
    }
}

// sb: scanned cross-cohort marks. sc: sort records keyed by (smaller cohort,
// larger cohort) carrying whether the pair is an incompatible witness.
@compute @workgroup_size(WG)
fn cross_keys(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let k = index_of(id, groups);
    if k >= p.n || sb[k + 1u] == sb[k] {
        return;
    }
    let first = sa[8u * k + 6u];
    let second = sa[8u * k + 7u];
    let slot = 3u * sb[k];
    sc[slot] = max(first, second);
    sc[slot + 1u] = min(first, second);
    sc[slot + 2u] = select(0u, 1u, sa[8u * k + 2u] != 0u);
}

// sa: sorted cross records. sb: scanned unique marks. sd: 1 at unique cohort
// pairs with an incompatible contacting witness.
@compute @workgroup_size(WG)
fn witness_marks(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let s = index_of(id, groups);
    if s >= p.n || sa[3u * s + 2u] == 0u {
        return;
    }
    atomicStore(&sd[sb[s + 1u] - 1u], 1u);
}

// sa: sorted cross records. sb: scanned unique marks. sd: scanned witness
// marks. sc: witnessed cohort pairs (smaller, larger) in order.
@compute @workgroup_size(WG)
fn witness_list(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let s = index_of(id, groups);
    if s >= p.n || sb[s + 1u] == sb[s] {
        return;
    }
    let slot = atomicLoad(&sd[sb[s]]);
    if atomicLoad(&sd[sb[s] + 1u]) == slot {
        return;
    }
    sc[2u * slot] = sa[3u * s + 1u];
    sc[2u * slot + 1u] = sa[3u * s];
}

// Counts the p.n ordered cohort pairs of sa absent from the p.b ordered cohort
// pairs of sb into status[p.c].
@compute @workgroup_size(WG)
fn witness_difference(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let s = index_of(id, groups);
    if s >= p.n {
        return;
    }
    let key = vec2<u32>(sa[2u * s], sa[2u * s + 1u]);
    var low = 0u;
    var high = p.b;
    while low < high {
        let middle = low + (high - low) / 2u;
        if before(vec2<u32>(sb[2u * middle], sb[2u * middle + 1u]), key) {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }
    if !(low < p.b && sb[2u * low] == key.x && sb[2u * low + 1u] == key.y) {
        atomicAdd(&status[p.c], 1u);
    }
}

// Final scalars of p.n contacts from the reduced row p.a of sb, into sc:
// words 0..8 the nine sums, 9..11 RMS excess (general lower, general upper,
// endpoint), 12..14 the ratios to the tolerances (p.x route, p.y endpoint),
// 15 the witnessed fraction of the p.c contacting cross-cohort pairs (p.b
// witnessed). Empty sets give zero.
@compute @workgroup_size(1)
fn score_finish(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x != 0u || id.y != 0u {
        return;
    }
    for (var column = 0u; column < 9u; column += 1u) {
        var value = 0u;
        if p.n + p.e > 0u {
            value = sb[ROW * p.a + column];
        }
        sc[column] = value;
    }
    for (var view = 0u; view < 3u; view += 1u) {
        var rms = 0.0;
        if p.n > 0u {
            rms = sqrt(f(sc[view]) / f32(p.n));
        }
        sc[9u + view] = bitcast<u32>(rms);
        sc[12u + view] = bitcast<u32>(rms / select(p.x, p.y, view == 2u));
    }
    var fraction = 0.0;
    if p.c > 0u {
        fraction = f32(p.b) / f32(p.c);
    }
    sc[15u] = bitcast<u32>(fraction);
}
