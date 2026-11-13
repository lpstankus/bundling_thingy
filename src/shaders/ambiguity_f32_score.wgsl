// Scoring of one frame's contact pairs in the f32 ambiguity pipeline.
//
// The pair log holds (lower ordinal, higher ordinal) in increasing order. Score
// records are five words: both ordinals, the general incompatibility flag, and
// the squared general excess from the low and high route distance estimates.
// Pairs whose routes share a representative have identical canonical curves
// and score zero; every other pair is assessed with the representatives of its
// routes, the later canonical route first. Baseline evidence is the baseline
// frame's incompatible records in the same order and layout.
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

const RECORD: u32 = 5u;
const ROW: u32 = 6u;
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
        if before(vec2<u32>(evidence[RECORD * middle], evidence[RECORD * middle + 1u]), key) {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }
    if low < count && evidence[RECORD * low] == key.x && evidence[RECORD * low + 1u] == key.y {
        return low;
    }
    return NONE;
}

// sa: score records. sb: 1 at pairs that need an assessment (plus zero at p.n).
// A pair with frozen baseline evidence (p.e records) reuses its flag and
// excesses; a pair of routes with one representative scores zero.
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
    let w = RECORD * k;
    sa[w] = a;
    sa[w + 1u] = b;
    sa[w + 2u] = 0u;
    sa[w + 3u] = 0u;
    sa[w + 4u] = 0u;
    let known = evidence_of(vec2<u32>(a, b), p.e);
    if known != NONE {
        for (var word = 2u; word < RECORD; word += 1u) {
            sa[w + word] = evidence[RECORD * known + word];
        }
    }
    sb[k] = select(0u, 1u, routes[a].rep != routes[b].rep && known == NONE);
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
    write_pair(16u * (sb[k] - p.a), routes[sa[RECORD * k]], routes[sa[RECORD * k + 1u]]);
}

fn excess(value: f32, tolerance: f32) -> f32 {
    let over = max(value - tolerance, 0.0);
    return over * over;
}

// Stores the completed assessments of needed pairs [p.a, p.a + p.b). p.x: route
// tolerance. The general flag is the distance stage's OUT_GENERAL bit alone.
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
    let w = RECORD * k;
    sa[w + 2u] = (flags >> 1u) & 1u;
    sa[w + 3u] = bitcast<u32>(excess(f(sc[slot]), p.x));
    sa[w + 4u] = bitcast<u32>(excess(f(sc[slot + 1u]), p.x));
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
// (zero in the baseline frame, p.f = 1). Columns 0..1 current low/high sums over
// every contact, 2..3 introduced, 4..5 resolved. sa: score records. sb: rows.
// status: S_A current incompatible, S_B introduced, S_C resolved. Every
// evidence record is incompatible.
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
        let w = RECORD * j;
        sb[out] = sa[w + 3u];
        sb[out + 1u] = sa[w + 4u];
        if sa[w + 2u] == 0u {
            return;
        }
        atomicAdd(&status[S_A], 1u);
        if p.f == 0u && evidence_of(vec2<u32>(sa[w], sa[w + 1u]), p.e) == NONE {
            atomicAdd(&status[S_B], 1u);
            sb[out + 2u] = sa[w + 3u];
            sb[out + 3u] = sa[w + 4u];
        }
        return;
    }
    let w = RECORD * (j - p.n);
    if in_pairs(vec2<u32>(evidence[w], evidence[w + 1u]), p.n) {
        return;
    }
    atomicAdd(&status[S_C], 1u);
    sb[out + 4u] = evidence[w + 3u];
    sb[out + 5u] = evidence[w + 4u];
}

// Sums rows [256 b, 256 b + 256) of the p.b rows at row p.a into row p.c + b.
@compute @workgroup_size(WG)
fn reduce(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let block = index_of(id, groups);
    if block >= p.n {
        return;
    }
    var sums: array<f32, 6>;
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

// sa: score records. sb: 1 at incompatible records, plus zero at p.n.
@compute @workgroup_size(WG)
fn record_marks(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let k = index_of(id, groups);
    if k > p.n {
        return;
    }
    sb[k] = select(0u, 1u, k < p.n && sa[RECORD * k + 2u] != 0u);
}

// sb: scanned incompatible marks. sc: baseline evidence records.
@compute @workgroup_size(WG)
fn evidence_write(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let k = index_of(id, groups);
    if k >= p.n || sb[k + 1u] == sb[k] {
        return;
    }
    for (var word = 0u; word < RECORD; word += 1u) {
        sc[RECORD * sb[k] + word] = sa[RECORD * k + word];
    }
}

// Final scalars of p.n contacts and p.e evidence rows from the reduced row p.a
// of sb, into sc: words 0..5 the six sums, 6..7 RMS excess (low, high) over
// every contact, 8..9 their ratios to the route tolerance p.x. Without contacts
// the RMS and ratios are zero; without rows every sum is zero.
@compute @workgroup_size(1)
fn score_finish(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x != 0u || id.y != 0u {
        return;
    }
    for (var column = 0u; column < ROW; column += 1u) {
        var value = 0u;
        if p.n + p.e > 0u {
            value = sb[ROW * p.a + column];
        }
        sc[column] = value;
    }
    for (var side = 0u; side < 2u; side += 1u) {
        var rms = 0.0;
        if p.n > 0u {
            rms = sqrt(f(sc[side]) / f32(p.n));
        }
        sc[6u + side] = bitcast<u32>(rms);
        sc[8u + side] = bitcast<u32>(rms / p.x);
    }
}
