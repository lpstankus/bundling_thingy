// Shared declarations of the GPU-resident f32 ambiguity kernels (backend gpu-f32).
//
// Every kernel reads its scalar arguments from one uniform register block, bound
// with a dynamic offset so many dispatches can share a submission. Bindings 1
// and 2 are read-only storage, 3 to 7 read-write storage, 8 the status words.
//
// Arithmetic is ordinary WGSL f32: results are point estimates, not certified
// enclosures. The kernels assume only this: an operation whose exact result is
// a normal number is accurate to a few ulp, and one whose exact result is below
// 2^-126 may return zero instead (flush-to-zero, including operands read as
// zero). No kernel may produce an overflow, infinity or NaN, and every divisor
// is a normal number. Coordinates are admitted only when finite and below 2^20
// in magnitude. Contact geometry compares unsquared, normalized distances (see
// the broad-phase kernels); whole-route distances use squared residuals, where
// flushing can only hide distances below 2^-63. Comparisons that define
// orderings or identity use integer order keys of the bit patterns and exact
// integer arithmetic.

struct P {
    n: u32, a: u32, b: u32, c: u32,
    d: u32, e: u32, f: u32, g: u32,
    h: u32, i: u32, j: u32, k: u32,
    x: f32, y: f32, z: f32, w: f32,
    // First workgroup of this dispatch slice; set by the runner only.
    o: u32, pad_0: u32, pad_1: u32, pad_2: u32,
}
struct RawPoint { x: u32, y: u32, edge_id: u32 }
struct Reservation { point_offset: u32, segment_offset: u32, count: u32 }
// One eligible route by ordinal (increasing ID). Canonical fields describe the
// frozen original; `rep` is the ordinal of the first route with identical
// canonical geometry and owns the knots.
struct Route {
    id: u32, point_offset: u32, count: u32, canon_start: u32,
    canon_count: u32, flags: u32, name: u32, position: u32,
    unique: u32, rep: u32, knot_count: u32, error: f32,
    cohort: u32, scratch: u32, pad_0: u32, pad_1: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(8) var<storage, read_write> status: array<atomic<u32>, 32>;

const WG: u32 = 256u;
const NONE: u32 = 0xffffffffu;
const INF_BITS: u32 = 0x7f800000u;
// status[0] error bits.
const E_NONFINITE: u32 = 1u;
const E_RANGE: u32 = 2u;
const E_LAYOUT: u32 = 4u;
const E_CANDIDATE: u32 = 8u;
const E_DUPLICATE: u32 = 16u;
const E_NEIGHBOR: u32 = 32u;
const E_CAPACITY: u32 = 64u;
const E_GRID: u32 = 128u;
const E_INTERNAL: u32 = 256u;
const E_ROUTE: u32 = 512u;
const E_OVERFLOW: u32 = 1024u;
// status[1..] counters; each stage documents the ones it uses.
const S_ERROR: u32 = 0u;
const S_PENDING: u32 = 1u;
const S_CHANGED: u32 = 2u;
const S_A: u32 = 3u;
const S_B: u32 = 4u;
const S_C: u32 = 5u;
const S_D: u32 = 6u;
const S_E: u32 = 7u;
const S_F: u32 = 8u;
const S_G: u32 = 9u;
const S_H: u32 = 10u;

// Element index of an invocation. A dispatch may be one slice of whole
// workgroups of a larger element range; `p.o` is the slice's first workgroup.
fn index_of(id: vec3<u32>, groups: vec3<u32>) -> u32 {
    return (p.o + id.y * groups.x) * WG + id.x;
}
fn fail(bits: u32) {
    atomicOr(&status[S_ERROR], bits);
}
// Monotone total order on non-NaN f32 bit patterns with -0 equal to +0.
fn okey(bits: u32) -> u32 {
    if (bits & 0x7fffffffu) == 0u {
        return 0x80000000u;
    }
    return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
fn is_finite_bits(bits: u32) -> bool {
    return (bits & 0x7f800000u) != 0x7f800000u;
}
// Magnitude below 2^20 (zero and subnormal values included).
fn admitted_bits(bits: u32) -> bool {
    return ((bits >> 23u) & 0xffu) <= 146u;
}
const TINY: f32 = 1e-30;
fn normal_zero(bits: u32) -> u32 {
    return select(bits, 0u, (bits & 0x7fffffffu) == 0u);
}
fn f(bits: u32) -> f32 {
    return bitcast<f32>(bits);
}
// Lengths below SMALL (2^-100) are not resolved: a vector whose larger
// component is below it is as long as that component.
const SMALL: f32 = 7.888609e-31;
// Euclidean length without squaring small numbers: the vector is scaled by its
// larger component first, avoiding square underflow amplified by division.
// A subordinate normalized component or its square may harmlessly flush.
fn norm(v: vec2<f32>) -> f32 {
    let m = max(abs(v.x), abs(v.y));
    if m < SMALL {
        return m;
    }
    let u = v / m;
    return m * sqrt(u.x * u.x + u.y * u.y);
}
fn sq(a: vec2<f32>, b: vec2<f32>) -> f32 {
    let d = a - b;
    return d.x * d.x + d.y * d.y;
}
// Squared distance from `q` to segment [a, b] (the weak Frechet boundary weight).
fn boundary(q: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let d = b - a;
    let r = q - a;
    let len = d.x * d.x + d.y * d.y;
    var t = 0.0;
    if len > TINY {
        t = clamp((r.x * d.x + r.y * d.y) / len, 0.0, 1.0);
    }
    let e = r - t * d;
    return e.x * e.x + e.y * e.y;
}
