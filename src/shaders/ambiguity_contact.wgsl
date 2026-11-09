// Certified CPU-f64 contact decisions on resident GPU candidate cells (stage S2).
// Derivation rules W*, A*, R*, P*, 5.* and C* are kept in the reviewed derivation notes.
// Each f32 operation is isolated by integer key stepping (W/consequence a); decisions compare
// order keys as integers (consequence c); unsafe operations sit behind admitted `if` guards (A*).

struct Params {
    cells: u32,
    source_base: u32,
    segment_bits: u32,
    partner_bits: u32,
    shift: u32,
    blocks: u32,
    pairs: u32,
    chunk_start: u32,
    chunk_count: u32,
    pair_cap: u32,
    // 1 when radius, squared radius and minimum passed host A2 admission; 0 sends every pair to the CPU.
    admitted: u32,
    // Width of the batch-relative source field in the 64-bit sort key (host-checked).
    source_bits: u32,
    radius: vec2<f32>,
    squared_radius: vec2<f32>,
    minimum: vec2<f32>,
    radius_is_zero: u32,
    // Cooperative mode: 1 = serial-small pairs sort their column order locally (no global column sort).
    local_columns: u32,
    // Cooperative label-propagation round cap (always ROUND_CAP in production).
    round_cap: u32,
    // Number of ordered paths and of batch sources, for the candidate index validation.
    path_count: u32,
    source_count: u32,
    padding: u32,
}

struct RawPoint { x: u32, y: u32, edge_id: u32 }
struct Reservation { point_offset: u32, segment_offset: u32, count: u32 }
struct Cell { source: u32, partner: u32, a: u32, b: u32 }
// base: first global segment of the ordinal; count: positive segments; closed: CPU first == last.
struct PathInfo { base: u32, count: u32, closed: u32, padding: u32 }
struct Cert {
    flags: u32,
    a_low_lo: f32, a_low_hi: f32, a_high_lo: f32, a_high_hi: f32,
    b_low_lo: f32, b_low_hi: f32, b_high_lo: f32, b_high_hi: f32,
}
struct PairOut { source: u32, partner: u32, outcome: u32, cells: u32 }

const CERTAIN: u32 = 1u;
const PRESENT: u32 = 2u;
const ROW_JOIN: u32 = 4u;
const COLUMN_JOIN: u32 = 8u;
const ERROR_POSSIBLE: u32 = 16u;
const CONTACT: u32 = 1u;
const NO_CONTACT: u32 = 2u;
const UNCERTAIN: u32 = 3u;
const UNCLASSIFIED: u32 = 0u;
// Single-invocation pair bound: larger pairs take the complete CPU route (watchdog safety).
override pair_cap: u32 = 4096u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> points: array<RawPoint>;
@group(0) @binding(2) var<storage, read> reservations: array<Reservation>;
@group(0) @binding(3) var<storage, read> ordinals: array<u32>;
@group(0) @binding(4) var<storage, read> point_scan: array<u32>;
@group(0) @binding(5) var<storage, read> path_base: array<u32>;
@group(0) @binding(6) var<storage, read_write> segment_end: array<u32>;
@group(0) @binding(7) var<storage, read_write> status: array<atomic<u32>, 4>;

@group(1) @binding(0) var<storage, read> path_info: array<PathInfo>;
@group(1) @binding(1) var<storage, read> segment_length: array<vec2<f32>>;
@group(1) @binding(2) var<storage, read_write> cells: array<Cell>;
@group(1) @binding(3) var<storage, read_write> sorted: array<Cell>;
@group(1) @binding(4) var<storage, read_write> certs: array<Cert>;
@group(1) @binding(5) var<storage, read_write> keys_in: array<vec2<u32>>;
@group(1) @binding(6) var<storage, read_write> perm_in: array<u32>;
@group(1) @binding(7) var<storage, read_write> keys_out: array<vec2<u32>>;
@group(1) @binding(8) var<storage, read_write> perm_out: array<u32>;
@group(1) @binding(9) var<storage, read_write> histogram: array<u32>;
@group(1) @binding(10) var<storage, read_write> pair_start: array<u32>;
@group(1) @binding(11) var<storage, read_write> pair_out: array<PairOut>;
@group(1) @binding(12) var<storage, read_write> scratch: array<u32>;
@group(1) @binding(13) var<storage, read_write> fallback_offsets: array<u32>;
@group(1) @binding(14) var<storage, read_write> fallback: array<Cell>;
// Classification audit words (one binding keeps every classifier within 8 storage buffers):
// [0, 31) settled after exactly r rounds, 31 settled after >= 31 rounds, 32 not settled at the cap, 33 audit failures, 34 label range failures,
// 35 cooperative pairs, 36 serial pairs, 37 classifier error bits (16 invalid segment index,
// 32 invalid/duplicate schedule, 64 cooperative cap violation); [64 + pair] exactly-once claims (O6).
@group(1) @binding(15) var<storage, read_write> audit: array<atomic<u32>>;
const CLAIMS: u32 = 64u;
const AUDIT_ERRORS: u32 = 37u;

const block_size: u32 = 256u;

fn invocation(id: vec3<u32>, groups: vec3<u32>) -> u32 {
    return id.y * groups.x * block_size + id.x;
}

// ---------------------------------------------------------------- keys and stepping (R1-R3)
const KEY_ZERO: u32 = 0x80000000u;
const NEG_TINY: u32 = 0x7effffffu; // key(-2^-125)
const POS_TINY: u32 = 0x81000000u; // key(+2^-125)

fn bits_key(bits: u32) -> u32 {
    if (bits & 0x7fffffffu) == 0u {
        return KEY_ZERO;
    }
    return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
fn key(value: f32) -> u32 {
    return bits_key(bitcast<u32>(value));
}
fn from_key(k: u32) -> f32 {
    return bitcast<f32>(select(~k, k & 0x7fffffffu, k >= 0x80000000u));
}
fn is_tiny(k: u32) -> bool {
    return k > NEG_TINY && k < POS_TINY;
}
fn down(q: f32, steps: u32) -> f32 {
    let k = key(q);
    var r = k - steps;
    if is_tiny(k) || is_tiny(r) {
        r = min(r, NEG_TINY);
    }
    return from_key(r);
}
fn up(q: f32, steps: u32) -> f32 {
    let k = key(q);
    var r = k + steps;
    if is_tiny(k) || is_tiny(r) {
        r = max(r, POS_TINY);
    }
    return from_key(r);
}
fn kmin(a: f32, b: f32) -> f32 {
    return select(b, a, key(a) <= key(b));
}
fn kmax(a: f32, b: f32) -> f32 {
    return select(b, a, key(a) >= key(b));
}
// Biased exponent field of the larger magnitude endpoint (integer inspection, A4).
fn exponent_of(value: f32) -> i32 {
    return i32((bitcast<u32>(value) >> 23u) & 0xffu);
}

// ---------------------------------------------------------------- intervals (R4-R7)
struct I { lo: f32, hi: f32 }

fn exact(value: f32) -> I {
    return I(value, value);
}
// Pipeline overrides only so the mutation test can weaken them; production values are R1/R2/R2s.
override STEP: u32 = 6u;
override STEP_DIV: u32 = 12u;
override STEP_SQRT: u32 = 24u;

fn i_add(a: I, b: I) -> I {
    return I(down(a.lo + b.lo, STEP), up(a.hi + b.hi, STEP));
}
fn i_sub(a: I, b: I) -> I {
    return I(down(a.lo - b.hi, STEP), up(a.hi - b.lo, STEP));
}
fn i_neg(a: I) -> I {
    return I(-a.hi, -a.lo);
}
fn i_mul(a: I, b: I) -> I {
    let p0 = a.lo * b.lo;
    let p1 = a.lo * b.hi;
    let p2 = a.hi * b.lo;
    let p3 = a.hi * b.hi;
    let lo = kmin(kmin(down(p0, STEP), down(p1, STEP)), kmin(down(p2, STEP), down(p3, STEP)));
    let hi = kmax(kmax(up(p0, STEP), up(p1, STEP)), kmax(up(p2, STEP), up(p3, STEP)));
    return I(lo, hi);
}
fn i_square(a: I) -> I {
    if key(a.lo) >= KEY_ZERO {
        return I(down(a.lo * a.lo, STEP), up(a.hi * a.hi, STEP));
    }
    if key(a.hi) <= KEY_ZERO {
        return I(down(a.hi * a.hi, STEP), up(a.lo * a.lo, STEP));
    }
    return I(0.0, kmax(up(a.lo * a.lo, STEP), up(a.hi * a.hi, STEP)));
}
// R5: fl64(a*b + c) with one rounding.
fn i_fma(a: I, b: I, c: I) -> I {
    return i_add(i_mul(a, b), c);
}
fn excludes_small(a: I) -> bool {
    // |x| >= 2^-100 for every x in a (keys of +-2^-100).
    return key(a.lo) >= 0x8d800000u || key(a.hi) <= 0x727fffffu;
}
fn max_exponent(a: I) -> i32 {
    return max(exponent_of(a.lo), exponent_of(a.hi));
}
fn min_exponent_nonzero(a: I) -> i32 {
    // a excludes [-2^-100, 2^-100]: the endpoint nearer zero sets the smallest magnitude.
    if key(a.lo) >= KEY_ZERO {
        return exponent_of(a.lo);
    }
    return exponent_of(a.hi);
}
// R2 under A4. `ok` is cleared before any unsafe division.
fn i_div(a: I, b: I, ok: ptr<function, bool>) -> I {
    // W2 range |y| in [2^-126, 2^126]: admitted divisors lie in [2^-100, 2^100].
    if !excludes_small(b) || max_exponent(b) > 227 || max_exponent(a) - min_exponent_nonzero(b) > 100 {
        *ok = false;
        return I(0.0, 0.0);
    }
    let q0 = a.lo / b.lo;
    let q1 = a.lo / b.hi;
    let q2 = a.hi / b.lo;
    let q3 = a.hi / b.hi;
    let lo = kmin(kmin(down(q0, STEP_DIV), down(q1, STEP_DIV)), kmin(down(q2, STEP_DIV), down(q3, STEP_DIV)));
    let hi = kmax(kmax(up(q0, STEP_DIV), up(q1, STEP_DIV)), kmax(up(q2, STEP_DIV), up(q3, STEP_DIV)));
    return I(lo, hi);
}
// R2s under A5: input enclosure of a CPU value known to be >= 0.
fn i_sqrt(a: I, ok: ptr<function, bool>) -> I {
    if exponent_of(a.hi) > 143 {
        *ok = false;
        return I(0.0, 0.0);
    }
    var hi = from_key(0xa6800000u); // 2^-50 (biased exponent 77)
    if key(a.hi) >= 0x8d800000u {
        hi = up(sqrt(a.hi), STEP_SQRT);
    }
    var lo = 0.0;
    if key(a.lo) >= 0x8d800000u {
        lo = down(sqrt(a.lo), STEP_SQRT);
    }
    return I(lo, hi);
}
fn i_min(a: I, b: I) -> I {
    return I(kmin(a.lo, b.lo), kmin(a.hi, b.hi));
}
fn i_max(a: I, b: I) -> I {
    return I(kmax(a.lo, b.lo), kmax(a.hi, b.hi));
}
fn i_hull(a: I, b: I) -> I {
    return I(kmin(a.lo, b.lo), kmax(a.hi, b.hi));
}
// Certified a <= b: 1 true, 0 false, 2 unknown.
fn le(a: I, b: I) -> u32 {
    if key(a.hi) <= key(b.lo) {
        return 1u;
    }
    if key(a.lo) > key(b.hi) {
        return 0u;
    }
    return 2u;
}
fn lt(a: I, b: I) -> u32 {
    if key(a.hi) < key(b.lo) {
        return 1u;
    }
    if key(a.lo) >= key(b.hi) {
        return 0u;
    }
    return 2u;
}

// ---------------------------------------------------------------- CPU expressions (P1-P4)
struct V { x: I, y: I, zx: bool, zy: bool }

fn coordinate_admitted(bits: u32) -> bool {
    let exponent = (bits >> 23u) & 0xffu;
    return (bits & 0x7fffffffu) == 0u || (exponent >= 103u && exponent <= 128u);
}
fn same_value(a: u32, b: u32) -> bool {
    return a == b || ((a | b) & 0x7fffffffu) == 0u;
}
fn difference_1(a: u32, b: u32) -> I {
    if same_value(a, b) {
        return exact(0.0);
    }
    let fa = bitcast<f32>(a);
    let fb = bitcast<f32>(b);
    let r = fa - fb;
    return I(down(r, STEP), up(r, STEP));
}
// P1: exact CPU difference p - q of admitted f32 points.
fn difference(p: vec2<u32>, q: vec2<u32>) -> V {
    return V(difference_1(p.x, q.x), difference_1(p.y, q.y), same_value(p.x, q.x), same_value(p.y, q.y));
}
// P2: dot = a.x.mul_add(b.x, a.y * b.y)
fn dot2(a: V, b: V) -> I {
    return i_fma(a.x, b.x, i_mul(a.y, b.y));
}
fn dot_is_zero(a: V, b: V) -> bool {
    return (a.zx || b.zx) && (a.zy || b.zy);
}
// P3/R6: compensated cross.
fn cross2(a: V, b: V) -> I {
    let p = i_mul(a.y, b.x);
    let t1 = i_fma(a.x, b.y, i_neg(p));
    let e = i_fma(i_neg(a.y), b.x, p);
    return i_add(t1, e);
}
fn cross_is_zero(a: V, b: V) -> bool {
    return (a.zx || b.zy) && (a.zy || b.zx);
}

struct Segment2 { start: vec2<u32>, end: vec2<u32>, length: I }

// P4: point_distance_squared with certified or hulled branches.
fn point_distance_squared(point: vec2<u32>, segment: Segment2, ok: ptr<function, bool>) -> I {
    let direction = difference(segment.end, segment.start);
    let relative = difference(point, segment.start);
    let length_squared = dot2(direction, direction);
    let projected = dot2(relative, direction);
    let at_start = le(projected, exact(0.0));
    let start_value = dot2(relative, relative);
    if at_start == 1u {
        return start_value;
    }
    let past_end = le(length_squared, projected);
    let residual = difference(point, segment.end);
    let end_value = dot2(residual, residual);
    var inner = end_value;
    if past_end != 1u {
        let perpendicular = cross2(relative, direction);
        let middle = i_div(i_square(perpendicular), length_squared, ok);
        if past_end == 0u {
            inner = middle;
        } else {
            inner = i_hull(end_value, middle);
        }
    }
    if at_start == 0u {
        return inner;
    }
    return i_hull(start_value, inner);
}

// ---------------------------------------------------------------- projection (P5-P7)
struct Projection {
    present: bool,
    low: I,
    high: I,
    end_inside: bool,
    error_possible: bool,
    reason: u32,
}
struct Clip { present: bool, low: I, high: I, reason: u32 }

// P6 include
fn include(p: ptr<function, Projection>, low: I, high: I) {
    if (*p).present {
        (*p).low = i_min((*p).low, low);
        (*p).high = i_max((*p).high, high);
    } else {
        (*p).present = true;
        (*p).low = low;
        (*p).high = high;
    }
}
// P5 clip_linear(interval, value, slope, low, high); interval given by enclosures.
fn clip(low_i: I, high_i: I, value: I, slope: I, slope_zero: bool, low: I, high: I) -> Clip {
    if slope_zero {
        let a = le(low, value);
        let b = le(value, high);
        if a == 2u || b == 2u {
            return Clip(false, low_i, high_i, 12u);
        }
        return Clip(a == 1u && b == 1u, low_i, high_i, 0u);
    }
    let positive = key(slope.lo) > KEY_ZERO;
    let negative = key(slope.hi) < KEY_ZERO;
    if !positive && !negative {
        return Clip(false, low_i, high_i, 3u);
    }
    var ok = true;
    let first = i_div(i_sub(low, value), slope, &ok);
    let last = i_div(i_sub(high, value), slope, &ok);
    if !ok {
        return Clip(false, low_i, high_i, 2u);
    }
    let lo = i_max(low_i, i_min(first, last));
    let hi = i_min(high_i, i_max(first, last));
    let nonempty = le(lo, hi);
    if nonempty == 2u {
        return Clip(false, lo, hi, 4u);
    }
    return Clip(nonempty == 1u, lo, hi, 0u);
}

fn projection(source: Segment2, goal: Segment2) -> Projection {
    var result = Projection(false, exact(0.0), exact(0.0), false, false, 0u);
    let direction = difference(source.end, source.start);
    let goal_direction = difference(goal.end, goal.start);
    let relative = difference(source.start, goal.start);
    let goal_squared = dot2(goal_direction, goal_direction);
    let strip = clip(exact(0.0), exact(1.0), dot2(relative, goal_direction), dot2(direction, goal_direction),
        dot_is_zero(direction, goal_direction), exact(0.0), goal_squared);
    if strip.reason != 0u {
        result.reason = strip.reason;
        return result;
    }
    let radius_length = i_mul(I(params.radius.x, params.radius.y), goal.length);
    if strip.present {
        let band = clip(strip.low, strip.high, cross2(relative, goal_direction), cross2(direction, goal_direction),
            cross_is_zero(direction, goal_direction), i_neg(radius_length), radius_length);
        if band.reason != 0u {
            result.reason = band.reason + 16u;
            return result;
        }
        if band.present {
            include(&result, band.low, band.high);
        }
    }
    let squared_radius = I(params.squared_radius.x, params.squared_radius.y);
    let direction_squared = dot2(direction, direction);
    for (var k = 0u; k < 2u; k += 1u) {
        let point = select(goal.start, goal.end, k == 1u);
        let offset = difference(point, source.start);
        var ok = true;
        let center = i_div(dot2(offset, direction), direction_squared, &ok);
        let distance_squared = i_div(i_square(cross2(offset, direction)), direction_squared, &ok);
        if !ok {
            result.reason = 2u;
            return result;
        }
        let remaining = i_sub(squared_radius, distance_squared);
        let nonnegative = le(exact(0.0), remaining);
        if nonnegative == 2u {
            result.reason = 6u;
            return result;
        }
        if nonnegative == 1u {
            let root = i_sqrt(I(kmax(remaining.lo, 0.0), remaining.hi), &ok);
            let half = i_div(root, source.length, &ok);
            if !ok {
                result.reason = 11u;
                return result;
            }
            let lo = i_max(exact(0.0), i_sub(center, half));
            let hi = i_min(exact(1.0), i_add(center, half));
            let nonempty = le(lo, hi);
            if nonempty == 2u {
                result.reason = 7u;
                return result;
            }
            if nonempty == 1u {
                include(&result, lo, hi);
            }
        }
    }
    for (var k = 0u; k < 2u; k += 1u) {
        let parameter = exact(f32(k));
        let point = select(source.start, source.end, k == 1u);
        var ok = true;
        let inside_value = point_distance_squared(point, goal, &ok);
        if !ok {
            result.reason = 13u;
            return result;
        }
        let inside = le(inside_value, squared_radius);
        if inside == 2u {
            result.reason = 8u;
            return result;
        }
        if inside == 0u && result.present {
            // NumericalFailure iff included: possible unless certainly excluded.
            let a = le(result.low, parameter);
            let b = le(parameter, result.high);
            if !(a == 0u || b == 0u) {
                result.error_possible = true;
            }
        }
        if inside == 1u {
            include(&result, parameter, parameter);
            if k == 1u {
                result.end_inside = true;
            }
        }
    }
    return result;
}

fn cell_segment(path: u32, index: u32) -> Segment2 {
    let global = path_info[path].base + index;
    let end_index = segment_end[global];
    let end = points[end_index];
    let start = points[end_index - 1u];
    let length = segment_length[global];
    return Segment2(vec2<u32>(start.x, start.y), vec2<u32>(end.x, end.y), I(length.x, length.y));
}
fn segment_admitted(s: Segment2) -> bool {
    return key(s.length.lo) > KEY_ZERO && coordinate_admitted(s.start.x) && coordinate_admitted(s.start.y)
        && coordinate_admitted(s.end.x) && coordinate_admitted(s.end.y);
}

// ---------------------------------------------------------------- kernels
// Positive segment end points in (ordinal, logical) order, from S1's resident scans.
@compute @workgroup_size(block_size)
fn segment_table(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = invocation(id, groups);
    if index >= params.cells || point_scan[index + 1u] == point_scan[index] {
        return;
    }
    let edge = points[index].edge_id;
    let ordinal = ordinals[edge];
    let logical = point_scan[index] - point_scan[reservations[edge].point_offset];
    segment_end[path_base[ordinal] + logical] = index;
}

// x << s as a 64-bit value (low, high) for any s; s >= 64 gives 0. WGSL u32 shifts are only used
// with amounts in [1, 31].
fn shift_left(x: u32, s: u32) -> vec2<u32> {
    if s == 0u {
        return vec2<u32>(x, 0u);
    }
    if s < 32u {
        return vec2<u32>(x << s, x >> (32u - s));
    }
    if s == 32u {
        return vec2<u32>(0u, x);
    }
    if s < 64u {
        return vec2<u32>(0u, x << (s - 32u));
    }
    return vec2<u32>(0u, 0u);
}
// Exact 64-bit key from four disjoint fields, most significant first:
//   source - base (source_bits) | partner (partner_bits) | first (segment_bits) | second (segment_bits).
// The host checks the total width. Key kernels check each field fits its width; validated S1 supplies
// actual path/segment identities. No field is pre-combined in u32.
fn fits_field(value: u32, width: u32) -> bool {
    if width == 32u {
        return true;
    }
    if width > 32u {
        return false;
    }
    return (value >> width) == 0u;
}
fn valid_fields(c: Cell) -> bool {
    return c.source >= params.source_base
        && fits_field(c.source - params.source_base, params.source_bits)
        && fits_field(c.partner, params.partner_bits)
        && fits_field(c.a, params.segment_bits)
        && fits_field(c.b, params.segment_bits);
}
// Batch source range, source < partner < paths and both segments below their own path's positive
// segment count. Checked before any indexed metadata access; failures set status[1] bit 2 and the
// host aborts before certification.
fn valid_indices(c: Cell) -> bool {
    if c.source < params.source_base || c.source - params.source_base >= params.source_count
        || c.partner <= c.source || c.partner >= params.path_count {
        return false;
    }
    return c.a < path_info[c.source].count && c.b < path_info[c.partner].count;
}
fn field_key(source: u32, partner: u32, first: u32, second: u32) -> vec2<u32> {
    let sb = params.segment_bits;
    let pb = params.partner_bits;
    return shift_left(source - params.source_base, pb + 2u * sb) | shift_left(partner, 2u * sb)
        | shift_left(first, sb) | vec2<u32>(second, 0u);
}
@compute @workgroup_size(block_size)
fn make_keys(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = invocation(id, groups);
    if index >= params.cells {
        return;
    }
    let c = cells[index];
    if !valid_fields(c) || !valid_indices(c) {
        atomicOr(&status[1], 2u);
    }
    keys_in[index] = field_key(c.source, c.partner, c.a, c.b);
    perm_in[index] = index;
}

// ---- radix sort: 4-bit digits, blocks of 1024 elements, 128 threads x 8 elements, stable.
const radix_threads: u32 = 128u;
const per_thread: u32 = 8u;
// Digit width (4, 6 or 8 bits; pipeline override). Local ranks are computed in chunks of 16 digit values
// so the per-thread tally stays [16][128] u32 = 8 KiB (portable 16 KiB workgroup limit) for every width.
override radix_bits: u32 = 6u;
var<workgroup> tally: array<array<u32, 128>, 16>;
var<workgroup> block_hist: array<atomic<u32>, 256>;

fn radix_bins() -> u32 {
    return 1u << radix_bits;
}
// Digit of the 64-bit key (k.y high word) at bit params.shift, crossing the word boundary when needed.
fn digit(k: vec2<u32>) -> u32 {
    let s = params.shift;
    let mask = radix_bins() - 1u;
    if s >= 32u {
        return (k.y >> (s - 32u)) & mask;
    }
    var v = k.x >> s;
    if s > 0u && s + radix_bits > 32u {
        v |= k.y << (32u - s);
    }
    return v & mask;
}
fn block_index(group: vec3<u32>, groups: vec3<u32>) -> u32 {
    return group.y * groups.x + group.x;
}

// Exact per-block digit counts (integer workgroup atomics; order-free).
@compute @workgroup_size(radix_threads)
fn radix_count(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let block = block_index(group, groups);
    if block >= params.blocks {
        return;
    }
    let bins = radix_bins();
    for (var d = local; d < bins; d += radix_threads) {
        atomicStore(&block_hist[d], 0u);
    }
    workgroupBarrier();
    for (var e = 0u; e < per_thread; e += 1u) {
        let index = block * radix_threads * per_thread + local * per_thread + e;
        if index < params.cells {
            atomicAdd(&block_hist[digit(keys_in[index])], 1u);
        }
    }
    workgroupBarrier();
    for (var d = local; d < bins; d += radix_threads) {
        histogram[d * params.blocks + block] = atomicLoad(&block_hist[d]);
    }
}

// Stable scatter: element order within a block is (thread, slot); the rank of an element among equal
// digits is (block base from the scanned histogram) + (equal digits in earlier threads, via a 16-bin
// chunked inclusive scan) + (equal digits earlier in the same thread).
@compute @workgroup_size(radix_threads)
fn radix_scatter(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let block = block_index(group, groups);
    if block >= params.blocks {
        return;
    }
    var keys: array<vec2<u32>, 8>;
    var digits: array<u32, 8>;
    var destination: array<u32, 8>;
    for (var e = 0u; e < per_thread; e += 1u) {
        let index = block * radix_threads * per_thread + local * per_thread + e;
        digits[e] = 0xffffffffu;
        if index < params.cells {
            keys[e] = keys_in[index];
            digits[e] = digit(keys[e]);
        }
    }
    let chunks = max(radix_bins() / 16u, 1u);
    for (var chunk = 0u; chunk < chunks; chunk += 1u) {
        // Explicit zero initialisation in every iteration (do not rely on implicit per-iteration init).
        var counts = array<u32, 16>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
        for (var e = 0u; e < per_thread; e += 1u) {
            if digits[e] != 0xffffffffu && digits[e] / 16u == chunk {
                counts[digits[e] % 16u] += 1u;
            }
        }
        for (var d = 0u; d < 16u; d += 1u) {
            tally[d][local] = counts[d];
        }
        workgroupBarrier();
        for (var step = 1u; step < radix_threads; step <<= 1u) {
            var values: array<u32, 16>;
            for (var d = 0u; d < 16u; d += 1u) {
                values[d] = tally[d][local] + select(0u, tally[d][local - min(step, local)], local >= step);
            }
            workgroupBarrier();
            for (var d = 0u; d < 16u; d += 1u) {
                tally[d][local] = values[d];
            }
            workgroupBarrier();
        }
        var running: array<u32, 16>;
        for (var d = 0u; d < 16u; d += 1u) {
            let bin = chunk * 16u + d;
            if bin < radix_bins() {
                running[d] = histogram[bin * params.blocks + block] + tally[d][local] - counts[d];
            }
        }
        for (var e = 0u; e < per_thread; e += 1u) {
            if digits[e] != 0xffffffffu && digits[e] / 16u == chunk {
                let d = digits[e] % 16u;
                destination[e] = running[d];
                running[d] += 1u;
            }
        }
        workgroupBarrier();
    }
    for (var e = 0u; e < per_thread; e += 1u) {
        let index = block * radix_threads * per_thread + local * per_thread + e;
        if index < params.cells {
            keys_out[destination[e]] = keys[e];
            perm_out[destination[e]] = perm_in[index];
        }
    }
}

@compute @workgroup_size(block_size)
fn gather_sorted(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = invocation(id, groups);
    if index >= params.cells {
        return;
    }
    sorted[index] = cells[perm_in[index]];
}

// P8/P9: certify one sorted cell in both projection directions.
@compute @workgroup_size(block_size)
fn certify_cells(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = invocation(id, groups);
    if index >= params.cells {
        return;
    }
    var cert = Cert(0u, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    if params.admitted == 0u {
        // Scalars outside A2: no arithmetic with them; the cell (and its pair) is uncertain.
        cert.flags = 30u << 8u;
        certs[index] = cert;
        return;
    }
    let c = sorted[index];
    let a = cell_segment(c.source, c.a);
    let b = cell_segment(c.partner, c.b);
    if !(segment_admitted(a) && segment_admitted(b)) {
        cert.flags = 1u << 8u;
        certs[index] = cert;
        return;
    }
    let along_a = projection(a, b);
    let along_b = projection(b, a);
    var reason = along_a.reason;
    if reason == 0u {
        reason = along_b.reason;
    }
    if reason != 0u {
        cert.flags = reason << 8u;
        certs[index] = cert;
        return;
    }
    var flags = CERTAIN;
    if along_a.error_possible || along_b.error_possible || along_a.present != along_b.present {
        flags = ERROR_POSSIBLE | (10u << 8u);
    }
    if along_a.present && along_b.present {
        flags |= PRESENT;
    }
    if along_a.end_inside {
        flags |= ROW_JOIN;
    }
    if along_b.end_inside {
        flags |= COLUMN_JOIN;
    }
    cert = Cert(flags, along_a.low.lo, along_a.low.hi, along_a.high.lo, along_a.high.hi,
        along_b.low.lo, along_b.low.hi, along_b.high.lo, along_b.high.hi);
    certs[index] = cert;
}

// Pair boundaries: histogram[i] = 1 where the pair key changes, then scanned on the host side.
@compute @workgroup_size(block_size)
fn pair_marks(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = invocation(id, groups);
    if index > params.cells {
        return;
    }
    var mark = 0u;
    if index < params.cells {
        let c = sorted[index];
        if index > 0u {
            // Defence in depth: the sorted run must be lexicographically nondecreasing by
            // (source, partner, a, b), so every pair is one contiguous run.
            let p = sorted[index - 1u];
            let less = p.source < c.source || (p.source == c.source && (p.partner < c.partner
                || (p.partner == c.partner && (p.a < c.a || (p.a == c.a && p.b <= c.b)))));
            if !less {
                atomicOr(&status[1], 1u);
            }
            // O7: an exactly repeated cell is a typed duplicate-cell error (bit 8), never a unique slice.
            if p.source == c.source && p.partner == c.partner && p.a == c.a && p.b == c.b {
                atomicOr(&status[1], 8u);
            }
        }
        mark = select(0u, 1u, index == 0u || sorted[index - 1u].source != c.source || sorted[index - 1u].partner != c.partner);
    }
    histogram[index] = mark;
}
@compute @workgroup_size(block_size)
fn pair_starts(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = invocation(id, groups);
    if index > params.cells {
        return;
    }
    if index == params.cells {
        pair_start[histogram[index]] = index;
        return;
    }
    if histogram[index + 1u] != histogram[index] {
        pair_start[histogram[index]] = index;
    }
}

// ---------------------------------------------------------------- whole-pair certificate (5.*)
// Scratch layout per pair at its sorted offset s: parent [s, s+n), column order [N + s, ...),
// lower/upper length accumulators as f32 bits at 2N + 4*(s + i) + {0: lower A, 1: upper A, 2: lower B, 3: upper B},
// line sort buffer [6N + s, ...). All regions are disjoint per pair, so one invocation owns its pair.
// Path halving; parents strictly decrease (5.6), so the loop terminates.
fn find(start: u32, x: u32) -> u32 {
    var v = x;
    while scratch[start + v] != v {
        let g = scratch[start + scratch[start + v]];
        scratch[start + v] = g;
        v = g;
    }
    return v;
}
fn unite(start: u32, x: u32, y: u32) {
    let rx = find(start, x);
    let ry = find(start, y);
    if rx < ry {
        scratch[start + ry] = rx;
    } else if ry < rx {
        scratch[start + rx] = ry;
    }
}
fn acc_index(slot: u32, field: u32) -> u32 {
    return 2u * params.cells + 4u * slot + field;
}
fn acc_add(slot: u32, field: u32, value: f32, upward: bool) {
    let index = acc_index(slot, field);
    let current = bitcast<f32>(scratch[index]);
    var next: f32;
    if upward {
        next = up(current + value, STEP);
    } else {
        next = down(current + value, STEP);
    }
    scratch[index] = bitcast<u32>(next);
}
fn cell_flags(index: u32) -> u32 {
    return certs[index].flags;
}
// Binary search for (a, b) in the pair's row-major cells.
fn locate(start: u32, count: u32, a: u32, b: u32) -> u32 {
    var low = 0u;
    var high = count;
    while low < high {
        let middle = low + (high - low) / 2u;
        let c = sorted[start + middle];
        if c.a < a || (c.a == a && c.b < b) {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }
    if low < count && sorted[start + low].a == a && sorted[start + low].b == b {
        return low;
    }
    return 0xffffffffu;
}
fn interval_of(index: u32, side_b: bool) -> vec4<f32> {
    let c = certs[index];
    if side_b {
        return vec4<f32>(c.b_low_lo, c.b_low_hi, c.b_high_lo, c.b_high_hi);
    }
    return vec4<f32>(c.a_low_lo, c.a_low_hi, c.a_high_lo, c.a_high_hi);
}
// Sort keys for scratch-index heap sorts: mode 0 = column order (b, a); mode 1/2 = (root, inner/outer
// low) on side A (false) or B (true). Roots are fully compressed before modes 1/2 are used.
fn sort_less(start: u32, x: u32, y: u32, mode: u32, side_b: bool) -> bool {
    if mode == 0u {
        let cx = sorted[start + x];
        let cy = sorted[start + y];
        return cx.b < cy.b || (cx.b == cy.b && cx.a < cy.a);
    }
    let rx = scratch[start + x];
    let ry = scratch[start + y];
    if rx != ry {
        return rx < ry;
    }
    let vx = interval_of(start + x, side_b);
    let vy = interval_of(start + y, side_b);
    let kx = select(key(vx.x), key(vx.y), mode == 1u);
    let ky = select(key(vy.x), key(vy.y), mode == 1u);
    return kx < ky || (kx == ky && x < y);
}
// Bounded in-place heap sort of scratch[base, base + m): O(m log m) comparisons.
fn heap_sort(start: u32, base: u32, m: u32, mode: u32, side_b: bool) {
    for (var i = 1u; i < m; i += 1u) {
        var child = i;
        while child > 0u {
            let parent = (child - 1u) / 2u;
            if sort_less(start, scratch[base + parent], scratch[base + child], mode, side_b) {
                let t = scratch[base + parent];
                scratch[base + parent] = scratch[base + child];
                scratch[base + child] = t;
                child = parent;
            } else {
                break;
            }
        }
    }
    for (var end = m; end > 1u; end -= 1u) {
        let t = scratch[base];
        scratch[base] = scratch[base + end - 1u];
        scratch[base + end - 1u] = t;
        var node = 0u;
        let size = end - 1u;
        loop {
            let left = 2u * node + 1u;
            if left >= size {
                break;
            }
            var largest = left;
            if left + 1u < size && sort_less(start, scratch[base + left], scratch[base + left + 1u], mode, side_b) {
                largest = left + 1u;
            }
            if sort_less(start, scratch[base + node], scratch[base + largest], mode, side_b) {
                let s = scratch[base + node];
                scratch[base + node] = scratch[base + largest];
                scratch[base + largest] = s;
                node = largest;
            } else {
                break;
            }
        }
    }
}
// P10 for one line of m patch indices at scratch[line, line + m): per component, the union measure of
// inner intervals (rounded down) and of outer intervals (rounded up), times the segment length enclosure.
fn accumulate_line(start: u32, line: u32, m: u32, length: vec2<f32>, side_b: bool) {
    let field = select(0u, 2u, side_b);
    heap_sort(start, line, m, 1u, side_b);
    var p = 0u;
    while p < m {
        let root = scratch[start + scratch[line + p]];
        var inner = 0.0;
        var end = 0.0;
        while p < m && scratch[start + scratch[line + p]] == root {
            let v = interval_of(start + scratch[line + p], side_b);
            if key(v.y) <= key(v.z) {
                let base = kmax(v.y, end);
                if key(v.z) > key(base) {
                    inner = down(inner + down(v.z - base, STEP), STEP);
                }
                end = kmax(end, v.z);
            }
            p += 1u;
        }
        if key(inner) > KEY_ZERO {
            acc_add(start + root, field, down(inner * length.x, STEP), false);
        }
    }
    heap_sort(start, line, m, 2u, side_b);
    p = 0u;
    while p < m {
        let root = scratch[start + scratch[line + p]];
        var outer = 0.0;
        var end = 0.0;
        while p < m && scratch[start + scratch[line + p]] == root {
            let v = interval_of(start + scratch[line + p], side_b);
            let base = kmax(v.x, end);
            if key(v.w) > key(base) {
                outer = up(outer + up(v.w - base, STEP), STEP);
            }
            end = kmax(end, v.w);
            p += 1u;
        }
        acc_add(start + root, field + 1u, up(outer * length.y, STEP), true);
    }
}

fn classify(pair: u32) -> u32 {
    let start = pair_start[pair];
    let count = pair_start[pair + 1u] - start;
    let first = sorted[start];
    if params.admitted == 0u {
        return UNCERTAIN | (30u << 8u);
    }
    if count > min(pair_cap, params.pair_cap) {
        return UNCERTAIN | (20u << 8u);
    }
    let info_a = path_info[first.source];
    let info_b = path_info[first.partner];
    let minimum = I(params.minimum.x, params.minimum.y);
    // A2 pre-check before any gate arithmetic: every length must be an admitted enclosure.
    for (var i = 0u; i < count; i += 1u) {
        let c = sorted[start + i];
        if key(segment_length[info_a.base + c.a].x) <= KEY_ZERO || key(segment_length[info_b.base + c.b].x) <= KEY_ZERO {
            return UNCERTAIN | (27u << 8u);
        }
    }
    // P11 gate, source side over distinct rows (row-major order).
    var lower_a = 0.0;
    var upper_a = 0.0;
    for (var i = 0u; i < count; i += 1u) {
        let c = sorted[start + i];
        if i == 0u || sorted[start + i - 1u].a != c.a {
            let len = segment_length[info_a.base + c.a];
            lower_a = down(lower_a + len.x, STEP);
            upper_a = up(upper_a + len.y, STEP);
        }
    }
    // Partner side: column order in scratch by (b, a) with a bounded heap sort.
    // Column order comes from the parallel (pair, b, a) radix sort, as global row-major indices.
    let column = params.cells + start;
    if params.local_columns != 0u {
        // Cooperative mode has no global column sort: bounded local heap sort by (b, a) (pairs here
        // are serial-small), stored as global row-major indices like the former column copy.
        for (var i = 0u; i < count; i += 1u) {
            scratch[column + i] = i;
        }
        heap_sort(start, column, count, 0u, false);
        for (var i = 0u; i < count; i += 1u) {
            scratch[column + i] += start;
        }
    }
    var lower_b = 0.0;
    var upper_b = 0.0;
    for (var i = 0u; i < count; i += 1u) {
        let c = sorted[scratch[column + i]];
        if i == 0u || sorted[scratch[column + i - 1u]].b != c.b {
            let len = segment_length[info_b.base + c.b];
            lower_b = down(lower_b + len.x, STEP);
            upper_b = up(upper_b + len.y, STEP);
        }
    }
    let growth = 1.0 + 1.0 / 8388608.0; // 1 + 2^-23, exact f32 (P11)
    upper_a = up(upper_a * growth, STEP);
    upper_b = up(upper_b * growth, STEP);
    let gate_true = key(lower_a) >= key(minimum.hi) && key(lower_b) >= key(minimum.hi);
    let gate_false = key(upper_a) < key(minimum.lo) || key(upper_b) < key(minimum.lo);
    if gate_false {
        return NO_CONTACT | (1u << 8u);
    }
    // 5.2/5.5: every cell certain and error-free.
    var any_present = false;
    for (var i = 0u; i < count; i += 1u) {
        let f = cell_flags(start + i);
        if (f & ERROR_POSSIBLE) != 0u {
            return UNCERTAIN | (22u << 8u);
        }
        if (f & CERTAIN) == 0u {
            return UNCERTAIN | (21u << 8u) | ((f >> 8u) << 16u);
        }
        any_present = any_present || (f & PRESENT) != 0u;
        scratch[start + i] = i;
        scratch[acc_index(start + i, 0u)] = 0u;
        scratch[acc_index(start + i, 1u)] = 0u;
        scratch[acc_index(start + i, 2u)] = 0u;
        scratch[acc_index(start + i, 3u)] = 0u;
    }
    if !any_present {
        return NO_CONTACT | (2u << 8u);
    }
    // 5.3 joins over certain patches; IncompleteCandidates when a true join lacks a patch.
    for (var i = 0u; i < count; i += 1u) {
        let f = cell_flags(start + i);
        if (f & PRESENT) == 0u {
            continue;
        }
        let c = sorted[start + i];
        if (f & ROW_JOIN) != 0u {
            var target_row = c.a + 1u;
            var exists = target_row < info_a.count;
            if !exists && info_a.closed != 0u {
                target_row = 0u;
                exists = true;
            }
            if exists {
                let k = locate(start, count, target_row, c.b);
                if k == 0xffffffffu || (cell_flags(start + k) & PRESENT) == 0u {
                    return UNCERTAIN | (23u << 8u);
                }
                unite(start, i, k);
            }
        }
        if (f & COLUMN_JOIN) != 0u {
            var target_column = c.b + 1u;
            var exists = target_column < info_b.count;
            if !exists && info_b.closed != 0u {
                target_column = 0u;
                exists = true;
            }
            if exists {
                let k = locate(start, count, c.a, target_column);
                if k == 0xffffffffu || (cell_flags(start + k) & PRESENT) == 0u {
                    return UNCERTAIN | (23u << 8u);
                }
                unite(start, i, k);
            }
        }
    }
    // P10 per row (side A) and per column (side B). Roots are compressed first.
    for (var k = 0u; k < count; k += 1u) {
        if (cell_flags(start + k) & PRESENT) != 0u {
            scratch[start + k] = find(start, k);
        }
    }
    let line = 6u * params.cells + start;
    var i = 0u;
    while i < count {
        let row = sorted[start + i].a;
        var m = 0u;
        var j = i;
        while j < count && sorted[start + j].a == row {
            if (cell_flags(start + j) & PRESENT) != 0u {
                scratch[line + m] = j;
                m += 1u;
            }
            j += 1u;
        }
        accumulate_line(start, line, m, segment_length[info_a.base + row], false);
        i = j;
    }
    i = 0u;
    while i < count {
        let col = sorted[scratch[column + i]].b;
        var m = 0u;
        var j = i;
        while j < count && sorted[scratch[column + j]].b == col {
            let k = scratch[column + j] - start;
            if (cell_flags(start + k) & PRESENT) != 0u {
                scratch[line + m] = k;
                m += 1u;
            }
            j += 1u;
        }
        accumulate_line(start, line, m, segment_length[info_b.base + col], true);
        i = j;
    }
    // 5.4 decision over component roots.
    let shrink = 1.0 - 1.0 / 4194304.0; // 1 - 2^-22, exact f32
    let grow = 1.0 + 1.0 / 4194304.0;
    var contact = false;
    var all_short = true;
    for (var r = 0u; r < count; r += 1u) {
        if (cell_flags(start + r) & PRESENT) == 0u || find(start, r) != r {
            continue;
        }
        let la = down(bitcast<f32>(scratch[acc_index(start + r, 0u)]) * shrink, STEP);
        let ua = up(bitcast<f32>(scratch[acc_index(start + r, 1u)]) * grow, STEP);
        let lb = down(bitcast<f32>(scratch[acc_index(start + r, 2u)]) * shrink, STEP);
        let ub = up(bitcast<f32>(scratch[acc_index(start + r, 3u)]) * grow, STEP);
        if key(la) >= key(minimum.hi) && key(lb) >= key(minimum.hi) {
            contact = true;
        }
        if !(key(ua) < key(minimum.lo) || key(ub) < key(minimum.lo)) {
            all_short = false;
        }
    }
    if all_short {
        return NO_CONTACT | (3u << 8u);
    }
    if contact && gate_true {
        return CONTACT | (4u << 8u);
    }
    if !gate_true && contact {
        return UNCERTAIN | (26u << 8u);
    }
    return UNCERTAIN | (25u << 8u);
}

@compute @workgroup_size(block_size)
fn init_pair_results(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let pair = invocation(id, groups);
    if pair < params.pairs {
        pair_out[pair] = PairOut(0u, 0u, UNCLASSIFIED, 0u);
    }
}

@compute @workgroup_size(64)
fn classify_pairs(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let slot = params.chunk_start + id.y * groups.x * 64u + id.x;
    if slot >= params.chunk_start + params.chunk_count || slot >= params.pairs {
        return;
    }
    // Pairs are dispatched in host-sorted size order (fallback_offsets holds the order here), so a
    // wave's invocations carry similar work. Every pair still gets exactly one invocation.
    let pair = pair_start[params.cells + 1u + slot];
    if pair >= params.pairs {
        atomicOr(&audit[AUDIT_ERRORS], 32u);
        return;
    }
    if atomicAdd(&audit[CLAIMS + pair], 1u) != 0u {
        atomicOr(&audit[AUDIT_ERRORS], 32u);
        return; // Only the claim winner may touch this pair's scratch or result.
    }
    atomicAdd(&audit[36], 1u);
    let first = sorted[pair_start[pair]];
    let outcome = classify(pair);
    pair_out[pair] = PairOut(first.source, first.partner, outcome, pair_start[pair + 1u] - pair_start[pair]);
}

// Fallback: per-pair uncertain cell counts (scanned), then complete copies of those pairs.
@compute @workgroup_size(block_size)
fn fallback_counts(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let pair = invocation(id, groups);
    if pair > params.pairs {
        return;
    }
    var n = 0u;
    if pair < params.pairs {
        let outcome = pair_out[pair].outcome & 255u;
        if outcome == UNCLASSIFIED || outcome > UNCERTAIN || atomicLoad(&audit[CLAIMS + pair]) != 1u {
            atomicOr(&status[1], 4u);
        } else if outcome == UNCERTAIN {
            n = pair_out[pair].cells;
        }
    }
    fallback_offsets[pair] = n;
}
@compute @workgroup_size(block_size)
fn fallback_copy(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = invocation(id, groups);
    if index >= params.cells {
        return;
    }
    let pair = histogram[index + 1u] - 1u;
    if (pair_out[pair].outcome & 255u) != UNCERTAIN {
        return;
    }
    fallback[fallback_offsets[pair] + index - pair_start[pair]] = sorted[index];
}

// ---------------------------------------------------------------- cooperative whole-pair classifier
// cooperative-design-v1 sections 3-6 and derivation-notes section 7 (C*). One workgroup owns one
// COMPLETE pair of n <= COOP_CAP cells. All per-pair arrays live in counted pair-private device scratch
// regions (base, width): pair data at base*N + width*s + j for j < width*n, disjoint across pairs because
// cell runs [s, s + n) are disjoint. Workgroup memory holds only control words and reduction lanes.
// Every barrier is reached by all lanes: stage guards and loop bounds are workgroupUniformLoad results,
// uniform-buffer parameters or constants; no lane returns before a barrier.
// Lanes per cooperative workgroup (pipeline override: 256, or 64 for pairs with P <= 128).
override WG: u32 = 256u;
const COOP_CAP: u32 = 4096u;
const ROUND_CAP: u32 = 128u;
const INVALID: u32 = 0xffffffffu;
const R_L0: u32 = 0u;    // width 1: final labels (export layout shared with the serial parent array)
const R_L1: u32 = 1u;    // width 1: second label buffer
const R_ACC: u32 = 2u;   // width 4: per-root accumulators (same layout as serial acc_index)
const R_COL: u32 = 6u;   // width 1: column order (local indices) by (b, a)
const R_NB: u32 = 7u;    // width 4: forward row, forward column, row predecessor, column predecessor
const R_SORT: u32 = 11u; // width 2: bitonic buffer, padded size P <= 2n
const R_SCAN0: u32 = 13u; // width 2: scan ping
const R_SCAN1: u32 = 15u; // width 2: scan pong
const R_KH: u32 = 17u;   // width 1: union group key (label << 12 | line)
const R_KL: u32 = 18u;   // width 1: union low-endpoint order key
const COOP_REGIONS: u32 = 19u;

var<workgroup> wg_flag: atomic<u32>;
var<workgroup> wg_min: atomic<u32>;
var<workgroup> wg_word: u32;
var<workgroup> wg_owner: u32;
var<workgroup> wg_info: array<u32, 8>;
var<workgroup> red_lo: array<f32, WG>;
var<workgroup> red_hi: array<f32, WG>;

fn reg(region: u32, width: u32, s: u32) -> u32 {
    return region * params.cells + width * s;
}
// Exact-zero-aware outward sums: x + 0 is exact, so no step is needed (sound, R1).
fn add_down(a: f32, b: f32) -> f32 {
    if key(a) == KEY_ZERO {
        return b;
    }
    if key(b) == KEY_ZERO {
        return a;
    }
    return down(a + b, STEP);
}
fn add_up(a: f32, b: f32) -> f32 {
    if key(a) == KEY_ZERO {
        return b;
    }
    if key(b) == KEY_ZERO {
        return a;
    }
    return up(a + b, STEP);
}
fn flags_reset(lid: u32) {
    workgroupBarrier();
    if lid == 0u {
        atomicStore(&wg_flag, 0u);
        atomicStore(&wg_min, INVALID);
    }
    workgroupBarrier();
}
fn flag_value(lid: u32) -> u32 {
    workgroupBarrier();
    if lid == 0u {
        wg_word = atomicLoad(&wg_flag);
    }
    return workgroupUniformLoad(&wg_word);
}
// Outward tree reduction of red_lo (down) and red_hi (up) over WG lanes; result in lane 0's slot.
fn reduce_sums(lid: u32) {
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride >>= 1u) {
        if lid < stride {
            red_lo[lid] = add_down(red_lo[lid], red_lo[lid + stride]);
            red_hi[lid] = add_up(red_hi[lid], red_hi[lid + stride]);
        }
        workgroupBarrier();
    }
}
fn present_local(s: u32, i: u32) -> bool {
    return (certs[s + i].flags & PRESENT) != 0u;
}
// Comparator for the bitonic sorts. INVALID (padding) is greatest. mode 0: column order (b, index), which
// equals (b, a, index) because row-major indices increase with a within one column. mode 1: (kh, kl, index).
fn coop_less(s: u32, mode: u32, x: u32, y: u32) -> bool {
    if y == INVALID {
        return x != INVALID;
    }
    if x == INVALID {
        return false;
    }
    if mode == 0u {
        let bx = sorted[s + x].b;
        let by = sorted[s + y].b;
        return bx < by || (bx == by && x < y);
    }
    let hx = scratch[reg(R_KH, 1u, s) + x];
    let hy = scratch[reg(R_KH, 1u, s) + y];
    if hx != hy {
        return hx < hy;
    }
    let lx = scratch[reg(R_KL, 1u, s) + x];
    let ly = scratch[reg(R_KL, 1u, s) + y];
    return lx < ly || (lx == ly && x < y);
}
// Bitonic sort of scratch[base, base + padded): log2(P)(log2(P)+1)/2 compare-exchange phases of P/2
// exchanges, each followed by a storage barrier. A total order on distinct elements: deterministic.
fn bitonic(lid: u32, s: u32, base: u32, padded: u32, mode: u32) {
    for (var k = 2u; k <= padded; k <<= 1u) {
        for (var j = k >> 1u; j > 0u; j >>= 1u) {
            for (var t = lid; t < padded / 2u; t += WG) {
                let i = 2u * t - (t & (j - 1u));
                let l = i + j;
                let x = scratch[base + i];
                let y = scratch[base + l];
                let ascending = (i & k) == 0u;
                if select(coop_less(s, mode, x, y), coop_less(s, mode, y, x), ascending) {
                    scratch[base + i] = y;
                    scratch[base + l] = x;
                }
            }
            storageBarrier();
        }
    }
}
fn lower_row(s: u32, n: u32, a: u32) -> u32 {
    var low = 0u;
    var high = n;
    while low < high {
        let middle = low + (high - low) / 2u;
        if sorted[s + middle].a < a {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }
    return low;
}
fn lower_column(s: u32, n: u32, b: u32) -> u32 {
    let col = reg(R_COL, 1u, s);
    var low = 0u;
    var high = n;
    while low < high {
        let middle = low + (high - low) / 2u;
        if sorted[s + scratch[col + middle]].b < b {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }
    return low;
}
// One union pass (C5): side B when side_b, inner (rounded down) or outer (rounded up) intervals. Writes the
// per-component totals to acc[label] field `field`. Elements excluded from the measure (absent cells and
// empty inner intervals) carry kh = INVALID and sort last.
fn union_pass(lid: u32, s: u32, n: u32, padded: u32, side_b: bool, inner: bool) {
    let labels = reg(R_L0, 1u, s);
    let kh = reg(R_KH, 1u, s);
    let kl = reg(R_KL, 1u, s);
    let buf = reg(R_SORT, 2u, s);
    for (var i = lid; i < padded; i += WG) {
        if i < n {
            var h = INVALID;
            var l = 0u;
            if present_local(s, i) {
                let v = interval_of(s + i, side_b);
                if !inner || key(v.y) <= key(v.z) {
                    let c = sorted[s + i];
                    var line: u32;
                    if side_b {
                        line = lower_column(s, n, c.b);
                    } else {
                        line = lower_row(s, n, c.a);
                    }
                    h = (scratch[labels + i] << 12u) | line;
                    l = select(key(v.x), key(v.y), inner);
                }
            }
            scratch[kh + i] = h;
            scratch[kl + i] = l;
        }
        scratch[buf + i] = select(INVALID, i, i < n);
    }
    storageBarrier();
    bitonic(lid, s, buf, padded, 1u);
    // Segmented inclusive prefix MAXIMUM of high-endpoint keys within (label, line) groups.
    var src = reg(R_SCAN0, 2u, s);
    var dst = reg(R_SCAN1, 2u, s);
    for (var p = lid; p < padded; p += WG) {
        let e = scratch[buf + p];
        var hk = KEY_ZERO;
        if e != INVALID && scratch[kh + e] != INVALID {
            let v = interval_of(s + e, side_b);
            hk = select(key(v.w), key(v.z), inner);
        }
        scratch[src + p] = hk;
    }
    storageBarrier();
    for (var d = 1u; d < padded; d <<= 1u) {
        for (var p = lid; p < padded; p += WG) {
            var v = scratch[src + p];
            if p >= d {
                let e = scratch[buf + p];
                let f = scratch[buf + p - d];
                if e != INVALID && f != INVALID && scratch[kh + e] != INVALID && scratch[kh + e] == scratch[kh + f] {
                    v = max(v, scratch[src + p - d]);
                }
            }
            scratch[dst + p] = v;
        }
        storageBarrier();
        let t = src;
        src = dst;
        dst = t;
    }
    // Contributions max(high - max(low, previous end), 0) * length, outward (P10), into dst.
    for (var p = lid; p < padded; p += WG) {
        let e = scratch[buf + p];
        var contribution = 0.0;
        if e != INVALID && scratch[kh + e] != INVALID {
            var previous = KEY_ZERO;
            if p > 0u {
                let f = scratch[buf + p - 1u];
                if f != INVALID && scratch[kh + f] == scratch[kh + e] {
                    previous = scratch[src + p - 1u];
                }
            }
            let v = interval_of(s + e, side_b);
            let c = sorted[s + e];
            var length: vec2<f32>;
            if side_b {
                length = segment_length[path_info[c.partner].base + c.b];
            } else {
                length = segment_length[path_info[c.source].base + c.a];
            }
            if inner {
                let base = kmax(v.y, from_key(previous));
                if key(v.z) > key(base) {
                    contribution = down(down(v.z - base, STEP) * length.x, STEP);
                }
            } else {
                let base = kmax(v.x, from_key(previous));
                if key(v.w) > key(base) {
                    contribution = up(up(v.w - base, STEP) * length.y, STEP);
                }
            }
        }
        scratch[dst + p] = bitcast<u32>(contribution);
    }
    storageBarrier();
    // Segmented inclusive outward SUM per component label (kh >> 12), rounded at every addition.
    let t0 = src;
    src = dst;
    dst = t0;
    for (var d = 1u; d < padded; d <<= 1u) {
        for (var p = lid; p < padded; p += WG) {
            var v = bitcast<f32>(scratch[src + p]);
            if p >= d {
                let e = scratch[buf + p];
                let f = scratch[buf + p - d];
                if e != INVALID && f != INVALID && scratch[kh + e] != INVALID && scratch[kh + f] != INVALID
                    && (scratch[kh + e] >> 12u) == (scratch[kh + f] >> 12u) {
                    let w = bitcast<f32>(scratch[src + p - d]);
                    v = select(add_up(v, w), add_down(v, w), inner);
                }
            }
            scratch[dst + p] = bitcast<u32>(v);
        }
        storageBarrier();
        let t = src;
        src = dst;
        dst = t;
    }
    // The last element of each label group writes that component's total (unique writer).
    let field = select(0u, 2u, side_b) + select(1u, 0u, inner);
    for (var p = lid; p < n; p += WG) {
        let e = scratch[buf + p];
        if e != INVALID && scratch[kh + e] != INVALID {
            var last = p + 1u >= n;
            if !last {
                let f = scratch[buf + p + 1u];
                last = f == INVALID || scratch[kh + f] == INVALID || (scratch[kh + f] >> 12u) != (scratch[kh + e] >> 12u);
            }
            if last {
                scratch[acc_index(s + (scratch[kh + e] >> 12u), field)] = scratch[src + p];
            }
        }
    }
    storageBarrier();
}

// Stage guards: wg_info[0] = state (0 running, else the final outcome word), [1] pair, [2] start,
// [3] count, [4] padded P, [5] log2 P, [6] gate_true.
fn set_state(lid: u32, outcome: u32) {
    if lid == 0u {
        wg_info[0] = outcome;
    }
}
fn state_of() -> u32 {
    return workgroupUniformLoad(&wg_info[0]);
}

@compute @workgroup_size(WG)
fn classify_coop(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let slot = params.chunk_start + group.y * groups.x + group.x;
    if slot >= params.chunk_start + params.chunk_count || slot >= params.pairs {
        return; // whole workgroup: uniform builtins and uniform buffer values
    }
    if lid == 0u {
        let pair = pair_start[params.cells + 1u + slot];
        var state = 0u;
        var start = 0u;
        var count = 0u;
        wg_owner = 0u;
        if pair >= params.pairs {
            atomicOr(&audit[AUDIT_ERRORS], 32u);
            state = UNCERTAIN | (31u << 8u);
        } else {
            if atomicAdd(&audit[CLAIMS + pair], 1u) != 0u {
                atomicOr(&audit[AUDIT_ERRORS], 32u);
            } else {
                wg_owner = 1u;
            }
            start = pair_start[pair];
            count = pair_start[pair + 1u] - start;
            // O9: the cooperative path is proved only for n <= COOP_CAP; the host never schedules more.
            if count > COOP_CAP || count > min(pair_cap, params.pair_cap) || count == 0u {
                atomicOr(&audit[AUDIT_ERRORS], 64u);
                state = UNCERTAIN | (20u << 8u);
            } else if params.admitted == 0u {
                state = UNCERTAIN | (30u << 8u);
            }
        }
        var padded = 1u;
        var logp = 0u;
        while padded < count {
            padded <<= 1u;
            logp += 1u;
        }
        wg_info[0] = state;
        wg_info[1] = pair;
        wg_info[2] = start;
        wg_info[3] = count;
        wg_info[4] = padded;
        wg_info[5] = logp;
        wg_info[6] = 0u;
    }
    if workgroupUniformLoad(&wg_owner) == 0u {
        return; // Uniform loser exit before any shared pair scratch or result writes.
    }
    if lid == 0u {
        atomicAdd(&audit[35], 1u);
    }
    let pair = workgroupUniformLoad(&wg_info[1]);
    let s = workgroupUniformLoad(&wg_info[2]);
    let n = workgroupUniformLoad(&wg_info[3]);
    let padded = workgroupUniformLoad(&wg_info[4]);
    let minimum = I(params.minimum.x, params.minimum.y);
    var state = state_of();
    var info_a = PathInfo(0u, 0u, 0u, 0u);
    var info_b = PathInfo(0u, 0u, 0u, 0u);
    if state == 0u {
        let first = sorted[s];
        info_a = path_info[first.source];
        info_b = path_info[first.partner];
    }
    // A2 pre-check (reason 27) before any gate arithmetic, over ALL cells (O5).
    if state == 0u {
        flags_reset(lid);
        for (var i = lid; i < n; i += WG) {
            let c = sorted[s + i];
            if key(segment_length[info_a.base + c.a].x) <= KEY_ZERO || key(segment_length[info_b.base + c.b].x) <= KEY_ZERO {
                atomicOr(&wg_flag, 1u);
            }
            if c.a >= info_a.count || c.b >= info_b.count {
                atomicOr(&wg_flag, 2u);
            }
        }
        let f = flag_value(lid);
        if (f & 2u) != 0u && lid == 0u {
            atomicOr(&audit[AUDIT_ERRORS], 16u);
        }
        if f != 0u {
            set_state(lid, UNCERTAIN | (27u << 8u));
        }
        state = state_of();
    }
    // P11 gate, source side: distinct rows over ALL cells (row-major), outward sums of actual-length
    // enclosures in any association (every addition rounded outward), then the CPU next_up growth.
    if state == 0u {
        var lo = 0.0;
        var hi = 0.0;
        for (var i = lid; i < n; i += WG) {
            let c = sorted[s + i];
            if i == 0u || sorted[s + i - 1u].a != c.a {
                let len = segment_length[info_a.base + c.a];
                lo = add_down(lo, len.x);
                hi = add_up(hi, len.y);
            }
        }
        red_lo[lid] = lo;
        red_hi[lid] = hi;
        reduce_sums(lid);
        if lid == 0u {
            let growth = 1.0 + 1.0 / 8388608.0;
            let upper = up(red_hi[0] * growth, STEP);
            if key(upper) < key(minimum.lo) {
                wg_info[0] = NO_CONTACT | (1u << 8u);
            }
            wg_info[6] = select(0u, 1u, key(red_lo[0]) >= key(minimum.hi));
        }
        state = state_of();
    }
    // Partner side: pair-local bitonic column order (replaces the second global sort), then distinct columns.
    if state == 0u {
        let buf = reg(R_SORT, 2u, s);
        for (var i = lid; i < padded; i += WG) {
            scratch[buf + i] = select(INVALID, i, i < n);
        }
        storageBarrier();
        bitonic(lid, s, buf, padded, 0u);
        let col = reg(R_COL, 1u, s);
        for (var p = lid; p < n; p += WG) {
            scratch[col + p] = scratch[buf + p];
        }
        storageBarrier();
        var lo = 0.0;
        var hi = 0.0;
        for (var p = lid; p < n; p += WG) {
            let c = sorted[s + scratch[col + p]];
            if p == 0u || sorted[s + scratch[col + p - 1u]].b != c.b {
                let len = segment_length[info_b.base + c.b];
                lo = add_down(lo, len.x);
                hi = add_up(hi, len.y);
            }
        }
        red_lo[lid] = lo;
        red_hi[lid] = hi;
        reduce_sums(lid);
        if lid == 0u {
            let growth = 1.0 + 1.0 / 8388608.0;
            let upper = up(red_hi[0] * growth, STEP);
            if key(upper) < key(minimum.lo) {
                wg_info[0] = NO_CONTACT | (1u << 8u);
            }
            if key(red_lo[0]) < key(minimum.hi) {
                wg_info[6] = 0u;
            }
        }
        state = state_of();
    }
    // 5.2/5.5: the first (lowest-index) uncertain or possibly failing cell decides the reason.
    if state == 0u {
        flags_reset(lid);
        for (var i = lid; i < n; i += WG) {
            let f = certs[s + i].flags;
            if (f & ERROR_POSSIBLE) != 0u || (f & CERTAIN) == 0u {
                atomicMin(&wg_min, i);
            }
            if (f & PRESENT) != 0u {
                atomicOr(&wg_flag, 1u);
            }
        }
        workgroupBarrier();
        if lid == 0u {
            let m = atomicLoad(&wg_min);
            if m != INVALID {
                let f = certs[s + m].flags;
                if (f & ERROR_POSSIBLE) != 0u {
                    wg_info[0] = UNCERTAIN | (22u << 8u);
                } else {
                    wg_info[0] = UNCERTAIN | (21u << 8u) | ((f >> 8u) << 16u);
                }
            } else if atomicLoad(&wg_flag) == 0u {
                wg_info[0] = NO_CONTACT | (2u << 8u);
            }
        }
        state = state_of();
    }
    // 5.3 / O1 / O2: certified undirected edges. A required neighbour must exist AND be certified PRESENT.
    let nb = reg(R_NB, 4u, s);
    let l0 = reg(R_L0, 1u, s);
    let l1 = reg(R_L1, 1u, s);
    if state == 0u {
        for (var i = lid; i < n; i += WG) {
            scratch[nb + 4u * i] = INVALID;
            scratch[nb + 4u * i + 1u] = INVALID;
            scratch[nb + 4u * i + 2u] = INVALID;
            scratch[nb + 4u * i + 3u] = INVALID;
            scratch[l0 + i] = select(INVALID, i, present_local(s, i));
            scratch[acc_index(s + i, 0u)] = 0u;
            scratch[acc_index(s + i, 1u)] = 0u;
            scratch[acc_index(s + i, 2u)] = 0u;
            scratch[acc_index(s + i, 3u)] = 0u;
        }
        storageBarrier(); // Publish initialization before cross-lane predecessor scatters.
        flags_reset(lid);
        for (var i = lid; i < n; i += WG) {
            let f = certs[s + i].flags;
            if (f & PRESENT) == 0u {
                continue;
            }
            let c = sorted[s + i];
            if (f & ROW_JOIN) != 0u {
                var target_row = c.a + 1u;
                var exists = target_row < info_a.count;
                if !exists && info_a.closed != 0u {
                    target_row = 0u;
                    exists = true;
                }
                if exists {
                    let k = locate(s, n, target_row, c.b);
                    if k == INVALID || !present_local(s, k) {
                        atomicOr(&wg_flag, 1u);
                    } else {
                        // Unique writers: (a, b) -> ((a + 1) mod rows, b) is injective for a < rows.
                        scratch[nb + 4u * i] = k;
                        scratch[nb + 4u * k + 2u] = i;
                    }
                }
            }
            if (f & COLUMN_JOIN) != 0u {
                var target_column = c.b + 1u;
                var exists = target_column < info_b.count;
                if !exists && info_b.closed != 0u {
                    target_column = 0u;
                    exists = true;
                }
                if exists {
                    let k = locate(s, n, c.a, target_column);
                    if k == INVALID || !present_local(s, k) {
                        atomicOr(&wg_flag, 1u);
                    } else {
                        scratch[nb + 4u * i + 1u] = k;
                        scratch[nb + 4u * k + 3u] = i;
                    }
                }
            }
        }
        storageBarrier(); // Publish the full graph and initial labels before propagation/audit.
        let f = flag_value(lid);
        if f != 0u {
            set_state(lid, UNCERTAIN | (23u << 8u));
        }
        state = state_of();
    }
    // C3/C4: synchronous minimum-label propagation with pointer jumping, immutable old snapshot.
    if state == 0u {
        let cap = min(ROUND_CAP, params.round_cap);
        var old = l0;
        var next = l1;
        var rounds = 0u;
        var settled = false;
        var range_bad = false;
        for (var r = 0u; r < cap; r += 1u) {
            flags_reset(lid);
            for (var i = lid; i < n; i += WG) {
                let mine = scratch[old + i];
                if mine == INVALID {
                    scratch[next + i] = INVALID;
                    continue;
                }
                var m = mine;
                for (var q = 0u; q < 5u; q += 1u) {
                    var v = i;
                    if q < 4u {
                        v = scratch[nb + 4u * i + q];
                        if v == INVALID {
                            continue;
                        }
                    }
                    let lv = scratch[old + v];
                    if lv >= n {
                        atomicOr(&wg_flag, 2u);
                        continue;
                    }
                    let jv = scratch[old + lv];
                    if jv >= n {
                        atomicOr(&wg_flag, 2u);
                        continue;
                    }
                    m = min(m, min(lv, jv));
                }
                scratch[next + i] = m;
                if m != mine {
                    atomicOr(&wg_flag, 1u);
                }
            }
            storageBarrier();
            let f = flag_value(lid);
            let t = old;
            old = next;
            next = t;
            rounds += 1u;
            if (f & 2u) != 0u {
                range_bad = true;
                break;
            }
            if f == 0u {
                settled = true;
                break;
            }
        }
        // C4 audit (always, even when the change flag reached zero): every present label is in range,
        // present, idempotent and not above its vertex, and every certified edge has equal endpoint labels.
        flags_reset(lid);
        for (var i = lid; i < n; i += WG) {
            let li = scratch[old + i];
            if li == INVALID {
                if present_local(s, i) {
                    atomicOr(&wg_flag, 1u);
                }
                continue;
            }
            if li >= n || li > i || !present_local(s, li) || scratch[old + li] != li {
                atomicOr(&wg_flag, 1u);
                continue;
            }
            for (var q = 0u; q < 4u; q += 1u) {
                let v = scratch[nb + 4u * i + q];
                if v != INVALID && scratch[old + v] != li {
                    atomicOr(&wg_flag, 1u);
                }
            }
        }
        let f = flag_value(lid);
        if lid == 0u {
            atomicAdd(&audit[select(min(rounds, 31u), 32u, !settled)], 1u);
            if range_bad {
                // Unreachable by the C3 invariant; a defect sends the whole pair to the CPU.
                atomicAdd(&audit[34], 1u);
                wg_info[0] = UNCERTAIN | (29u << 8u);
            } else if f != 0u {
                atomicAdd(&audit[33], 1u);
                wg_info[0] = UNCERTAIN | (28u << 8u);
            }
        }
        // Export the audited labels in the shared layout (R_L0), for the unions and oracle comparison.
        if old != l0 {
            for (var i = lid; i < n; i += WG) {
                scratch[l0 + i] = scratch[old + i];
            }
        }
        storageBarrier();
        state = state_of();
    }
    // C5: per component and ORIGINAL segment interval unions, inner (down) and outer (up), both sides.
    if state == 0u {
        union_pass(lid, s, n, padded, false, true);
        union_pass(lid, s, n, padded, false, false);
        union_pass(lid, s, n, padded, true, true);
        union_pass(lid, s, n, padded, true, false);
        // 5.4 / O8 decision per audited component root.
        flags_reset(lid);
        let shrink = 1.0 - 1.0 / 4194304.0;
        let grow = 1.0 + 1.0 / 4194304.0;
        for (var r = lid; r < n; r += WG) {
            if scratch[l0 + r] != r {
                continue;
            }
            let la = down(bitcast<f32>(scratch[acc_index(s + r, 0u)]) * shrink, STEP);
            let ua = up(bitcast<f32>(scratch[acc_index(s + r, 1u)]) * grow, STEP);
            let lb = down(bitcast<f32>(scratch[acc_index(s + r, 2u)]) * shrink, STEP);
            let ub = up(bitcast<f32>(scratch[acc_index(s + r, 3u)]) * grow, STEP);
            if key(la) >= key(minimum.hi) && key(lb) >= key(minimum.hi) {
                atomicOr(&wg_flag, 1u);
            }
            if !(key(ua) < key(minimum.lo) || key(ub) < key(minimum.lo)) {
                atomicOr(&wg_flag, 2u);
            }
        }
        workgroupBarrier();
        if lid == 0u {
            let f = atomicLoad(&wg_flag);
            let contact = (f & 1u) != 0u;
            let all_short = (f & 2u) == 0u;
            let gate_true = wg_info[6] != 0u;
            if all_short {
                wg_info[0] = NO_CONTACT | (3u << 8u);
            } else if contact && gate_true {
                wg_info[0] = CONTACT | (4u << 8u);
            } else if contact {
                wg_info[0] = UNCERTAIN | (26u << 8u);
            } else {
                wg_info[0] = UNCERTAIN | (25u << 8u);
            }
        }
        state = state_of();
    }
    if lid == 0u && pair < params.pairs {
        let first = sorted[s];
        pair_out[pair] = PairOut(first.source, first.partner, state, n);
    }
}

// ---------------------------------------------------------------- scan (S1 scan, with block guards)
struct ScanParameters { count: u32, padding_0: u32, padding_1: u32, padding_2: u32 }
@group(2) @binding(0) var<uniform> scan_parameters: ScanParameters;
@group(2) @binding(1) var<storage, read_write> values: array<u32>;
@group(2) @binding(2) var<storage, read_write> sums: array<u32>;
const scan_block: u32 = 512u;
var<workgroup> partial: array<u32, block_size>;
fn checked_add(a: u32, b: u32) -> u32 {
    let sum = a + b;
    if sum < a {
        atomicOr(&status[0], 1u);
    }
    return sum;
}
@compute @workgroup_size(block_size)
fn scan_blocks(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
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
fn scan_add(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let block = group.y * groups.x + group.x;
    let count = scan_parameters.count;
    let blocks = count / scan_block + min(count % scan_block, 1u);
    if block >= blocks {
        return;
    }
    let first = block * scan_block + 2u * local;
    let offset = sums[block];
    for (var index = first; index < min(first + 2u, count); index += 1u) {
        values[index] = checked_add(values[index], offset);
    }
}
