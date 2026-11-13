// Shared exclusive scans and stable radix sorts of the gpu-f32 backend.

@group(0) @binding(3) var<storage, read_write> ua: array<u32>;
@group(0) @binding(4) var<storage, read_write> ub: array<u32>;
@group(0) @binding(5) var<storage, read_write> uc: array<u32>;
@group(0) @binding(6) var<storage, read_write> ud: array<u32>;

const SCAN_BLOCK: u32 = 512u;
const RADIX_BLOCK: u32 = 1024u;

fn checked_add(a: u32, b: u32) -> u32 {
    let sum = a + b;
    if sum < a {
        fail(E_OVERFLOW);
    }
    return sum;
}
// Values of one scan level: uc at p.a (p.e = 0) or the scratch ud at p.a.
fn scan_load(index: u32) -> u32 {
    if p.e == 0u {
        return uc[p.a + index];
    }
    return ud[p.a + index];
}
fn scan_store(index: u32, value: u32) {
    if p.e == 0u {
        uc[p.a + index] = value;
    } else {
        ud[p.a + index] = value;
    }
}

var<workgroup> partial: array<u32, WG>;

// Exclusive in-place scan of p.n values within blocks of 512; block totals go
// to ud at p.b.
@compute @workgroup_size(WG)
fn scan_blocks(
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let block = p.o + group.y * groups.x + group.x;
    let first = block * SCAN_BLOCK + 2u * local;
    var a = 0u;
    var b = 0u;
    if first < p.n {
        a = scan_load(first);
    }
    if first + 1u < p.n {
        b = scan_load(first + 1u);
    }
    var total = checked_add(a, b);
    partial[local] = total;
    workgroupBarrier();
    for (var step = 1u; step < WG; step <<= 1u) {
        var addend = 0u;
        if local >= step {
            addend = partial[local - step];
        }
        workgroupBarrier();
        total = checked_add(total, addend);
        partial[local] = total;
        workgroupBarrier();
    }
    let exclusive = total - a - b;
    if first < p.n {
        scan_store(first, exclusive);
    }
    if first + 1u < p.n {
        scan_store(first + 1u, exclusive + a);
    }
    if local == WG - 1u && block * SCAN_BLOCK < p.n {
        ud[p.b + block] = total;
    }
}

// Adds each block's scanned total (ud at p.b) to its values.
@compute @workgroup_size(WG)
fn scan_add(
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let block = p.o + group.y * groups.x + group.x;
    let first = block * SCAN_BLOCK + 2u * local;
    if first >= p.n {
        return;
    }
    let offset = ud[p.b + block];
    scan_store(first, checked_add(scan_load(first), offset));
    if first + 1u < p.n {
        scan_store(first + 1u, checked_add(scan_load(first + 1u), offset));
    }
}

// Records are three words (key low, key high, payload). p.e selects the source
// (0: ua, 1: ub); the other buffer receives the pass. The digit is eight bits of
// key word p.c at shift p.d. uc holds counts by digit, then block (p.b blocks).
fn record(index: u32, word: u32) -> u32 {
    if p.e == 0u {
        return ua[3u * index + word];
    }
    return ub[3u * index + word];
}
fn digit(index: u32) -> u32 {
    return (record(index, p.c) >> p.d) & 255u;
}

@compute @workgroup_size(WG)
fn radix_count(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let block = index_of(id, groups);
    if block >= p.b {
        return;
    }
    var counts: array<u32, 256>;
    for (var d = 0u; d < 256u; d += 1u) {
        counts[d] = 0u;
    }
    let end = min(p.n, (block + 1u) * RADIX_BLOCK);
    for (var index = block * RADIX_BLOCK; index < end; index += 1u) {
        counts[digit(index)] += 1u;
    }
    for (var d = 0u; d < 256u; d += 1u) {
        uc[d * p.b + block] = counts[d];
    }
}

// uc: scanned counts. Stable within and across blocks.
@compute @workgroup_size(WG)
fn radix_scatter(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let block = index_of(id, groups);
    if block >= p.b {
        return;
    }
    var cursors: array<u32, 256>;
    for (var d = 0u; d < 256u; d += 1u) {
        cursors[d] = uc[d * p.b + block];
    }
    let end = min(p.n, (block + 1u) * RADIX_BLOCK);
    for (var index = block * RADIX_BLOCK; index < end; index += 1u) {
        let d = digit(index);
        let slot = 3u * cursors[d];
        cursors[d] += 1u;
        if p.e == 0u {
            ub[slot] = ua[3u * index];
            ub[slot + 1u] = ua[3u * index + 1u];
            ub[slot + 2u] = ua[3u * index + 2u];
        } else {
            ua[slot] = ub[3u * index];
            ua[slot + 1u] = ub[3u * index + 1u];
            ua[slot + 2u] = ub[3u * index + 2u];
        }
    }
}
