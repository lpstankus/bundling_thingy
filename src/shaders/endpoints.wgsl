override block_size: u32;

// Reuse control-point memory after resampling. Four slots per path hold at most two endpoints.
@group(0) @binding(0) var<storage, read_write> table: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> paths: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> points: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> endpoints: array<u32>;

fn insert(index: u32) {
    let point = points[index];
    let bits = bitcast<vec2<u32>>(select(point, vec2<f32>(0.0), point == vec2<f32>(0.0)));
    var hash = bits.x * 0x9e3779b9u ^ bits.y;
    hash = (hash ^ (hash >> 16u)) * 0x7feb352du;
    hash = (hash ^ (hash >> 15u)) * 0x846ca68bu;
    var slot = (hash ^ (hash >> 16u)) % arrayLength(&table);
    loop {
        let old = atomicLoad(&table[slot]);
        if old != 0u {
            if all(points[old - 1u] == point) {
                return;
            }
            slot = (slot + 1u) % arrayLength(&table);
            continue;
        }
        let result = atomicCompareExchangeWeak(&table[slot], 0u, index + 1u);
        if result.exchanged {
            // State words 9 and 2 are the indirect instance count and endpoint statistic.
            let output = atomicAdd(&state[9], 1u);
            atomicAdd(&state[2], 1u);
            endpoints[output] = index;
            return;
        }
    }
}

@compute @workgroup_size(block_size)
fn deduplicate(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = id.y * groups.x * block_size + id.x;
    if index >= arrayLength(&paths) || atomicLoad(&state[3]) != 0u {
        return;
    }
    let path = paths[index];
    if path.w < 2u {
        return;
    }
    insert(path.x);
    insert(path.x + path.w - 1u);
}
