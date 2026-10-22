override map_width: u32;
override map_height: u32;

const SAMPLE_SPACING: f32 = 1.0; // Density samples are one texel apart at every resolution.

// Normalized Gaussian, sigma = 6 texels, truncated at radius 18.
const GAUSSIAN = array<f32, 19>(
    0.0666251319, 0.0657061792, 0.0630246744, 0.0587964725, 0.0533492351,
    0.0470805347, 0.0404101852, 0.0337346772, 0.0273904106, 0.0216300135,
    0.0166131238, 0.0124102942, 0.0090167311, 0.0063716575, 0.0043791719,
    0.0029273040, 0.0019031803, 0.0012034507, 0.0007401384,
);

@group(0) @binding(0) var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> horizontal: array<f32>;
@group(0) @binding(3) var<storage, read_write> blurred: array<f32>;
@group(0) @binding(4) var<storage, read_write> maximum: atomic<u32>;
@group(0) @binding(5) var density: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var<storage, read> starts: array<u32>;

var<workgroup> local_maximum: array<f32, 256>;

fn grid_position(point: vec2<f32>) -> vec2<f32> {
    let uv = vec2<f32>(point.x * 0.5 + 0.5, 0.5 - point.y * 0.5);
    return uv * vec2<f32>(f32(map_width - 1u), f32(map_height - 1u));
}

fn count_sample(position: vec2<f32>) {
    let last = vec2<f32>(f32(map_width - 1u), f32(map_height - 1u));
    let cell = vec2<u32>(round(clamp(position, vec2<f32>(0.0), last)));
    atomicAdd(&counts[cell.y * map_width + cell.x], 1u);
}

@compute @workgroup_size(256)
fn count_edges(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x + 1u >= arrayLength(&starts) {
        return;
    }
    let start = starts[id.x];
    let end = starts[id.x + 1u];
    if start >= end {
        return;
    }

    // Match Graph::mesh: an edge needs at least one nonzero segment to render.
    var first_segment = start + 1u;
    while first_segment < end && length(points[first_segment] - points[first_segment - 1u]) == 0.0 {
        first_segment += 1u;
    }
    if first_segment >= end {
        return;
    }

    // Keep sampling phase across segment boundaries and restart at each edge.
    var previous = grid_position(points[first_segment - 1u]);
    var remaining = SAMPLE_SPACING;
    count_sample(previous);
    for (var index = first_segment; index < end; index += 1u) {
        let current = grid_position(points[index]);
        let delta = current - previous;
        let distance = length(delta);
        var along = remaining;
        while along <= distance {
            count_sample(previous + delta * (along / distance));
            along += SAMPLE_SPACING;
        }
        remaining = along - distance;
        previous = current;
    }
    if remaining < SAMPLE_SPACING {
        count_sample(previous);
    }
}

@compute @workgroup_size(16, 16)
fn blur_x(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(map_width, map_height)) {
        return;
    }

    var value = 0.0;
    for (var offset = -18; offset <= 18; offset += 1) {
        let x = i32(id.x) + offset;
        // The density outside the graph's map is zero.
        if x >= 0 && x < i32(map_width) {
            value += f32(atomicLoad(&counts[id.y * map_width + u32(x)])) * GAUSSIAN[u32(abs(offset))];
        }
    }
    horizontal[id.y * map_width + id.x] = value;
}

@compute @workgroup_size(16, 16)
fn blur_y(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    var value = 0.0;
    if all(id.xy < vec2<u32>(map_width, map_height)) {
        for (var offset = -18; offset <= 18; offset += 1) {
            let y = i32(id.y) + offset;
            if y >= 0 && y < i32(map_height) {
                value += horizontal[u32(y) * map_width + id.x] * GAUSSIAN[u32(abs(offset))];
            }
        }
        blurred[id.y * map_width + id.x] = value;
    }

    // Out-of-bounds lanes still participate in every workgroup barrier.
    local_maximum[lane] = value;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if lane < stride {
            local_maximum[lane] = max(local_maximum[lane], local_maximum[lane + stride]);
        }
        workgroupBarrier();
    }
    if lane == 0u {
        // Positive finite f32 values have the same ordering as their u32 bits.
        atomicMax(&maximum, bitcast<u32>(local_maximum[0]));
    }
}

@compute @workgroup_size(16, 16)
fn normalize(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(map_width, map_height)) {
        return;
    }

    let peak = bitcast<f32>(atomicLoad(&maximum));
    var value = 0.0;
    if peak > 0.0 {
        value = clamp(blurred[id.y * map_width + id.x] / peak, 0.0, 1.0);
    }
    textureStore(density, vec2<i32>(id.xy), vec4<f32>(value, 0.0, 0.0, 1.0));
}
