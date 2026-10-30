@group(0) @binding(0) var<storage, read> input_image: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_image: array<f32>;
@group(0) @binding(2) var<uniform> direction: vec2<u32>;
@group(0) @binding(3) var<storage, read_write> maximum: atomic<u32>;

// A 32x32 output tile plus an 18-texel halo along the blur direction.
var<workgroup> tile: array<f32, 2176>;

fn pair(center: u32, offset: u32) -> f32 {
    return tile[center - offset] + tile[center + offset];
}

@compute @workgroup_size(32, 8)
fn blur(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let base = group.xy * 32u;
    let size = vec2<u32>(32u) + 36u * direction;
    for (var index = lane; index < size.x * size.y; index += 256u) {
        let position = vec2<i32>(base)
            + vec2<i32>(i32(index % size.x), i32(index / size.x)) - 18 * vec2<i32>(direction);
        var value = 0.0;
        if all(position >= vec2<i32>(0)) && all(position < vec2<i32>(map.size)) {
            let bits = input_image[u32(position.y) * map.size.x + u32(position.x)];
            // Horizontal reads integer counts. Vertical reads the horizontal f32 results.
            value = select(bitcast<f32>(bits), f32(bits), direction.x != 0u);
        }
        tile[index] = value;
    }
    workgroupBarrier();

    var peak = 0.0;
    for (var row = 0u; row < 4u; row += 1u) {
        let local_position = local.xy + vec2<u32>(0u, row * 8u);
        let position = base + local_position;
        let center = local_position + 18u * direction;
        let index = center.y * size.x + center.x;
        let stride = direction.x + direction.y * size.x;
        // Gaussian sigma 6, radius 18. Unroll paired taps to avoid dynamic weight indexing.
        var value = tile[index] * 0.0666251319;
        value += pair(index, stride) * 0.0657061792;
        value += pair(index, 2u * stride) * 0.0630246744;
        value += pair(index, 3u * stride) * 0.0587964725;
        value += pair(index, 4u * stride) * 0.0533492351;
        value += pair(index, 5u * stride) * 0.0470805347;
        value += pair(index, 6u * stride) * 0.0404101852;
        value += pair(index, 7u * stride) * 0.0337346772;
        value += pair(index, 8u * stride) * 0.0273904106;
        value += pair(index, 9u * stride) * 0.0216300135;
        value += pair(index, 10u * stride) * 0.0166131238;
        value += pair(index, 11u * stride) * 0.0124102942;
        value += pair(index, 12u * stride) * 0.0090167311;
        value += pair(index, 13u * stride) * 0.0063716575;
        value += pair(index, 14u * stride) * 0.0043791719;
        value += pair(index, 15u * stride) * 0.0029273040;
        value += pair(index, 16u * stride) * 0.0019031803;
        value += pair(index, 17u * stride) * 0.0012034507;
        value += pair(index, 18u * stride) * 0.0007401384;
        if all(position < map.size) {
            output_image[position.y * map.size.x + position.x] = value;
            peak = max(peak, value);
        }
    }

    if direction.y != 0u {
        // Finish all tile reads before reusing its first 256 values for reduction.
        workgroupBarrier();
        tile[lane] = peak;
        workgroupBarrier();
        for (var stride = 128u; stride > 0u; stride /= 2u) {
            if lane < stride {
                tile[lane] = max(tile[lane], tile[lane + stride]);
            }
            workgroupBarrier();
        }
        if lane == 0u {
            // Positive finite f32 values retain their ordering as u32 bits.
            atomicMax(&maximum, bitcast<u32>(tile[0]));
        }
    }
}
