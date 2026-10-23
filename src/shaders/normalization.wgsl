override map_width: u32;
override map_height: u32;

@group(0) @binding(0) var<storage, read> input_image: array<f32>;
@group(0) @binding(1) var<storage, read_write> maximum: atomic<u32>;
@group(0) @binding(2) var output_image: texture_storage_2d<rgba16float, write>;

var<workgroup> local_maximum: array<f32, 128>;

@compute @workgroup_size(32, 4)
fn find_maximum(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    var value = 0.0;
    if all(id.xy < vec2<u32>(map_width, map_height)) {
        value = input_image[id.y * map_width + id.x];
    }

    // Out-of-bounds lanes still participate in every workgroup barrier.
    local_maximum[lane] = value;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride /= 2u) {
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

@compute @workgroup_size(32, 4)
fn normalize(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(map_width, map_height)) {
        return;
    }

    let peak = bitcast<f32>(atomicLoad(&maximum));
    var value = 0.0;
    if peak > 0.0 {
        value = clamp(input_image[id.y * map_width + id.x] / peak, 0.0, 1.0);
    }
    textureStore(output_image, vec2<i32>(id.xy), vec4<f32>(value, 0.0, 0.0, 1.0));
}
