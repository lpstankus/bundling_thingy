@group(0) @binding(0) var<storage, read> input_image: array<f32>;
@group(0) @binding(1) var<storage, read_write> maximum: atomic<u32>;
@group(0) @binding(2) var output_image: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(32, 4)
fn normalize(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= map.size) {
        return;
    }

    let peak = bitcast<f32>(atomicLoad(&maximum));
    var value = 0.0;
    if peak > 0.0 {
        value = clamp(input_image[id.y * map.size.x + id.x] / peak, 0.0, 1.0);
    }
    textureStore(output_image, vec2<i32>(id.xy), vec4<f32>(value, 0.0, 0.0, 1.0));
}
