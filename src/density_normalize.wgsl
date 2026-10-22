override map_width: u32;
override map_height: u32;

@group(0) @binding(0) var<storage, read> input_image: array<f32>;
@group(0) @binding(1) var<storage, read> maximum: u32;
@group(0) @binding(2) var output_image: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16)
fn normalize(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(map_width, map_height)) {
        return;
    }

    let peak = bitcast<f32>(maximum);
    var value = 0.0;
    if peak > 0.0 {
        value = clamp(input_image[id.y * map_width + id.x] / peak, 0.0, 1.0);
    }
    textureStore(output_image, vec2<i32>(id.xy), vec4<f32>(value, 0.0, 0.0, 1.0));
}
