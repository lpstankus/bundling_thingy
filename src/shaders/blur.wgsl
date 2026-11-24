override map_width: u32;
override map_height: u32;
override axis: u32;

@group(0) @binding(0) var<storage, read> input_image: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_image: array<f32>;
@group(0) @binding(2) var<storage, read> kernel: array<f32>;
@group(0) @binding(3) var<uniform> kernel_radius: u32;

@compute @workgroup_size(32, 4)
fn blur(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<u32>(map_width, map_height);
    if any(id.xy >= size) {
        return;
    }

    // CUBu convolves its point histogram with a parabolic kernel in two passes.
    let radius = i32(kernel_radius);
    let direction = select(vec2<i32>(0, 1), vec2<i32>(1, 0), axis == 0u);
    var value = 0.0;
    for (var offset = -radius; offset <= radius; offset += 1) {
        let position = vec2<i32>(id.xy) + direction * offset;
        if all(position >= vec2<i32>(0)) && all(position < vec2<i32>(size)) {
            let index = u32(position.y) * map_width + u32(position.x);
            value += input_image[index] * kernel[u32(abs(offset))];
        }
    }
    output_image[id.y * map_width + id.x] = value;
}
