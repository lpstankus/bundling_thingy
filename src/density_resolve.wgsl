override map_width: u32;
override map_height: u32;

@group(0) @binding(0) var<storage, read> counts: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_image: array<f32>;

@compute @workgroup_size(16, 16)
fn resolve_counts(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(map_width, map_height)) {
        return;
    }

    let index = id.y * map_width + id.x;
    output_image[index] = f32(counts[index]);
}
