override map_width: u32;
override map_height: u32;

@group(0) @binding(0) var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> output_image: array<f32>;

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
fn count_points(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = id.y * groups.x * 256u + id.x;
    if index >= arrayLength(&points) {
        return;
    }
    count_sample(grid_position(points[index]));
}

@compute @workgroup_size(32, 4)
fn resolve_counts(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(map_width, map_height)) {
        return;
    }

    let index = id.y * map_width + id.x;
    output_image[index] = f32(atomicLoad(&counts[index]));
}
