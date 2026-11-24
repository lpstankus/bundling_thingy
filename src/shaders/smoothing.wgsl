override block_size: u32;
override accumulate_density: bool;

struct SmoothingParameters {
    window: u32,
    strength: f32,
    padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> input_points: array<Point>;
@group(0) @binding(1) var<storage, read_write> output_points: array<Point>;
@group(0) @binding(3) var<storage, read_write> density: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> paths: array<Path>;
@group(0) @binding(10) var<uniform> smoothing: SmoothingParameters;

// Laplacian smoothing along each path. The window shrinks symmetrically near the ends,
// so fixed endpoints never pull their neighbors sideways.
@compute @workgroup_size(block_size)
fn smooth_paths(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = id.y * groups.x * block_size + id.x;
    if index >= arrayLength(&input_points) {
        return;
    }
    let current = input_points[index];
    let position = point_position(current);
    let path = paths[current.edge_id];
    let along = index - path.point_offset;
    let reach = min(smoothing.window, min(along, path.count - 1u - along));
    var smoothed = position;
    if reach > 0u {
        // Offsets from the center avoid summing large absolute coordinates.
        var offset = vec2<f32>(0.0);
        for (var step = 1u; step <= reach; step += 1u) {
            offset += point_position(input_points[index - step]) - position;
            offset += point_position(input_points[index + step]) - position;
        }
        smoothed = position + smoothing.strength * offset / f32(2u * reach + 1u);
    }
    output_points[index] = Point(smoothed.x, smoothed.y, current.edge_id);
    if accumulate_density {
        let uv = vec2<f32>(smoothed.x * 0.5 + 0.5, 0.5 - smoothed.y * 0.5);
        let cell = vec2<u32>(round(uv * vec2<f32>(map.size - vec2<u32>(1u))));
        atomicAdd(&density[cell.y * map.size.x + cell.x], 1u);
    }
}
