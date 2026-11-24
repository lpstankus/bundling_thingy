override map_width: u32;
override map_height: u32;
override spacing: f32;

@group(0) @binding(0) var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> ranges: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> output_points: array<vec2<f32>>;

@compute @workgroup_size(256)
fn smooth_paths(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = id.y * groups.x * 256u + id.x;
    if index >= arrayLength(&points) {
        return;
    }
    // Find this point's edge so the averaging window never crosses into another edge.
    var first = 0u;
    var last = arrayLength(&ranges);
    while last - first > 1u {
        let middle = first + (last - first) / 2u;
        if ranges[middle].x <= index {
            first = middle;
        } else {
            last = middle;
        }
    }
    let range = ranges[first];
    let point = points[index];
    if index == range.x || index == range.y - 1u {
        output_points[index] = point;
        return;
    }
    // Released CUBu defaults: strength 0.2, window radius 5% of the image size.
    let radius = u32(0.05 * f32(max(map_width, map_height)) / spacing);
    let begin = max(range.x, index - min(index, radius));
    let end = min(range.y - 1u, index + min(radius, range.y - 1u - index));
    var mean = vec2<f32>(0.0);
    for (var neighbor = begin; neighbor <= end; neighbor += 1u) {
        mean += points[neighbor];
    }
    mean /= f32(end - begin + 1u);
    output_points[index] = mix(point, mean, 0.2);
}
