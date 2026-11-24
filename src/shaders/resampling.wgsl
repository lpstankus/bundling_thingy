override map_width: u32;
override map_height: u32;
override spacing: f32;

@group(0) @binding(0) var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> ranges: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> counts: array<u32>;
@group(0) @binding(3) var<storage, read> output_ranges: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> output_points: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> output_segments: array<u32>;

fn pixel_length(delta: vec2<f32>) -> f32 {
    let scale = 0.5 * vec2<f32>(f32(map_width - 1u), f32(map_height - 1u));
    return length(delta * scale);
}

@compute @workgroup_size(64)
fn count_samples(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let edge = id.y * groups.x * 64u + id.x;
    if edge >= arrayLength(&ranges) {
        return;
    }
    let range = ranges[edge];
    var distance = 0.0;
    for (var index = range.x + 1u; index < range.y; index += 1u) {
        distance += pixel_length(points[index] - points[index - 1u]);
    }
    // Clamp before converting so impossible allocations fail validation on the CPU.
    let closed = all(points[range.x] == points[range.y - 1u]) && distance > 0.0;
    counts[edge] = max(select(2u, 3u, closed), u32(min(ceil(distance / spacing), 4294967040.0)) + 1u);
}

@compute @workgroup_size(64)
fn resample(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let edge = id.y * groups.x * 64u + id.x;
    if edge >= arrayLength(&ranges) {
        return;
    }
    let range = ranges[edge];
    let output_range = output_ranges[edge];
    let count = output_range.y - output_range.x;
    var source = range.x + 1u;
    var consumed = 0.0;
    output_points[output_range.x] = points[range.x];
    for (var index = 1u; index < count; index += 1u) {
        let output_index = output_range.x + index;
        output_segments[output_index - edge - 1u] = output_index - 1u;
        if index == count - 1u {
            output_points[output_index] = points[range.y - 1u];
            continue;
        }
        let sample_distance = f32(index) * spacing;
        var segment_length = pixel_length(points[source] - points[source - 1u]);
        while source < range.y - 1u && consumed + segment_length < sample_distance {
            consumed += segment_length;
            source += 1u;
            segment_length = pixel_length(points[source] - points[source - 1u]);
        }
        // Preserve a drawable point on closed paths shorter than the sampling step.
        if count == 3u && all(points[range.x] == points[range.y - 1u])
            && consumed + segment_length <= sample_distance {
            var representative = range.x + 1u;
            while representative < range.y - 1u && all(points[representative] == points[range.x]) {
                representative += 1u;
            }
            output_points[output_index] = points[representative];
            continue;
        }
        let fraction = clamp((sample_distance - consumed) / max(segment_length, 1e-8), 0.0, 1.0);
        output_points[output_index] = mix(points[source - 1u], points[source], fraction);
    }
}
