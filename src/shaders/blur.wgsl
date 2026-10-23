override map_width: u32;
override map_height: u32;
override axis: u32;

// Normalized Gaussian, sigma = 6 texels, truncated at radius 18.
const GAUSSIAN = array<f32, 19>(
    0.0666251319, 0.0657061792, 0.0630246744, 0.0587964725, 0.0533492351,
    0.0470805347, 0.0404101852, 0.0337346772, 0.0273904106, 0.0216300135,
    0.0166131238, 0.0124102942, 0.0090167311, 0.0063716575, 0.0043791719,
    0.0029273040, 0.0019031803, 0.0012034507, 0.0007401384,
);

@group(0) @binding(0) var<storage, read> input_image: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_image: array<f32>;

@compute @workgroup_size(32, 4)
fn blur(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<u32>(map_width, map_height);
    if any(id.xy >= size) {
        return;
    }

    let direction = select(vec2<i32>(0, 1), vec2<i32>(1, 0), axis == 0u);
    var value = 0.0;
    for (var offset = -18; offset <= 18; offset += 1) {
        let position = vec2<i32>(id.xy) + direction * offset;
        if all(position >= vec2<i32>(0)) && all(position < vec2<i32>(size)) {
            let index = u32(position.y) * map_width + u32(position.x);
            value += input_image[index] * GAUSSIAN[u32(abs(offset))];
        }
    }
    output_image[id.y * map_width + id.x] = value;
}
