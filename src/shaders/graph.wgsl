// Scalar position fields keep the storage-array stride at 12 bytes.
struct Point {
    x: f32,
    y: f32,
    edge_id: u32,
}

struct Edge {
    source: vec2<f32>,
    destination: vec2<f32>,
}

// Sample ranges and draw reservations stay separate from immutable endpoints.
struct Path {
    point_offset: u32,
    segment_offset: u32,
    count: u32,
}

fn point_position(point: Point) -> vec2<f32> {
    return vec2<f32>(point.x, point.y);
}
