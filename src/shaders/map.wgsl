struct MapDimensions {
    size: vec2<u32>,
    padding: vec2<u32>,
}

@group(0) @binding(9) var<uniform> map: MapDimensions;
