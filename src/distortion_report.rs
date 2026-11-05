use crate::{Options, gpu::SIZE};
use bundling_wgpu::geometry::Layout;
use color_eyre::eyre::Result;
use serde_json::{Value, json};

pub fn write(opts: &Options, layout: &Layout, images: &[Value]) -> Result<()> {
    let report = json!({
        "schema_version": 1, "metric_version": 1, "metric": "distortion",
        "definition": "Square root of the equal-edge mean integral of squared clip-coordinate displacement under directed normalized arc-length correspondence, divided by the fixed canvas diagonal.",
        "interpretation": "Lower means less geometric change from the original GPU-resampled geometry. Detours, retracing, reversal and loop start changes affect correspondence. Splitting edges changes equal-edge weighting. This is not Frechet or Hausdorff distance, topology, ambiguity or overall quality. Color, MSAA, opacity and endpoint dots are excluded. GPU ordering can cause run-to-run variation.",
        "input": opts.input.display().to_string(),
        "baseline": {
            "source": "gpu_resampled_iteration_0",
            "correspondence": "directed_edge_normalized_arc_length",
            "weighting": "equal_edges", "preprocessing_distortion_included": false
        },
        "geometry": {
            "geometry_contract_version": 1, "coordinate_space": "clip",
            "width": SIZE, "height": SIZE,
            "normalization": "initial graph bounds, centered uniform 95% fit, unchanged through iterations",
            "diagonal_clip_units": 2.0 * std::f64::consts::SQRT_2,
            "pixels_per_clip_unit": f64::from(SIZE) / 2.0,
            "edges_total": layout.edge_count().saturating_add(layout.skipped_count()),
            "edges_evaluated": layout.edge_count(), "edges_skipped": layout.skipped_count(),
            "active_samples": layout.sample_count()
        },
        "bundling": {"neighbor_radius": opts.neighbor_radius,
            "iterations": if opts.neighbor_radius.is_some() { opts.iterations } else { 0 }},
        "images": images
    });
    crate::report::publish(&opts.out, "distortion.json", &report)
}
