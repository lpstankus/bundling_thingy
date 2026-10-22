#![allow(clippy::arithmetic_side_effects)]

use super::{Graph, Point, ResampledGraph};
use glam::DVec2;

#[test]
fn construction_filters_degenerate_edges_before_normalization() {
    let mut graph = Graph::parse("0: 10000 10000\n1: -10000 0 -10000 0\n2: 0 0 0 0 2 0 2 0")
        .expect("parse graph fixture");
    assert_eq!(graph.starts, [0, 2]);
    assert_eq!(graph.points, [Point::ZERO, Point::new(2.0, 0.0)]);
    graph.normalize().expect("normalize graph fixture");
    assert!((graph.points[0].x + 0.95).abs() < 0.00001);
    assert!((graph.points[1].x - 0.95).abs() < 0.00001);
    for text in ["", "0: 0 0", "0: 0 0 0 0 0 0", "0: NaN 0 1 0"] {
        assert!(Graph::parse(text).is_err());
    }
}

#[test]
fn resampling_tracks_render_dimensions_and_preserves_endpoints() {
    let graph = Graph::parse("0: -1 0 1 0").expect("parse graph fixture");
    for (width, height) in [(29, 23), (57, 45), (1, 23), (29, 1), (1, 1)] {
        let sampled = graph
            .resample(width, height)
            .expect("resample graph fixture");
        assert_eq!(sampled.dimensions(), [width, height]);
        assert_eq!(
            sampled.points().len(),
            usize::try_from(width.max(2)).expect("small dimension")
        );
        assert_eq!(sampled.points().first(), Some(&Point::new(-1.0, 0.0)));
        assert_eq!(sampled.points().last(), Some(&Point::new(1.0, 0.0)));
        verify_segments(&sampled);
    }
    assert!(graph.resample(0, 23).is_err());
    assert!(graph.resample(29, 0).is_err());
}

#[test]
fn sampling_phase_survives_segment_subdivision() {
    let reference = Graph::parse("0: -1 0 1 0")
        .expect("parse graph fixture")
        .resample(29, 23)
        .expect("resample graph fixture");
    for text in ["0: -1 0 -0.25 0 0 0 0.5 0 1 0", "0: -1 0 -1 0 1 0 1 0"] {
        let sampled = Graph::parse(text)
            .expect("parse graph fixture")
            .resample(29, 23)
            .expect("resample graph fixture");
        assert_eq!(reference.points(), sampled.points());
        assert_eq!(reference.segments(), sampled.segments());
    }
}

#[test]
fn resampling_keeps_diagonals_bends_and_short_closed_paths_drawable() {
    let diagonal = Graph::parse("0: -1 -1 1 1")
        .expect("parse graph fixture")
        .resample(25, 33)
        .expect("resample graph fixture");
    assert_eq!(diagonal.points().len(), 41);
    verify_segments(&diagonal);
    for text in [
        "0: -1 -1 -1 1 1 1",
        "0: -0.9 -0.7 -0.1234 0.4567 0.8765 -0.3333",
        "0: 0 0 0.005 0 0 0",
    ] {
        verify_segments(
            &Graph::parse(text)
                .expect("parse graph fixture")
                .resample(25, 33)
                .expect("resample graph fixture"),
        );
    }
}

#[test]
fn segment_indices_never_join_separate_edges() {
    let sampled = Graph::parse("0: -1 -0.5 -0.5 -0.5\n1: 0.5 0.5 1 0.5")
        .expect("parse graph fixture")
        .resample(29, 23)
        .expect("resample graph fixture");
    assert_eq!(sampled.points().len(), 16);
    assert_eq!(sampled.segments().len(), 14);
    assert!(!sampled.segments().contains(&7));
    verify_segments(&sampled);
}

#[test]
fn normalization_cannot_leave_collapsed_edges_in_gpu_geometry() {
    let mut graph = Graph::parse("0: 0 0 0.000000000000000000000000000001 0\n1: -1 0 1 0")
        .expect("parse graph fixture");
    graph.normalize().expect("normalize graph fixture");
    verify_segments(&graph.resample(29, 23).expect("resample graph fixture"));

    let mut graph =
        Graph::parse("0: 0 0 0.000000000000000000000000000001 0 0.005 0 0 0\n1: -1 0 1 0")
            .expect("parse closed path with a collapsing first segment");
    graph.normalize().expect("normalize closed path");
    let sampled = graph.resample(29, 23).expect("resample closed path");
    assert_eq!(sampled.points().len(), 31);
    verify_segments(&sampled);
}

fn verify_segments(graph: &ResampledGraph) {
    let [width, height] = graph.dimensions();
    let scale = DVec2::new(f64::from(width - 1) / 2.0, f64::from(height - 1) / 2.0);
    for &index in graph.segments() {
        let index = usize::try_from(index).expect("small point index");
        let first = graph.points()[index];
        let second = graph.points()[index + 1];
        assert!(first.is_finite() && second.is_finite());
        assert!((second - first).length() > 0.0);
        let distance = ((second.as_dvec2() - first.as_dvec2()) * scale).length();
        assert!(distance <= 1.0001, "gap is {distance} texels");
    }
}
