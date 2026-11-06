use crate::{Options, gpu::SIZE};
use bundling_wgpu::{
    ambiguity::{self, Bounds, CohortView, Evaluator, Row, View},
    ambiguity_cohorts::{Membership, Route},
    ambiguity_index, ambiguity_similarity,
    geometry::Snapshot,
    weak_frechet::{Budget, Statistics},
};
use color_eyre::eyre::{Result, WrapErr, ensure, eyre};
use serde::Serialize;
use std::mem::size_of;

#[derive(Debug, clap::Args, Serialize)]
pub struct Parameters {
    /// Whole-route tolerance in export pixels. Operational, not perceptually calibrated.
    #[arg(long = "ambiguity-route-tolerance-pixels", default_value_t = 16.0, value_parser = positive_pixels)]
    route_tolerance_pixels: f64,
    /// Matched-endpoint tolerance in export pixels.
    #[arg(long = "ambiguity-endpoint-tolerance-pixels", default_value_t = 4.0, value_parser = positive_pixels)]
    endpoint_tolerance_pixels: f64,
    /// Sustained-contact radius in export pixels. Zero is supported.
    #[arg(long = "ambiguity-contact-radius-pixels", default_value_t = 2.0, value_parser = nonnegative_pixels)]
    contact_radius_pixels: f64,
    /// Required contact length along both curves in export pixels.
    #[arg(long = "ambiguity-minimum-contact-pixels", default_value_t = 24.0, value_parser = positive_pixels)]
    minimum_contact_pixels: f64,
    /// Certified correspondence error in export pixels. Zero uses full curves.
    #[arg(long = "ambiguity-simplification-pixels", default_value_t = 0.1, value_parser = nonnegative_pixels)]
    simplification_pixels: f64,
    /// Work units per preparation or snapshot, shared across all pair evaluations.
    #[arg(long = "ambiguity-work-limit", default_value_t = 2_000_000_000, value_parser = clap::value_parser!(u64).range(1..))]
    work_limit: u64,
    /// Requested evaluation workspace bytes, excluding shared snapshots and allocator overhead.
    #[arg(long = "ambiguity-workspace-bytes", default_value_t = 1_073_741_824, value_parser = positive_capacity)]
    workspace_bytes: usize,
    /// Global stored-pair capacity, including fixed tables and temporary candidate records.
    #[arg(long = "ambiguity-stored-pair-limit", default_value_t = 5_000_000, value_parser = positive_capacity)]
    stored_pair_limit: usize,
}

fn positive_capacity(text: &str) -> std::result::Result<usize, String> {
    let value = text.parse::<usize>().map_err(|e| e.to_string())?;
    if value == 0 {
        return Err("capacity must be positive".to_owned());
    }
    Ok(value)
}
fn pixels(text: &str, allow_zero: bool) -> std::result::Result<f64, String> {
    let value = text.parse::<f64>().map_err(|e| e.to_string())?;
    let clip = value / pixels_per_clip();
    let squared = clip * clip;
    if !value.is_finite() || value < 0.0 || (!allow_zero && value == 0.0) {
        return Err(
            "pixels must be finite and positive, or zero for radius/simplification".to_owned(),
        );
    }
    if !squared.is_finite() || (value > 0.0 && (squared == 0.0 || !clip.recip().is_finite())) {
        return Err("pixel value is not representable in clip-distance arithmetic".to_owned());
    }
    Ok(value)
}
fn positive_pixels(text: &str) -> std::result::Result<f64, String> {
    pixels(text, false)
}
fn nonnegative_pixels(text: &str) -> std::result::Result<f64, String> {
    pixels(text, true)
}
fn pixels_per_clip() -> f64 {
    f64::from(SIZE) / 2.0
}
impl Parameters {
    pub fn validate(&self) -> Result<()> {
        for (value, zero) in [
            (self.route_tolerance_pixels, false),
            (self.endpoint_tolerance_pixels, false),
            (self.contact_radius_pixels, true),
            (self.minimum_contact_pixels, false),
            (self.simplification_pixels, true),
        ] {
            pixels(&value.to_string(), zero).map_err(|error| eyre!(error))?;
        }
        ensure!(
            self.workspace_bytes > 0 && self.stored_pair_limit > 0 && self.work_limit > 0,
            "ambiguity resource limits must be positive"
        );
        Ok(())
    }
    fn config(&self, routes: usize) -> Result<(ambiguity::Config, Capacities)> {
        let pairs = routes
            .checked_mul(routes.saturating_sub(1))
            .and_then(|n| n.checked_div(2))
            .ok_or_else(|| eyre!("possible pair count overflow"))?;
        let global = self.stored_pair_limit;
        let baseline = pairs.min(global / 5);
        let cohort = pairs.min(global / 10);
        let cache = pairs.min(global / 100).min(65_536);
        let fixed = cohort
            .checked_mul(3)
            .and_then(|n| n.checked_add(baseline))
            .and_then(|n| n.checked_add(cache))
            .ok_or_else(|| eyre!("fixed capacity overflow"))?;
        let capacities = Capacities {
            possible_edge_pairs: pairs,
            baseline_evidence_records: baseline,
            cohort_records_per_table: cohort,
            cohort_table_count: 3,
            cache_records: cache,
            fixed_pair_records: fixed,
            candidate_record_headroom: global.saturating_sub(fixed),
        };
        Ok((
            ambiguity::Config {
                similarity: ambiguity_similarity::Config {
                    route_tolerance: self.route_tolerance_pixels / pixels_per_clip(),
                    endpoint_tolerance: self.endpoint_tolerance_pixels / pixels_per_clip(),
                    sigma: self.simplification_pixels / pixels_per_clip(),
                },
                radius: self.contact_radius_pixels / pixels_per_clip(),
                minimum_contact: self.minimum_contact_pixels / pixels_per_clip(),
                piece_length: self.route_tolerance_pixels / pixels_per_clip(),
                baseline_capacity: baseline,
                cohort_capacity: cohort,
                cache_capacity: cache,
            },
            capacities,
        ))
    }
}
#[derive(Serialize)]
struct Capacities {
    possible_edge_pairs: usize,
    baseline_evidence_records: usize,
    cohort_records_per_table: usize,
    cohort_table_count: usize,
    cache_records: usize,
    fixed_pair_records: usize,
    candidate_record_headroom: usize,
}
#[derive(Serialize)]
struct General {
    current_incompatible_pairs: usize,
    baseline_incompatible_pairs: usize,
    introduced_pairs: usize,
    resolved_pairs: usize,
    squared_excess_pixels_squared: Bounds,
    baseline_squared_excess_pixels_squared: Bounds,
    introduced_squared_excess_pixels_squared: Bounds,
    resolved_squared_excess_pixels_squared: Bounds,
    rms_excess_pixels: Bounds,
    ratio: Bounds,
}
fn scaled(bounds: Bounds, factor: f64) -> Result<Bounds> {
    let result = Bounds {
        lower: bounds.lower * factor,
        upper: bounds.upper * factor,
    };
    ensure!(
        result.lower.is_finite() && result.upper.is_finite(),
        "nonfinite ambiguity report conversion"
    );
    Ok(result)
}
impl General {
    fn new(view: View, baseline: View) -> Result<Self> {
        let scale = pixels_per_clip();
        let square = scale * scale;
        Ok(Self {
            current_incompatible_pairs: view.current,
            baseline_incompatible_pairs: view.baseline,
            introduced_pairs: view.introduced,
            resolved_pairs: view.resolved,
            squared_excess_pixels_squared: scaled(view.squared_excess, square)?,
            baseline_squared_excess_pixels_squared: scaled(baseline.squared_excess, square)?,
            introduced_squared_excess_pixels_squared: scaled(
                view.introduced_squared_excess,
                square,
            )?,
            resolved_squared_excess_pixels_squared: scaled(view.resolved_squared_excess, square)?,
            rms_excess_pixels: scaled(view.rms_excess, scale)?,
            ratio: view.ratio,
        })
    }
}
#[derive(Serialize)]
struct Endpoints {
    current_incompatible_pairs: usize,
    baseline_incompatible_pairs: usize,
    introduced_pairs: usize,
    resolved_pairs: usize,
    squared_excess_pixels_squared: f64,
    baseline_squared_excess_pixels_squared: f64,
    introduced_squared_excess_pixels_squared: f64,
    resolved_squared_excess_pixels_squared: f64,
    rms_excess_pixels: f64,
    ratio: f64,
}
impl From<General> for Endpoints {
    fn from(value: General) -> Self {
        Self {
            current_incompatible_pairs: value.current_incompatible_pairs,
            baseline_incompatible_pairs: value.baseline_incompatible_pairs,
            introduced_pairs: value.introduced_pairs,
            resolved_pairs: value.resolved_pairs,
            squared_excess_pixels_squared: value.squared_excess_pixels_squared.lower,
            baseline_squared_excess_pixels_squared: value
                .baseline_squared_excess_pixels_squared
                .lower,
            introduced_squared_excess_pixels_squared: value
                .introduced_squared_excess_pixels_squared
                .lower,
            resolved_squared_excess_pixels_squared: value
                .resolved_squared_excess_pixels_squared
                .lower,
            rms_excess_pixels: value.rms_excess_pixels.lower,
            ratio: value.ratio.lower,
        }
    }
}
#[derive(Serialize)]
pub struct Image {
    file: String,
    iteration: u32,
    contact_pairs: u64,
    baseline_contact_pairs: u64,
    current_collapsed_paths: usize,
    general: General,
    endpoints: Endpoints,
    cohorts: CohortView,
    #[serde(with = "ResourceStatistics")]
    resources: Statistics,
    #[serde(with = "IndexStatistics")]
    index: ambiguity_index::Statistics,
}
impl Image {
    pub fn new(file: String, iteration: u32, row: &Row, baseline: &Row) -> Result<Self> {
        Ok(Self {
            file,
            iteration,
            contact_pairs: row.contacts,
            baseline_contact_pairs: baseline.contacts,
            current_collapsed_paths: row.collapsed,
            general: General::new(row.general, baseline.general)?,
            endpoints: General::new(row.endpoints, baseline.endpoints)?.into(),
            cohorts: row.cohorts,
            resources: row.resources,
            index: row.index,
        })
    }
}
#[derive(Serialize)]
struct Report<'a> {
    #[serde(flatten)]
    metadata: serde_json::Value,
    parameters: &'a Parameters,
    capacities: Capacities,
    memberships: &'a [Membership],
    #[serde(with = "ResourceStatistics")]
    preparation_resources: Statistics,
    images: &'a [Image],
}

pub fn run(
    opts: &Options,
    baseline: &Snapshot<'_>,
    operation: impl FnOnce(&mut Evaluator<'_, '_>, &mut Vec<Image>) -> Result<()>,
) -> Result<()> {
    let params = &opts.ambiguity;
    let count = baseline.layout().edge_count();
    let (config, capacities) = params.config(count)?;
    let rows = if opts.neighbor_radius.is_some() {
        usize::try_from(opts.iterations)?
            .checked_add(1)
            .ok_or_else(|| eyre!("image count overflow"))?
    } else {
        1
    };
    let route_bytes = count
        .checked_mul(
            size_of::<Route<'_>>()
                .checked_add(size_of::<ambiguity_index::Path<'_>>())
                .ok_or_else(|| eyre!("route size overflow"))?,
        )
        .ok_or_else(|| eyre!("route buffer overflow"))?;
    let row_bytes = rows
        .checked_mul(
            size_of::<Image>()
                .checked_add(32)
                .ok_or_else(|| eyre!("image size overflow"))?,
        )
        .ok_or_else(|| eyre!("image buffer overflow"))?;
    let caller_bytes = route_bytes
        .checked_add(row_bytes)
        .ok_or_else(|| eyre!("integration buffer overflow"))?;
    let mut budget = Budget::with_record_limit(
        params.work_limit,
        params.workspace_bytes,
        params.stored_pair_limit,
    );
    budget
        .reserve(caller_bytes)
        .wrap_err("ambiguity route/report buffers exceed --ambiguity-workspace-bytes")?;
    let outcome = (|| -> Result<()> {
        let mut originals = Vec::new();
        originals.try_reserve_exact(count)?;
        originals.extend(
            baseline
                .edges()
                .map(|(edge_id, points)| Route { edge_id, points }),
        );
        let mut images = Vec::new();
        images.try_reserve_exact(rows)?;
        // The inner Result preserves GPU/I/O failures; the outer Result owns evaluator cleanup.
        ambiguity::with_evaluator(&originals, &config, &mut budget, |evaluator| {
            Ok((|| -> Result<()> {
                operation(evaluator, &mut images)?;
                let cohorts = evaluator.memberships();
                let report = Report {
                    metadata: serde_json::json!({
                        "schema_version": 1, "metric_version": 1, "metric": "ambiguity",
                        "baseline": {"source": "gpu_resampled_iteration_0", "preprocessing_distortion_included": false},
                        "input": opts.input.display().to_string(),
                        "geometry": {"geometry_contract_version": 1, "coordinate_space": "clip", "distance_output_units": "export_pixels", "squared_distance_output_units": "export_pixels_squared", "width": SIZE, "height": SIZE, "pixels_per_clip_unit": pixels_per_clip(), "normalization": "initial graph bounds, centered uniform 95% fit, fixed across snapshots", "active_samples": baseline.layout().sample_count()},
                        "counts": {"eligible_routes": count, "skipped_reservations": baseline.layout().skipped_count(), "total_reservations": count.saturating_add(baseline.layout().skipped_count()), "original_constants": evaluator.original_constant_count(), "closed_stored_start_routes": cohorts.closed_route_count, "unique_canonical_routes": cohorts.unique_route_count, "cohorts": cohorts.cohort_count},
                        "definitions": {
                            "general": "For sustained current contacts, sqrt(mean(max(d-T,0)^2))/T, with original whole-route continuous endpoint-constrained weak Frechet distance d in the joint selected orientation. Introduced/resolved incompatible edge counts and squared-excess sums are co-primary alongside RMS.",
                            "endpoints": "Separate sqrt(mean(max(e-U,0)^2))/U over the same contacts, where e is the larger matched endpoint distance in that same orientation. Introduced/resolved counts and sums are co-primary.",
                            "cohorts": "Deduplicated unordered cohort pairs witnessed by an incompatible contacting edge pair. Introduced cohort associations are primary; denominator counts all distinct cross-cohort pairs contacting in this snapshot.",
                            "interpretation": "Lower excess means less original-route mismatch among current sustained contacts. Ratios are dimensionless and can exceed 1. Compatible contacts can dilute RMS. No combined score or overall quality claim.",
                            "empty_contacts": "count 0, sums 0, RMS 0, ratios 0; empty cohort denominator gives fraction 0"
                        },
                        "policies": {
                            "defaults": "Operational defaults, not calibrated perceptual constants. All pixel thresholds convert in f64 using SIZE/2; GPU f32 geometry is unchanged.",
                            "orientation": "Minimize max(d/T,e/U) jointly over forward/reversed matching, then smaller d, smaller e, then forward. Endpoint tolerance gives stronger endpoint influence. Inclusive thresholds, no epsilon halo.",
                            "canonicalization": "Normalize signed zero; remove consecutive duplicates and exact redundant forward-collinear vertices while retaining retracing. Lexicographically smaller forward/reverse sequence, geometry order with stable edge ID ties. Closed routes retain stored starts.",
                            "cohorts": "First-fit complete-link over all unique canonical originals, with full membership mapping and duplicate multiplicity retained. Partition depends on thresholds and canonical order; crossing a partition alone is not incompatibility.",
                            "approximation": "Endpoint-preserving strong index-correspondence certificate per shortcut; bound original weak distance by simplified distance plus/minus the sum of certificate errors. Refine undecidable orientations/classifications on full curves. Bounds describe frozen original geometry and can tighten without geometric change. General output has explicit lower/upper bounds; endpoint values are exact under ordinary f64 arithmetic, not exact-real arithmetic.",
                            "limitations": "Stored starts constrain closed-route matching. Weak Frechet permits retracing. Connected contact participation can include shallow crossings/backtracking and is not monotone one-to-one traversal. No tangent filter or endpoint exemption. Constants are eligible but have no positive-length contact.",
                            "contact": "One connected continuous radius-proximity component must have projected arc-interval union length at least the minimum on both curves. Point/tangent patches connect cells. Exact closed current paths use periodic adjacency.",
                            "capacities": "Explicit fail limits, never truncation. Baseline min(possible pairs,floor(global/5)); each of three cohort tables min(possible pairs,floor(global/10)); cache min(possible pairs,floor(global/100),65536). Remaining global records support index/candidate batches. Increase --ambiguity-stored-pair-limit for exhausted baseline/cohort tables or candidates. Only recomputable cache entries may evict.",
                            "work": "One work limit per preparation/baseline/current snapshot; no per-pair reset. Total work is charged before operations. Detailed measured index categories accompany each row. Separate weak-cell, heap and simplifier visit counters are unavailable; their work is included in total. Sort comparisons are not work units.",
                            "workspace": "Requested capacities include canonical/cohort data, evaluator metadata/tables, index/kernel temporaries, original/current route views and typed report rows with 32 filename bytes each. Shared GPU/CPU snapshots, allocator overhead, small report metadata and streaming serializer/I/O buffers are separate from requested evaluation workspace; this is not process RSS."
                        },
                        "resources": {"integration_route_view_bytes": route_bytes, "integration_report_buffer_bytes": row_bytes, "integration_reserved_bytes": caller_bytes, "image_capacity": rows, "original_route_view_capacity": count, "current_route_view_capacity": count, "filename_capacity_bytes_per_image": 32, "shared_snapshot_bytes_each": baseline.layout().sample_count().checked_mul(size_of::<[f32;2]>()), "maximum_shared_snapshots": 2, "geometry_staging_bytes": crate::resampling::GEOMETRY_STAGING_BYTES, "index_piece_length_clip_units": config.piece_length, "index_piece_length_pixels": params.route_tolerance_pixels},
                        "bundling": {"neighbor_radius": opts.neighbor_radius, "iterations": rows.saturating_sub(1)}
                    }),
                    parameters: params,
                    capacities,
                    memberships: cohorts.memberships,
                    preparation_resources: evaluator.preparation_resources(),
                    images: &images,
                };
                crate::report::publish(&opts.out, "ambiguity.json", &report)
            })())
        }).wrap_err("ambiguity preparation/evaluation failed; check --ambiguity-work-limit, --ambiguity-workspace-bytes and --ambiguity-stored-pair-limit; baseline/cohort capacities derive from the stored-pair limit")??;
        Ok(())
    })();
    budget.release(caller_bytes)?;
    outcome.wrap_err_with(|| format!(
        "ambiguity limits: --ambiguity-work-limit={}, --ambiguity-workspace-bytes={}, --ambiguity-stored-pair-limit={}; derived baseline evidence capacity={}, each of three cohort tables={}, recomputable cache={}. Raise the work or workspace setting for that failure; for stored-pair or baseline/cohort table exhaustion raise --ambiguity-stored-pair-limit, which scales the fixed tables and candidate headroom; resources after scoped cleanup={:?}",
        params.work_limit, params.workspace_bytes, params.stored_pair_limit,
        config.baseline_capacity, config.cohort_capacity, config.cache_capacity, budget.statistics()
    ))
}

#[derive(Serialize)]
#[serde(remote = "ambiguity_index::Statistics")]
struct IndexStatistics {
    constant_paths: usize,
    short_paths: usize,
    pieces_generated: u64,
    entries_visited: u64,
    candidate_examinations: u64,
    candidate_records: u64,
    narrow_pairs: u64,
    contacts: u64,
}

#[derive(Serialize)]
#[serde(remote = "Statistics")]
struct ResourceStatistics {
    work_consumed: u64,
    workspace_reserved: usize,
    workspace_peak: usize,
    records_reserved: usize,
    records_peak: usize,
}
