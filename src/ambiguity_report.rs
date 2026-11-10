use crate::{Options, gpu::SIZE};
use bundling_wgpu::{
    ambiguity::{self, Bounds, CohortView, Evaluator, Row, View},
    ambiguity_cohorts::{Membership, Route},
    ambiguity_gpu::{self, resident},
    ambiguity_index, ambiguity_similarity,
    geometry::Snapshot,
    weak_frechet::{Budget, Statistics, WorkBreakdown},
};
use color_eyre::eyre::{Result, WrapErr, ensure, eyre};
use serde::Serialize;
use std::mem::size_of;

#[derive(Debug, clap::Args, Serialize)]
pub struct Parameters {
    /// Whole-route tolerance in export pixels. Operational, not perceptually calibrated.
    #[arg(requires = "evaluate_ambiguity", long = "ambiguity-route-tolerance-pixels", default_value_t = 16.0, value_parser = positive_pixels)]
    route_tolerance_pixels: f64,
    /// Matched-endpoint tolerance in export pixels.
    #[arg(requires = "evaluate_ambiguity", long = "ambiguity-endpoint-tolerance-pixels", default_value_t = 4.0, value_parser = positive_pixels)]
    endpoint_tolerance_pixels: f64,
    /// Sustained-contact radius in export pixels. Zero is supported.
    #[arg(requires = "evaluate_ambiguity", long = "ambiguity-contact-radius-pixels", default_value_t = 2.0, value_parser = nonnegative_pixels)]
    contact_radius_pixels: f64,
    /// Required contact length along both curves in export pixels.
    #[arg(requires = "evaluate_ambiguity", long = "ambiguity-minimum-contact-pixels", default_value_t = 24.0, value_parser = positive_pixels)]
    minimum_contact_pixels: f64,
    /// Correspondence simplification error in export pixels. Zero uses full
    /// curves. Certified for cpu and gpu; an f32 estimate for gpu-f32.
    #[arg(requires = "evaluate_ambiguity", long = "ambiguity-simplification-pixels", default_value_t = 0.1, value_parser = nonnegative_pixels)]
    simplification_pixels: f64,
    /// Work units per preparation or snapshot, shared across all pair
    /// evaluations (cpu and gpu backends; rejected with gpu-f32, which bounds
    /// work per GPU dispatch instead).
    #[arg(requires = "evaluate_ambiguity", long = "ambiguity-work-limit", default_value_t = 2_000_000_000, value_parser = clap::value_parser!(u64).range(1..))]
    work_limit: u64,
    /// Requested evaluation workspace bytes, excluding shared snapshots and
    /// allocator overhead (cpu and gpu backends; rejected with gpu-f32).
    #[arg(requires = "evaluate_ambiguity", long = "ambiguity-workspace-bytes", default_value_t = 1_073_741_824, value_parser = positive_capacity)]
    workspace_bytes: usize,
    /// Global stored-pair capacity, including fixed tables and temporary
    /// candidate records (cpu and gpu backends; rejected with gpu-f32).
    #[arg(requires = "evaluate_ambiguity", long = "ambiguity-stored-pair-limit", default_value_t = 5_000_000, value_parser = positive_capacity)]
    stored_pair_limit: usize,
    /// Ambiguity backend. cpu and gpu produce the same CPU-f64 decisions and
    /// scores; gpu-f32 evaluates everything on the GPU in f32.
    #[arg(requires = "evaluate_ambiguity", long = "ambiguity-backend", value_enum, default_value_t = Backend::Cpu)]
    pub backend: Backend,
    /// Combined byte limit for all GPU ambiguity buffers (gpu: per frame;
    /// gpu-f32: everything it owns, including the frozen baseline), in MiB.
    #[arg(requires = "evaluate_ambiguity", long = "ambiguity-gpu-memory-mib", default_value_t = 1024, value_parser = clap::value_parser!(u64).range(1..=1 << 40))]
    gpu_memory_mib: u64,
    /// Cap on GPU candidate records per batch; defaults to the stored-pair
    /// limit (gpu backend; rejected with gpu-f32).
    #[arg(requires = "evaluate_ambiguity", long = "ambiguity-gpu-batch-records", value_parser = clap::value_parser!(u32).range(1..))]
    gpu_batch_records: Option<u32>,
    /// gpu-f32 only: largest free-space cell count of one whole-route distance
    /// search, enforced exactly.
    #[arg(requires = "evaluate_ambiguity", long = "ambiguity-gpu-search-cells", default_value_t = resident::SEARCH_CELLS_LIMIT, value_parser = clap::value_parser!(u64).range(1..=resident::SEARCH_CELLS_LIMIT))]
    #[serde(skip)]
    gpu_search_cells: u64,
    /// gpu-f32 only: contact pairs one frame may log.
    #[arg(requires = "evaluate_ambiguity", long = "ambiguity-gpu-contact-pairs", default_value_t = 5_000_000, value_parser = clap::value_parser!(u64).range(1..=1 << 28))]
    #[serde(skip)]
    gpu_contact_pairs: u64,
    /// gpu-f32 only: MiB of host metadata it may allocate from input or GPU
    /// counts (readbacks, membership tables, batch plans).
    #[arg(requires = "evaluate_ambiguity", long = "ambiguity-gpu-host-mib", default_value_t = 256, value_parser = clap::value_parser!(u64).range(1..=1 << 20))]
    #[serde(skip)]
    gpu_host_mib: u64,
}

/// Resource flags of the cpu and gpu backends that gpu-f32 does not implement.
const REFERENCE_ONLY: [(&str, &str); 4] = [
    ("work_limit", "--ambiguity-work-limit"),
    ("workspace_bytes", "--ambiguity-workspace-bytes"),
    ("stored_pair_limit", "--ambiguity-stored-pair-limit"),
    ("gpu_batch_records", "--ambiguity-gpu-batch-records"),
];
/// Resource flags only gpu-f32 implements.
const RESIDENT_ONLY: [(&str, &str); 3] = [
    ("gpu_search_cells", "--ambiguity-gpu-search-cells"),
    ("gpu_contact_pairs", "--ambiguity-gpu-contact-pairs"),
    ("gpu_host_mib", "--ambiguity-gpu-host-mib"),
];

/// Reject resource flags given explicitly for a backend that would ignore them.
pub fn check_backend_flags(matches: &clap::ArgMatches, backend: Backend) -> Result<()> {
    let (flags, name) = if backend == Backend::GpuF32 {
        (REFERENCE_ONLY.as_slice(), "gpu-f32")
    } else {
        (RESIDENT_ONLY.as_slice(), "the cpu and gpu backends")
    };
    for (id, flag) in flags {
        ensure!(
            matches.value_source(id) != Some(clap::parser::ValueSource::CommandLine),
            "{flag} has no effect with {name}; remove it"
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    /// CPU AABB hierarchy and CPU contact kernel (reference).
    Cpu,
    /// GPU candidates and certified GPU contact decisions on resident geometry,
    /// with the complete CPU pair route for every uncertain pair.
    Gpu,
    /// Every numerical stage on the GPU in f32, with no CPU fallback. Results
    /// are f32 estimates and can differ from the reference near thresholds.
    #[value(name = "gpu-f32")]
    #[serde(rename = "gpu-f32")]
    GpuF32,
}

/// Contact backend identity and GPU accounting for one frame.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct CandidateStage {
    backend: Backend,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    gpu: Option<ambiguity_gpu::Statistics>,
}
impl CandidateStage {
    const CPU: Self = Self {
        backend: Backend::Cpu,
        gpu: None,
    };
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
    /// GPU candidate limits. The radius comes from each enumeration.
    pub fn gpu_settings(&self) -> Result<ambiguity_gpu::Settings> {
        Ok(ambiguity_gpu::Settings {
            memory_bytes: self
                .gpu_memory_mib
                .checked_mul(1 << 20)
                .ok_or_else(|| eyre!("GPU candidate memory overflow"))?,
            // The CPU cannot retain more candidate records than the stored-pair limit.
            batch_records: Some(
                self.gpu_batch_records
                    .unwrap_or_else(|| u32::try_from(self.stored_pair_limit).unwrap_or(u32::MAX)),
            ),
        })
    }
    /// Thresholds and limits of the gpu-f32 backend. The combined GPU byte limit
    /// covers every buffer it owns; the contact-pair limit caps logged contacts.
    pub fn resident_settings(&self) -> Result<resident::Settings> {
        Ok(resident::Settings {
            route_tolerance: self.route_tolerance_pixels / pixels_per_clip(),
            endpoint_tolerance: self.endpoint_tolerance_pixels / pixels_per_clip(),
            sigma: self.simplification_pixels / pixels_per_clip(),
            radius: self.contact_radius_pixels / pixels_per_clip(),
            minimum_contact: self.minimum_contact_pixels / pixels_per_clip(),
            memory_bytes: self
                .gpu_memory_mib
                .checked_mul(1 << 20)
                .ok_or_else(|| eyre!("GPU memory overflow"))?,
            pair_capacity: self.gpu_contact_pairs,
            host_bytes: self
                .gpu_host_mib
                .checked_mul(1 << 20)
                .ok_or_else(|| eyre!("host metadata limit overflow"))?,
            search_cells: self.gpu_search_cells,
        })
    }
    /// What gpu-f32 was asked for and what it applies. The flags of the other
    /// backends are named as not applicable instead of being echoed.
    fn resident_parameters(&self) -> Result<serde_json::Value> {
        let settings = self.resident_settings()?;
        // The thresholds the kernels compare with: the nearest f32.
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "Reports the same f32 rounding of each threshold that the device receives."
        )]
        let applied = |value: f64| f64::from(value as f32);
        Ok(serde_json::json!({
            "requested": {
                "route_tolerance_pixels": self.route_tolerance_pixels,
                "endpoint_tolerance_pixels": self.endpoint_tolerance_pixels,
                "contact_radius_pixels": self.contact_radius_pixels,
                "minimum_contact_pixels": self.minimum_contact_pixels,
                "simplification_pixels": self.simplification_pixels,
                "backend": self.backend,
                "gpu_memory_mib": self.gpu_memory_mib,
                "gpu_contact_pairs": self.gpu_contact_pairs,
                "gpu_search_cells": self.gpu_search_cells,
                "gpu_host_mib": self.gpu_host_mib
            },
            "effective": {
                "thresholds_clip_units_f32": {
                    "route_tolerance": applied(settings.route_tolerance),
                    "endpoint_tolerance": applied(settings.endpoint_tolerance),
                    "contact_radius": applied(settings.radius),
                    "minimum_contact": applied(settings.minimum_contact),
                    "simplification": applied(settings.sigma)
                },
                "owned_gpu_buffer_bytes_limit": settings.memory_bytes,
                "contact_pairs_per_frame": settings.pair_capacity,
                "search_cells_per_distance_search": settings.search_cells,
                "host_metadata_bytes_limit": settings.host_bytes,
                "work_units_per_dispatch_and_submission": resident::WORK_TARGET,
                "work_units_per_invocation": resident::INVOCATION_TARGET
            },
            "not_applicable": REFERENCE_ONLY.map(|(_, flag)| flag)
        }))
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
    candidates: CandidateStage,
}
impl Image {
    pub fn new(
        file: String,
        iteration: u32,
        (row, baseline): (&Row, &Row),
        candidates: &CandidateStage,
    ) -> Result<Self> {
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
            candidates: *candidates,
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

/// Evaluate one frame with the selected contact backend.
/// # Errors
/// Evaluation failures. GPU resource, device and protocol failures abort and
/// never switch to the CPU index; per-pair numerical fallback is expected.
pub fn evaluate(
    evaluator: &mut Evaluator<'_, '_>,
    paths: &[ambiguity_index::Path<'_>],
    candidates: Option<(&ambiguity_gpu::Candidates, ambiguity_gpu::Geometry)>,
) -> Result<(Row, CandidateStage)> {
    let Some((candidates, geometry)) = candidates else {
        return Ok((
            evaluator.evaluate(paths).map_err(|error| eyre!(error))?,
            CandidateStage::CPU,
        ));
    };
    let mut frame = candidates.frame(geometry);
    let row = evaluator
        .evaluate_with(paths, ambiguity_index::BroadPhase::Gpu(&mut frame))
        .map_err(|error| eyre!(error))?;
    Ok((row, gpu_stage(&frame)))
}

fn gpu_stage(frame: &ambiguity_gpu::Frame) -> CandidateStage {
    let timings = frame.timings();
    tracing::info!(
        validation_wait_seconds = timings.validation.as_secs_f64(),
        setup_wait_seconds = timings.setup.as_secs_f64(),
        count_wait_seconds = timings.count.as_secs_f64(),
        fill_wait_seconds = timings.fill.as_secs_f64(),
        contact_seconds = timings.contact.as_secs_f64(),
        statistics = ?frame.statistics(),
        "GPU ambiguity contact"
    );
    CandidateStage {
        backend: Backend::Gpu,
        gpu: Some(frame.statistics()),
    }
}

/// Route views and typed report rows reserved by this integration.
fn integration_bytes(count: usize, rows: usize) -> Result<(usize, usize, usize)> {
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
    Ok((route_bytes, row_bytes, caller_bytes))
}

pub fn run(
    opts: &Options,
    baseline: &Snapshot<'_>,
    candidates: Option<(&ambiguity_gpu::Candidates, ambiguity_gpu::Geometry)>,
    operation: impl FnOnce(&mut Evaluator<'_, '_>, CandidateStage, &mut Vec<Image>) -> Result<()>,
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
    let (route_bytes, row_bytes, caller_bytes) = integration_bytes(count, rows)?;
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
        let mut frame = candidates.map(|(candidates, geometry)| candidates.frame(geometry));
        // The inner Result preserves GPU/I/O failures; the outer Result owns evaluator cleanup.
        ambiguity::with_evaluator_using(&originals, &config, frame.as_mut(), &mut budget, |evaluator, frame| {
            Ok((|| -> Result<()> {
                let stage = frame.map_or(CandidateStage::CPU, |frame| gpu_stage(frame));
                operation(evaluator, stage, &mut images)?;
                let cohorts = evaluator.memberships();
                let report = Report {
                    metadata: serde_json::json!({
                        "schema_version": 2, "metric_version": 1, "metric": "ambiguity",
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
                            "distance_argument_order": "Both cohort inference and pair scoring pass the later route in canonical membership order first, so their f64 predicates agree near thresholds.",
                            "original_preparations": "Certified original routes are prepared on demand and retained by the original scope until evaluation ends. Their compact knots, source ranges and metadata share the workspace budget. Stored baseline contributions are reused for the same edge pairs; same-cohort contacts have proven zero excess.",
                            "canonicalization": "Normalize signed zero; remove consecutive duplicates and exact redundant forward-collinear vertices while retaining retracing. Lexicographically smaller forward/reverse sequence, geometry order with stable edge ID ties. Closed routes retain stored starts.",
                            "cohorts": "First-fit complete-link over all unique canonical originals, with full membership mapping and duplicate multiplicity retained. Partition depends on thresholds and canonical order; crossing a partition alone is not incompatibility.",
                            "approximation": "Endpoint-preserving strong index-correspondence certificate per shortcut; bound original weak distance by simplified distance plus/minus the sum of certificate errors. Refine undecidable orientations/classifications on full curves. Bounds describe frozen original geometry and can tighten without geometric change. General output has explicit lower/upper bounds; endpoint values are exact under ordinary f64 arithmetic, not exact-real arithmetic.",
                            "limitations": "Stored starts constrain closed-route matching. Weak Frechet permits retracing. Connected contact participation can include shallow crossings/backtracking and is not monotone one-to-one traversal. No tangent filter or endpoint exemption. Constants are eligible but have no positive-length contact.",
                            "rendering": "Geometry only: line color, opacity, MSAA and endpoint dots do not enter these scores.",
                            "edge_ids": "Membership IDs are stable drawable GPU path IDs from the uploaded layout, not semantic cohort labels.",
                            "contact": "One connected continuous radius-proximity component must have projected arc-interval union length at least the minimum on both curves. Point/tangent patches connect cells. Exact closed current paths use periodic adjacency.",
                            "candidates": "ambiguity_backend selects the contact backend. Both produce the same contact decisions in the same source/partner order and feed the unchanged similarity, orientation, baseline evidence, cohorts and aggregation. cpu: f64 AABB hierarchy over bounded pieces, then the CPU contact kernel for every candidate pair the participation bound admits. gpu: before any work, the resident GPU points and reservations are read back and compared bit for bit with this frame's CPU paths; the enumeration's own radius, minimum, ordered IDs, positive-segment offsets and CPU-f64 segment lengths are used; mismatches fail. S1 enumerates a conservative superset of candidate cells with widened f32 boxes on a clamped uniform grid; candidate work items examine at most images[].candidates.partner_entries_per_item entries and are dispatched at most items_per_dispatch per submission. S2 sorts each whole-source batch by an exact key and certifies every cell and complete source/partner pair with outward f32 intervals that enclose the ordinary CPU-f64 values; certified contacts and certified negatives are decided on the GPU. Uncertain certificates, unsupported scalars or lengths, pairs above 4096 cells and component labels that do not settle within 128 rounds take the complete CPU pair route (participation bound, then contact kernel): this per-pair numerical fallback is expected. Resource, device and protocol failures abort with no report; there is no silent switch to the CPU index. index.narrow_pairs counts actual CPU contact-kernel pairs after the participation bound and index.contacts every emitted contact. images[].candidates keeps the S1 counters; images[].candidates.contact reports GPU decisions, CPU fallback pairs and cells by reason, schedule buckets, rounds, work estimates against the grouping target, submissions and maximum host submit-through-completion waits (not per-pair GPU durations). GPU work is not charged CPU work units. images[].candidates.memory reports the combined GPU byte limit, the simultaneous allocated peak of all owned buffers, device stored-record slots and owner-local host peaks; renderer-owned geometry is reported as borrowed and excluded, as are driver, pipeline and allocator overhead, so this is a buffer-byte limit, not a VRAM guarantee. Each frame's GPU buffers are destroyed when its enumeration ends.",
                            "short_path_count": "index.short_paths counts positive paths whose ordinary f64 total is below the minimum. This is diagnostic only; they remain indexed and use the conservative candidate participation test.",
                            "capacities": "Explicit fail limits, never truncation. Baseline min(possible pairs,floor(global/5)); each of three cohort tables min(possible pairs,floor(global/10)); cache min(possible pairs,floor(global/100),65536). Remaining global records support index/candidate batches. With the gpu backend, resident candidate cells, sorted cells, pair results, readback staging that holds those records, and host pair decisions and fallback runs each count as stored records while they exist; certificates, keys, permutations, offsets, schedules and scratch count only as GPU bytes. Increase --ambiguity-stored-pair-limit for exhausted baseline/cohort tables or candidates. Only recomputable cache entries may evict.",
                            "work": "One work limit per preparation/baseline/current snapshot; no per-pair reset. Total work is charged before operations. The disjoint work_breakdown fields sum to work_consumed in charged work units: weak_cell_visits counts settled nonstale free-space cell visits, heap_operations counts individual pushes/pops, simplifier_point_visits counts interior point examinations for shortcut correspondence certificates, and other contains all remaining charges, including label initialization and geometry validation. Counts record successful pre-operation charges; failed charges change no counters. Detailed index counters accompany each row and overlap total work; do not add them to this full breakdown. Sort comparisons are not work units.",
                            "workspace": "Requested capacities include canonical/cohort data, evaluator metadata/tables, index/kernel temporaries, original/current route views and typed report rows with 32 filename bytes each. Shared GPU/CPU snapshots, allocator overhead, small report metadata and streaming serializer/I/O buffers are separate from requested evaluation workspace; this is not process RSS."
                        },
                        "resources": {"integration_route_view_bytes": route_bytes, "integration_report_buffer_bytes": row_bytes, "integration_reserved_bytes": caller_bytes, "image_capacity": rows, "original_route_view_capacity": count, "current_route_view_capacity": count, "filename_capacity_bytes_per_image": 32, "shared_snapshot_bytes_each": baseline.layout().sample_count().checked_mul(size_of::<[f32;2]>()), "maximum_shared_snapshots": 2, "geometry_staging_bytes": crate::resampling::GEOMETRY_STAGING_BYTES, "index_piece_length_clip_units": config.piece_length, "index_piece_length_pixels": params.route_tolerance_pixels},
                        "bundling": {"neighbor_radius": opts.neighbor_radius, "iterations": rows.saturating_sub(1)}
                    }),
                    parameters: params,
                    capacities,
                    memberships: cohorts.memberships(),
                    preparation_resources: evaluator.preparation_resources(),
                    images: &images,
                };
                crate::report::publish(&opts.out, "ambiguity.json", &report)
            })())
        }).wrap_err("ambiguity preparation/evaluation failed; check --ambiguity-work-limit, --ambiguity-workspace-bytes and --ambiguity-stored-pair-limit (baseline/cohort capacities derive from the stored-pair limit); GPU contact errors name their own limit or device/protocol cause")??;
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
    #[serde(with = "ClassifiedWork")]
    work_breakdown: WorkBreakdown,
    workspace_reserved: usize,
    workspace_peak: usize,
    records_reserved: usize,
    records_peak: usize,
}

#[derive(Serialize)]
#[serde(remote = "WorkBreakdown")]
struct ClassifiedWork {
    other: u64,
    weak_cell_visits: u64,
    heap_operations: u64,
    simplifier_point_visits: u64,
}

/// One gpu-f32 frame in export units.
#[derive(Serialize)]
pub struct ResidentImage {
    file: String,
    iteration: u32,
    contact_pairs: u64,
    baseline_contact_pairs: u64,
    current_collapsed_paths: u64,
    general: General,
    endpoints: Endpoints,
    cohorts: resident::CohortScores,
    device: resident::FrameStatistics,
}

/// Widen a device view to the report's types; values stay f32 estimates.
fn resident_view(view: &resident::View, baseline: &resident::View) -> Result<View> {
    let bounds = |interval: resident::Interval| Bounds {
        lower: f64::from(interval.lower),
        upper: f64::from(interval.upper),
    };
    Ok(View {
        squared_excess: bounds(view.squared_excess),
        rms_excess: bounds(view.rms_excess),
        ratio: bounds(view.ratio),
        current: usize::try_from(view.current)?,
        baseline: usize::try_from(baseline.current)?,
        introduced: usize::try_from(view.introduced)?,
        resolved: usize::try_from(view.resolved)?,
        introduced_squared_excess: bounds(view.introduced_squared_excess),
        resolved_squared_excess: bounds(view.resolved_squared_excess),
    })
}

impl ResidentImage {
    pub fn new(
        file: String,
        iteration: u32,
        frame: &resident::Frame,
        baseline: &resident::Frame,
    ) -> Result<Self> {
        let view = |current: &resident::View, original: &resident::View| -> Result<General> {
            General::new(
                resident_view(current, original)?,
                resident_view(original, original)?,
            )
        };
        Ok(Self {
            file,
            iteration,
            contact_pairs: frame.contacts,
            baseline_contact_pairs: baseline.contacts,
            current_collapsed_paths: frame.statistics.collapsed_paths,
            general: view(&frame.general, &baseline.general)?,
            endpoints: view(&frame.endpoints, &baseline.endpoints)?.into(),
            cohorts: frame.cohorts,
            device: frame.statistics,
        })
    }
}

#[derive(Serialize)]
struct ResidentMembership {
    edge_id: u32,
    cohort: u32,
}

/// Publish the gpu-f32 report atomically.
pub fn publish_resident(
    opts: &Options,
    session: &resident::Session,
    images: &[ResidentImage],
    samples: u32,
) -> Result<()> {
    let preparation = session.preparation();
    let memberships: Vec<ResidentMembership> = session
        .memberships()
        .iter()
        .map(|[edge_id, cohort]| ResidentMembership {
            edge_id: *edge_id,
            cohort: *cohort,
        })
        .collect();
    let report = serde_json::json!({
        "schema_version": 2, "metric_version": 1, "metric": "ambiguity",
        "numerical_contract": {
            "backend": "gpu-f32", "precision": "f32", "contract_version": resident::CONTRACT_VERSION,
            "cpu_fallback": false, "certified_bounds": false, "cpu_f64_parity": false,
            "summary": "Every numerical stage runs in WGSL f32 on the GPU. Values are ordinary f32 point estimates; the device may flush subnormal intermediates to zero. Results are not bit-compatible with the cpu and gpu backends and classifications at a threshold can differ from them. Range, resource, convergence and protocol failures abort without a report."
        },
        "baseline": {"source": "gpu_resampled_iteration_0", "storage": "device copy frozen before bundling", "preprocessing_distortion_included": false},
        "input": opts.input.display().to_string(),
        "geometry": {"geometry_contract_version": 1, "coordinate_space": "clip", "distance_output_units": "export_pixels", "squared_distance_output_units": "export_pixels_squared", "width": SIZE, "height": SIZE, "pixels_per_clip_unit": pixels_per_clip(), "normalization": "initial graph bounds, centered uniform 95% fit, fixed across snapshots", "active_samples": samples},
        "counts": {"eligible_routes": preparation.eligible_routes, "skipped_reservations": preparation.skipped_reservations, "total_reservations": preparation.eligible_routes.saturating_add(preparation.skipped_reservations), "original_constants": preparation.original_constants, "closed_stored_start_routes": preparation.closed_stored_start_routes, "unique_canonical_routes": preparation.unique_canonical_routes, "cohorts": preparation.cohorts},
        "definitions": {
            "general": "For sustained current contacts, sqrt(mean(max(d-T,0)^2))/T, with original whole-route continuous endpoint-constrained weak Frechet distance d in the joint selected orientation. Introduced/resolved incompatible edge counts and squared-excess sums are co-primary alongside RMS.",
            "endpoints": "Separate sqrt(mean(max(e-U,0)^2))/U over the same contacts, where e is the larger matched endpoint distance in that same orientation. Introduced/resolved counts and sums are co-primary.",
            "cohorts": "Deduplicated unordered cohort pairs witnessed by an incompatible contacting edge pair. Introduced cohort associations are primary; denominator counts all distinct cross-cohort pairs contacting in this snapshot.",
            "interpretation": "Lower excess means less original-route mismatch among current sustained contacts. Ratios are dimensionless and can exceed 1. Compatible contacts can dilute RMS. No combined score or overall quality claim.",
            "empty_contacts": "count 0, sums 0, RMS 0, ratios 0; empty cohort denominator gives fraction 0"
        },
        "policies": {
            "residency": "The host parses the input, uploads thresholds as f32, sizes and schedules bounded dispatches, enforces limits, and reads counts used for sizing and batching, the membership table and the finished counts and scalars. Baseline points are copied on the device before bundling. Layout validation, positive segments, canonical originals and their order, simplification, cohort inference, contact, whole-route distances, orientation, baseline evidence, cohort associations, counts and reductions all run on the device. Pixel conversion of the finished f32 scalars for this report is done in f64 on the host.",
            "ranges": "Coordinates must be finite and below 2^20 clip units in magnitude; thresholds must lie within [2^-20, 2^20] clip units (radius and simplification may be zero); a route may store at most 32768 points. Violations abort before use.",
            "bounds": "general lower/upper values are f32 estimates d-e and d+e from the index-correspondence simplification error e, clamped below by the endpoint distance; they are not certified enclosures. Undecidable orientations and tolerance-straddling pairs are refined on the full canonical curves, where lower equals upper. Endpoint values are single f32 estimates.",
            "orientation": "Minimize max(d/T,e/U) jointly over forward/reversed matching, then smaller d, smaller e, then forward. Inclusive thresholds, no epsilon halo. Cohort inference and scoring assess the representatives of both routes with the later canonical route first, with the same kernels.",
            "canonicalization": "Normalize signed zero; remove consecutive duplicates and exact redundant forward-collinear vertices (exact integer determinant) while retaining retracing. Lexicographically smaller forward/reverse sequence, geometry order with stable edge ID ties, all by exact integer comparison of bit patterns. Closed routes retain stored starts.",
            "cohorts": "First-fit complete-link over all unique canonical originals in canonical order, with full membership mapping and duplicate multiplicity retained. Every earlier unique route with compatible endpoints is assessed; a route joins the earliest cohort all of whose members are compatible. Crossing a partition alone is not incompatibility.",
            "distance": "Continuous endpoint-constrained weak Frechet distance: point, single-segment, synchronized-vertex and staircase certificates, otherwise the exact minimax search over free-space cells. The search settles cells in increasing bottleneck order from the first cell and stops at the last, in bounded resumable steps with a fixed-capacity frontier; a search that exceeds the capacity runs tiled atomic label relaxation to a complete unchanged round instead. Both give the same f32 minimax value. The endpoint-nearer orientation is assessed first and the other only when its endpoint key does not already exceed the first's upper key. No sampling, rasterization or discrete Frechet substitute.",
            "contact": "One connected continuous radius-proximity component must have projected arc-interval union length at least the minimum on both curves. Point/tangent boundary joins connect cells; exact closed current paths use periodic adjacency. Endpoint distance tests seed both intervals and determine joins. Only a cell without any endpoint hit may use a crossing seed, and its unexpanded segment boxes must overlap. This prevents orientation noise near collinearity from stretching an endpoint-seeded interval. Crossing signs and interval endpoints are f32 estimates. Endpoint-seeded patches extend with strip/disk clips. A crossing-only cell uses each computed extent directly, retaining a zero-length seed only where no extent is available; it never takes the hull of two independently rounded crossing estimates. Extent endpoints can be sensitive near tangency or parallelism and have no uniform arc-length error bound. Missing joined neighbors cause an explicit failure. Distances are compared unsquared after scale-based unit normalization. Every divisor is normal with a bounded quotient: segment normalization uses a 2^-100 scale threshold, while clipping and crossing permit divisors down to 2^-126. Segments below the scale threshold are points and never infinite strips. Subnormal operands/results may flush to zero. Exact shared stored vertices have distance zero by direct comparison; other radius-zero contacts follow the evaluated f32 distance and crossing tests and may depend on rounding. No neighbor cap or downsampling. Components are the fixpoint of atomic label propagation. At radius zero, segments collinear within f32 resolution may contribute arc length along the near-overlap.",
            "contact_lengths": "A component's length on a route is the sum of its cells' uncovered interval lengths. Each contribution is an f32 value in units of the minimum, capped at one and truncated to a multiple of 2^-32, and the truncated values are added as 64-bit integers, so the sum does not depend on evaluation order. It is exact for the truncated contributions, not an exact arc length: K contributions lose less than K * 2^-32 of the minimum, below 1/64 of it at the largest batch. A decision exactly at the minimum can differ from an unquantized evaluation.",
            "reductions": "Counts are exact integers. Sums are f32, added in pair order within blocks of 256 and then across blocks level by level; RMS and ratios are f32.",
            "baseline_evidence": "The baseline frame's incompatible contact records and witnessed cohort pairs stay on the device. A later contact of the same edge pair reuses its stored flags and excesses; same-cohort contacts score zero; every other cross-cohort contact is assessed again with the same kernels.",
            "limits": "Explicit fail limits, never truncation. --ambiguity-gpu-memory-mib bounds all owned GPU buffers including the frozen baseline copy (renderer geometry is borrowed; driver, pipeline and allocator overhead are outside it). --ambiguity-gpu-contact-pairs bounds the contact pairs one frame logs. --ambiguity-gpu-search-cells bounds one distance search, compared exactly with its cell count. --ambiguity-gpu-host-mib bounds the host metadata this backend allocates from input or device counts (readback vectors, membership and ID tables, batch and chunk plans, pair listings); report rows, the JSON serializer and the renderer are outside it. A single route whose cells exceed a batch fails; batches are whole lower-route ranges, so no component is split. The flags listed under parameters.not_applicable belong to the cpu and gpu backends and are rejected when given with gpu-f32.",
            "work": "Every dispatch declares a worst-case unit count; a dispatch above parameters.effective.work_units_per_dispatch_and_submission is refused rather than run, and pending work is submitted before a submission would exceed it. A work unit is one innermost-loop iteration or straight-line block touching at most 32 buffer elements. Long work is resumable on the device (simplification, first-fit, frontier searches, chain walks). Uniform-bound dispatches are sliced into whole workgroups. A dispatch with a caller-proven Total bound is encoded whole or refused before execution. Work is never truncated. Round limits for simplification, relaxation, label propagation and chain walks are resource limits, not convergence proofs: an input that exceeds one fails with no report. These are operation-count bounds, not wall-time guarantees; device.work.maximum_wait_seconds is the longest observed submission on this machine. There is no global work budget in this backend.",
            "limitations": "Stored starts constrain closed-route matching. Weak Frechet permits retracing. Connected contact participation can include shallow crossings/backtracking and is not monotone one-to-one traversal. No tangent filter or endpoint exemption. Constants are eligible but have no positive-length contact.",
            "rendering": "Geometry only: line color, opacity, MSAA and endpoint dots do not enter these scores.",
            "edge_ids": "Membership IDs are stable drawable GPU path IDs from the uploaded layout, not semantic cohort labels."
        },
        "bundling": {"neighbor_radius": opts.neighbor_radius, "iterations": images.len().saturating_sub(1)},
        "parameters": opts.ambiguity.resident_parameters()?,
        "memberships": memberships,
        "preparation": preparation,
        "images": images,
    });
    crate::report::publish(&opts.out, "ambiguity.json", &report)
}
