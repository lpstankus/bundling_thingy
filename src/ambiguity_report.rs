use crate::{Options, gpu::Gpu, gpu::SIZE};
use bundling_wgpu::ambiguity::{self, Frame, Geometry, MAXIMUM_SCALE, MINIMUM_SCALE, View};
use color_eyre::eyre::{Result, WrapErr, ensure, eyre};
use serde::Serialize;
use std::mem::size_of;

const MIB: u64 = 1 << 20;

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
    /// f32 index-correspondence simplification tolerance in export pixels; zero uses full curves.
    #[arg(requires = "evaluate_ambiguity", long = "ambiguity-simplification-pixels", default_value_t = 0.1, value_parser = nonnegative_pixels)]
    simplification_pixels: f64,
    /// Combined MiB of every GPU buffer the evaluation owns, including the frozen baseline.
    #[arg(requires = "evaluate_ambiguity", long = "ambiguity-gpu-memory-mib", default_value_t = 1024, value_parser = clap::value_parser!(u64).range(1..=1 << 40))]
    gpu_memory_mib: u64,
    /// Contact pairs one frame may log.
    #[arg(requires = "evaluate_ambiguity", long = "ambiguity-gpu-contact-pairs", default_value_t = 5_000_000, value_parser = clap::value_parser!(u64).range(1..=1 << 28))]
    gpu_contact_pairs: u64,
    /// Largest free-space cell count of one whole-route distance search.
    #[arg(requires = "evaluate_ambiguity", long = "ambiguity-gpu-search-cells", default_value_t = 1 << 22, value_parser = clap::value_parser!(u64).range(1..=1 << 22))]
    gpu_search_cells: u64,
    /// MiB of host metadata, including the prepaid report rows.
    #[arg(requires = "evaluate_ambiguity", long = "ambiguity-gpu-host-mib", default_value_t = 256, value_parser = clap::value_parser!(u64).range(1..=1 << 20))]
    gpu_host_mib: u64,
    /// Declared kernel-work units of the preparation, the baseline frame or one later frame.
    #[arg(requires = "evaluate_ambiguity", long = "ambiguity-gpu-work-limit", default_value_t = 1 << 40, value_parser = clap::value_parser!(u64).range(1..))]
    gpu_work_limit: u64,
}

/// Parse export pixels whose clip value is zero (when allowed) or within the
/// supported f32 threshold range.
fn pixels(text: &str, allow_zero: bool) -> std::result::Result<f64, String> {
    let value = text.parse::<f64>().map_err(|e| e.to_string())?;
    let clip = value / pixels_per_clip();
    let is_zero = value == 0.0 && allow_zero;
    if is_zero || (MINIMUM_SCALE..=MAXIMUM_SCALE).contains(&clip) {
        return Ok(value);
    }
    Err(format!(
        "pixels must be finite with a clip value within [2^-20, 2^20] ({} to {} pixels){}",
        MINIMUM_SCALE * pixels_per_clip(),
        MAXIMUM_SCALE * pixels_per_clip(),
        if allow_zero { ", or zero" } else { "" }
    ))
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
    fn settings(&self, host_bytes: u64) -> Result<ambiguity::Settings> {
        let scale = pixels_per_clip();
        Ok(ambiguity::Settings {
            route_tolerance: self.route_tolerance_pixels / scale,
            endpoint_tolerance: self.endpoint_tolerance_pixels / scale,
            sigma: self.simplification_pixels / scale,
            radius: self.contact_radius_pixels / scale,
            minimum_contact: self.minimum_contact_pixels / scale,
            memory_bytes: mebibytes(self.gpu_memory_mib)?,
            pair_capacity: self.gpu_contact_pairs,
            host_bytes,
            search_cells: self.gpu_search_cells,
            work_limit: self.gpu_work_limit,
        })
    }
    fn limits(&self) -> String {
        format!(
            "--ambiguity-gpu-memory-mib={}, --ambiguity-gpu-contact-pairs={}, --ambiguity-gpu-search-cells={}, --ambiguity-gpu-host-mib={}, --ambiguity-gpu-work-limit={}",
            self.gpu_memory_mib,
            self.gpu_contact_pairs,
            self.gpu_search_cells,
            self.gpu_host_mib,
            self.gpu_work_limit
        )
    }
}

fn mebibytes(value: u64) -> Result<u64> {
    value
        .checked_mul(MIB)
        .ok_or_else(|| eyre!("MiB limit overflow"))
}

/// One uploaded f32 threshold in clip units and export pixels.
#[derive(Serialize)]
struct Threshold {
    clip: f32,
    pixels: f64,
}
impl Threshold {
    fn new(clip: f64) -> Self {
        let uploaded = if clip == 0.0 {
            0.0
        } else {
            ambiguity::nearest_f32(clip)
        };
        Self {
            clip: uploaded,
            pixels: f64::from(uploaded) * pixels_per_clip(),
        }
    }
}

/// Values actually used: f32 thresholds as uploaded and every resource limit.
#[derive(Serialize)]
struct Effective {
    route_tolerance: Threshold,
    endpoint_tolerance: Threshold,
    contact_radius: Threshold,
    minimum_contact: Threshold,
    simplification: Threshold,
    memory_bytes: u64,
    contact_pairs: u64,
    search_cells: u64,
    host_bytes: u64,
    engine_host_bytes: u64,
    work_limit_units_per_phase: u64,
    invocation_work_units: u64,
    dispatch_and_submission_work_units: u64,
}
impl Effective {
    fn new(settings: &ambiguity::Settings, host_bytes: u64) -> Self {
        Self {
            route_tolerance: Threshold::new(settings.route_tolerance),
            endpoint_tolerance: Threshold::new(settings.endpoint_tolerance),
            contact_radius: Threshold::new(settings.radius),
            minimum_contact: Threshold::new(settings.minimum_contact),
            simplification: Threshold::new(settings.sigma),
            memory_bytes: settings.memory_bytes,
            contact_pairs: settings.pair_capacity,
            search_cells: settings.search_cells,
            host_bytes,
            engine_host_bytes: settings.host_bytes,
            work_limit_units_per_phase: settings.work_limit,
            invocation_work_units: ambiguity::INVOCATION_TARGET,
            dispatch_and_submission_work_units: ambiguity::WORK_TARGET,
        }
    }
}

/// Low and high estimates converted to export units.
#[derive(Serialize)]
struct Estimate {
    low: f64,
    high: f64,
}
fn scaled(estimate: ambiguity::Estimate, factor: f64) -> Result<Estimate> {
    let result = Estimate {
        low: f64::from(estimate.low) * factor,
        high: f64::from(estimate.high) * factor,
    };
    ensure!(
        result.low.is_finite() && result.high.is_finite(),
        "nonfinite ambiguity report conversion"
    );
    Ok(result)
}

#[derive(Serialize)]
struct General {
    current_incompatible_pairs: u64,
    baseline_incompatible_pairs: u64,
    introduced_pairs: u64,
    resolved_pairs: u64,
    squared_excess_pixels_squared: Estimate,
    baseline_squared_excess_pixels_squared: Estimate,
    introduced_squared_excess_pixels_squared: Estimate,
    resolved_squared_excess_pixels_squared: Estimate,
    rms_excess_pixels: Estimate,
    ratio: Estimate,
}
impl General {
    fn new(view: &View, baseline: &View) -> Result<Self> {
        let scale = pixels_per_clip();
        let square = scale * scale;
        Ok(Self {
            current_incompatible_pairs: view.current,
            baseline_incompatible_pairs: baseline.current,
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
            ratio: scaled(view.ratio, 1.0)?,
        })
    }
}

#[derive(Serialize)]
pub struct Image {
    file: String,
    iteration: u32,
    contact_pairs: u64,
    baseline_contact_pairs: u64,
    current_collapsed_paths: u64,
    general: General,
    device: ambiguity::FrameStatistics,
}
impl Image {
    fn new(file: String, iteration: u32, frame: &Frame, baseline: &Frame) -> Result<Self> {
        Ok(Self {
            file,
            iteration,
            contact_pairs: frame.contacts,
            baseline_contact_pairs: baseline.contacts,
            current_collapsed_paths: frame.statistics.collapsed_paths,
            general: General::new(&frame.general, &baseline.general)?,
            device: frame.statistics,
        })
    }
}

#[derive(Serialize)]
struct Requested<'a> {
    requested: &'a Parameters,
    effective: Effective,
}

#[derive(Serialize)]
struct Report<'a> {
    #[serde(flatten)]
    metadata: serde_json::Value,
    parameters: Requested<'a>,
    preparation: ambiguity::Preparation,
    images: &'a [Image],
}

/// One ambiguity report: a device session and exactly one row per exported image.
///
/// The report rows are prepaid from the one host limit before the device
/// session is built; the session receives the remainder. Any failed record
/// invalidates the session, and only a complete session publishes.
pub struct Session<'a> {
    opts: &'a Options,
    effective: Effective,
    report_bytes: u64,
    rows: usize,
    images: Vec<Image>,
    device: ambiguity::Session,
    valid: bool,
}
impl<'a> Session<'a> {
    /// Freeze the baseline and prepare the originals; call before bundling.
    /// # Errors
    /// Host limit, preparation and baseline failures.
    pub fn new(opts: &'a Options, gpu: &Gpu) -> Result<Self> {
        let params = &opts.ambiguity;
        let rows = if opts.neighbor_radius.is_some() {
            usize::try_from(opts.iterations)?
                .checked_add(1)
                .ok_or_else(|| eyre!("image count overflow"))?
        } else {
            1
        };
        // Each row prepays the 32-byte requested filename capacity of `image_name`.
        let report_bytes = u64::try_from(size_of::<Image>().saturating_add(32))?
            .checked_mul(u64::try_from(rows)?)
            .ok_or_else(|| eyre!("image buffer overflow"))?;
        let host_bytes = mebibytes(params.gpu_host_mib)?;
        let engine_host_bytes = host_bytes.checked_sub(report_bytes).ok_or_else(|| {
            eyre!("ambiguity report rows need {report_bytes} bytes, above --ambiguity-gpu-host-mib")
        })?;
        let settings = params.settings(engine_host_bytes)?;
        let mut images = Vec::new();
        images.try_reserve_exact(rows)?;
        let session = gpu.ambiguity_session(&settings).wrap_err_with(|| {
            format!(
                "ambiguity preparation/baseline failed; limits {}",
                params.limits()
            )
        })?;
        Ok(Self {
            opts,
            effective: Effective::new(&settings, host_bytes),
            report_bytes,
            rows,
            images,
            device: session,
            valid: true,
        })
    }

    /// Record image `iteration`: the baseline row first, then one row per
    /// later frame with the renderer's current geometry.
    /// # Errors
    /// Out-of-order rows, evaluation and conversion failures. Every failure
    /// invalidates this session.
    pub fn record(
        &mut self,
        file: String,
        iteration: u32,
        current: Option<&Geometry>,
    ) -> Result<()> {
        ensure!(
            self.valid,
            "ambiguity session is invalid after a failed record"
        );
        self.valid = false;
        let index = self.images.len();
        let is_next = index < self.rows
            && usize::try_from(iteration).is_ok_and(|iteration| iteration == index)
            && current.is_some() == (index > 0);
        ensure!(
            is_next,
            "ambiguity row {iteration} is missing, repeated, extra or out of order"
        );
        let baseline = self.device.baseline();
        let frame = match current {
            Some(geometry) => self.device.evaluate(geometry).wrap_err_with(|| {
                format!(
                    "evaluate ambiguity frame {iteration}; limits {}",
                    self.opts.ambiguity.limits()
                )
            })?,
            None => baseline,
        };
        self.images
            .push(Image::new(file, iteration, &frame, &baseline)?);
        self.valid = true;
        Ok(())
    }

    /// Publish a complete, valid session.
    /// # Errors
    /// Invalid or incomplete sessions, serialization and I/O failures.
    pub fn finish(self) -> Result<()> {
        ensure!(
            self.valid && self.images.len() == self.rows,
            "ambiguity session is invalid or incomplete"
        );
        let opts = self.opts;
        let preparation = self.device.preparation();
        let engine_peak = self
            .images
            .iter()
            .map(|image| image.device.host_peak_bytes)
            .fold(preparation.host_peak_bytes, u64::max);
        let total_reservations = preparation
            .eligible_routes
            .saturating_add(preparation.skipped_reservations);
        let report = Report {
            metadata: serde_json::json!({
                "schema_version": 3, "metric_version": 1, "metric": "ambiguity",
                "numerical_contract": {"precision": "f32", "contract_version": ambiguity::CONTRACT_VERSION, "certified_bounds": false, "cpu_fallback": false, "cpu_f64_parity": false},
                "baseline": {"source": "gpu_resampled_iteration_0", "storage": "device copy frozen before bundling", "preprocessing_distortion_included": false},
                "input": opts.input.display().to_string(),
                "geometry": {"geometry_contract_version": 1, "coordinate_space": "clip", "distance_output_units": "export_pixels", "squared_distance_output_units": "export_pixels_squared", "width": SIZE, "height": SIZE, "pixels_per_clip_unit": pixels_per_clip(), "normalization": "initial graph bounds, centered uniform 95% fit, fixed across snapshots", "active_samples": preparation.active_samples},
                "counts": {"eligible_routes": preparation.eligible_routes, "skipped_reservations": preparation.skipped_reservations, "total_reservations": total_reservations, "original_constants": preparation.original_constants, "closed_stored_start_routes": preparation.closed_stored_start_routes, "unique_canonical_routes": preparation.unique_canonical_routes},
                "definitions": {
                    "general": "For sustained current contacts, sqrt(mean(max(d-T,0)^2))/T, with original whole-route continuous endpoint-constrained weak Frechet distance d in the joint selected orientation, evaluated in f32. Introduced/resolved incompatible edge counts and squared-excess sums are co-primary alongside RMS.",
                    "interpretation": "Lower excess means less original-route mismatch among current sustained contacts. Ratios are dimensionless and can exceed 1. Compatible contacts can dilute RMS. No combined score or overall quality claim.",
                    "empty_contacts": "A frame without current contacts reports contact_pairs, current_incompatible_pairs and introduced_pairs 0 and exact +0 current and introduced squared sums, RMS and ratio on both estimate sides. Baseline fields keep the frozen baseline frame's device values. In a later frame resolved_pairs counts every baseline incompatible pair, and its sums are the f32 reduction of those evidence weights in their row order, which can differ in the last bits from the baseline aggregate. A baseline without contacts has no evidence, so every later resolved field is zero."
                },
                "policies": {
                    "defaults": "Operational defaults, not calibrated perceptual constants. Pixel thresholds convert to clip units in f64 using SIZE/2 and are rounded to the nearest f32 for upload; parameters.effective lists the uploaded values.",
                    "orientation": "Minimize max(d/T,e/U) jointly over forward/reversed matching, then smaller d, smaller e, then forward, comparing f32 estimates. Endpoint distance e enters only this orientation key and tie-break; there is no separate endpoint score.",
                    "distance_argument_order": "Pair scoring passes the later route in canonical original order first, so both evaluations of a pair use one argument order.",
                    "canonicalization": "Normalize signed zero; remove consecutive duplicates and exact redundant forward-collinear vertices (exact integer determinant) while retaining retracing. Lexicographically smaller forward/reverse bit-key sequence, geometry order with stable edge ID ties. Closed routes retain stored starts. Routes with identical canonical geometry share one representative and score zero against each other.",
                    "simplification": "Device index-correspondence shortcuts in f32: a skipped vertex must lie within sigma of the chord point at its index fraction. With route errors e_a, e_b, low=max(endpoint distance, d_s-e_a-e_b, 0) and high=max(endpoint distance, d_s+e_a+e_b). These are f32 estimates, not certified bounds. An orientation order the estimates cannot decide is recomputed on both full canonical curves; a chosen orientation with low<=T<high is recomputed on its full canonical curves, giving low=high. Sigma zero uses full curves.",
                    "distance": "Continuous endpoint-constrained weak Frechet distance with backward moves over free-space cell boundaries with squared f32 boundary weights: exact fast branches for points, single segments, chains, synchronized vertices and staircases, otherwise a bounded resumable frontier search in increasing bottleneck order or tiled atomic label relaxation to a complete unchanged round. Squared residuals below 2^-126 can flush to zero, hiding distances below about 2^-63 clip units; boundary segments with computed squared length <=1e-30 are treated as points.",
                    "contact": "One connected continuous radius-proximity component must have projected arc-interval union length at least the minimum on both curves. Point/tangent patches connect cells; exact closed current paths use periodic adjacency. Unsquared distances after scale normalization; segments whose larger coordinate difference is below 2^-100 are points for clipping only. The broad phase widens grid coverage by half the radius plus 2^-14 of coordinate magnitude to select candidate cells; no predicate uses a radius halo. Exact shared stored vertices are at distance zero.",
                    "contact_lengths": "Each cell's uncovered interval length on each route is an f32 value in units of the minimum, capped at one, truncated to 2^-32 and summed as 64-bit integers, so sums are order independent. K contributions lose less than K*2^-32 of the minimum and only lower lengths; a decision exactly at the minimum can differ from an unquantized evaluation.",
                    "reductions": "Contact pairs are sorted by (lower, higher) ordinal. Reduction rows are the N contact rows in that order, then the E baseline evidence rows; columns are current, introduced and resolved low/high squared excess. Sums are f32, sequential within blocks of 256 rows, then level by level. RMS=sqrt(sum/f32(N)) and ratio=RMS/T in f32. Counts are exact integers; f32(N) is exact up to 2^24 and above it WGSL allows either adjacent f32, a relative error below 2^-23. The host only scales finished scalars to pixels in f64.",
                    "baseline_evidence": "The baseline frame's general-incompatible contact records stay on the device; a later pair with evidence reuses its flag and weights. Introduced pairs are current incompatible pairs without evidence; resolved pairs are evidence pairs without current contact.",
                    "residency": "Every preparation, contact, distance, classification, evidence and reduction stage runs on the device. The host reads status words, counts, per-route work and cell counts for planning, search chunk heads and the ten finished scalars; it never reads ambiguity geometry. Other metrics read their own snapshots.",
                    "ranges": "Coordinates must be finite with magnitude below 2^20 clip units; thresholds zero (radius and simplification only) or within [2^-20, 2^20] clip units; at most 32768 stored points per route; device counts within u32; one distance search within --ambiguity-gpu-search-cells cells; contact sort keys within 63 bits; each source route's cells within one contact batch. Unsupported inputs fail by name and are never truncated.",
                    "limits": "Explicit fail limits, never truncation: combined owned GPU bytes (--ambiguity-gpu-memory-mib, including the frozen baseline, workspaces, evidence, staging and uniforms, checked per buffer against the adapter's buffer and binding limits for its usage), contact pairs per frame, search cells and one host metadata limit shared with the prepaid report rows. Renderer buffers and driver, pipeline and allocator overhead are excluded.",
                    "work": "Each dispatch charges declared kernel-work units, including padding lanes and every slice, to the phase budget --ambiguity-gpu-work-limit before it is encoded. Phases are the original preparation, the baseline frame and each later frame. Independently, one invocation may declare at most 2^18 units and one dispatch or submission at most 2^31, and bounded iterations have round limits. These are resource limits, not convergence proofs or wall-time guarantees.",
                    "limitations": "Stored starts constrain closed-route matching. Weak Frechet permits retracing. Connected contact participation can include shallow crossings/backtracking and is not monotone one-to-one traversal. No tangent filter or endpoint exemption. Constants are eligible but have no positive-length contact. f32 results can classify pairs near a threshold differently from an f64 evaluation; exact zero distances are adapter behavior, not a portable WGSL guarantee.",
                    "rendering": "Geometry only: line color, opacity, MSAA and endpoint dots do not enter these scores.",
                    "edge_ids": "Edge IDs are stable drawable GPU path IDs from the uploaded layout, not semantic route labels."
                },
                "resources": {"report_row_bytes": self.report_bytes, "image_capacity": self.rows, "filename_capacity_bytes_per_image": 32, "combined_host_peak_bytes": engine_peak.saturating_add(self.report_bytes)},
                "bundling": {"neighbor_radius": opts.neighbor_radius, "iterations": self.rows.saturating_sub(1)}
            }),
            parameters: Requested {
                requested: &opts.ambiguity,
                effective: self.effective,
            },
            preparation,
            images: &self.images,
        };
        crate::report::publish(&opts.out, "ambiguity.json", &report)
    }
}
