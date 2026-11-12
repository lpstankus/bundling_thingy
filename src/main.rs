mod ambiguity_report;
mod density;
mod distortion_report;
mod gpu;
mod graph;
mod neighbor_mean;
mod noisiness;
mod palette;
mod profiling;
mod report;
mod resampling;
mod shader;
mod window;

use bundling_wgpu::distortion::{Aggregate, Curve};
use clap::{Parser, builder::TypedValueParser};
use color_eyre::eyre::{Result, WrapErr};
use gpu::{Gpu, SIZE};
use graph::Graph;
use palette::{ColorBy, Palette};
use std::{
    fs::{self, File},
    path::{Path, PathBuf},
};

#[derive(Debug, Parser)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Clap parses independent CLI switches."
)]
#[command(
    args_override_self = true,
    after_help = "Without --headless: Left/Right/Space switch graphs; Escape exits."
)]
pub struct Options {
    /// Export a PNG without opening a window.
    #[arg(long)]
    headless: bool,

    /// Measure exported RGB occupancy and write noisiness.json.
    #[arg(long, requires = "headless")]
    evaluate_noisiness: bool,

    /// Measure GPU geometry displacement and write distortion.json.
    #[arg(long, requires = "headless")]
    evaluate_distortion: bool,

    /// Measure all three geometric ambiguity views and write ambiguity.json.
    #[arg(long, requires = "headless")]
    evaluate_ambiguity: bool,

    #[command(flatten)]
    ambiguity: ambiguity_report::Parameters,

    /// Accepted for compatibility; headless exports always resample.
    #[arg(long = "resample", requires = "headless", hide = true)]
    _resample: bool,

    /// TRL input for headless rendering.
    #[arg(long, default_value = "data/fr-airlines.trl", requires = "headless")]
    input: PathBuf,

    /// Output directory for headless rendering.
    #[arg(long, default_value = "out/rust", requires = "headless")]
    out: PathBuf,

    /// Color palette for paths and endpoint dots.
    #[arg(long, value_enum, default_value_t = Palette::default())]
    palette: Palette,

    /// Value used to select each path's gradient color.
    #[arg(long, value_enum, default_value_t = ColorBy::default())]
    color_by: ColorBy,

    /// Draw dots at the start and end of each path.
    #[arg(long)]
    endpoint_dots: bool,

    /// Mean-update radius in texels. Endpoints remain fixed.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    neighbor_radius: Option<u32>,

    /// Mean updates per graph. Headless exports every iteration.
    #[arg(long, default_value_t = 1, requires = "neighbor_radius", value_parser = clap::value_parser!(u32).range(1..))]
    iterations: u32,

    /// MSAA samples. Use 2 for speed or 4 for smoother edges.
    #[arg(long, default_value_t = 4, value_parser = clap::builder::PossibleValuesParser::new(["2", "4"]).try_map(|value| value.parse::<u32>()))]
    msaa: u32,

    /// Enable GPU debugging and validation.
    #[arg(long)]
    validate_gpu: bool,

    /// Maximum graph buffer memory in MiB, excluding density and render targets.
    #[arg(long, default_value_t = 1024, value_parser = clap::value_parser!(u64).range(1..))]
    max_graph_memory_mib: u64,
}

fn main() -> Result<()> {
    color_eyre::install()?;
    profiling::init()?;
    let matches = <Options as clap::CommandFactory>::command().get_matches();
    let opts = <Options as clap::FromArgMatches>::from_arg_matches(&matches)?;

    if opts.headless {
        headless(&opts, &matches)
    } else {
        window::run(opts)
    }
}

#[tracing::instrument(name = "headless", skip_all, fields(input = %opts.input.display()), err)]
fn headless(opts: &Options, matches: &clap::ArgMatches) -> Result<()> {
    fs::create_dir_all(&opts.out)
        .wrap_err_with(|| format!("create output directory {}", opts.out.display()))?;

    report::invalidate(&opts.out, "ambiguity.json")?;
    report::invalidate(&opts.out, "distortion.json")?;
    report::invalidate(&opts.out, "noisiness.json")?;
    if opts.evaluate_ambiguity {
        opts.ambiguity.validate()?;
        ambiguity_report::check_backend_flags(matches, opts.ambiguity.backend)?;
    }
    let mut gpu = pollster::block_on(Gpu::new(None, opts))?;
    let noisiness = if opts.evaluate_noisiness {
        gpu.render()?;
        let blank = gpu.readback()?;
        Some(noisiness::Report::new(
            noisiness::background(&blank, SIZE, SIZE).wrap_err("calibrate noisiness background")?,
        ))
    } else {
        None
    };
    let graph = Graph::load(&opts.input)?;
    let samples = gpu.upload(&graph)?;
    // The gpu-f32 ambiguity backend keeps all geometry on the device.
    let resident =
        opts.evaluate_ambiguity && opts.ambiguity.backend == ambiguity_report::Backend::GpuF32;
    let layout = (opts.evaluate_distortion || opts.evaluate_ambiguity && !resident)
        .then(|| gpu.geometry_layout())
        .transpose()?;
    let baseline = layout
        .as_ref()
        .map(|layout| gpu.geometry_snapshot(layout))
        .transpose()?;
    if opts.evaluate_distortion {
        baseline
            .as_ref()
            .ok_or_else(|| color_eyre::eyre::eyre!("missing distortion baseline"))?
            .validate_baseline()?;
    }
    if resident {
        // Freezes the baseline points on the device before any bundling step.
        let mut session = gpu.ambiguity_resident(&opts.ambiguity.resident_settings()?)?;
        let mut images = Vec::new();
        image_flow(
            opts,
            &mut gpu,
            baseline.as_ref(),
            noisiness,
            None,
            &mut Vec::new(),
            Some((&mut session, &mut images)),
        )?;
        ambiguity_report::publish_resident(opts, &session, &images, samples)?;
    } else if opts.evaluate_ambiguity {
        ambiguity_report::run(
            opts,
            baseline
                .as_ref()
                .ok_or_else(|| color_eyre::eyre::eyre!("missing ambiguity baseline"))?,
            |evaluator, images| {
                image_flow(
                    opts,
                    &mut gpu,
                    baseline.as_ref(),
                    noisiness,
                    Some(evaluator),
                    images,
                    None,
                )
            },
        )?;
    } else {
        image_flow(
            opts,
            &mut gpu,
            baseline.as_ref(),
            noisiness,
            None,
            &mut Vec::new(),
            None,
        )?;
    }
    println!("rendered {} samples to {}", samples, opts.out.display());
    Ok(())
}

impl Options {
    const fn geometry_evaluation(&self) -> bool {
        self.evaluate_distortion || self.evaluate_ambiguity
    }
}

fn image_flow(
    opts: &Options,
    gpu: &mut Gpu,
    baseline: Option<&bundling_wgpu::geometry::Snapshot<'_>>,
    mut noisiness: Option<noisiness::Report>,
    mut evaluator: Option<&mut bundling_wgpu::ambiguity::Evaluator<'_, '_>>,
    ambiguity_images: &mut Vec<ambiguity_report::Image>,
    mut resident: Option<(
        &mut bundling_wgpu::ambiguity_gpu::resident::Session,
        &mut Vec<ambiguity_report::ResidentImage>,
    )>,
) -> Result<()> {
    let mut distortion_images = Vec::new();
    let iterations = if opts.neighbor_radius.is_some() {
        opts.iterations
    } else {
        0
    };
    for iteration in 0..=iterations {
        let _timing = tracing::info_span!("image", iteration).entered();
        if iteration > 0 {
            gpu.iterate()?;
        }
        let name = image_name(iteration, iterations)?;
        write_image(gpu, &opts.out, &name, iteration, noisiness.as_mut())?;
        let current = if iteration > 0 {
            baseline
                .map(|original| gpu.geometry_snapshot(original.layout()))
                .transpose()?
        } else {
            None
        };
        if opts.evaluate_distortion {
            let mut aggregate = Aggregate::default();
            let result = if let (Some(original), Some(changed)) = (baseline, current.as_ref()) {
                let _measure = tracing::info_span!("distortion.measure", iteration).entered();
                for ((_, a), (_, b)) in original.edges().zip(changed.edges()) {
                    aggregate.add(&Curve::new(a)?, &Curve::new(b)?)?;
                }
                let result = aggregate.finish(SIZE)?;
                (result.distortion, result.rms_displacement_pixels)
            } else {
                (0.0, 0.0)
            };
            distortion_images.push(serde_json::json!({
                "file": name, "iteration": iteration,
                "distortion": result.0, "rms_displacement_pixels": result.1
            }));
            println!(
                "{name} distortion={} rms_displacement_pixels={}",
                result.0, result.1
            );
        }
        if let Some(evaluator) = evaluator.as_mut() {
            let row = if let Some(current) = &current {
                let mut paths = Vec::new();
                paths.try_reserve_exact(current.layout().edge_count())?;
                paths.extend(
                    current
                        .edges()
                        .map(|(id, points)| bundling_wgpu::ambiguity_index::Path { id, points }),
                );
                evaluator.evaluate(&paths).wrap_err("evaluate ambiguity snapshot; check --ambiguity-work-limit, --ambiguity-workspace-bytes and --ambiguity-stored-pair-limit")?
            } else {
                evaluator.baseline()
            };
            ambiguity_images.push(ambiguity_report::Image::new(
                name.clone(),
                iteration,
                &row,
                &evaluator.baseline(),
            )?);
        }
        if let Some((session, images)) = resident.as_mut() {
            let frame = if iteration > 0 {
                session
                    .evaluate(&gpu.candidate_geometry()?)
                    .wrap_err("evaluate gpu-f32 ambiguity frame")?
            } else {
                session.baseline()
            };
            images.push(ambiguity_report::ResidentImage::new(
                name,
                iteration,
                &frame,
                &session.baseline(),
            )?);
        }
    }
    if let Some(noisiness) = noisiness {
        noisiness.write(opts)?;
    }
    if opts.evaluate_distortion {
        distortion_report::write(
            opts,
            baseline
                .ok_or_else(|| color_eyre::eyre::eyre!("missing distortion layout"))?
                .layout(),
            &distortion_images,
        )?;
    }
    Ok(())
}

fn image_name(iteration: u32, iterations: u32) -> Result<String> {
    use std::fmt::Write;
    let mut name = String::new();
    name.try_reserve_exact(32)?;
    match iteration {
        0 => name.push_str("original.png"),
        _ if iterations == 1 => name.push_str("neighbor-mean.png"),
        _ => write!(&mut name, "iteration-{iteration:02}.png")?,
    }
    Ok(name)
}

fn write_image(
    gpu: &Gpu,
    directory: &Path,
    file: &str,
    iteration: u32,
    report: Option<&mut noisiness::Report>,
) -> Result<()> {
    let path = directory.join(file);
    gpu.render()?;
    let pixels = gpu.readback()?;
    let _timing = tracing::info_span!("png.write", output = %path.display()).entered();
    let mut encoder = png::Encoder::new(
        File::create(&path).wrap_err_with(|| format!("create {}", path.display()))?,
        SIZE,
        SIZE,
    );
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer
        .write_image_data(&pixels)
        .and_then(|()| writer.finish())
        .wrap_err_with(|| format!("write {}", path.display()))?;
    if let Some(report) = report {
        report.record(&pixels, file, iteration)?;
    }
    println!("wrote {}", path.display());
    Ok(())
}
