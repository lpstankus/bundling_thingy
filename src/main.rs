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

    /// Measure general geometric ambiguity and write ambiguity.json.
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
    let opts = Options::parse();

    if opts.headless {
        headless(&opts)
    } else {
        window::run(opts)
    }
}

#[tracing::instrument(name = "headless", skip_all, fields(input = %opts.input.display()), err)]
fn headless(opts: &Options) -> Result<()> {
    fs::create_dir_all(&opts.out)
        .wrap_err_with(|| format!("create output directory {}", opts.out.display()))?;

    report::invalidate(&opts.out, "ambiguity.json")?;
    report::invalidate(&opts.out, "ambiguity.json.tmp")?;
    report::invalidate(&opts.out, "distortion.json")?;
    report::invalidate(&opts.out, "noisiness.json")?;
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
    let layout = opts
        .evaluate_distortion
        .then(|| gpu.distortion_layout())
        .transpose()?;
    let baseline = layout
        .as_ref()
        .map(|layout| gpu.distortion_snapshot(layout))
        .transpose()?;
    if let Some(baseline) = &baseline {
        baseline.validate_baseline()?;
    }
    let mut ambiguity = opts
        .evaluate_ambiguity
        .then(|| ambiguity_report::Session::new(opts, &gpu))
        .transpose()?;
    image_flow(
        opts,
        &mut gpu,
        baseline.as_ref(),
        noisiness,
        ambiguity.as_mut(),
    )?;
    if let Some(ambiguity) = ambiguity {
        ambiguity.finish()?;
    }
    println!("rendered {} samples to {}", samples, opts.out.display());
    Ok(())
}

fn image_flow(
    opts: &Options,
    gpu: &mut Gpu,
    baseline: Option<&bundling_wgpu::geometry::Snapshot<'_>>,
    mut noisiness: Option<noisiness::Report>,
    mut ambiguity: Option<&mut ambiguity_report::Session<'_>>,
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
        if let Some(original) = baseline {
            let current = (iteration > 0)
                .then(|| gpu.distortion_snapshot(original.layout()))
                .transpose()?;
            let mut aggregate = Aggregate::default();
            let result = if let Some(changed) = &current {
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
        if let Some(session) = ambiguity.as_mut() {
            // Bundling alternates point buffers: take the current one each frame.
            let current = (iteration > 0)
                .then(|| gpu.ambiguity_geometry())
                .transpose()?;
            session.record(name, iteration, current.as_ref())?;
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
