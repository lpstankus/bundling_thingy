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

    report::invalidate(&opts.out, "distortion.json")?;
    report::invalidate(&opts.out, "noisiness.json")?;
    let mut distortion_images = Vec::new();

    let mut gpu = pollster::block_on(Gpu::new(None, opts))?;

    let mut report = if opts.evaluate_noisiness {
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
    let layout = if opts.evaluate_distortion {
        Some(gpu.distortion_layout()?)
    } else {
        None
    };
    let baseline = layout
        .as_ref()
        .map(|layout| gpu.distortion_snapshot(layout))
        .transpose()?;
    if let Some(baseline) = &baseline {
        baseline.validate_baseline()?;
    }
    write_image(&gpu, &opts.out, "original.png", 0, report.as_mut())?;
    if baseline.is_some() {
        println!("original.png distortion=0 rms_displacement_pixels=0");
        distortion_images.push(serde_json::json!({"file": "original.png", "iteration": 0, "distortion": 0.0, "rms_displacement_pixels": 0.0}));
    }
    if opts.neighbor_radius.is_some() {
        for iteration in 1..=opts.iterations {
            let _timing = tracing::info_span!("iteration", iteration).entered();
            gpu.iterate()?;
            let name = if opts.iterations == 1 {
                "neighbor-mean.png".to_owned()
            } else {
                format!("iteration-{iteration:02}.png")
            };
            write_image(&gpu, &opts.out, &name, iteration, report.as_mut())?;
            if let Some(baseline) = &baseline {
                let _capture =
                    tracing::info_span!("distortion.capture", file = %name, iteration).entered();
                let current = gpu.distortion_snapshot(baseline.layout())?;
                let _measure =
                    tracing::info_span!("distortion.measure", file = %name, iteration).entered();
                let mut aggregate = bundling_wgpu::distortion::Aggregate::default();
                for ((_, original), (_, changed)) in baseline.edges().zip(current.edges()) {
                    aggregate.add(
                        &bundling_wgpu::distortion::Curve::new(original)?,
                        &bundling_wgpu::distortion::Curve::new(changed)?,
                    )?;
                }
                let result = aggregate.finish(SIZE)?;
                distortion_images.push(serde_json::json!({"file": name, "iteration": iteration, "distortion": result.distortion, "rms_displacement_pixels": result.rms_displacement_pixels}));
                println!(
                    "{name} distortion={} rms_displacement_pixels={}",
                    result.distortion, result.rms_displacement_pixels
                );
            }
        }
    }
    if let Some(report) = report {
        report.write(opts)?;
    }
    if let Some(layout) = &layout {
        distortion_report::write(opts, layout, &distortion_images)?;
    }
    println!("rendered {} samples to {}", samples, opts.out.display());
    Ok(())
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
