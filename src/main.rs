mod advection;
mod bundling;
mod density;
mod gpu;
mod graph;
mod palette;
mod profiling;
mod resampling;
mod smoothing;
mod window;

use bundling::BundlingMethod;
use clap::Parser;
use color_eyre::eyre::{Result, WrapErr};
use gpu::{Gpu, SIZE};
use graph::Graph;
use palette::Palette;
use std::{
    fs::{self, File},
    path::PathBuf,
};

#[derive(Debug, Parser)]
#[command(
    args_override_self = true,
    after_help = "Without --headless: Left/Right/Space switch graphs; Escape exits."
)]
pub struct Options {
    /// Export a PNG without opening a window.
    #[arg(long)]
    headless: bool,

    /// Accepted for compatibility; headless exports always resample.
    #[arg(long = "resample", requires = "headless", hide = true)]
    _resample: bool,

    /// TRL input for headless rendering.
    #[arg(long, default_value = "data/fr-airlines.trl", requires = "headless")]
    input: PathBuf,

    /// Output directory for headless rendering.
    #[arg(long, default_value = "out/rust", requires = "headless")]
    out: PathBuf,

    /// Color palette for normalized path density.
    #[arg(long, value_enum, default_value_t = Palette::default())]
    palette: Palette,

    /// Number of bundling steps. Use zero to render the original paths.
    #[arg(long, default_value_t = 50)]
    iterations: u32,

    /// Initial movement distance in texels, reduced with the kernel each iteration.
    #[arg(long, default_value_t = 4.0, value_parser = parse_step_size)]
    step_size: f32,

    /// Initial parabolic kernel radius, reduced geometrically toward 2 texels.
    #[arg(long, default_value_t = 80, value_parser = clap::value_parser!(u16).range(2..))]
    kernel_radius: u16,

    /// Bundling sample spacing in texels. Display density always samples at one texel.
    #[arg(long, default_value_t = density::SAMPLE_SPACING, value_parser = parse_step_size)]
    sampling_step: f32,

    /// Bundling method used by the viewer and headless exports.
    #[arg(long = "bund_method", value_enum, default_value_t = BundlingMethod::Cubu)]
    bund_method: BundlingMethod,

    /// Enable GPU debugging and validation.
    #[arg(long)]
    validate_gpu: bool,
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

fn parse_step_size(value: &str) -> Result<f32, String> {
    let step = value.parse::<f32>().map_err(|error| error.to_string())?;
    if !step.is_finite() || step <= 0.0 {
        return Err("step size must be positive and finite".into());
    }
    Ok(step)
}

#[tracing::instrument(name = "headless", skip_all, fields(input = %opts.input.display()), err)]
fn headless(opts: &Options) -> Result<()> {
    fs::create_dir_all(&opts.out)
        .wrap_err_with(|| format!("create output directory {}", opts.out.display()))?;

    let mut gpu = pollster::block_on(Gpu::new(
        None,
        opts.validate_gpu,
        opts.palette,
        opts.kernel_radius,
    ))?;

    let mut graph = Graph::load(&opts.input)?;
    graph.normalize()?;
    let [width, height] = gpu.dimensions();
    let graph = graph.resample(width, height)?;
    gpu.upload(&graph)?;
    gpu.bundle(
        opts.bund_method,
        opts.iterations,
        opts.step_size,
        opts.sampling_step,
    )?;
    gpu.render()?;
    let pixels = gpu.readback()?;

    let path = opts
        .out
        .join(if opts.iterations == 0 {
            "original"
        } else {
            opts.bund_method.name()
        })
        .with_extension("png");
    let timing = tracing::info_span!("png.write", output = %path.display()).entered();
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
    drop(timing);

    println!(
        "wrote {} with {} samples",
        path.display(),
        gpu.sample_count()
    );
    Ok(())
}
