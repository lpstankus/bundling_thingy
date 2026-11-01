mod density;
mod gpu;
mod graph;
mod neighbor_mean;
mod palette;
mod profiling;
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

    let mut gpu = pollster::block_on(Gpu::new(None, opts))?;

    let graph = Graph::load(&opts.input)?;
    let samples = gpu.upload(&graph)?;
    write_image(&gpu, &opts.out.join("original.png"))?;
    if opts.neighbor_radius.is_some() {
        for iteration in 1..=opts.iterations {
            let _timing = tracing::info_span!("iteration", iteration).entered();
            gpu.iterate()?;
            let name = if opts.iterations == 1 {
                "neighbor-mean.png".to_owned()
            } else {
                format!("iteration-{iteration:02}.png")
            };
            write_image(&gpu, &opts.out.join(name))?;
        }
    }
    println!("rendered {} samples to {}", samples, opts.out.display());
    Ok(())
}

fn write_image(gpu: &Gpu, path: &Path) -> Result<()> {
    gpu.render()?;
    let pixels = gpu.readback()?;
    let _timing = tracing::info_span!("png.write", output = %path.display()).entered();
    let mut encoder = png::Encoder::new(
        File::create(path).wrap_err_with(|| format!("create {}", path.display()))?,
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
    println!("wrote {}", path.display());
    Ok(())
}
