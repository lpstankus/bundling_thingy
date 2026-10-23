mod density;
mod gpu;
mod graph;
mod palette;
mod window;

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

    /// Enable GPU debugging and validation.
    #[arg(long)]
    validate_gpu: bool,
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let opts = Options::parse();

    if opts.headless {
        headless(&opts)
    } else {
        window::run(opts)
    }
}

fn headless(opts: &Options) -> Result<()> {
    fs::create_dir_all(&opts.out)
        .wrap_err_with(|| format!("create output directory {}", opts.out.display()))?;

    let mut gpu = pollster::block_on(Gpu::new(None, opts.validate_gpu, opts.palette))?;

    let mut graph = Graph::load(&opts.input)?;
    graph.normalize()?;
    let [width, height] = gpu.dimensions();
    let graph = graph.resample(width, height)?;
    gpu.upload(&graph)?;
    gpu.render()?;
    let pixels = gpu.readback()?;

    let path = opts.out.join("original.png");
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

    println!(
        "wrote {} with {} samples",
        path.display(),
        graph.points().len()
    );
    Ok(())
}
