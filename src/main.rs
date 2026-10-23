mod gpu;
mod graph;
mod window;

use clap::Parser;
use color_eyre::eyre::{Result, WrapErr};
use gpu::{Gpu, SIZE};
use graph::{Graph, GraphLog};
use std::{
    fs::{self, File},
    path::PathBuf,
    time::Instant,
};

#[derive(Debug, Parser)]
#[command(
    args_override_self = true,
    after_help = "Without --headless or --benchmark: Left/Right/Space switch graphs; Escape exits."
)]
pub struct Options {
    #[command(flatten)]
    mode: ModeOptions,

    /// Resample paths before headless rendering or benchmarking.
    #[arg(long, requires = "batch")]
    resample: bool,

    /// TRL input for headless rendering or benchmarking.
    #[arg(long, default_value = "data/fr-airlines.trl", requires = "batch")]
    input: PathBuf,

    /// Output directory for headless rendering.
    #[arg(
        long,
        default_value = "out/rust",
        requires = "headless",
        conflicts_with = "benchmark"
    )]
    out: PathBuf,

    /// Enable GPU debugging and validation.
    #[arg(long)]
    validate_gpu: bool,
}

#[derive(Debug, clap::Args)]
#[group(id = "batch", multiple = true)]
struct ModeOptions {
    /// Export a PNG without opening a window.
    #[arg(long)]
    headless: bool,

    /// Run the pipeline and print timings without a window or PNG export.
    #[arg(long, visible_alias = "test")]
    benchmark: bool,
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let opts = Options::parse();

    if opts.mode.headless || opts.mode.benchmark {
        headless(&opts)
    } else {
        window::run(opts)
    }
}

fn headless(opts: &Options) -> Result<()> {
    let total_started = Instant::now();
    let mut stage_started = total_started;
    let mut timings = Vec::new();
    let mut record_time = |stage| {
        timings.push((stage, stage_started.elapsed()));
        stage_started = Instant::now();
    };
    let log = if opts.mode.benchmark {
        GraphLog::Silent
    } else {
        GraphLog::Summary
    };

    let mut gpu = pollster::block_on(Gpu::new(None, opts.validate_gpu))?;
    record_time("GPU setup");

    let mut graph = Graph::load(&opts.input, log)?;
    record_time("Load");
    graph.normalize()?;
    record_time("Normalize");
    if opts.resample {
        graph.resample(0.02, log)?;
        record_time("Resample");
    }

    let mesh = graph.mesh(0.002)?;
    record_time("Mesh");
    if !mesh.is_empty() {
        gpu.upload(&mesh)?;
        record_time("Upload");
        gpu.render()?;
        record_time("Render");
    }

    if opts.mode.benchmark {
        let total = total_started.elapsed();
        for (stage, elapsed) in timings {
            println!("{stage}: {:.3} ms", elapsed.as_secs_f64() * 1000.0);
        }
        println!("Total: {:.3} ms", total.as_secs_f64() * 1000.0);
        return Ok(());
    }

    if mesh.is_empty() {
        println!("Skipping empty mesh: original");
        return Ok(());
    }

    let pixels = gpu.readback()?;

    fs::create_dir_all(&opts.out)
        .wrap_err_with(|| format!("create output directory {}", opts.out.display()))?;
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

    println!("wrote {} with {} vertices", path.display(), mesh.len());
    Ok(())
}
