mod gpu;
mod graph;
mod window;
use color_eyre::eyre::{OptionExt, Result, WrapErr, bail};
use gpu::{Gpu, SIZE};
use graph::Graph;
use std::{
    fs::{self, File},
    io::BufWriter,
    path::PathBuf,
};

#[derive(Debug)]
pub struct Options {
    headless: bool,
    resample: bool,
    validate_gpu: bool,
    input: PathBuf,
    out: PathBuf,
}
impl Options {
    fn parse() -> Result<Option<Self>> {
        let mut o = Self {
            headless: false,
            resample: false,
            validate_gpu: false,
            input: "data/fr-airlines.trl".into(),
            out: "out/rust".into(),
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--headless" => o.headless = true,
                "--resample" => o.resample = true,
                "--validate-gpu" => o.validate_gpu = true,
                "--input" => o.input = args.next().ok_or_eyre("--input needs a path")?.into(),
                "--out" => o.out = args.next().ok_or_eyre("--out needs a path")?.into(),
                "--help" => {
                    println!(
                        "bundling-wgpu [--headless] [--input FILE] [--out DIR] [--resample] [--validate-gpu]\nWithout --headless: Left/Right/Space switch graphs; Escape exits."
                    );
                    return Ok(None);
                }
                _ => bail!("unknown option {arg}"),
            }
        }
        Ok(Some(o))
    }
}
fn headless(opts: &Options) -> Result<()> {
    fs::create_dir_all(&opts.out)
        .wrap_err_with(|| format!("create output directory {}", opts.out.display()))?;
    let mut gpu = pollster::block_on(Gpu::new(None, opts.validate_gpu))?;
    let mut graph = Graph::load(&opts.input)?;
    graph.normalize()?;
    if opts.resample {
        graph.resample(0.02)?;
    }
    gpu.upload(&graph.mesh(0.002)?)?;
    gpu.render()?;
    let pixels = gpu.readback()?;
    let path = opts.out.join("original.png");
    let mut encoder = png::Encoder::new(
        BufWriter::new(File::create(&path).wrap_err_with(|| format!("create {}", path.display()))?),
        SIZE,
        SIZE,
    );
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()?
        .write_image_data(&pixels)
        .wrap_err_with(|| format!("write {}", path.display()))?;
    println!("wrote {}", path.display());
    Ok(())
}
fn main() -> Result<()> {
    color_eyre::install()?;
    let Some(opts) = Options::parse()? else {
        return Ok(());
    };
    if opts.headless {
        headless(&opts)
    } else {
        window::run(opts)
    }
}
