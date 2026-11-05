use color_eyre::eyre::{OptionExt, Result, WrapErr, ensure};
use serde::Serialize;

use crate::{Options, gpu::SIZE};
use clap::ValueEnum;

#[derive(Debug, Serialize)]
pub struct Measurement {
    colored_pixels: u64,
    background_pixels: u64,
    total_pixels: u64,
    pub noisiness: f64,
    colored_to_background_ratio: Option<f64>,
}

fn validate_buffer(pixels: &[u8], width: u32, height: u32) -> Result<u64> {
    ensure!(width > 0 && height > 0, "image dimensions must be nonzero");
    let total = u64::from(width).checked_mul(u64::from(height));
    let length = total.and_then(|count| count.checked_mul(4));
    let length = length.and_then(|count| usize::try_from(count).ok());
    ensure!(
        length == Some(pixels.len()),
        "invalid RGBA buffer length for {width}x{height}"
    );
    total.ok_or_eyre("image pixel count overflow")
}

pub fn background(pixels: &[u8], width: u32, height: u32) -> Result<[u8; 3]> {
    validate_buffer(pixels, width, height)?;
    let mut chunks = pixels.as_chunks::<4>().0.iter();
    let Some([r, g, b, 255]) = chunks.next() else {
        color_eyre::eyre::bail!("blank frame must be opaque");
    };
    let rgb = [*r, *g, *b];
    for pixel in chunks {
        ensure!(
            *pixel == [*r, *g, *b, 255],
            "blank frame must have uniform RGB and opaque alpha"
        );
    }
    Ok(rgb)
}

pub fn measure(pixels: &[u8], width: u32, height: u32, rgb: [u8; 3]) -> Result<Measurement> {
    let total_pixels = validate_buffer(pixels, width, height)?;
    let mut colored_pixels = 0_u64;
    for pixel in pixels.as_chunks::<4>().0 {
        let [r, g, b, _] = pixel;
        if [*r, *g, *b] != rgb {
            colored_pixels = colored_pixels.saturating_add(1);
        }
    }
    let background_pixels = total_pixels.saturating_sub(colored_pixels);
    // Counts are bounded by u32 dimensions. Conversion only affects fraction precision.
    #[expect(
        clippy::cast_precision_loss,
        clippy::as_conversions,
        reason = "JSON fractions use f64; integer counts remain exact."
    )]
    let (colored, background, total) = (
        colored_pixels as f64,
        background_pixels as f64,
        total_pixels as f64,
    );
    Ok(Measurement {
        colored_pixels,
        background_pixels,
        total_pixels,
        noisiness: colored / total,
        colored_to_background_ratio: (background_pixels > 0).then(|| colored / background),
    })
}

#[derive(Serialize)]
struct Image {
    file: String,
    iteration: u32,
    #[serde(flatten)]
    measurement: Measurement,
    change_from_original: f64,
}

pub struct Report {
    rgb: [u8; 3],
    images: Vec<Image>,
}

impl Report {
    pub const fn new(rgb: [u8; 3]) -> Self {
        Self {
            rgb,
            images: Vec::new(),
        }
    }

    pub fn record(&mut self, pixels: &[u8], file: &str, iteration: u32) -> Result<()> {
        let measurement =
            measure(pixels, SIZE, SIZE, self.rgb).wrap_err_with(|| format!("measure {file}"))?;
        let original = self
            .images
            .first()
            .map_or(measurement.noisiness, |image| image.measurement.noisiness);
        let change_from_original = measurement.noisiness - original;
        println!(
            "noisiness {file}: {:.6}, change {change_from_original:+.6}",
            measurement.noisiness
        );
        self.images.push(Image {
            file: file.to_owned(),
            iteration,
            measurement,
            change_from_original,
        });
        Ok(())
    }

    pub fn write(&self, opts: &Options) -> Result<()> {
        let report = serde_json::json!({
            "schema_version": 1, "metric_version": 1, "metric": "noisiness",
            "definition": "Fraction of exported pixels whose RGB differs exactly from the calibrated background. Alpha is ignored.",
            "interpretation": "Comparisons require identical rendering settings. Lower occupancy is not total drawing quality. For bundling comparisons, edge-id coloring and disabled endpoint dots are recommended; density coloring can change with geometry.",
            "input": opts.input.display().to_string(),
            "render": {
                "render_contract_version": 1,
                "width": SIZE, "height": SIZE, "format": "rgba8unorm",
                "background_rgb": self.rgb, "comparison": "rgb_exact_inequality",
                "palette": opts.palette.to_possible_value().map(|value| value.get_name().to_owned()),
                "color_by": opts.color_by.to_possible_value().map(|value| value.get_name().to_owned()),
                "endpoint_dots": opts.endpoint_dots, "msaa": opts.msaa,
                "path_style": { "half_width_clip": 0.001, "opacity": 0.1, "endpoint_radius_pixels": 3, "endpoint_outline_width_pixels": 1 },
                "normalization": "Initial graph bounds; uniform fit to 95% canvas extent; centered; unchanged through iterations."
            },
            "bundling": { "neighbor_radius": opts.neighbor_radius, "iterations": self.images.last().map_or(0, |image| image.iteration) },
            "images": self.images
        });
        crate::report::publish(&opts.out, "noisiness.json", &report)
    }
}
