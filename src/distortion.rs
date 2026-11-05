//! Exact displacement under directed, normalized arc-length correspondence.

use color_eyre::eyre::{Result, ensure};

/// A borrowed curve with finite coordinates and a finite arc length.
///
/// One point or a sequence of identical points defines a constant curve.
/// Positions borrow compact GPU-compatible f32 pairs. Lengths, interpolation
/// and integration use f64 throughout.
pub struct Curve<'a> {
    points: &'a [[f32; 2]],
    length: f64,
    origin: [f64; 2],
}

impl<'a> Curve<'a> {
    /// # Errors
    /// Rejects empty curves, nonfinite coordinates or nonfinite total length.
    pub fn new(points: &'a [[f32; 2]]) -> Result<Self> {
        let Some(first) = points.first() else {
            color_eyre::eyre::bail!("distortion curve must not be empty");
        };
        for point in points {
            ensure!(
                point.iter().all(|coordinate| coordinate.is_finite()),
                "distortion curve contains a nonfinite coordinate"
            );
        }
        let mut length = 0.0;
        for pair in points.windows(2) {
            if let [a, b] = pair {
                length += distance(*a, *b);
            }
        }
        ensure!(length.is_finite(), "nonfinite distortion curve length");
        Ok(Self {
            points,
            length,
            origin: first.map(f64::from),
        })
    }

    #[must_use]
    pub const fn length(&self) -> f64 {
        self.length
    }
}

fn distance([ax, ay]: [f32; 2], [bx, by]: [f32; 2]) -> f64 {
    (f64::from(bx) - f64::from(ax)).hypot(f64::from(by) - f64::from(ay))
}

struct Cursor<'a> {
    segments: std::slice::Windows<'a, [f32; 2]>,
    total: f64,
    traversed: f64,
    start: [f64; 2],
    end: [f64; 2],
    start_t: f64,
    end_t: f64,
}

impl<'a> Cursor<'a> {
    fn new(curve: &Curve<'a>) -> Self {
        let first = curve.origin;
        Self {
            segments: curve.points.windows(2),
            total: curve.length,
            traversed: 0.0,
            start: first,
            end: first,
            start_t: 0.0,
            end_t: 0.0,
        }
    }

    // Consume events even if adding a tiny length does not change its f64
    // cumulative sum, or normalization rounds it to the preceding event.
    fn advance(&mut self, t: f64) {
        while self.end_t <= t && self.end_t < 1.0 {
            let Some(pair) = self.segments.next() else {
                self.start = self.end;
                self.start_t = self.end_t;
                self.end_t = 1.0;
                return;
            };
            let [a, b] = pair else { continue };
            let length = distance(*a, *b);
            if length == 0.0 {
                continue;
            }
            self.start = a.map(f64::from);
            self.end = b.map(f64::from);
            self.start_t = self.end_t;
            self.traversed += length;
            self.end_t = if self.segments.len() == 0 {
                1.0
            } else {
                self.traversed / self.total
            };
        }
    }

    fn position(&self, t: f64) -> [f64; 2] {
        if t >= self.end_t {
            return self.end;
        }
        if t <= self.start_t {
            return self.start;
        }
        let fraction = (t - self.start_t) / (self.end_t - self.start_t);
        let [ax, ay] = self.start;
        let [bx, by] = self.end;
        [fraction.mul_add(bx - ax, ax), fraction.mul_add(by - ay, ay)]
    }
}

#[derive(Clone, Copy, Default)]
struct Sum {
    value: f64,
    correction: f64,
}

impl Sum {
    fn add(&mut self, term: f64) {
        let adjusted = term - self.correction;
        let next = self.value + adjusted;
        self.correction = (next - self.value) - adjusted;
        self.value = next;
    }
}

/// Integral of squared separation, in clip-coordinate units squared.
/// Each segment is visited a bounded number of times, without allocations.
/// # Errors
/// Rejects nonfinite integration results or invalid normalized intervals.
pub fn squared_integral(original: &Curve<'_>, current: &Curve<'_>) -> Result<f64> {
    let mut original = Cursor::new(original);
    let mut current = Cursor::new(current);
    let mut integral = Sum::default();
    let mut t = 0.0;
    loop {
        if t >= 1.0 {
            break;
        }
        original.advance(t);
        current.advance(t);
        let next = original.end_t.min(current.end_t);
        ensure!(next > t && next <= 1.0, "invalid distortion interval");
        let [px, py] = original.position(t);
        let [qx, qy] = current.position(t);
        let [rx, ry] = original.position(next);
        let [sx, sy] = current.position(next);
        let [ax, ay] = [px - qx, py - qy];
        let [bx, by] = [rx - sx, ry - sy];
        let [sum_x, sum_y] = [ax + bx, ay + by];
        let [delta_x, delta_y] = [bx - ax, by - ay];
        let midpoint_squared = sum_x.mul_add(sum_x, sum_y * sum_y);
        let change_squared = delta_x.mul_add(delta_x, delta_y * delta_y);
        integral.add((next - t) * (midpoint_squared / 4.0 + change_squared / 12.0));
        t = next;
    }
    ensure!(
        integral.value.is_finite() && integral.value >= 0.0,
        "nonfinite or negative distortion integral"
    );
    Ok(integral.value)
}

/// Equal-edge aggregation. Each call adds one directed edge, regardless of
/// its length or sample count. No per-edge arrays or square roots are kept.
#[derive(Default)]
pub struct Aggregate {
    sum: Sum,
    edges: u32,
}

pub struct Measurement {
    pub distortion: f64,
    pub rms_displacement_pixels: f64,
}

impl Aggregate {
    /// # Errors
    /// Rejects invalid integrals, overflowing edge counts or nonfinite totals.
    pub fn add(&mut self, original: &Curve<'_>, current: &Curve<'_>) -> Result<()> {
        let integral = squared_integral(original, current)?;
        let Some(edges) = self.edges.checked_add(1) else {
            color_eyre::eyre::bail!("too many distortion edges");
        };
        let mut sum = self.sum;
        sum.add(integral);
        ensure!(sum.value.is_finite(), "nonfinite distortion aggregate");
        self.sum = sum;
        self.edges = edges;
        Ok(())
    }

    /// # Errors
    /// Rejects an empty aggregate, a zero viewport or nonfinite measurements.
    pub fn finish(&self, size: u32) -> Result<Measurement> {
        ensure!(self.edges > 0, "distortion requires at least one edge");
        ensure!(size > 0, "distortion viewport must be nonzero");
        let rms = (self.sum.value / f64::from(self.edges)).sqrt();
        let distortion = rms / (2.0 * std::f64::consts::SQRT_2);
        let rms_displacement_pixels = rms * (f64::from(size) / 2.0);
        ensure!(
            distortion.is_finite() && rms_displacement_pixels.is_finite(),
            "nonfinite distortion measurement"
        );
        Ok(Measurement {
            distortion,
            rms_displacement_pixels,
        })
    }
}
