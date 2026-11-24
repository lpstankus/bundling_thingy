use color_eyre::eyre::{Result, ensure};

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum BundlingMethod {
    /// `CUBu` with Laplacian path smoothing.
    Cubu,
    /// `CUBu` with path smoothing disabled.
    #[value(name = "cubu_nosmoothing")]
    CubuNoSmoothing,
}

impl BundlingMethod {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cubu => "cubu",
            Self::CubuNoSmoothing => "cubu_nosmoothing",
        }
    }
}
pub struct CubuSchedule {
    remaining: u32,
    radius: f64,
    step_size: f64,
    reduction: f64,
}

pub struct CubuIteration {
    pub radius: f64,
    pub kernel_radius: u16,
    pub step_size: f64,
    pub final_iteration: bool,
}

impl CubuSchedule {
    pub fn new(radius: u16, step_size: f32, iterations: u32) -> Result<Self> {
        ensure!(
            radius >= 2,
            "initial kernel radius must be at least 2 texels"
        );
        ensure!(
            step_size.is_finite() && step_size > 0.0,
            "initial step size must be positive and finite"
        );
        let radius = f64::from(radius);
        // Released CUBu: multiply h by (2 / h_initial)^(1 / iterations) after each step.
        let reduction = (2.0 / radius).powf(1.0 / f64::from(iterations.max(1)));
        Ok(Self {
            remaining: iterations,
            radius,
            step_size: f64::from(step_size),
            reduction,
        })
    }
}

impl Iterator for CubuSchedule {
    type Item = CubuIteration;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let iteration = CubuIteration {
            radius: self.radius,
            kernel_radius: discrete_radius(self.radius),
            step_size: self.step_size,
            final_iteration: self.remaining == 1,
        };
        self.radius *= self.reduction;
        self.step_size *= self.reduction;
        self.remaining = self.remaining.saturating_sub(1);
        Some(iteration)
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "The schedule stays finite and within the initial u16 radius. CUBu truncates h/2 to get an even filter radius."
)]
fn discrete_radius(radius: f64) -> u16 {
    ((radius / 2.0) as u16).saturating_mul(2).max(2)
}
