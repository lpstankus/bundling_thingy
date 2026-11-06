//! Index-correspondence simplification and coherent endpoint-aware matching.
use crate::weak_frechet::{self, Budget, Error, Orientation, WorkCategory};
use std::mem::size_of;

/// Opaque certified geometry bound to an immutable original.
///
/// Scoped preparations belong to `with_prepared`; retained preparations belong to their caller.
/// Adjacent source indices describe each retained shortcut's inclusive source range.
pub struct Prepared<'a> {
    original: &'a [[f32; 2]],
    points: Vec<[f32; 2]>,
    source_indices: Vec<usize>,
    error: f64,
}

/// Runs a callback with certified geometry, releasing storage even when it fails.
/// The certificate uses ordinary f64 arithmetic, not exact-real arithmetic.
/// # Errors
/// Rejects invalid input, invalid sigma, allocation failures and exhausted budgets.
pub fn with_prepared<R>(
    input: &[[f32; 2]],
    sigma: f64,
    budget: &mut Budget,
    operation: impl FnOnce(&Prepared, &mut Budget) -> Result<R, Error>,
) -> Result<R, Error> {
    if input.is_empty() || !sigma.is_finite() || sigma < 0.0 {
        return Err(Error::InvalidCurve);
    }
    u32::try_from(input.len()).map_err(|_| Error::SizeOverflow)?;
    budget.charge(u64::try_from(input.len()).map_err(|_| Error::SizeOverflow)?)?;
    if input.iter().flatten().any(|x| !x.is_finite()) {
        return Err(Error::InvalidCurve);
    }
    let bytes = input
        .len()
        .checked_mul(
            size_of::<[f32; 2]>()
                .checked_add(size_of::<usize>())
                .ok_or(Error::SizeOverflow)?,
        )
        .ok_or(Error::SizeOverflow)?;
    budget.reserve(bytes)?;
    let result = (|| {
        let mut prepared = Prepared {
            original: input,
            points: Vec::new(),
            source_indices: Vec::new(),
            error: 0.0,
        };
        prepared
            .points
            .try_reserve_exact(input.len())
            .map_err(|_| Error::Allocation)?;
        prepared
            .source_indices
            .try_reserve_exact(input.len())
            .map_err(|_| Error::Allocation)?;
        let mut start = 0;
        prepared.points.push(
            input
                .first()
                .ok_or(Error::InvalidCurve)?
                .map(|x| if x == 0.0 { 0.0 } else { x }),
        );
        prepared.source_indices.push(0);
        while increment(start)? < input.len() {
            let mut end = if sigma == 0.0 {
                increment(start)?
            } else {
                input.len().checked_sub(1).ok_or(Error::InvalidCurve)?
            };
            let error = loop {
                let mut maximum = 0.0_f64;
                for i in increment(start)?..end {
                    budget.charge_classified(1, WorkCategory::SimplifierPointVisits)?;
                    let fraction = index_value(i.checked_sub(start).ok_or(Error::InvalidIndex)?)?
                        / index_value(end.checked_sub(start).ok_or(Error::InvalidIndex)?)?;
                    let [ax, ay] = point(input, start)?;
                    let [bx, by] = point(input, end)?;
                    let [px, py] = point(input, i)?;
                    let dx = px - fraction.mul_add(bx - ax, ax);
                    let dy = py - fraction.mul_add(by - ay, ay);
                    maximum = maximum.max(dx.hypot(dy));
                }
                if maximum <= sigma {
                    break maximum;
                }
                end = start
                    .checked_add(end.checked_sub(start).ok_or(Error::InvalidIndex)? / 2)
                    .ok_or(Error::SizeOverflow)?;
            };
            budget.charge(1)?;
            prepared.error = prepared.error.max(error);
            prepared.points.push(
                input
                    .get(end)
                    .ok_or(Error::InvalidIndex)?
                    .map(|x| if x == 0.0 { 0.0 } else { x }),
            );
            prepared.source_indices.push(end);
            start = end;
        }
        operation(&prepared, budget)
    })();
    budget.release(bytes)?;
    result
}

#[derive(Clone, Copy)]
pub struct Assessment {
    pub orientation: Orientation,
    pub route_lower: f64,
    pub route_upper: f64,
    pub endpoints: f64,
    pub general_incompatible: bool,
    pub endpoint_incompatible: bool,
}

fn increment(index: usize) -> Result<usize, Error> {
    index.checked_add(1).ok_or(Error::SizeOverflow)
}
fn index_value(index: usize) -> Result<f64, Error> {
    Ok(f64::from(
        u32::try_from(index).map_err(|_| Error::SizeOverflow)?,
    ))
}
fn point(points: &[[f32; 2]], index: usize) -> Result<[f64; 2], Error> {
    Ok(points.get(index).ok_or(Error::InvalidIndex)?.map(f64::from))
}
impl Prepared<'_> {
    #[must_use]
    pub fn points(&self) -> &[[f32; 2]] {
        &self.points
    }
    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }
    #[must_use]
    pub const fn error(&self) -> f64 {
        self.error
    }
}

/// Positive route and endpoint thresholds and nonnegative simplification tolerance.
/// Invalid values produce `Error::InvalidCurve` before preparing any storage.
pub struct Config {
    pub route_tolerance: f64,
    pub endpoint_tolerance: f64,
    pub sigma: f64,
}
#[derive(Clone, Copy)]
struct Bounds {
    lower: f64,
    upper: f64,
    endpoints: f64,
}
impl Bounds {
    fn key(self, route: f64, config: &Config) -> Result<f64, Error> {
        let key = (route / config.route_tolerance).max(self.endpoints / config.endpoint_tolerance);
        if !key.is_finite() {
            return Err(Error::InvalidScale);
        }
        Ok(key)
    }
    fn precedes(self, other: Self, config: &Config, forward: bool) -> Result<bool, Error> {
        let key = self
            .key(self.upper, config)?
            .total_cmp(&other.key(other.lower, config)?);
        if !key.is_eq() {
            return Ok(key.is_lt());
        }
        let route = self.upper.total_cmp(&other.lower);
        if !route.is_eq() {
            return Ok(route.is_lt());
        }
        let endpoints = self.endpoints.total_cmp(&other.endpoints);
        Ok(endpoints.is_lt() || (endpoints.is_eq() && forward))
    }
}
fn bounds(
    a: &Prepared,
    b: &Prepared,
    orientation: Orientation,
    budget: &mut Budget,
) -> Result<Bounds, Error> {
    let d = weak_frechet::distance(a.points(), b.points(), orientation, budget)?;
    let epsilon = a.error + b.error;
    Ok(Bounds {
        lower: d.endpoints.max(d.route - epsilon).max(0.0),
        upper: d.endpoints.max(d.route + epsilon),
        endpoints: d.endpoints,
    })
}
fn full(
    a: &[[f32; 2]],
    b: &[[f32; 2]],
    orientation: Orientation,
    budget: &mut Budget,
) -> Result<Bounds, Error> {
    let d = weak_frechet::distance(a, b, orientation, budget)?;
    Ok(Bounds {
        lower: d.route,
        upper: d.route,
        endpoints: d.endpoints,
    })
}

/// Assesses both orientations using certified simplified distances.
///
/// Ambiguous orientation ordering or route classification triggers full-original
/// fallback. Bounds use ordinary f64 rounding, without an exact-real guarantee.
/// All temporary reservations end before return, including on error. Work remains
/// charged to the caller's shared budget. Stored starts of closed curves matter.
/// # Errors
/// Rejects invalid configuration or geometry, unsupported index sizes, allocation
/// failures and exhausted work or workspace budgets using the kernel error type.
/// Returns `Error::InvalidScale` if a normalized orientation key is nonfinite.
pub fn assess(
    a: &[[f32; 2]],
    b: &[[f32; 2]],
    config: &Config,
    budget: &mut Budget,
) -> Result<Assessment, Error> {
    if !config.route_tolerance.is_finite()
        || config.route_tolerance <= 0.0
        || !config.endpoint_tolerance.is_finite()
        || config.endpoint_tolerance <= 0.0
        || !config.sigma.is_finite()
        || config.sigma < 0.0
    {
        return Err(Error::InvalidCurve);
    }
    with_prepared(a, config.sigma, budget, |pa, budget| {
        with_prepared(b, config.sigma, budget, |pb, budget| {
            assess_prepared(pa, pb, config, budget)
        })
    })
}

/// Assess preparations bound to their immutable originals. Full fallback uses
/// those originals and never another route with the same cohort membership.
/// # Errors
/// Invalid thresholds or exhausted shared resources abort the assessment.
pub fn assess_prepared(
    pa: &Prepared<'_>,
    pb: &Prepared<'_>,
    config: &Config,
    budget: &mut Budget,
) -> Result<Assessment, Error> {
    if !config.route_tolerance.is_finite()
        || config.route_tolerance <= 0.0
        || !config.endpoint_tolerance.is_finite()
        || config.endpoint_tolerance <= 0.0
        || !config.sigma.is_finite()
        || config.sigma < 0.0
    {
        return Err(Error::InvalidCurve);
    }
    let a = pa.original;
    let b = pb.original;

    let needs_original = |prepared: &Prepared<'_>| {
        prepared.error > config.sigma
            || (config.sigma == 0.0 && prepared.points.len() != prepared.original.len())
    };
    let (mut forward, mut reverse) = if needs_original(pa) || needs_original(pb) {
        (
            full(a, b, Orientation::Forward, budget)?,
            full(a, b, Orientation::Reversed, budget)?,
        )
    } else {
        (
            bounds(pa, pb, Orientation::Forward, budget)?,
            bounds(pa, pb, Orientation::Reversed, budget)?,
        )
    };
    if !forward.precedes(reverse, config, true)? && !reverse.precedes(forward, config, false)? {
        forward = full(a, b, Orientation::Forward, budget)?;
        reverse = full(a, b, Orientation::Reversed, budget)?;
    }
    let (mut chosen, orientation) = if forward.precedes(reverse, config, true)? {
        (forward, Orientation::Forward)
    } else {
        (reverse, Orientation::Reversed)
    };
    if chosen.lower <= config.route_tolerance && chosen.upper > config.route_tolerance {
        chosen = full(a, b, orientation, budget)?;
    }
    Ok(Assessment {
        orientation,
        route_lower: chosen.lower,
        route_upper: chosen.upper,
        endpoints: chosen.endpoints,
        general_incompatible: chosen.lower > config.route_tolerance,
        endpoint_incompatible: chosen.endpoints > config.endpoint_tolerance,
    })
}

/// Retain compact certified knots and source indices. The caller owns the
/// reservation until `release`; construction restores it on every failure.
/// # Errors
/// Invalid geometry, sigma, allocation or shared resource limits abort.
pub fn prepare<'a>(
    input: &'a [[f32; 2]],
    sigma: f64,
    budget: &mut Budget,
) -> Result<Prepared<'a>, Error> {
    with_prepared(input, sigma, budget, |temporary, budget| {
        let bytes = temporary
            .points
            .len()
            .checked_mul(
                size_of::<[f32; 2]>()
                    .checked_add(size_of::<usize>())
                    .ok_or(Error::SizeOverflow)?,
            )
            .ok_or(Error::SizeOverflow)?;
        budget.reserve(bytes)?;
        let result = (|| {
            budget
                .charge(u64::try_from(temporary.points.len()).map_err(|_| Error::SizeOverflow)?)?;
            let mut points = Vec::new();
            points
                .try_reserve_exact(temporary.points.len())
                .map_err(|_| Error::Allocation)?;
            let mut source_indices = Vec::new();
            source_indices
                .try_reserve_exact(temporary.source_indices.len())
                .map_err(|_| Error::Allocation)?;
            points.extend_from_slice(&temporary.points);
            source_indices.extend_from_slice(&temporary.source_indices);
            Ok(Prepared {
                original: input,
                points,
                source_indices,
                error: temporary.error,
            })
        })();
        if result.is_err() {
            budget.release(bytes)?;
        }
        result
    })
}
impl Prepared<'_> {
    /// Release this owner's compact retained storage.
    /// # Errors
    /// Reports a ledger ownership violation.
    pub fn release(self, budget: &mut Budget) -> Result<(), Error> {
        let bytes = self
            .points
            .len()
            .checked_mul(
                size_of::<[f32; 2]>()
                    .checked_add(size_of::<usize>())
                    .ok_or(Error::SizeOverflow)?,
            )
            .ok_or(Error::SizeOverflow)?;
        drop(self);
        budget.release(bytes)
    }
}
