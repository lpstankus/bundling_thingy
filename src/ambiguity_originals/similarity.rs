//! Index-correspondence simplification and coherent endpoint-aware matching.
//!
//! Only the parent `Originals` pairs a preparation with the canonical original
//! it was built from, using the same validated sigma for every preparation.
use crate::weak_frechet::{self, Budget, Error, Orientation, Storage, WorkCategory, Workspace};

/// Compact certified knots of one canonical original and their maximum error.
pub struct Prepared {
    points: Vec<[f32; 2]>,
    error: f64,
}

/// Certify shortcuts of a nonempty finite canonical original.
///
/// Full-size temporary knots are reserved locally. The compact copy is reserved
/// while they are live and stays charged to the ledger only after success.
/// The certificate uses ordinary f64 arithmetic, not exact-real arithmetic.
/// # Errors
/// Unsupported sizes, allocation failures and exhausted budgets.
pub(super) fn prepare(
    input: &[[f32; 2]],
    sigma: f64,
    budget: &mut Budget,
) -> Result<Prepared, Error> {
    u32::try_from(input.len()).map_err(|_| Error::SizeOverflow)?;
    let mut temporary = Workspace::new(budget);
    let mut points = temporary.vector(input.len(), Storage::Bytes)?;
    let mut maximum_error = 0.0_f64;
    {
        let budget = &mut *temporary.budget;
        let mut start = 0;
        points.push(
            input
                .first()
                .ok_or(Error::InvalidCurve)?
                .map(|x| if x == 0.0 { 0.0 } else { x }),
        );
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
            maximum_error = maximum_error.max(error);
            points.push(
                input
                    .get(end)
                    .ok_or(Error::InvalidIndex)?
                    .map(|x| if x == 0.0 { 0.0 } else { x }),
            );
            start = end;
        }
    }
    let mut retained = temporary.nested();
    retained.charge(u64::try_from(points.len()).map_err(|_| Error::SizeOverflow)?)?;
    let mut compact = retained.vector(points.len(), Storage::Bytes)?;
    compact.extend_from_slice(&points);
    retained.commit();
    Ok(Prepared {
        points: compact,
        error: maximum_error,
    })
}

/// Original route distance bounds for the selected joint orientation.
#[derive(Clone, Copy)]
pub struct Assessment {
    pub lower: f64,
    pub upper: f64,
    pub incompatible: bool,
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
/// Positive route and endpoint thresholds and nonnegative simplification tolerance,
/// validated by `ambiguity::Config::new`.
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
    let d = weak_frechet::distance(&a.points, &b.points, orientation, budget)?;
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

/// Assess both orientations using certified simplified distances.
///
/// Each original must be the canonical route its preparation was built from with
/// this configuration's sigma, so certificate errors never exceed sigma. Ambiguous
/// orientation ordering or route classification triggers fallback on those
/// complete originals. Bounds use ordinary f64 rounding, without an exact-real
/// guarantee. Stored starts of closed curves matter.
/// # Errors
/// Unsupported index sizes, allocation failures and exhausted shared resources.
/// Returns `Error::InvalidScale` if a normalized orientation key is nonfinite.
pub(super) fn assess(
    (a, pa): (&[[f32; 2]], &Prepared),
    (b, pb): (&[[f32; 2]], &Prepared),
    config: &Config,
    budget: &mut Budget,
) -> Result<Assessment, Error> {
    let mut forward = bounds(pa, pb, Orientation::Forward, budget)?;
    let mut reverse = bounds(pa, pb, Orientation::Reversed, budget)?;
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
        lower: chosen.lower,
        upper: chosen.upper,
        incompatible: chosen.lower > config.route_tolerance,
    })
}
