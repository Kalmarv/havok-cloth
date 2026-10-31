//! Elden Ring's cloth wind: the force fields the game's effects place in the world, sampled where a sim cloth is, and
//! the wind action that turns the wind into forces on the cloth's particles on every substep.
//!
//! The wind action is the game's own (not part of the cloth data: a tagfile declares no action of it), run as an
//! [`Action`] of the simulate operator. Its arithmetic is single precision in the game's operation order, and its sines
//! are the C runtime's single-precision `sinf`, as the game's are on Windows; tests/game_capture.rs holds it to the
//! forces the running game left on every call of a capture, bit for bit.
//!
//! What is here is what the game does with the fields it samples once for the whole cloth (a field of kind 0 that is
//! not local): the sample at the centre of the particles' bounds ([`ForceField::sample`]), the wind of all of them
//! ([`combine`]) and the forces ([`WindAction`]). Fields of other kinds, and local ones, act per particle on another
//! path, not reproduced: [`global_wind`] refuses them.

use crate::simd::V4;
use crate::simulate::{Action, SimState};

/// 2π as the game's sampler holds it.
const TWO_PI: f32 = f32::from_bits(0x40C9_0FDB);

/// `maxss x, floor` behind the game's NaN guard: `x` if it is above `floor`, else `floor`, and NaN for NaN.
fn at_least(x: f32, floor: f32) -> f32 {
    if x.is_nan() {
        f32::NAN
    } else if x > floor {
        x
    } else {
        floor
    }
}

/// `minss x, ceiling` behind the game's NaN guard.
fn at_most(x: f32, ceiling: f32) -> f32 {
    if x.is_nan() {
        f32::NAN
    } else if x < ceiling {
        x
    } else {
        ceiling
    }
}

/// The C runtime's `modff` fractional part: the sign of `x` kept, -0 for a negative whole number.
fn fraction(x: f32) -> f32 {
    (x - x.trunc()).copysign(x)
}

/// `x` clamped to [0, 1]: the upper bound first, NaN kept.
fn clamp01(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let m = if 1.0 < x { 1.0 } else { x };
    if 0.0 > m { 0.0 } else { m }
}

/// The time the force fields read: the game's effect clock (seconds) in single precision, or 0 while the clock is
/// stopped.
pub fn clock(seconds: f64, stopped: bool) -> f32 {
    if stopped { 0.0 } else { seconds as f32 }
}

/// The centre of the particles' bounds, where the game samples the fields for the whole cloth: the halfway point of
/// the per-lane minimum and maximum of the positions the step starts from (the positions the last step ended on).
pub fn bounds_centre(positions: &[V4]) -> V4 {
    let Some(&first) = positions.first() else {
        return V4::ZERO;
    };
    let (lo, hi) = positions
        .iter()
        .fold((first, first), |(lo, hi), &p| (lo.min(p), hi.max(p)));
    lo.add(hi).scale(0.5)
}

/// A force field's gust: a wave that travels along the wind, rising, falling and resting in turn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gust {
    /// The gust's speed and strength (none at or below 0).
    pub strength: f32,
    /// Its wavelength, which also sets the sideways wobble of its front (up to 5).
    pub length: f32,
    /// Its amplitude's scale.
    pub scale: f32,
    /// What its wave adds to the sine before halving it.
    pub offset: f32,
    /// The time it rises (the rise, fall and rest add up to its period).
    pub rise: f32,
    /// The time it falls.
    pub fall: f32,
    /// The time it rests.
    pub rest: f32,
}

/// A force field as the game samples it: where it is and how it blows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForceField {
    /// The field's kind (0: a directional wind).
    pub kind: u32,
    /// Whether the field acts per particle rather than once for the cloth.
    pub local: bool,
    /// The point its waves are measured from (only `x` and `z` enter).
    pub origin: [f32; 3],
    /// The direction it blows (`x` and `z` measure along it; all four lanes scale the wind).
    pub direction: V4,
    /// Its weight.
    pub weight: f32,
    /// The fade's scale: the fade, the base wave's amplitude, is `(scale * (high - low)) * 0.5 + mid`.
    pub fade_scale: f32,
    /// The fade's low end.
    pub fade_low: f32,
    /// The fade's high end.
    pub fade_high: f32,
    /// The fade's middle.
    pub fade_mid: f32,
    /// The distance along the wind over which it falls off to nothing, where it falls off.
    pub falloff: Option<f32>,
    /// The base wave's length (taken as at least 0.001).
    pub wave_length: f32,
    /// What the base wave adds to its sine before halving it.
    pub wave_offset: f32,
    /// The gust.
    pub gust: Gust,
}

impl ForceField {
    /// The fade: the base wave's amplitude.
    fn fade(&self) -> f32 {
        ((self.fade_scale * (self.fade_high - self.fade_low)) * 0.5) + self.fade_mid
    }

    /// The field's wind at `at` at time `clock` ([`clock`]): its direction times a base wave travelling along it,
    /// plus the gust, times the falloff and the weight.
    pub fn sample(&self, at: V4, clock: f32) -> V4 {
        // the sampler's two trailing arguments, as the cloth's sample passes them
        let (lift, swing) = (0.5_f32, 0.5_f32);
        let d = self.direction;
        let dx = at.x() - self.origin[0];
        let dz = at.z() - self.origin[2];
        let along = (d.z() * dz) + (d.x() * dx);
        let k1 = TWO_PI / at_least(self.wave_length, 0.001);
        let wave = (((((k1 * clock) - (k1 * along)).sin() + self.wave_offset) * 0.5) * swing) + lift;
        let mut amplitude = self.fade() * wave;
        let g = self.gust;
        if g.strength > 0.0 {
            let wobble = at_most(g.length, 5.0);
            let falling = g.fall + g.rise;
            let period = falling + g.rest;
            let k2 = TWO_PI / at_least(g.length, 0.001);
            let across = (dz * -d.x()) + (dx * d.z());
            let front = ((fraction(across / (g.length * 2.0)) * TWO_PI).sin() * wobble) + along;
            let travelled =
                (((g.length * -0.25) + (g.strength * clock)) - front) / at_least(period * g.strength, 0.001);
            let mut envelope = fraction(travelled);
            if 0.0 > travelled {
                envelope = 1.0 - envelope;
            }
            let (rise, fall) = (g.rise / period, falling / period);
            envelope = if rise > envelope {
                envelope / rise
            } else if fall > envelope {
                1.0 - ((envelope - rise) / (fall - rise))
            } else {
                0.0
            };
            let phase = ((k2 * g.strength) * clock) - (k2 * along);
            let gust =
                ((((((phase.sin() + g.offset) * 0.5) * swing) + lift) * envelope) * g.strength) * g.scale;
            amplitude += gust;
        }
        let fall = match self.falloff {
            Some(radius) => clamp01(1.0 - (along.abs() / at_least(radius, 1e-4))),
            None => 1.0,
        };
        d.scale((fall * amplitude) * self.weight)
    }
}

/// The wind of several fields' samples: per lane, the largest of them and 0 plus the smallest of them and 0 (NaN where
/// one is).
pub fn combine(samples: impl IntoIterator<Item = V4>) -> V4 {
    let guarded = |a: f32, b: f32, pick: fn(f32, f32) -> f32| {
        if a.is_nan() || b.is_nan() {
            f32::NAN
        } else {
            pick(a, b)
        }
    };
    let (mut hi, mut lo) = (V4::ZERO, V4::ZERO);
    for w in samples {
        hi = V4(std::array::from_fn(|i| {
            guarded(hi.0[i], w.0[i], |a, b| if a > b { a } else { b })
        }));
        lo = V4(std::array::from_fn(|i| {
            guarded(lo.0[i], w.0[i], |a, b| if a < b { a } else { b })
        }));
    }
    lo.add(hi)
}

/// A field the cloth's wind for the whole cloth cannot take.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Unsupported {
    /// A field of another kind than 0, or a local one: the game applies it per particle, on a path not reproduced.
    #[error("force field {index}: kind {kind}, local {local}: applied per particle, not reproduced")]
    PerParticle {
        /// The field's place in the list.
        index: usize,
        /// Its kind.
        kind: u32,
        /// Whether it is local.
        local: bool,
    },
}

/// The wind on a cloth whose particles' bounds centre is `centre` ([`bounds_centre`]): every field with weight
/// sampled there, combined. Refuses a field it does not reproduce.
pub fn global_wind(fields: &[ForceField], centre: V4, clock: f32) -> Result<V4, Unsupported> {
    let mut samples = Vec::with_capacity(fields.len());
    for (index, f) in fields.iter().enumerate() {
        if f.weight == 0.0 {
            continue;
        }
        if f.kind != 0 || f.local {
            return Err(Unsupported::PerParticle {
                index,
                kind: f.kind,
                local: f.local,
            });
        }
        samples.push(f.sample(centre, clock));
    }
    Ok(combine(samples))
}

/// The game's wind action for one step: the wind, its stiffness and its stride, and the particles' masses.
///
/// Per particle `i`, with `n` the wind's direction (its four lanes over their length), `s` its speed (the length of its
/// first three) and `x`, `x'` the position and previous position: the force `((-(1 / dt)) * (x - x') + n * s) * c`, added
/// to the forces, where `c = (1 / total mass) * m_i * (|n . N_i| * k)` with the sim's normals `N` (where the sim cloth
/// computes them: the ones the last step ended with), or `(1 / total mass) * m_i * (k * 0.7)` without. An inner particle for which `fraction(stride * i) + stride` is not
/// below 1 takes the force of the particle before it again.
#[derive(Clone, Copy, Debug)]
pub struct WindAction<'a> {
    /// The wind ([`global_wind`]).
    pub wind: V4,
    /// The action's stiffness: how strongly the particles follow the wind.
    pub stiffness: f32,
    /// The action's stride (0: every particle computed).
    pub stride: f32,
    /// The particles' masses.
    pub mass: &'a [f32],
    /// The sim cloth's total mass.
    pub total_mass: f32,
    /// Whether the sim cloth computes particle normals (the state's are then taken).
    pub normals: bool,
}

impl<'a> WindAction<'a> {
    /// Elden Ring 1.17.3's action after cloth initialization, with the sim's own masses and normals policy.
    ///
    /// The action constructor at RVA `0x26b7f60` sets stiffness to 10. Cloth initialization at `0xc4c4e0`
    /// enables global wind through `0x2673920`, then loads 0.6 at `0xc4c5c8` and passes it to `0x26738e0`.
    /// That setter clamps the value to [0, 1] before writing the stride. These are executable policy,
    /// not fields of a cloth tagfile. The game-capture test checks both against every recorded action call.
    pub fn elden_ring(wind: V4, mass: &'a [f32], total_mass: f32, normals: bool) -> Self {
        Self {
            wind,
            stiffness: 10.0,
            stride: 0.6,
            mass,
            total_mass,
            normals,
        }
    }
}

impl Action for WindAction<'_> {
    fn apply(&mut self, state: &SimState, time_step: f32, forces: &mut [V4]) {
        let w = self.wind;
        let speed = (((w.x() * w.x()) + (w.y() * w.y())) + (w.z() * w.z())).sqrt();
        let length = ((((w.x() * w.x()) + (w.y() * w.y())) + (w.z() * w.z())) + (w.w() * w.w())).sqrt();
        let n = if length == 0.0 {
            V4::ZERO
        } else if length.is_infinite() {
            V4::splat(f32::NAN)
        } else {
            V4(w.0.map(|x| x / length))
        };
        let inv_total = 1.0 / self.total_mass;
        let neg_inv_dt = -(1.0 / time_step);
        let wind = n.scale(speed);
        let flat = self.stiffness * 0.7;
        let normals = self.normals.then_some(&state.normals);
        let count = state.positions.len();
        let mut last = V4::ZERO;
        for (i, out) in forces.iter_mut().enumerate().take(count) {
            // an inner particle whose stride sum is not below 1 (or is NaN) repeats the last force
            let sum = fraction(self.stride * i as f32) + self.stride;
            let reuse = i != 0 && i != count - 1 && sum.partial_cmp(&1.0) != Some(std::cmp::Ordering::Less);
            if !reuse {
                let mut c = inv_total * self.mass[i];
                c = match normals {
                    Some(normals) => {
                        let p = n.mul(normals[i]);
                        let dot = (p.y() + p.x()) + p.z();
                        let along = if 0.0 < dot { dot } else { 0.0 - dot };
                        c * (along * self.stiffness)
                    }
                    None => c * flat,
                };
                let velocity = state.positions[i].sub(state.previous[i]);
                last = velocity.scale(neg_inv_dt).add(wind).scale(c);
            }
            *out = last.add(*out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulate::StiffnessModel;

    fn state(positions: Vec<V4>, previous: Vec<V4>) -> SimState {
        SimState {
            normals: vec![V4::ZERO; positions.len()],
            positions,
            previous,
            last_time_step: 0.0,
            damping: 1.0,
            gravity: V4::ZERO,
            global_damping_per_second: 0.0,
            slow_motion: 1.0,
            world_stepping: 1.0,
            stiffness_model: StiffnessModel::Legacy,
            transitions: Vec::new(),
            volumes: Vec::new(),
            collidables: Vec::new(),
            landscape_planes: Vec::new(),
        }
    }

    /// With the stride 0.6 the game's rope of ten particles computes particles 0, 2, 5, 7 and 9, and each of 1, 3, 4,
    /// 6 and 8 takes the force of the particle before it (every call of a capture of Elden Ring's rope shows this
    /// pattern in the forces the game left).
    #[test]
    fn the_stride_repeats_the_forces_the_game_repeats() {
        let positions: Vec<V4> = (0..10).map(|i| V4([i as f32, 0.0, 0.0, 1.0])).collect();
        let previous = vec![V4([0.0, 0.0, 0.0, 1.0]); 10];
        let mass = [0.5; 10];
        let mut forces = vec![V4::ZERO; 10];
        WindAction {
            wind: V4::ZERO,
            stiffness: 10.0,
            stride: 0.6,
            mass: &mass,
            total_mass: 5.0,
            normals: false,
        }
        .apply(&state(positions, previous), 1.0 / 30.0, &mut forces);
        let repeated: Vec<usize> = (1..10).filter(|&i| forces[i] == forces[i - 1]).collect();
        assert_eq!(repeated, [1, 3, 4, 6, 8]);
    }

    /// Without a stride every particle has its own force, and the forces add to the ones given.
    #[test]
    fn without_a_stride_every_particle_is_computed_and_added() {
        let positions: Vec<V4> = (0..4).map(|i| V4([i as f32, 0.0, 0.0, 1.0])).collect();
        let previous = vec![V4([0.0, 0.0, 0.0, 1.0]); 4];
        let mut forces = vec![V4([1.0, 0.0, 0.0, 0.0]); 4];
        WindAction {
            wind: V4::ZERO,
            stiffness: 1.0,
            stride: 0.0,
            mass: &[1.0; 4],
            total_mass: 4.0,
            normals: false,
        }
        .apply(&state(positions, previous), 1.0, &mut forces);
        // c = (1/4) * 1 * (1 * 0.7); the force -(x - x') * c, plus the 1 given
        let c = 0.25_f32 * 0.7;
        for (i, f) in forces.iter().enumerate() {
            assert_eq!(f.x(), (-(i as f32)) * c + 1.0, "particle {i}");
        }
    }

    /// The wind of several fields: the strongest push each way, per lane.
    #[test]
    fn combined_wind_takes_the_largest_and_the_smallest() {
        let w = combine([
            V4([1.0, -2.0, 0.0, 0.0]),
            V4([3.0, 1.0, 0.0, 0.0]),
            V4([-1.0, -1.0, 0.0, 0.0]),
        ]);
        assert_eq!(w, V4([3.0 + -1.0, 1.0 + -2.0, 0.0, 0.0]));
        assert!(combine([V4([f32::NAN, 0.0, 0.0, 0.0])]).x().is_nan());
    }

    #[test]
    fn fractions_keep_the_sign() {
        assert_eq!(fraction(2.5), 0.5);
        assert_eq!(fraction(-2.5), -0.5);
        assert!(fraction(-2.0).is_sign_negative() && fraction(-2.0) == 0.0);
        assert!(fraction(2.0).is_sign_positive());
    }

    #[test]
    fn a_field_of_another_kind_is_refused() {
        let field = ForceField {
            kind: 2,
            local: false,
            origin: [0.0; 3],
            direction: V4([0.0, 0.0, 1.0, 0.0]),
            weight: 1.0,
            fade_scale: 0.0,
            fade_low: 12.0,
            fade_high: 15.0,
            fade_mid: 13.5,
            falloff: None,
            wave_length: 1.0,
            wave_offset: 1.0,
            gust: Gust {
                strength: 0.0,
                length: 1.0,
                scale: 1.0,
                offset: 1.0,
                rise: 5.0,
                fall: 10.0,
                rest: 60.0,
            },
        };
        assert_eq!(
            global_wind(&[field], V4::ZERO, 0.0),
            Err(Unsupported::PerParticle {
                index: 0,
                kind: 2,
                local: false
            })
        );
        let weightless = ForceField { weight: 0.0, ..field };
        assert_eq!(global_wind(&[weightless], V4::ZERO, 0.0), Ok(V4::ZERO));
    }
}
