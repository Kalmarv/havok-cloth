//! The constraint sets: each moves particle positions (and, when velocity is preserved, previous positions) towards
//! what the set asks for, scaled by the set's stiffness for this substep.

use crate::buffer::Buffer;
use crate::simd::{V4, min_ps, rcp23, rcp23_or_zero};
use havok_formats::cloth::{
    BendLink, BendStiffnessLink, CompressibleLink, Link, LocalRange, LocalRangeShape, TransitionParticle,
};

/// `FLT_EPSILON` in every lane: what the local range constraints add to a difference before normalising it.
const EPSILON4: V4 = V4::splat(f32::EPSILON);

/// The particles a constraint set moves.
#[derive(Debug)]
pub struct Particles<'a> {
    /// Current positions.
    pub positions: &'a mut [V4],
    /// Previous positions.
    pub previous: &'a mut [V4],
    /// Each particle's inverse mass.
    pub inv_mass: &'a [f32],
}

/// Two particles' positions moved apart (`f > 0` pulls them together) along the unit direction `n` from A to B,
/// each by its inverse mass: `d = n*f`, `A += d*invMassA`, `B -= d*invMassB`.
fn move_pair(p: &mut Particles<'_>, a: usize, b: usize, n: V4, f: f32) {
    let d = n.scale(f);
    let da = d.scale(p.inv_mass[a]);
    let db = d.scale(p.inv_mass[b]);
    p.positions[a] = p.positions[a].add(da);
    p.positions[b] = p.positions[b].sub(db);
}

/// The unit direction from particle `a` to `b` and their distance, by the fast normalisation.
fn direction(p: &Particles<'_>, a: usize, b: usize) -> (V4, f32) {
    p.positions[b].sub(p.positions[a]).normalized_fast_with_length()
}

/// Standard links: each pulls its two particles towards its rest length, `f = ((L - rest) * linkStiffness) * s`.
pub fn standard_links(links: &[Link], p: &mut Particles<'_>, s: f32) {
    if s <= 0.0 {
        return;
    }
    for link in links {
        let (a, b) = (usize::from(link.particle_a), usize::from(link.particle_b));
        let (n, length) = direction(p, a, b);
        let f = ((length - link.rest_length) * link.stiffness) * s;
        move_pair(p, a, b, n, f);
    }
}

/// Stretch links: only particle B moves, and only when the link is longer than its rest length:
/// `B += n * ((min(rest - L, 0) * linkStiffness) * s)`.
pub fn stretch_links(links: &[Link], p: &mut Particles<'_>, s: f32) {
    if s <= 0.0 {
        return;
    }
    for link in links {
        let (a, b) = (usize::from(link.particle_a), usize::from(link.particle_b));
        let (n, length) = direction(p, a, b);
        let error = min_ps(link.rest_length - length, 0.0);
        let f = (error * link.stiffness) * s;
        p.positions[b] = p.positions[b].add_scaled(n, f);
    }
}

/// Compressible links: no force between the compression length and the rest length; beyond the rest length they pull
/// to it, below the compression length they push to that.
pub fn compressible_links(links: &[CompressibleLink], p: &mut Particles<'_>, s: f32) {
    if s <= 0.0 {
        return;
    }
    for link in links {
        let (a, b) = (usize::from(link.particle_a), usize::from(link.particle_b));
        let (n, length) = direction(p, a, b);
        let compressed = length < link.compression_length;
        if length > link.rest_length || compressed {
            let target = if compressed {
                link.compression_length
            } else {
                link.rest_length
            };
            let f = ((length - target) * link.stiffness) * s;
            move_pair(p, a, b, n, f);
        }
    }
}

/// Bend links: a link shorter than its bend length pushes out by its bend stiffness, one longer than its stretch length
/// pulls in by its stretch stiffness: `f = (max(L - stretchMax, 0)*stretchStiffness - max(bendMin - L, 0)*bendStiffness) * s`.
pub fn bend_links(links: &[BendLink], p: &mut Particles<'_>, s: f32) {
    if s <= 0.0 {
        return;
    }
    for link in links {
        let (a, b) = (usize::from(link.particle_a), usize::from(link.particle_b));
        let (n, length) = direction(p, a, b);
        let bend = crate::simd::max_ps(0.0, link.bend_min_length - length);
        let stretch = crate::simd::max_ps(0.0, length - link.stretch_max_length);
        let f = (stretch * link.stretch_stiffness - bend * link.bend_stiffness) * s;
        move_pair(p, a, b, n, f);
    }
}

/// How a bend stiffness set treats its rest curvature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BendRestPose {
    /// `useRestPoseConfig` false: no rest curvature.
    Flat,
    /// `useRestPoseConfig` true; `clamp` holds `maxRestPoseHeightSq` when `clampBendStiffness` is set.
    Curved {
        /// The squared height beyond which a link's stiffness is dropped, when clamping.
        clamp: Option<f32>,
    },
}

/// The bending vector of a link: `((wA*pA + wB*pB) + wC*pC) + wD*pD`.
fn bending_vector(link: &BendStiffnessLink, positions: &[V4], ix: [usize; 4]) -> V4 {
    let [w0, w1, w2, w3] = link.weights;
    positions[ix[0]]
        .scale(w0)
        .add_scaled(positions[ix[1]], w1)
        .add_scaled(positions[ix[2]], w2)
        .add_scaled(positions[ix[3]], w3)
}

/// Bend stiffness: each link moves its four particles along its bending vector, particle `X` by
/// `((stiffness * weightX) * invMassX)`, in the order A, B, C, D.
pub fn bend_stiffness(links: &[BendStiffnessLink], rest_pose: BendRestPose, p: &mut Particles<'_>, s: f32) {
    if s <= 0.0 {
        return;
    }
    for link in links {
        let ix = link.particles.map(usize::from);
        let mut stiffness = link.bend_stiffness * s;
        let mut r = bending_vector(link, p.positions, ix);
        if let BendRestPose::Curved { clamp } = rest_pose {
            let [pa, pb, pc, pd] = ix.map(|i| p.positions[i]);
            let ca = pa.sub(pc);
            let cb = pb.sub(pc);
            let cd = pd.sub(pc);
            let (nc, lc) = cd.cross(ca).normalized_fast_with_length();
            let (nd, ld) = cb.cross(cd).normalized_fast_with_length();
            let offset = nc.add(nd).normalized_fast();
            let inv_base_sq = rcp23_or_zero(cd.dot3(cd));
            let current = (lc * ld) * inv_base_sq;
            if let Some(max_height_sq) = clamp {
                let ratio = link.rest_curvature * current;
                if ratio * ratio > max_height_sq {
                    stiffness = 0.0;
                }
            }
            r = r.add_scaled(offset.scale(current), link.rest_curvature);
        }
        for (&i, &w) in ix.iter().zip(&link.weights) {
            let k = (stiffness * w) * p.inv_mass[i];
            p.positions[i] = p.positions[i].add_scaled(r, k);
        }
    }
}

/// The previous position kept so that a constraint's correction does not change the particle's velocity.
fn preserved_velocity(p: &Particles<'_>, i: usize, preserve: bool) -> Option<V4> {
    preserve.then(|| p.positions[i].sub(p.previous[i]))
}

fn restore_velocity(p: &mut Particles<'_>, i: usize, velocity: Option<V4>) {
    if let Some(v) = velocity {
        p.previous[i] = p.positions[i].sub(v);
    }
}

/// Local range constraints: each keeps a particle within `maximumDistance` of a vertex of the reference buffer, and,
/// with normals, between `minNormalDistance` and `maxNormalDistance` along that vertex's normal (a sphere), or within
/// `maximumDistance` of the normal's line (a cylinder). `stiffness` is the set's; `s` the substep's.
#[allow(
    clippy::too_many_arguments,
    reason = "the set's members and the solver state, as the set applies them"
)]
pub fn local_range(
    constraints: &[LocalRange],
    shape: LocalRangeShape,
    apply_normal_component: bool,
    stiffness: f32,
    reference: &Buffer,
    p: &mut Particles<'_>,
    s: f32,
    preserve_velocity: bool,
) {
    if s <= 0.0 {
        return;
    }
    let k = stiffness * s;
    let with_normals = apply_normal_component && !reference.normals.is_empty();
    for c in constraints {
        let i = usize::from(c.particle);
        let v = usize::from(c.reference_vertex);
        let Some(origin) = reference.world_position(v) else {
            continue;
        };
        let velocity = preserved_velocity(p, i, preserve_velocity);
        let mut out = p.positions[i];
        if with_normals {
            let normal = reference.world_normal(v).unwrap_or(V4::ZERO);
            match shape {
                LocalRangeShape::Sphere => {
                    let (n, distance) = out.sub(origin).add(EPSILON4).normalized_fast_with_length();
                    let error = min_ps((c.maximum_distance - distance) * k, 0.0);
                    out = out.add_scaled(n, error);
                    let distance = distance + error;
                    let along = distance * n.dot3(normal);
                    let below = min_ps(along - c.min_normal_distance, 0.0);
                    out = out.sub_scaled(normal, below);
                    let above = min_ps(c.max_normal_distance - along, 0.0);
                    out = out.add_scaled(normal, above);
                }
                LocalRangeShape::Cylinder => {
                    let along = out.sub(origin).dot3(normal);
                    let centre = origin.add_scaled(normal, along);
                    let (n, distance) = out.sub(centre).add(EPSILON4).normalized_fast_with_length();
                    let error = min_ps((c.maximum_distance - distance) * k, 0.0);
                    out = out.add_scaled(n, error);
                    let below = min_ps((along - c.min_normal_distance) * k, 0.0);
                    out = out.sub_scaled(normal, below);
                    let above = min_ps((c.max_normal_distance - along) * k, 0.0);
                    out = out.add_scaled(normal, above);
                }
            }
        } else {
            let (n, distance) = out.sub(origin).add(EPSILON4).normalized_fast_with_length();
            let error = min_ps((c.maximum_distance - distance) * k, 0.0);
            out = out.add_scaled(n, error);
        }
        p.positions[i] = out;
        restore_velocity(p, i, velocity);
    }
}

/// Where a transition constraint set is between animation and simulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransitionState {
    /// Simulated: the set does nothing.
    Simulated,
    /// Animated: every particle sits on its reference vertex.
    Animated,
    /// Blending from simulation to animation.
    ToAnimation,
    /// Blending from animation to simulation.
    ToSimulation,
}

/// A transition constraint set's state for one sim cloth instance.
#[derive(Clone, Debug, PartialEq)]
pub struct Transition {
    /// The state.
    pub state: TransitionState,
    /// Time since the blend began.
    pub time: f32,
    /// Per particle, the distance at which a particle started blending to animation (`f32::MAX` until then).
    pub max_distance: Vec<f32>,
}

impl Transition {
    /// A set in the simulated state, for a sim cloth of `particles` particles.
    pub fn new(particles: usize) -> Transition {
        Transition {
            state: TransitionState::Simulated,
            time: 0.0,
            max_distance: vec![f32::MAX; particles],
        }
    }

    /// Begin blending to simulation.
    pub fn to_simulation(&mut self) {
        self.state = TransitionState::ToSimulation;
        self.time = 0.0;
    }

    /// Begin blending to animation.
    pub fn to_animation(&mut self) {
        self.state = TransitionState::ToAnimation;
        self.time = 0.0;
        self.max_distance.fill(f32::MAX);
    }

    /// Advance a blend by `dt`, ending it after the set's period plus its longest delay.
    pub fn step(&mut self, dt: f32, to_anim_plus_delay: f32, to_sim_plus_delay: f32) {
        match self.state {
            TransitionState::ToAnimation => {
                self.time += dt;
                if self.time > to_anim_plus_delay {
                    self.state = TransitionState::Animated;
                }
            }
            TransitionState::ToSimulation => {
                self.time += dt;
                if self.time > to_sim_plus_delay {
                    self.state = TransitionState::Simulated;
                }
            }
            TransitionState::Simulated | TransitionState::Animated => {}
        }
    }
}

/// The members of a transition constraint set the solve uses.
#[derive(Clone, Copy, Debug)]
pub struct TransitionPeriods {
    /// `toAnimPeriod`.
    pub to_anim: f32,
    /// `toSimPeriod`.
    pub to_sim: f32,
}

/// Transition constraints: in the animated state each particle sits on its reference vertex; while blending, a
/// particle is pulled to within a distance of its reference vertex that shrinks (to animation) or grows (to
/// simulation) over the period, after the particle's delay. The game divides here where other constraints take the
/// refined reciprocal: the bound is `((1 - (1/period)*t) * startDistance) * (1/s)` blending to animation and
/// `(((1/period)*t) * toSimMaxDistance) * (1/s)` blending to simulation, and a particle beyond it moves by
/// `(bound - distance) / distance` of its offset from the reference vertex.
///
/// Blending to simulation, the first particle whose blend has finished ends the pass: the particles after it in the
/// set's list are left alone for that solve.
#[allow(
    clippy::too_many_arguments,
    reason = "the set's members and the solver state, as the set applies them"
)]
pub fn transition(
    particles: &[TransitionParticle],
    periods: TransitionPeriods,
    reference: &Buffer,
    instance: &mut Transition,
    p: &mut Particles<'_>,
    s: f32,
    preserve_velocity: bool,
) {
    match instance.state {
        TransitionState::Simulated => return,
        TransitionState::ToAnimation | TransitionState::ToSimulation if s <= 0.0 => return,
        _ => {}
    }
    let inv_stiffness = 1.0 / s;
    let time = instance.time;
    match instance.state {
        TransitionState::Simulated => {}
        TransitionState::Animated => {
            for pp in particles {
                if let Some(origin) = reference.world_position(usize::from(pp.reference_vertex)) {
                    p.positions[usize::from(pp.particle)] = origin;
                }
            }
        }
        TransitionState::ToAnimation => {
            let period = periods.to_anim;
            let per_period = 1.0 / period;
            for pp in particles {
                let i = usize::from(pp.particle);
                let Some(origin) = reference.world_position(usize::from(pp.reference_vertex)) else {
                    continue;
                };
                let velocity = preserved_velocity(p, i, preserve_velocity);
                let diff = p.positions[i].sub(origin);
                let distance = diff.length3();
                let t = time - pp.to_anim_delay;
                if t <= 0.0 {
                    if let Some(m) = instance.max_distance.get_mut(i) {
                        *m = distance;
                    }
                } else if t >= period {
                    p.positions[i] = origin;
                } else {
                    let start = instance.max_distance.get(i).copied().unwrap_or(f32::MAX);
                    let max_distance = ((1.0 - per_period * t) * start) * inv_stiffness;
                    if distance > max_distance {
                        let step = (max_distance - distance) / distance;
                        p.positions[i] = p.positions[i].add_scaled(diff, step);
                        restore_velocity(p, i, velocity);
                    }
                }
            }
        }
        TransitionState::ToSimulation => {
            let period = periods.to_sim;
            let per_period = 1.0 / period;
            for pp in particles {
                let i = usize::from(pp.particle);
                let Some(origin) = reference.world_position(usize::from(pp.reference_vertex)) else {
                    continue;
                };
                let t = time - pp.to_sim_delay;
                if t <= 0.0 {
                    p.positions[i] = origin;
                } else if t >= period {
                    break;
                } else {
                    let velocity = preserved_velocity(p, i, preserve_velocity);
                    let diff = p.positions[i].sub(origin);
                    let distance = diff.length3();
                    let max_distance = ((per_period * t) * pp.to_sim_max_distance) * inv_stiffness;
                    if distance > max_distance {
                        let step = (max_distance - distance) / distance;
                        p.positions[i] = p.positions[i].add_scaled(diff, step);
                        restore_velocity(p, i, velocity);
                    }
                }
            }
        }
    }
}
