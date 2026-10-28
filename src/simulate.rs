//! The simulate operator: integration, the constraint solve and collision, substep by substep, for one sim cloth.

use crate::buffer::Buffer;
use crate::collide::{self, RuntimeCollidable};
use crate::constraints::{self, BendRestPose, Particles, Transition, TransitionPeriods};
use crate::matrix::Mat4;
use crate::simd::V4;
use crate::volume::{self, VolumeState};
use havok_formats::cloth::{ConstraintKind, ConstraintSet, SimCloth, SimulateConfig};

/// How a constraint set's stiffness is scaled for a substep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StiffnessModel {
    /// Every set at full stiffness on every substep.
    Legacy,
    /// Scaled by the number of substeps (`adaptConstraintStiffness`).
    Adaptive,
}

/// The run-time state of one sim cloth: what the simulate operator reads and writes from frame to frame.
#[derive(Clone, Debug, PartialEq)]
pub struct SimState {
    /// Current particle positions.
    pub positions: Vec<V4>,
    /// Previous particle positions.
    pub previous: Vec<V4>,
    /// Particle normals (when the sim cloth computes them).
    pub normals: Vec<V4>,
    /// The substep length of the last frame (0 before the first).
    pub last_time_step: f32,
    /// The velocity kept per substep (`(1 - globalDampingPerSecond)^dt`).
    pub damping: f32,
    /// Gravity (the sim cloth's, unless overridden).
    pub gravity: V4,
    /// Global damping per second (the sim cloth's, unless overridden).
    pub global_damping_per_second: f32,
    /// The slow motion factor (1: real time).
    pub slow_motion: f32,
    /// The world stepping factor (1: one step a frame).
    pub world_stepping: f32,
    /// Per constraint set, the transition state of a transition set.
    pub transitions: Vec<Option<Transition>>,
    /// Per constraint set, the state of a volume set (its last frame and eigenvectors).
    pub volumes: Vec<Option<VolumeState>>,
    /// The collidables as the solver moves them.
    pub collidables: Vec<RuntimeCollidable>,
}

impl SimState {
    /// The state of a sim cloth at rest: every particle at its first rest pose position, at rest (previous equals
    /// current), transitions simulated, collidables where the data puts them.
    pub fn new(sim: &SimCloth) -> SimState {
        let positions: Vec<V4> = sim
            .poses
            .first()
            .map(|p| p.positions.iter().map(|&x| V4(x)).collect())
            .unwrap_or_else(|| vec![V4::ZERO; sim.particles.len()]);
        SimState {
            previous: positions.clone(),
            normals: vec![V4::ZERO; positions.len()],
            positions,
            last_time_step: 0.0,
            damping: 1.0,
            gravity: V4(sim.gravity),
            global_damping_per_second: sim.global_damping_per_second,
            slow_motion: 1.0,
            world_stepping: 1.0,
            transitions: sim
                .constraint_sets
                .iter()
                .map(|c| {
                    matches!(c.kind, ConstraintKind::Transition { .. })
                        .then(|| Transition::new(sim.particles.len()))
                })
                .collect(),
            volumes: sim
                .constraint_sets
                .iter()
                .map(|c| matches!(c.kind, ConstraintKind::VolumeMx { .. }).then(VolumeState::default))
                .collect(),
            collidables: sim.collidables.iter().map(RuntimeCollidable::new).collect(),
        }
    }
}

/// The velocity kept per substep of length `dt` for a damping of `per_second`: 1 for none, 0 for 1 or more,
/// `powf(1 - per_second, dt)` between.
pub fn damping(per_second: f32, dt: f32) -> f32 {
    if per_second >= 1.0 {
        0.0
    } else if per_second == 0.0 {
        1.0
    } else {
        (1.0 - per_second).powf(dt)
    }
}

/// What happens before the operators run each frame: the substep length for a frame of `frame_dt` with `sub_steps`
/// substeps; a change of length rescales every particle's previous position so its velocity is kept, and the damping
/// is recomputed.
pub fn prepare(state: &mut SimState, frame_dt: f32, sub_steps: u8, model: StiffnessModel) {
    let slow = match model {
        StiffnessModel::Legacy => 1.0,
        StiffnessModel::Adaptive => state.slow_motion,
    };
    let dt = frame_dt / (f32::from(sub_steps) * slow);
    if state.last_time_step == dt {
        return;
    }
    if state.last_time_step != 0.0 {
        let ratio = 1.0 - dt / state.last_time_step;
        for (cur, prev) in state.positions.iter().zip(state.previous.iter_mut()) {
            *prev = prev.add_scaled(cur.sub(*prev), ratio);
        }
    }
    state.last_time_step = dt;
    state.damping = damping(state.global_damping_per_second, dt);
}

/// A force a game adds to a sim cloth on every substep (a Havok action, such as wind): `apply` adds each particle's
/// force to `forces` (which start at zero on every substep), before integration.
pub trait Action {
    /// Adds this substep's forces, `time_step` long, to `forces` (one per particle).
    fn apply(&mut self, state: &SimState, time_step: f32, forces: &mut [V4]);
}

/// Verlet integration over `dt`: `next = (x + (x - prev)*damping) + (((force + gravity*mass) * invMass) * dt^2)`,
/// then `prev = x`, `x = next`, all four lanes. `forces` holds the actions' forces of this substep (empty: none).
pub fn integrate(state: &mut SimState, mass: &[f32], inv_mass: &[f32], forces: &[V4], dt: f32) {
    let dt2 = dt * dt;
    let k = state.damping;
    let g = state.gravity;
    for (p, (((cur, prev), &m), &im)) in state
        .positions
        .iter_mut()
        .zip(state.previous.iter_mut())
        .zip(mass)
        .zip(inv_mass)
        .enumerate()
    {
        // the force accumulator starts at +0 in every lane: + 0 turns a -0 product into +0
        let force = forces.get(p).copied().unwrap_or(V4::ZERO).add(g.scale(m));
        let step = force.scale(im).scale(dt2);
        let x = *cur;
        let next = x.add_scaled(x.sub(*prev), k).add(step);
        *prev = x;
        *cur = next;
    }
}

/// The stiffness of constraint set `set` on substep `i` of `n`: under the adaptive model with more than one substep
/// (or a slow motion or stepping factor), local range sets act only on the last substep, transition sets ramp with
/// the substep, and the rest are softened by `powf(n * slowMotion * worldStepping, -1.725)`.
pub fn set_stiffness(state: &SimState, model: StiffnessModel, set: &ConstraintSet, i: u32, n: u32) -> f32 {
    if model == StiffnessModel::Legacy || (n == 1 && state.slow_motion == 1.0 && state.world_stepping == 1.0)
    {
        return 1.0;
    }
    match set.kind {
        ConstraintKind::LocalRange { .. } => {
            if i + 1 == n {
                1.0
            } else {
                0.0
            }
        }
        ConstraintKind::Transition { .. } => (i + 1) as f32 / n as f32,
        _ => (n as f32 * state.slow_motion * state.world_stepping).powf(-1.725),
    }
}

/// The particle data the solver reads.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleData {
    /// Masses.
    pub mass: Vec<f32>,
    /// Inverse masses.
    pub inv_mass: Vec<f32>,
    /// Radii.
    pub radius: Vec<f32>,
    /// Frictions.
    pub friction: Vec<f32>,
}

impl ParticleData {
    /// The particle data of a sim cloth.
    pub fn new(sim: &SimCloth) -> ParticleData {
        ParticleData {
            mass: sim.particles.iter().map(|p| p.mass).collect(),
            inv_mass: sim.particles.iter().map(|p| p.inv_mass).collect(),
            radius: sim.particles.iter().map(|p| p.radius).collect(),
            friction: sim.particles.iter().map(|p| p.friction).collect(),
        }
    }
}

/// Applies constraint set `index` of `sim` with stiffness `s`.
#[allow(
    clippy::too_many_arguments,
    reason = "the set, the solver state and the buffers it reads"
)]
fn apply_set(
    sim: &SimCloth,
    index: usize,
    state: &mut SimState,
    data: &ParticleData,
    buffers: &[Buffer],
    s: f32,
    preserve_velocity: bool,
) {
    let Some(set) = sim.constraint_sets.get(index) else {
        return;
    };
    let SimState {
        positions,
        previous,
        transitions,
        volumes,
        ..
    } = state;
    let mut p = Particles {
        positions,
        previous,
        inv_mass: &data.inv_mass,
    };
    match &set.kind {
        ConstraintKind::StandardLink(links) => constraints::standard_links(links, &mut p, s),
        ConstraintKind::StretchLink(links) => constraints::stretch_links(links, &mut p, s),
        ConstraintKind::CompressibleLink(links) => constraints::compressible_links(links, &mut p, s),
        ConstraintKind::BendLink(links) => constraints::bend_links(links, &mut p, s),
        ConstraintKind::BendStiffness {
            links,
            max_rest_pose_height_sq,
            clamp_bend_stiffness,
            use_rest_pose_config,
        } => {
            let rest_pose = if *use_rest_pose_config {
                BendRestPose::Curved {
                    clamp: clamp_bend_stiffness.then_some(*max_rest_pose_height_sq),
                }
            } else {
                BendRestPose::Flat
            };
            constraints::bend_stiffness(links, rest_pose, &mut p, s);
        }
        ConstraintKind::LocalRange {
            reference_mesh_buffer,
            stiffness,
            shape_type,
            apply_normal_component,
            constraints: entries,
        } => {
            if let Some(reference) = usize::try_from(*reference_mesh_buffer)
                .ok()
                .and_then(|b| buffers.get(b))
            {
                constraints::local_range(
                    entries,
                    *shape_type,
                    *apply_normal_component,
                    *stiffness,
                    reference,
                    &mut p,
                    s,
                    preserve_velocity,
                );
            }
        }
        ConstraintKind::Transition {
            particles,
            to_anim_period,
            to_sim_period,
            reference_mesh_buffer,
            ..
        } => {
            let reference = usize::try_from(*reference_mesh_buffer)
                .ok()
                .and_then(|b| buffers.get(b));
            if let (Some(reference), Some(Some(instance))) = (reference, transitions.get_mut(index)) {
                let periods = TransitionPeriods {
                    to_anim: *to_anim_period,
                    to_sim: *to_sim_period,
                };
                constraints::transition(
                    particles,
                    periods,
                    reference,
                    instance,
                    &mut p,
                    s,
                    preserve_velocity,
                );
            }
        }
        ConstraintKind::VolumeMx {
            frame_batches,
            frame_singles,
            apply_batches,
            apply_singles,
        } => {
            if let Some(Some(v)) = volumes.get_mut(index) {
                volume::apply(
                    p.positions,
                    v,
                    frame_batches,
                    frame_singles,
                    apply_batches,
                    apply_singles,
                    s,
                );
            }
        }
        ConstraintKind::Other { .. } => {}
    }
}

/// Sets each per-instance collidable's velocities to carry it, over `time_step`, to its bone in `bones` (the transform
/// set the sim cloth's collidable transform map names) times its offset.
pub fn drive_collidables(sim: &SimCloth, state: &mut SimState, bones: &[Mat4], time_step: f32) {
    let Some(map) = &sim.collidable_transform_map else {
        return;
    };
    for ((collidable, &bone), offset) in state
        .collidables
        .iter_mut()
        .zip(&map.transform_indices)
        .zip(&map.offsets)
    {
        let Some(world_from_bone) = usize::try_from(bone).ok().and_then(|b| bones.get(b)) else {
            continue;
        };
        let target = world_from_bone.mul_affine(&Mat4::from_columns(*offset));
        collidable.drive_to(&target, time_step);
    }
}

/// One frame of the simulate operator for `sim`: `frame_dt` long, with `config` and its stiffness model. `bones` is
/// the transform set that drives the collidables (when the sim cloth maps them to bones); `actions` add forces on
/// every substep, in order.
#[allow(
    clippy::too_many_arguments,
    reason = "the operator's data, the sim cloth's state and the frame's inputs"
)]
pub fn simulate(
    sim: &SimCloth,
    config: &SimulateConfig,
    state: &mut SimState,
    data: &ParticleData,
    buffers: &[Buffer],
    bones: &[Mat4],
    actions: &mut [&mut dyn Action],
    frame_dt: f32,
) {
    let model = if config.adapt_constraint_stiffness {
        StiffnessModel::Adaptive
    } else {
        StiffnessModel::Legacy
    };
    let n = u32::from(config.sub_steps.max(1));
    let time_step = frame_dt / state.slow_motion;
    let sub = time_step / n as f32;
    let preserve_velocity = model == StiffnessModel::Adaptive
        && (n != 1 || state.slow_motion != 1.0 || state.world_stepping != 1.0);
    drive_collidables(sim, state, bones, time_step);
    let mut runtime = state.collidables.clone();
    let fixed: Vec<(usize, V4, V4)> = sim
        .fixed_particles
        .iter()
        .map(|&f| {
            let f = usize::from(f);
            (f, state.previous[f], state.positions[f])
        })
        .collect();
    for i in 0..n {
        for c in &mut runtime {
            c.step_forward(sub);
        }
        let t = (i + 1) as f32 / n as f32;
        for &(f, prev, cur) in &fixed {
            let blended = prev.scale(1.0 - t).add(cur.scale(t));
            state.previous[f] = blended;
            state.positions[f] = blended;
        }
        let mut forces = Vec::new();
        if !actions.is_empty() {
            forces = vec![V4::ZERO; state.positions.len()];
            for action in actions.iter_mut() {
                action.apply(state, sub, &mut forces);
            }
        }
        integrate(state, &data.mass, &data.inv_mass, &forces, sub);
        for _ in 0..config.solve_iterations {
            let collide_and_solve = |state: &mut SimState| collide::collide(sim, &runtime, state, data, sub);
            if config.constraint_execution.is_empty() {
                for set in 0..sim.constraint_sets.len() {
                    let s = set_stiffness(state, model, &sim.constraint_sets[set], i, n);
                    apply_set(sim, set, state, data, buffers, s, preserve_velocity);
                }
                collide_and_solve(state);
            } else {
                for &entry in &config.constraint_execution {
                    match usize::try_from(entry) {
                        Ok(set) => {
                            let Some(constraint) = sim.constraint_sets.get(set) else {
                                continue;
                            };
                            let s = set_stiffness(state, model, constraint, i, n);
                            apply_set(sim, set, state, data, buffers, s, preserve_velocity);
                        }
                        Err(_) => collide_and_solve(state),
                    }
                }
            }
        }
    }
    if sim.collidable_transform_map.is_some() {
        for (instance, stepped) in state.collidables.iter_mut().zip(&runtime) {
            instance.transform = stepped.transform;
        }
    }
    if sim.do_normals {
        update_normals(
            &sim.triangle_indices,
            &sim.triangle_flips,
            &state.positions,
            &mut state.normals,
        );
    }
}

/// The particle normals: every triangle's cross product `(B - A) x (C - A)`, its sign flipped where the triangle's bit
/// of `flips` is set (the most significant bit of a byte is the first of its eight triangles), added to the normals of
/// its three particles in triangle order, then each normal fast-normalised.
pub fn update_normals(triangles: &[u16], flips: &[u8], positions: &[V4], normals: &mut [V4]) {
    normals.fill(V4::ZERO);
    for (t, tri) in triangles.as_chunks::<3>().0.iter().enumerate() {
        let [a, b, c] = tri.map(usize::from);
        let pa = positions[a];
        let mut n = positions[b].sub(pa).cross(positions[c].sub(pa));
        let byte = flips.get(t / 8).copied().unwrap_or(0);
        if byte & (0x80 >> (t % 8)) != 0 {
            n = n.scale(-1.0);
        }
        for v in [a, b, c] {
            normals[v] = normals[v].add(n);
        }
    }
    for n in normals.iter_mut() {
        *n = n.normalized_fast();
    }
}
