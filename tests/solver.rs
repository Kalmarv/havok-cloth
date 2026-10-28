//! The solver's steps on hand-built cloths, against expectations computed independently: closed forms in f64,
//! conservation laws, and what each constraint promises (a link restored to its length, a pinned particle unmoved, a
//! particle kept in its range).
#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    reason = "test fixtures; exact comparisons are the point where they appear"
)]

use havok_cloth::buffer::Buffer;
use havok_cloth::collide::{self, RuntimeCollidable, WorldShape};
use havok_cloth::constraints::{
    self, BendRestPose, Particles, Transition, TransitionPeriods, TransitionState,
};
use havok_cloth::matrix::Mat4;
use havok_cloth::simd::V4;
use havok_cloth::simulate::{self, ParticleData, SimState, StiffnessModel};
use havok_formats::cloth::{
    BendLink, BendStiffnessLink, CompressibleLink, ConstraintKind, ConstraintSet, Link, LocalRange,
    LocalRangeShape, Particle, Shape, SimCloth, SimulateConfig, TransitionParticle,
};

/// The fast normalisation's relative error bound (the 12-bit reciprocal square root estimate).
const FAST: f64 = 1.0 / 2048.0;

fn v(x: f32, y: f32, z: f32) -> V4 {
    V4::new(x, y, z, 0.0)
}

fn dist(a: V4, b: V4) -> f64 {
    let d = a.sub(b).0;
    d[..3]
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt()
}

fn particles<'a>(positions: &'a mut [V4], previous: &'a mut [V4], inv_mass: &'a [f32]) -> Particles<'a> {
    Particles {
        positions,
        previous,
        inv_mass,
    }
}

fn link(a: u16, b: u16, rest: f32, stiffness: f32) -> Link {
    Link {
        particle_a: a,
        particle_b: b,
        rest_length: rest,
        stiffness,
    }
}

/// A sim cloth of `masses.len()` particles at `positions`, the first `fixed` of them pinned.
fn sim_cloth(positions: &[V4], masses: &[f32], sets: Vec<ConstraintKind>) -> SimCloth {
    let fixed: Vec<u16> = masses
        .iter()
        .enumerate()
        .filter(|&(_, &m)| m == 0.0)
        .map(|(i, _)| u16::try_from(i).unwrap())
        .collect();
    SimCloth {
        name: "test".into(),
        gravity: [0.0, -10.0, 0.0, 0.0],
        global_damping_per_second: 0.0,
        particles: masses
            .iter()
            .map(|&m| Particle {
                mass: m,
                inv_mass: if m == 0.0 { 0.0 } else { 1.0 / m },
                radius: 0.01,
                friction: 0.0,
            })
            .collect(),
        fixed_particles: fixed,
        do_normals: false,
        sim_op_ids: Vec::new(),
        poses: vec![havok_formats::cloth::Pose {
            name: "rest".into(),
            positions: positions.iter().map(|p| p.0).collect(),
        }],
        constraint_sets: sets
            .into_iter()
            .map(|kind| ConstraintSet {
                name: String::new(),
                constraint_id: None,
                kind,
            })
            .collect(),
        anti_pinch_constraint_sets: Vec::new(),
        collidable_transform_map: None,
        collidables: Vec::new(),
        max_particle_radius: 0.01,
        static_collision_masks: vec![u32::MAX; masses.len()],
        actions: Vec::new(),
        total_mass: masses.iter().sum(),
        transfer_motion: havok_formats::cloth::TransferMotion {
            transform_set: 0,
            transform: 0,
            transfer_translation: false,
            min_translation_speed: 0.0,
            max_translation_speed: 0.0,
            min_translation_blend: 0.0,
            max_translation_blend: 1.0,
            transfer_rotation: false,
            min_rotation_speed: 0.0,
            max_rotation_speed: 0.0,
            min_rotation_blend: 0.0,
            max_rotation_blend: 1.0,
        },
        transfer_motion_enabled: false,
        landscape_collision_enabled: false,
        landscape_collision: havok_formats::cloth::LandscapeCollision {
            radius: 0.0,
            stuck_particle_detection: false,
            stuck_particles_stretch_factor_sq: 9.0,
            pinch_detection: false,
            pinch_detection_priority: 0,
            pinch_detection_radius: 0.0,
            collision_tolerance: 0.2,
        },
        landscape_collidable_particles: 0,
        triangle_indices: Vec::new(),
        triangle_flips: Vec::new(),
        pinch_detection_enabled: false,
        per_particle_pinch_detection: Vec::new(),
        collidable_pinching: Vec::new(),
        min_pinched_particle: 0,
        max_pinched_particle: 0,
        max_collision_pairs: 0,
        virtual_collision_points: Default::default(),
    }
}

fn config(sub_steps: u8, execution: Vec<i32>, adapt: bool) -> SimulateConfig {
    SimulateConfig {
        name: "Default Config".into(),
        constraint_execution: execution,
        instance_collidables_used: Vec::new(),
        sub_steps,
        solve_iterations: 1,
        use_all_instance_collidables: true,
        adapt_constraint_stiffness: adapt,
    }
}

// ---- integration --------------------------------------------------------------------------------------------

#[test]
fn free_fall_follows_the_verlet_closed_form_exactly() {
    // dt = 1/64, g = -8, mass 2: every product and sum is a multiple of 2^-10 well inside f32's range, so the f32
    // integration is exact and must equal the closed form x_n = x_0 + g dt^2 n(n+1)/2 to the bit
    let mut sim = sim_cloth(&[v(0.0, 0.0, 0.0)], &[2.0], Vec::new());
    sim.gravity = [0.0, -8.0, 0.0, 0.0];
    let mut state = SimState::new(&sim);
    let data = ParticleData::new(&sim);
    let dt = 1.0f32 / 64.0;
    state.damping = 1.0;
    for n in 1..=120u32 {
        simulate::integrate(&mut state, &data.mass, &data.inv_mass, &[], dt);
        let want = -8.0 * f64::from(dt) * f64::from(dt) * f64::from(n * (n + 1)) / 2.0;
        assert_eq!(f64::from(state.positions[0].y()), want, "step {n}");
        assert_eq!(state.positions[0].x(), 0.0);
    }
}

#[test]
fn a_pinned_particle_never_moves_under_integration() {
    let sim = sim_cloth(&[v(1.0, 2.0, 3.0)], &[0.0], Vec::new());
    let mut state = SimState::new(&sim);
    let data = ParticleData::new(&sim);
    for _ in 0..100 {
        simulate::integrate(&mut state, &data.mass, &data.inv_mass, &[], 1.0 / 30.0);
    }
    assert_eq!(state.positions[0], v(1.0, 2.0, 3.0));
    assert_eq!(state.previous[0], v(1.0, 2.0, 3.0));
}

#[test]
fn damping_scales_the_velocity_each_step() {
    let mut sim = sim_cloth(&[v(0.0, 0.0, 0.0)], &[1.0], Vec::new());
    sim.gravity = [0.0; 4];
    let mut state = SimState::new(&sim);
    let data = ParticleData::new(&sim);
    state.previous[0] = v(-1.0, 0.0, 0.0); // velocity 1 per step in x
    state.damping = 0.5;
    let mut last = state.positions[0].x();
    for n in 1..=10 {
        simulate::integrate(&mut state, &data.mass, &data.inv_mass, &[], 0.1);
        let x = state.positions[0].x();
        assert_eq!(f64::from(x - last), 0.5f64.powi(n), "step {n}");
        last = x;
    }
}

#[test]
fn the_damping_factor_is_none_full_or_a_power() {
    assert_eq!(simulate::damping(0.0, 0.016), 1.0);
    assert_eq!(simulate::damping(1.0, 0.016), 0.0);
    assert_eq!(simulate::damping(1.5, 0.016), 0.0);
    let k = f64::from(simulate::damping(0.1, 0.5));
    assert!((k - 0.9f64.powf(0.5)).abs() < 1e-6);
}

#[test]
fn a_new_timestep_keeps_the_velocity_per_second() {
    let sim = sim_cloth(&[v(0.0, 0.0, 0.0)], &[1.0], Vec::new());
    let mut state = SimState::new(&sim);
    simulate::prepare(&mut state, 1.0 / 30.0, 1, StiffnessModel::Legacy);
    assert_eq!(state.last_time_step, 1.0 / 30.0);
    state.previous[0] = v(-0.1, 0.0, 0.0); // 0.1 per 1/30 s = 3 units per second
    simulate::prepare(&mut state, 1.0 / 60.0, 1, StiffnessModel::Legacy);
    let per_step = f64::from(state.positions[0].x() - state.previous[0].x());
    assert!((per_step - 0.05).abs() < 1e-8, "{per_step}");
    // substeps divide the frame
    simulate::prepare(&mut state, 1.0 / 30.0, 2, StiffnessModel::Legacy);
    assert_eq!(state.last_time_step, 1.0 / 60.0);
}

// ---- links --------------------------------------------------------------------------------------------------

#[test]
fn a_standard_link_restores_its_length_and_keeps_the_centre_of_mass() {
    // masses 1 and 3: the stiffness that restores the length in one pass is 1 / (invA + invB)
    let mut pos = [v(0.0, 0.0, 0.0), v(2.0, 1.0, 0.0)];
    let mut prev = pos;
    let inv = [1.0, 1.0 / 3.0];
    let stiffness = 1.0 / (inv[0] + inv[1]);
    let com = |p: &[V4]| {
        (
            f64::from(p[0].x()) + 3.0 * f64::from(p[1].x()),
            f64::from(p[0].y()) + 3.0 * f64::from(p[1].y()),
        )
    };
    let before = com(&pos);
    constraints::standard_links(
        &[link(0, 1, 1.0, stiffness)],
        &mut particles(&mut pos, &mut prev, &inv),
        1.0,
    );
    let l = dist(pos[0], pos[1]);
    assert!((l - 1.0).abs() < 4.0 * FAST * 5f64.sqrt(), "{l}");
    let after = com(&pos);
    assert!((after.0 - before.0).abs() < 1e-6 && (after.1 - before.1).abs() < 1e-6);
    // the previous positions are not touched
    assert_eq!(prev, [v(0.0, 0.0, 0.0), v(2.0, 1.0, 0.0)]);
}

#[test]
fn a_link_to_a_pinned_particle_moves_only_the_free_one() {
    let mut pos = [v(0.0, 0.0, 0.0), v(0.0, -3.0, 0.0)];
    let mut prev = pos;
    let inv = [0.0, 1.0];
    constraints::standard_links(
        &[link(0, 1, 1.0, 1.0)],
        &mut particles(&mut pos, &mut prev, &inv),
        1.0,
    );
    assert_eq!(pos[0], V4::ZERO);
    assert!((dist(pos[0], pos[1]) - 1.0).abs() < 3.0 * FAST * 3.0);
    assert_eq!(pos[1].x(), 0.0);
}

#[test]
fn a_zero_stiffness_does_nothing() {
    let mut pos = [v(0.0, 0.0, 0.0), v(5.0, 0.0, 0.0)];
    let mut prev = pos;
    let inv = [1.0, 1.0];
    let before = pos;
    constraints::standard_links(
        &[link(0, 1, 1.0, 0.5)],
        &mut particles(&mut pos, &mut prev, &inv),
        0.0,
    );
    constraints::stretch_links(
        &[link(0, 1, 1.0, 1.0)],
        &mut particles(&mut pos, &mut prev, &inv),
        -1.0,
    );
    assert_eq!(pos, before);
}

#[test]
fn a_stretch_link_only_shortens_and_only_moves_its_second_particle() {
    let inv = [1.0, 1.0];
    let mut pos = [v(0.0, 0.0, 0.0), v(0.0, 0.5, 0.0)];
    let mut prev = pos;
    // shorter than its rest length: left alone
    constraints::stretch_links(
        &[link(0, 1, 1.0, 1.0)],
        &mut particles(&mut pos, &mut prev, &inv),
        1.0,
    );
    assert_eq!(pos, [v(0.0, 0.0, 0.0), v(0.0, 0.5, 0.0)]);
    // longer: B comes back to the rest length, A stays
    let mut pos = [v(1.0, 1.0, 1.0), v(1.0, 5.0, 1.0)];
    constraints::stretch_links(
        &[link(0, 1, 2.0, 1.0)],
        &mut particles(&mut pos, &mut prev, &inv),
        1.0,
    );
    assert_eq!(pos[0], v(1.0, 1.0, 1.0));
    assert!((dist(pos[0], pos[1]) - 2.0).abs() < 4.0 * 4.0 * FAST);
}

#[test]
fn a_compressible_link_acts_only_outside_its_band() {
    let inv = [1.0, 1.0];
    let l = CompressibleLink {
        particle_a: 0,
        particle_b: 1,
        rest_length: 2.0,
        compression_length: 1.0,
        stiffness: 0.5,
    };
    for (length, want) in [(1.5f32, 1.5f64), (3.0, 2.0), (0.5, 1.0)] {
        let mut pos = [v(0.0, 0.0, 0.0), v(length, 0.0, 0.0)];
        let mut prev = pos;
        constraints::compressible_links(&[l], &mut particles(&mut pos, &mut prev, &inv), 1.0);
        let got = dist(pos[0], pos[1]);
        assert!((got - want).abs() < 4.0 * want * FAST, "{length}: {got}");
    }
}

#[test]
fn a_bend_link_pushes_below_its_bend_length_and_pulls_above_its_stretch_length() {
    let inv = [1.0, 1.0];
    let l = BendLink {
        particle_a: 0,
        particle_b: 1,
        bend_min_length: 1.0,
        stretch_max_length: 2.0,
        bend_stiffness: 0.5,
        stretch_stiffness: 0.5,
    };
    for (length, want) in [(1.5f32, 1.5f64), (3.0, 2.0), (0.5, 1.0)] {
        let mut pos = [v(0.0, 0.0, 0.0), v(0.0, 0.0, length)];
        let mut prev = pos;
        constraints::bend_links(&[l], &mut particles(&mut pos, &mut prev, &inv), 1.0);
        let got = dist(pos[0], pos[1]);
        assert!((got - want).abs() < 4.0 * want * FAST, "{length}: {got}");
    }
}

// ---- bend stiffness ---------------------------------------------------------------------------------------------

fn hinge(a_height: f32) -> [V4; 4] {
    // A and B on either side of the edge C-D (along z); flat when A's height is 0
    [
        v(-1.0, a_height, 0.5),
        v(1.0, 0.0, 0.5),
        v(0.0, 0.0, 0.0),
        v(0.0, 0.0, 1.0),
    ]
}

#[test]
fn bend_stiffness_leaves_a_flat_hinge_alone_and_conserves_momentum_otherwise() {
    // the data's convention: weights of A and B positive, C and D negative, the stiffness negative (every link of both
    // installs); the bending vector is zero when A + B = C + D
    let link = BendStiffnessLink {
        weights: [0.5, 0.5, -0.5, -0.5],
        bend_stiffness: -0.8,
        rest_curvature: 0.0,
        particles: [0, 1, 2, 3],
    };
    let inv = [1.0, 0.5, 0.25, 1.0];
    let mut pos = hinge(0.0);
    let mut prev = pos;
    constraints::bend_stiffness(
        &[link],
        BendRestPose::Flat,
        &mut particles(&mut pos, &mut prev, &inv),
        1.0,
    );
    assert_eq!(pos, hinge(0.0));
    let mut pos = hinge(0.4);
    let before = pos;
    constraints::bend_stiffness(
        &[link],
        BendRestPose::Flat,
        &mut particles(&mut pos, &mut prev, &inv),
        1.0,
    );
    // sum of mass * displacement is zero because the weights sum to zero
    let mut momentum = [0.0f64; 3];
    for i in 0..4 {
        let m = 1.0 / f64::from(inv[i]);
        for (k, out) in momentum.iter_mut().enumerate() {
            *out += m * f64::from(pos[i].0[k] - before[i].0[k]);
        }
    }
    assert!(momentum.iter().all(|x| x.abs() < 1e-6), "{momentum:?}");
    // and the hinge is flatter: A comes down, the edge goes up
    assert!(pos[0].y() < 0.4 && pos[2].y() > 0.0, "{pos:?}");
}

#[test]
fn bend_stiffness_rest_curvature_zero_is_the_flat_form_and_its_clamp_switches_a_link_off() {
    let flat_link = BendStiffnessLink {
        weights: [0.5, 0.5, -0.5, -0.5],
        bend_stiffness: -0.8,
        rest_curvature: 0.0,
        particles: [0, 1, 2, 3],
    };
    let inv = [1.0, 1.0, 1.0, 1.0];
    // a rest curvature of 0 adds nothing: the curved form equals the flat form bit for bit
    let mut flat = hinge(0.5);
    let mut curved = hinge(0.5);
    let mut prev = flat;
    constraints::bend_stiffness(
        &[flat_link],
        BendRestPose::Flat,
        &mut particles(&mut flat, &mut prev, &inv),
        1.0,
    );
    constraints::bend_stiffness(
        &[flat_link],
        BendRestPose::Curved { clamp: None },
        &mut particles(&mut curved, &mut prev, &inv),
        1.0,
    );
    assert_eq!(flat, curved);
    // with a rest curvature, a clamp below the curvature product drops the link's stiffness: nothing moves
    let link = BendStiffnessLink {
        rest_curvature: 0.3,
        ..flat_link
    };
    let mut pos = hinge(0.5);
    constraints::bend_stiffness(
        &[link],
        BendRestPose::Curved { clamp: Some(0.0) },
        &mut particles(&mut pos, &mut prev, &inv),
        1.0,
    );
    assert_eq!(pos, hinge(0.5));
    // unclamped it moves, and keeps the momentum (the weights still sum to zero)
    let before = hinge(0.5);
    constraints::bend_stiffness(
        &[link],
        BendRestPose::Curved { clamp: None },
        &mut particles(&mut pos, &mut prev, &inv),
        1.0,
    );
    assert_ne!(pos, before);
    for k in 0..3 {
        let sum: f64 = (0..4).map(|i| f64::from(pos[i].0[k] - before[i].0[k])).sum();
        assert!(sum.abs() < 1e-6, "{sum}");
    }
}

// ---- local range ----------------------------------------------------------------------------------------------

fn reference(at: V4, normal: V4) -> Buffer {
    let mut b = Buffer::new(1);
    b.positions[0] = at;
    b.normals[0] = normal;
    b
}

fn local(max: f32, min_n: f32, max_n: f32) -> LocalRange {
    LocalRange {
        particle: 0,
        reference_vertex: 0,
        maximum_distance: max,
        max_normal_distance: max_n,
        min_normal_distance: min_n,
    }
}

#[test]
fn a_local_range_sphere_pulls_a_particle_back_to_its_maximum_distance() {
    let inv = [1.0];
    let buffer = reference(v(1.0, 1.0, 1.0), v(0.0, 1.0, 0.0));
    let mut pos = [v(1.0, 1.0, 4.0)];
    let mut prev = pos;
    constraints::local_range(
        &[local(1.0, -10.0, 10.0)],
        LocalRangeShape::Sphere,
        true,
        1.0,
        &buffer,
        &mut particles(&mut pos, &mut prev, &inv),
        1.0,
        false,
    );
    assert!(
        (dist(pos[0], v(1.0, 1.0, 1.0)) - 1.0).abs() < 3.0 * FAST * 3.0,
        "{pos:?}"
    );
    // inside the sphere and the normal band: unchanged
    let mut pos = [v(1.0, 1.2, 1.3)];
    constraints::local_range(
        &[local(1.0, -10.0, 10.0)],
        LocalRangeShape::Sphere,
        true,
        1.0,
        &buffer,
        &mut particles(&mut pos, &mut prev, &inv),
        1.0,
        false,
    );
    assert_eq!(pos, [v(1.0, 1.2, 1.3)]);
}

#[test]
fn a_local_range_keeps_a_particle_in_its_normal_band() {
    let inv = [1.0];
    let buffer = reference(v(0.0, 0.0, 0.0), v(0.0, 1.0, 0.0));
    for (start, want_y) in [(v(0.1, 0.8, 0.0), 0.5f64), (v(0.1, -0.8, 0.0), -0.25)] {
        let mut pos = [start];
        let mut prev = pos;
        constraints::local_range(
            &[local(5.0, -0.25, 0.5)],
            LocalRangeShape::Sphere,
            true,
            1.0,
            &buffer,
            &mut particles(&mut pos, &mut prev, &inv),
            1.0,
            false,
        );
        assert!(
            (f64::from(pos[0].y()) - want_y).abs() < 2e-3,
            "{start:?} -> {pos:?}"
        );
    }
}

#[test]
fn a_local_range_cylinder_limits_the_distance_from_the_normals_line() {
    let inv = [1.0];
    let buffer = reference(v(0.0, 0.0, 0.0), v(0.0, 1.0, 0.0));
    let mut pos = [v(3.0, 0.2, 4.0)];
    let mut prev = pos;
    constraints::local_range(
        &[local(1.0, -10.0, 10.0)],
        LocalRangeShape::Cylinder,
        true,
        1.0,
        &buffer,
        &mut particles(&mut pos, &mut prev, &inv),
        1.0,
        false,
    );
    let radial = (f64::from(pos[0].x()).powi(2) + f64::from(pos[0].z()).powi(2)).sqrt();
    assert!((radial - 1.0).abs() < 3.0 * FAST * 5.0, "{pos:?}");
    assert!((pos[0].y() - 0.2).abs() < 1e-6);
}

#[test]
fn a_local_range_preserving_velocity_moves_the_previous_position_with_the_particle() {
    let inv = [1.0];
    let buffer = reference(v(0.0, 0.0, 0.0), v(0.0, 1.0, 0.0));
    let mut pos = [v(0.0, 0.0, 3.0)];
    let mut prev = [v(0.0, 0.0, 2.5)];
    constraints::local_range(
        &[local(1.0, -10.0, 10.0)],
        LocalRangeShape::Sphere,
        false,
        1.0,
        &buffer,
        &mut particles(&mut pos, &mut prev, &inv),
        1.0,
        true,
    );
    assert!((f64::from(pos[0].z() - prev[0].z()) - 0.5).abs() < 1e-6);
}

// ---- transitions ----------------------------------------------------------------------------------------------

#[test]
fn an_animated_transition_puts_particles_on_their_references_and_a_simulated_one_does_nothing() {
    let mut buffer = Buffer::new(2);
    buffer.positions = vec![v(1.0, 2.0, 3.0), v(4.0, 5.0, 6.0)];
    let set = [
        TransitionParticle {
            particle: 0,
            reference_vertex: 1,
            to_anim_delay: 0.0,
            to_sim_delay: 0.0,
            to_sim_max_distance: 1.0,
        },
        TransitionParticle {
            particle: 1,
            reference_vertex: 0,
            to_anim_delay: 0.0,
            to_sim_delay: 0.0,
            to_sim_max_distance: 1.0,
        },
    ];
    let periods = TransitionPeriods {
        to_anim: 0.1,
        to_sim: 0.1,
    };
    let inv = [1.0, 1.0];
    let mut pos = [V4::ZERO, V4::ZERO];
    let mut prev = pos;
    let mut instance = Transition::new(2);
    constraints::transition(
        &set,
        periods,
        &buffer,
        &mut instance,
        &mut particles(&mut pos, &mut prev, &inv),
        1.0,
        false,
    );
    assert_eq!(pos, [V4::ZERO, V4::ZERO]);
    instance.state = TransitionState::Animated;
    constraints::transition(
        &set,
        periods,
        &buffer,
        &mut instance,
        &mut particles(&mut pos, &mut prev, &inv),
        1.0,
        false,
    );
    // the reference position goes through the buffer's matrix as a point: its translation column's w (1) is added
    assert_eq!(pos, [V4::new(4.0, 5.0, 6.0, 1.0), V4::new(1.0, 2.0, 3.0, 1.0)]);
}

#[test]
fn blending_to_simulation_stops_at_the_first_particle_whose_blend_is_over() {
    let mut buffer = Buffer::new(3);
    buffer.positions = vec![v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(2.0, 0.0, 0.0)];
    let p = |i: u16, delay: f32| TransitionParticle {
        particle: i,
        reference_vertex: i,
        to_anim_delay: 0.0,
        to_sim_delay: delay,
        to_sim_max_distance: 0.0,
    };
    // particle 0 still before its delay (snapped), 1 past its period (ends the pass), 2 before its delay (not reached)
    let set = [p(0, 1.0), p(1, -1.0), p(2, 1.0)];
    let periods = TransitionPeriods {
        to_anim: 0.1,
        to_sim: 0.1,
    };
    let inv = [1.0; 3];
    let mut pos = [v(9.0, 9.0, 9.0); 3];
    let mut prev = pos;
    let mut instance = Transition::new(3);
    instance.to_simulation();
    instance.time = 0.05;
    constraints::transition(
        &set,
        periods,
        &buffer,
        &mut instance,
        &mut particles(&mut pos, &mut prev, &inv),
        1.0,
        false,
    );
    assert_eq!(
        pos,
        [V4::new(0.0, 0.0, 0.0, 1.0), v(9.0, 9.0, 9.0), v(9.0, 9.0, 9.0)]
    );
}

#[test]
fn a_transition_ends_after_its_period_plus_delay() {
    let mut t = Transition::new(1);
    t.to_animation();
    t.step(0.06, 0.1, 0.1);
    assert_eq!(t.state, TransitionState::ToAnimation);
    t.step(0.06, 0.1, 0.1);
    assert_eq!(t.state, TransitionState::Animated);
    t.to_simulation();
    t.step(0.2, 0.1, 0.1);
    assert_eq!(t.state, TransitionState::Simulated);
}

// ---- collision ------------------------------------------------------------------------------------------------

fn still(shape: Shape) -> RuntimeCollidable {
    RuntimeCollidable {
        transform: Mat4::IDENTITY,
        linear_velocity: V4::ZERO,
        angular_velocity: V4::ZERO,
        shape,
        enabled: true,
        virtual_collision_points: false,
    }
}

#[test]
fn particles_inside_a_sphere_end_on_its_surface_plus_their_radius() {
    // five particles: the first four are queried together, the fifth alone. The four-wide query normalises with
    // FLT_EPSILON added to the squared length, the single one without: modelled here in f64.
    let starts = [
        v(0.1, 0.0, 0.0),
        v(0.0, 0.2, 0.0),
        v(0.0, 0.0, -0.3),
        v(0.2, 0.2, 0.0),
        v(0.0, -0.1, 0.1),
    ];
    let mut sim = sim_cloth(&starts, &[1.0; 5], Vec::new());
    sim.static_collision_masks = vec![1; 5];
    let mut state = SimState::new(&sim);
    let data = ParticleData::new(&sim);
    let collidables = [still(Shape::Sphere {
        pos: [0.0, 0.0, 0.0, 1.0],
    })];
    collide::collide(&sim, &collidables, &mut state, &data, 1.0 / 60.0);
    for (i, (p, start)) in state.positions.iter().zip(&starts).enumerate() {
        let s = start.0.map(f64::from);
        let sq: f64 = s[..3].iter().map(|x| x * x).sum();
        let len = if i < 4 {
            (sq + f64::from(f32::EPSILON)).sqrt()
        } else {
            sq.sqrt()
        };
        let n: Vec<f64> = s[..3].iter().map(|x| x / len).collect();
        // contact point c = n * 1; newP = P + n * (r - (P - c) . n)
        let pc: f64 = (0..3).map(|k| (s[k] - n[k]) * n[k]).sum();
        let step = 0.01 - pc;
        for k in 0..3 {
            let want = s[k] + n[k] * step;
            assert!(
                (f64::from(p.0[k]) - want).abs() < 2e-6,
                "particle {i} lane {k}: {p:?} vs {want}"
            );
        }
    }
}

#[test]
fn the_static_collision_mask_chooses_the_collidables() {
    let starts = [v(0.1, 0.0, 0.0), v(0.0, 0.2, 0.0)];
    let mut sim = sim_cloth(&starts, &[1.0; 2], Vec::new());
    // particle 0 collides with collidable 0 only, particle 1 with collidable 1 only
    sim.static_collision_masks = vec![0b01, 0b10];
    let mut state = SimState::new(&sim);
    let data = ParticleData::new(&sim);
    let collidables = [
        still(Shape::Sphere {
            pos: [0.0, 0.0, 0.0, 1.0],
        }),
        still(Shape::Plane {
            equation: [0.0, 1.0, 0.0, -5.0],
        }),
    ];
    collide::collide(&sim, &collidables, &mut state, &data, 1.0 / 60.0);
    assert!((dist(state.positions[0], V4::ZERO) - 1.01).abs() < 1e-5);
    // the plane y = 5 pushes particle 1 above it
    assert!((state.positions[1].y() - 5.01).abs() < 1e-5);
    // a disabled collidable does nothing
    let mut state = SimState::new(&sim);
    let mut off = collidables.clone();
    off[0].enabled = false;
    collide::collide(&sim, &off, &mut state, &data, 1.0 / 60.0);
    assert_eq!(state.positions[0], v(0.1, 0.0, 0.0));
}

#[test]
fn a_tapered_capsule_pushes_particles_out_of_its_cone_and_caps() {
    let shape = Shape::TaperedCapsule {
        small: [0.0, 0.0, 0.0, 0.0],
        big: [0.0, 2.0, 0.0, 0.0],
        cone_apex: [0.0; 4],
        cone_axis: [0.0; 4],
        l_vec: [0.0; 4],
        d_vec: [0.0; 4],
        tan_theta_vec_neg: [0.0; 4],
        small_radius: 0.2,
        big_radius: 0.6,
        l: 0.0,
        d: 0.0,
        cos_theta: 0.0,
        sin_theta: 0.0,
        tan_theta: 0.0,
        tan_theta_sqr: 0.0,
    };
    let world = WorldShape::new(&shape, &Mat4::IDENTITY);
    // on the axis below the small end, beyond the big end, and beside the cone
    let below = world.closest(v(0.0, -1.0, 0.0), 0.0);
    assert!((below.distance - 0.8).abs() < 1e-5, "{below:?}");
    let above = world.closest(v(0.0, 3.0, 0.0), 0.0);
    assert!((above.distance - 0.4).abs() < 1e-5, "{above:?}");
    let beside = world.closest(v(1.0, 1.0, 0.0), 0.0);
    // the cone's surface distance from (1, 1): the exact tangent cone of the two spheres, in f64
    let (r, big_r, l) = (0.2f64, 0.6f64, 2.0f64);
    let sin = (big_r - r) / l;
    let cos = (1.0 - sin * sin).sqrt();
    // the cone's surface line is offset from the axis by r/cos at the small centre, widening by tan
    let apex = -r / sin; // below the small centre
    let tan = sin / cos;
    let radius_at = |y: f64| (y - apex) * tan;
    let want = (1.0 - radius_at(1.0)) * cos;
    assert!(
        (f64::from(beside.distance) - want).abs() < 1e-5,
        "{} vs {want}",
        beside.distance
    );
}

#[test]
fn friction_moves_the_previous_position_by_the_tangential_slip() {
    // a particle moving along a still plane: full friction removes the tangential velocity, none keeps it
    let collidable = still(Shape::Plane {
        equation: [0.0, 1.0, 0.0, 0.0],
    });
    let shape = WorldShape::new(&collidable.shape, &collidable.transform);
    for (friction, want_vx) in [(0.0f32, 0.5f32), (1.0, 0.0)] {
        let mut p = v(0.5, -0.01, 0.0);
        let mut prev = v(0.0, -0.01, 0.0);
        let c = shape.closest(p, 0.0);
        collide::solve_contact(&c, &collidable, 0.0, friction, &mut p, &mut prev, 1.0 / 60.0);
        assert!(
            (p.x() - prev.x() - want_vx).abs() < 1e-6,
            "friction {friction}: {p:?} {prev:?}"
        );
        assert_eq!(p.y(), 0.0);
    }
}

#[test]
fn a_convex_geometry_tetrahedron_pushes_a_particle_out_through_its_surface_face() {
    // one tetrahedron: the cube corner x, y, z >= 0, x + y + z <= 1; the surface face is x + y + z = 1 (row 0)
    let rows = [
        [1.0f32, 1.0, 1.0, -1.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
    ];
    let mut columns = [0.0f32; 16];
    for (r, row) in rows.iter().enumerate() {
        for (c, value) in row.iter().enumerate() {
            columns[c * 4 + r] = *value;
        }
    }
    let shape = Shape::ConvexGeometry {
        tetrahedra_grid: vec![0, 1],
        grid_cells: vec![0],
        tetrahedra_equations: vec![columns],
        local_from_world: Mat4::IDENTITY.to_columns(),
        world_from_local: Mat4::IDENTITY.to_columns(),
        aabb_min: [0.0, 0.0, 0.0, 0.0],
        aabb_max: [1.0, 1.0, 1.0, 0.0],
        centroid: [0.0, 0.0, 0.0, 0.0],
        inv_cell_size: [1.0, 1.0, 1.0, 0.0],
        grid_res: 1,
    };
    let world = WorldShape::new(&shape, &Mat4::IDENTITY);
    let c = world.closest(v(0.2, 0.2, 0.2), 0.0);
    // the plane's value at the particle: 0.6 - 1
    assert!((c.distance - (-0.4)).abs() < 1e-6, "{c:?}");
    // the ray from the centroid (the origin) through the particle meets the face at (1/3, 1/3, 1/3)
    for k in 0..3 {
        assert!((c.point.0[k] - 1.0 / 3.0).abs() < 1e-5, "{c:?}");
    }
    // outside the face: no contact
    assert!(world.closest(v(0.6, 0.6, 0.6), 0.0).distance > 0.0);
}

// ---- the simulate operator --------------------------------------------------------------------------------------

#[test]
fn a_pinned_particle_follows_its_animation_through_the_substeps() {
    // the pinned particle was moved by the animation to (3, 0, 0) from (0, 0, 0): the substeps step it there linearly
    let sim = sim_cloth(&[v(0.0, 0.0, 0.0), v(0.0, -1.0, 0.0)], &[0.0, 1.0], Vec::new());
    let data = ParticleData::new(&sim);
    let mut state = SimState::new(&sim);
    state.positions[0] = v(3.0, 0.0, 0.0);
    simulate::simulate(
        &sim,
        &config(3, Vec::new(), false),
        &mut state,
        &data,
        &[],
        &[],
        &mut [],
        1.0 / 30.0,
    );
    assert_eq!(state.positions[0], v(3.0, 0.0, 0.0));
    assert_eq!(state.previous[0], v(3.0, 0.0, 0.0));
}

#[test]
fn a_pendulum_keeps_its_length_and_swings_through_the_bottom() {
    let sets = vec![ConstraintKind::StandardLink(vec![link(0, 1, 1.0, 1.0)])];
    let sim = sim_cloth(&[v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)], &[0.0, 1.0], sets);
    let data = ParticleData::new(&sim);
    let mut state = SimState::new(&sim);
    let cfg = config(1, vec![0, -1], false);
    let mut lowest = 0.0f32;
    for _ in 0..60 {
        simulate::prepare(&mut state, 1.0 / 60.0, cfg.sub_steps, StiffnessModel::Legacy);
        simulate::simulate(&sim, &cfg, &mut state, &data, &[], &[], &mut [], 1.0 / 60.0);
        assert!((dist(state.positions[0], state.positions[1]) - 1.0).abs() < 2e-3);
        lowest = lowest.min(state.positions[1].y());
    }
    // released level, it falls through the bottom and has crossed to the other side within a second
    assert!(lowest < -0.95, "{lowest}");
    assert!(state.positions[1].x() < 0.0, "{:?}", state.positions[1]);
}

#[test]
fn adaptive_stiffness_softens_links_by_the_substep_count_and_ramps_transitions() {
    let sim = sim_cloth(&[v(0.0, 0.0, 0.0)], &[1.0], Vec::new());
    let state = SimState::new(&sim);
    let set = |kind| ConstraintSet {
        name: String::new(),
        constraint_id: None,
        kind,
    };
    let links = set(ConstraintKind::StandardLink(Vec::new()));
    let range = set(ConstraintKind::LocalRange {
        reference_mesh_buffer: 0,
        stiffness: 1.0,
        shape_type: LocalRangeShape::Sphere,
        apply_normal_component: false,
        constraints: Vec::new(),
    });
    assert_eq!(
        simulate::set_stiffness(&state, StiffnessModel::Legacy, &links, 0, 3),
        1.0
    );
    assert_eq!(
        simulate::set_stiffness(&state, StiffnessModel::Adaptive, &links, 0, 1),
        1.0
    );
    let s = f64::from(simulate::set_stiffness(
        &state,
        StiffnessModel::Adaptive,
        &links,
        1,
        3,
    ));
    assert!((s - 3f64.powf(-1.725)).abs() < 1e-6);
    assert_eq!(
        simulate::set_stiffness(&state, StiffnessModel::Adaptive, &range, 1, 3),
        0.0
    );
    assert_eq!(
        simulate::set_stiffness(&state, StiffnessModel::Adaptive, &range, 2, 3),
        1.0
    );
}

#[test]
fn normals_of_a_flat_sheet_point_one_way_and_flip_with_their_bit() {
    let positions = [
        v(0.0, 0.0, 0.0),
        v(1.0, 0.0, 0.0),
        v(0.0, 1.0, 0.0),
        v(1.0, 1.0, 0.0),
    ];
    let mut normals = [V4::ZERO; 4];
    simulate::update_normals(&[0, 1, 2, 1, 3, 2], &[0], &positions, &mut normals);
    for n in &normals {
        assert!(
            (n.z() - 1.0).abs() < 1.0 / 1024.0 && n.x() == 0.0 && n.y() == 0.0,
            "{n:?}"
        );
    }
    // the first triangle's bit is the byte's top bit: flipping it cancels it at the shared vertices 1 and 2
    simulate::update_normals(&[0, 1, 2, 1, 3, 2], &[0x80], &positions, &mut normals);
    assert!(normals[0].z() < 0.0);
    assert_eq!(normals[1], V4::ZERO);
    assert!(normals[3].z() > 0.0);
}

#[test]
fn the_same_inputs_give_bit_identical_results() {
    let sets = vec![
        ConstraintKind::StandardLink(vec![link(0, 1, 1.0, 0.9), link(1, 2, 1.0, 0.9)]),
        ConstraintKind::StretchLink(vec![link(0, 2, 2.0, 1.0)]),
    ];
    let mut sim = sim_cloth(
        &[v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(2.0, 0.0, 0.0)],
        &[0.0, 1.0, 1.0],
        sets,
    );
    sim.global_damping_per_second = 0.2;
    let data = ParticleData::new(&sim);
    let collidables = [still(Shape::Sphere {
        pos: [1.0, -1.0, 0.0, 0.7],
    })];
    let run = || {
        let mut state = SimState::new(&sim);
        state.collidables = collidables.to_vec();
        let cfg = config(2, vec![0, 1, -1], true);
        for frame in 0..200 {
            let dt = if frame % 7 == 0 { 1.0 / 30.0 } else { 1.0 / 60.0 };
            simulate::prepare(&mut state, dt, cfg.sub_steps, StiffnessModel::Legacy);
            simulate::simulate(&sim, &cfg, &mut state, &data, &[], &[], &mut [], dt);
        }
        state
    };
    let a = run();
    let b = run();
    let bits = |s: &SimState| {
        s.positions
            .iter()
            .flat_map(|p| p.0.map(f32::to_bits))
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&a), bits(&b));
    assert!(a.positions.iter().all(|p| p.is_finite()));
}

/// An action that pushes every particle along x with a constant force and counts its calls.
struct Push {
    force: f32,
    calls: Vec<f32>,
}

impl simulate::Action for Push {
    fn apply(&mut self, _state: &SimState, time_step: f32, forces: &mut [V4]) {
        self.calls.push(time_step);
        for f in forces.iter_mut() {
            *f = f.add(V4::new(self.force, 0.0, 0.0, 0.0));
        }
    }
}

#[test]
fn actions_add_their_forces_on_every_substep() {
    // no gravity, unit mass, a force of 4 along x: each substep of 1/64 adds 4/4096 to the velocity, all exact
    let mut sim = sim_cloth(&[v(0.0, 0.0, 0.0)], &[1.0], Vec::new());
    sim.gravity = [0.0; 4];
    let data = ParticleData::new(&sim);
    let mut state = SimState::new(&sim);
    state.damping = 1.0;
    state.last_time_step = 1.0 / 64.0;
    let mut push = Push {
        force: 4.0,
        calls: Vec::new(),
    };
    let cfg = config(2, Vec::new(), false);
    for _ in 0..3 {
        simulate::simulate(
            &sim,
            &cfg,
            &mut state,
            &data,
            &[],
            &[],
            &mut [&mut push],
            1.0 / 32.0,
        );
    }
    // two substeps a frame, each 1/64 long
    assert_eq!(push.calls, vec![1.0 / 64.0; 6]);
    // after n substeps x = F dt^2 n(n+1)/2 with n = 6
    assert_eq!(f64::from(state.positions[0].x()), 4.0 / 4096.0 * 21.0);
}

#[test]
fn a_virtual_point_inside_a_shape_pushes_its_particle_along_the_triangle_normal() {
    use havok_formats::cloth::{
        VirtualCollisionPoints, VirtualDictionaryEntry, VirtualTriangleFan, VirtualTriangleSection,
    };
    // particle 0's triangle (0, 1, 2) in the plane z = 0; one virtual point at its midpoint (1, 1, 0), 0.02 inside a
    // unit sphere centred below it; the particles themselves are outside the sphere
    let mut sim = sim_cloth(
        &[v(0.0, 0.0, 0.0), v(2.0, 0.0, 0.0), v(0.0, 2.0, 0.0)],
        &[1.0, 1.0, 1.0],
        Vec::new(),
    );
    sim.max_particle_radius = 1.0;
    sim.virtual_collision_points = VirtualCollisionPoints {
        points: 1,
        triangle_fan_indices: vec![0, 0xffff, 0xffff],
        triangle_fans: vec![VirtualTriangleFan {
            particle: 0,
            vcp_start: 0,
            triangles: 1,
        }],
        triangles: vec![VirtualTriangleSection {
            opposite: [1, 2],
            dictionary: 0,
        }],
        triangle_dictionary: vec![VirtualDictionaryEntry { start: 0, count: 1 }],
        triangle_barycentrics: vec![(0.5, 0.5)],
        edge_fan_indices: vec![0xffff; 3],
        ..Default::default()
    };
    let data = ParticleData::new(&sim);
    let mut state = SimState::new(&sim);
    let collidable = RuntimeCollidable {
        transform: Mat4::IDENTITY,
        linear_velocity: V4::ZERO,
        angular_velocity: V4::ZERO,
        shape: Shape::Sphere {
            pos: [1.0, 1.0, -0.98, 1.0],
        },
        enabled: true,
        virtual_collision_points: true,
    };
    collide::collide(&sim, &[collidable], &mut state, &data, 1.0 / 60.0);
    // depth: (0.98 - 1) - radius 0.01 = -0.03, along the triangle's normal (0, 0, 1)
    let p = state.positions[0];
    assert!((p.z() - 0.03).abs() < 1e-6, "{p:?}");
    assert_eq!((p.x(), p.y()), (0.0, 0.0));
    assert_eq!(state.positions[1], v(2.0, 0.0, 0.0));
    assert_eq!(state.positions[2], v(0.0, 2.0, 0.0));
    assert_eq!(
        state.previous[0],
        v(0.0, 0.0, 0.0),
        "previous positions are left alone"
    );
}

#[test]
fn landscape_planes_push_particles_onto_their_plane() {
    // six particles 0.05 below the plane z = 0 (normal +z); the first five have bit 31 of their collision mask (four
    // queried together, one alone), the sixth has not
    let positions: Vec<V4> = (0..6).map(|i| v(i as f32, 0.0, -0.05)).collect();
    let mut sim = sim_cloth(&positions, &[1.0; 6], Vec::new());
    sim.static_collision_masks = vec![0x8000_0000, 0x8000_0000, 0x8000_0000, 0x8000_0000, 0x8000_0000, 0];
    let data = ParticleData::new(&sim);
    let mut state = SimState::new(&sim);
    state.landscape_planes = vec![V4::new(0.0, 0.0, 1.0, 0.0); 6];
    collide::collide(&sim, &[], &mut state, &data, 1.0 / 60.0);
    for i in 0..5 {
        let p = state.positions[i];
        assert!(p.z().abs() < 1e-7, "particle {i} at {p:?}");
        assert_eq!((p.x(), p.y()), (i as f32, 0.0));
    }
    assert_eq!(
        state.positions[5],
        v(5.0, 0.0, -0.05),
        "no landscape bit: untouched"
    );
}
