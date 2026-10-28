//! Game captures: cloth steps recorded from the running game, replayed through this crate's operators and compared
//! bit for bit (see tests/inputs/README.md for where the captures live; the tests skip without them). No game data is
//! included in the repository.
//!
//! `captures.tsv` (a header line, then one capture per line, tab-separated): the frames file, the tagfile of the
//! Havok cloth corpus that holds the cloth data, the cloth data's name, and the processor the game ran on (as Windows
//! names it in `PROCESSOR_IDENTIFIER`).
//!
//! A frames file is `CLF3`, then four `u32`: steps, particles, bones, collidables; then per step (every value
//! little-endian, a vector four `f32`, a matrix four columns of four `f32`):
//! - `f32` the frame's timestep;
//! - three particle states: before the simulate operator's notify, after it, and after the step; each the last
//!   substep length (`f32`), the damping (`f32`), the positions and the previous positions (a vector per particle);
//! - the bones of the transform set the step reads (a matrix each);
//! - the collidables after the notify, then after the step: per collidable its transform (a matrix), linear and
//!   angular velocity (a vector each);
//! - `u32` the number of action calls of the step, then per call its substep length (`f32`) and the forces the action
//!   left (a vector per particle).
#![allow(
    clippy::unwrap_used,
    reason = "a real-data test reports what it cannot read as its failure"
)]
mod inputs;

use havok_cloth::buffer::Buffer;
use havok_cloth::deform::{self, LocalBlocks};
use havok_cloth::matrix::Mat4;
use havok_cloth::operators;
use havok_cloth::simd::V4;
use havok_cloth::simulate::{self, Action, ParticleData, SimState, StiffnessModel};
use havok_formats::Tagfile;
use havok_formats::cloth::{self, Cloth, OperatorKind, SimulateConfig};
use inputs::{Game, OracleSet, Query, expect_found, resolve};
use std::path::Path;

/// One particle state of the sim cloth.
struct Particles {
    last_time_step: f32,
    damping: f32,
    positions: Vec<V4>,
    previous: Vec<V4>,
}

/// A collidable as the game left it: transform, linear and angular velocity.
type Collidable = (Mat4, V4, V4);

/// One recorded step.
struct Step {
    dt: f32,
    before: Particles,
    after_notify: Particles,
    after: Particles,
    bones: Vec<Mat4>,
    collidables_after_notify: Vec<Collidable>,
    collidables_after: Vec<Collidable>,
    /// The action calls, in order: substep length and the forces after the call.
    actions: Vec<(f32, Vec<V4>)>,
}

/// Reads little-endian values from the front of a byte slice.
struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn u32(&mut self) -> u32 {
        let (head, rest) = self
            .0
            .split_first_chunk::<4>()
            .expect("the frames file ends early");
        self.0 = rest;
        u32::from_le_bytes(*head)
    }

    fn f32(&mut self) -> f32 {
        f32::from_bits(self.u32())
    }

    fn v4(&mut self) -> V4 {
        V4([self.f32(), self.f32(), self.f32(), self.f32()])
    }

    fn mat(&mut self) -> Mat4 {
        Mat4::from_columns(std::array::from_fn(|_| self.f32()))
    }

    fn vectors(&mut self, n: usize) -> Vec<V4> {
        (0..n).map(|_| self.v4()).collect()
    }
}

fn read_steps(path: &Path) -> (usize, Vec<Step>) {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..4], b"CLF3", "{}: not a frames file", path.display());
    let mut r = Reader(&bytes[4..]);
    let [steps, particles, bones, collidables] = [(); 4].map(|()| r.u32() as usize);
    let state = |r: &mut Reader| Particles {
        last_time_step: r.f32(),
        damping: r.f32(),
        positions: r.vectors(particles),
        previous: r.vectors(particles),
    };
    let moved =
        |r: &mut Reader| -> Vec<Collidable> { (0..collidables).map(|_| (r.mat(), r.v4(), r.v4())).collect() };
    let steps = (0..steps)
        .map(|_| {
            let dt = r.f32();
            let before = state(&mut r);
            let after_notify = state(&mut r);
            let after = state(&mut r);
            let bones = (0..bones).map(|_| r.mat()).collect();
            let collidables_after_notify = moved(&mut r);
            let collidables_after = moved(&mut r);
            let calls = r.u32() as usize;
            let actions = (0..calls).map(|_| (r.f32(), r.vectors(particles))).collect();
            Step {
                dt,
                before,
                after_notify,
                after,
                bones,
                collidables_after_notify,
                collidables_after,
                actions,
            }
        })
        .collect();
    assert!(
        r.0.is_empty(),
        "{}: {} bytes after the last step",
        path.display(),
        r.0.len()
    );
    (particles, steps)
}

/// The distance between two floats in units in the last place (0: the same bits, or both zeros).
fn ulps(a: f32, b: f32) -> u32 {
    if a == b {
        return 0;
    }
    let key = |x: f32| {
        let i = x.to_bits() as i32;
        if i < 0 { i32::MIN - i } else { i }
    };
    key(a).abs_diff(key(b))
}

/// The largest distance in units in the last place over the `x y z` lanes of two vector lists (the game leaves
/// the particles' `w` lanes undefined).
fn worst3(ours: &[V4], game: &[V4]) -> u32 {
    assert_eq!(ours.len(), game.len());
    ours.iter()
        .zip(game)
        .flat_map(|(a, b)| (0..3).map(move |k| ulps(a.0[k], b.0[k])))
        .max()
        .unwrap_or(0)
}

/// One capture: its steps and the cloth data they belong to.
struct Capture {
    name: String,
    processor: String,
    cloth: Cloth,
    steps: Vec<Step>,
}

/// The game's captures; empty (with a line saying so) where they are not configured.
fn captures(game: Game) -> Vec<Capture> {
    let query = Query::Oracle(OracleSet::Capture, game, "havok-cloth");
    let Some(dir) = resolve(query) else {
        return Vec::new();
    };
    let corpus = Query::Oracle(OracleSet::Corpus, game, "havok-cloth");
    let Some(corpus_dir) = resolve(corpus) else {
        return Vec::new();
    };
    let list = std::fs::read_to_string(dir.join("captures.tsv")).unwrap();
    let mut out = Vec::new();
    for line in list.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let columns: Vec<&str> = line.split('\t').collect();
        let [frames, tagfile, cloth_name, processor] = columns.as_slice() else {
            panic!("captures.tsv: not four columns: {line}");
        };
        let bytes = std::fs::read(corpus_dir.join(tagfile)).unwrap();
        let physics = cloth::read(&Tagfile::read(&bytes, None).unwrap()).unwrap();
        let cloth = physics
            .cloths
            .into_iter()
            .find(|c| c.name == *cloth_name)
            .unwrap_or_else(|| panic!("{tagfile} has no cloth data {cloth_name}"));
        let (particles, steps) = read_steps(&dir.join(frames));
        let sim = simulated(&cloth).0;
        assert_eq!(
            particles,
            cloth.sim_cloths[sim].particles.len(),
            "{frames}: particle count"
        );
        out.push(Capture {
            name: (*frames).to_owned(),
            processor: (*processor).to_owned(),
            cloth,
            steps,
        });
    }
    expect_found(query, "captures", out.len());
    out
}

/// The sim cloth the cloth data's (first) simulate operator steps, and its first configuration.
fn simulated(c: &Cloth) -> (usize, SimulateConfig) {
    c.operators
        .iter()
        .find_map(|op| match &op.kind {
            OperatorKind::Simulate { sim_cloth, configs } => Some((*sim_cloth as usize, configs[0].clone())),
            _ => None,
        })
        .expect("a captured cloth data has a simulate operator")
}

/// A sim state holding one recorded particle state.
fn state_from(c: &Cloth, sim: usize, p: &Particles) -> SimState {
    let mut state = SimState::new(&c.sim_cloths[sim]);
    state.positions.clone_from(&p.positions);
    state.previous.clone_from(&p.previous);
    state.last_time_step = p.last_time_step;
    state.damping = p.damping;
    state
}

/// The operators of the cloth's first state up to the simulate operator, from the recorded state after the notify:
/// the skins into the buffers the moves read (the recorded bones are the transform set they skin by), then the moves
/// of the pinned particles to their skinned vertices. Returns the buffers.
fn skin_and_move(c: &Cloth, step: &Step, state: &mut SimState, fixed_particles: bool) -> Vec<Buffer> {
    let mut buffers: Vec<Buffer> = c
        .buffers
        .iter()
        .map(|b| Buffer::new(b.vertices as usize))
        .collect();
    let moved_from: Vec<u32> = c
        .operators
        .iter()
        .filter_map(|op| match &op.kind {
            OperatorKind::MoveParticles { reference_buffer, .. } => Some(*reference_buffer),
            _ => None,
        })
        .collect();
    for &index in &c.states[0].operators {
        match &c.operators[index as usize].kind {
            OperatorKind::ObjectSpaceSkin {
                components,
                bone_from_skin_mesh,
                transform_subset,
                output_buffer,
                deformer,
                local,
                local_unpacked,
                ..
            } if moved_from.contains(output_buffer) => {
                let out = &mut buffers[*output_buffer as usize];
                let composite = deform::composite_matrices(
                    &out.mesh_from_world,
                    &step.bones,
                    transform_subset,
                    bone_from_skin_mesh,
                );
                let blocks = if local.is_empty() {
                    LocalBlocks::Unpacked(local_unpacked)
                } else {
                    LocalBlocks::Packed(local)
                };
                deform::deform(deformer, blocks, *components, &composite, out);
            }
            OperatorKind::MoveParticles {
                pairs,
                reference_buffer,
                ..
            } => {
                let reference = buffers[*reference_buffer as usize].clone();
                operators::move_particles(
                    &reference,
                    pairs,
                    &mut state.positions,
                    &mut state.previous,
                    fixed_particles,
                );
            }
            OperatorKind::Simulate { .. } => break,
            _ => {}
        }
    }
    buffers
}

/// The game's recorded action forces, played back one call per substep; counts the calls whose substep length is
/// not the one the game passed.
struct Recorded<'a> {
    calls: &'a [(f32, Vec<V4>)],
    next: usize,
    other_time_steps: usize,
}

impl Action for Recorded<'_> {
    fn apply(&mut self, _state: &SimState, time_step: f32, forces: &mut [V4]) {
        if let Some((t, f)) = self.calls.get(self.next) {
            if t.to_bits() != time_step.to_bits() {
                self.other_time_steps += 1;
            }
            forces.copy_from_slice(&f[..forces.len()]);
        }
        self.next += 1;
    }
}

/// The first capture of the scarf of Sekiro's player (`BD_M_9040_muffler`, 299 steps) gave these on the processor
/// it was made on: the notify's rescaling of the previous positions, and the pinned particles moved onto the skin,
/// bit for bit in every step.
#[test]
fn the_notify_and_the_pinned_particles_match_the_game_bit_for_bit() {
    for capture in captures(Game::Sekiro) {
        let c = &capture.cloth;
        let (sim, config) = simulated(c);
        let model = if config.adapt_constraint_stiffness {
            StiffnessModel::Adaptive
        } else {
            StiffnessModel::Legacy
        };
        let fixed_particles = !c.sim_cloths[sim].fixed_particles.is_empty();
        // the particles the move operators place on the skin (they end the step where they were placed)
        let placed: Vec<usize> = c
            .operators
            .iter()
            .filter_map(|op| match &op.kind {
                OperatorKind::MoveParticles { pairs, .. } => Some(pairs),
                _ => None,
            })
            .flatten()
            .map(|&(_, particle)| usize::from(particle))
            .collect();
        for (n, step) in capture.steps.iter().enumerate() {
            let mut notified = state_from(c, sim, &step.before);
            simulate::prepare(&mut notified, step.dt, config.sub_steps, model);
            assert_eq!(
                worst3(&notified.previous, &step.after_notify.previous),
                0,
                "{} step {n}: previous",
                capture.name
            );
            assert_eq!(
                notified.last_time_step.to_bits(),
                step.after_notify.last_time_step.to_bits(),
                "{} step {n}: substep length",
                capture.name
            );
            assert_eq!(
                notified.damping.to_bits(),
                step.after_notify.damping.to_bits(),
                "{} step {n}: damping",
                capture.name
            );

            let mut state = state_from(c, sim, &step.after_notify);
            skin_and_move(c, step, &mut state, fixed_particles);
            for &p in &placed {
                assert_eq!(
                    worst3(&state.positions[p..=p], &step.after.positions[p..=p]),
                    0,
                    "{} step {n}: pinned particle {p}",
                    capture.name
                );
            }
        }
    }
}

/// The whole simulate step, with the game's own action forces played back: positions, previous positions and the
/// collidables after the step. Compared only on the processor the capture was made on (the approximate reciprocals
/// differ between processors).
#[test]
#[ignore = "not yet bit for bit: on the first Sekiro capture the collidables' drive differs in the last bits in 101 \
            of 299 steps, and the positions have not been compared with the action forces played back"]
fn the_simulate_step_matches_the_game_bit_for_bit() {
    for capture in captures(Game::Sekiro) {
        if std::env::var("PROCESSOR_IDENTIFIER").ok().as_deref() != Some(capture.processor.as_str()) {
            eprintln!(
                "{}: made on {}, not compared on another processor",
                capture.name, capture.processor
            );
            continue;
        }
        let c = &capture.cloth;
        let (sim, config) = simulated(c);
        let sim_cloth = &c.sim_cloths[sim];
        let data = ParticleData::new(sim_cloth);
        let mut differ = Vec::new();
        for (n, step) in capture.steps.iter().enumerate() {
            let mut state = state_from(c, sim, &step.after_notify);
            for (rc, (t, l, a)) in state.collidables.iter_mut().zip(&step.collidables_after_notify) {
                rc.transform = *t;
                rc.linear_velocity = *l;
                rc.angular_velocity = *a;
            }
            let buffers = skin_and_move(c, step, &mut state, !sim_cloth.fixed_particles.is_empty());
            let mut recorded = Recorded {
                calls: &step.actions,
                next: 0,
                other_time_steps: 0,
            };
            if step.actions.is_empty() {
                simulate::simulate(
                    sim_cloth,
                    &config,
                    &mut state,
                    &data,
                    &buffers,
                    &step.bones,
                    &mut [],
                    step.dt,
                );
            } else {
                let mut actions: [&mut dyn Action; 1] = [&mut recorded];
                simulate::simulate(
                    sim_cloth,
                    &config,
                    &mut state,
                    &data,
                    &buffers,
                    &step.bones,
                    &mut actions,
                    step.dt,
                );
            }
            assert_eq!(
                recorded.next,
                step.actions.len(),
                "{} step {n}: action calls",
                capture.name
            );
            assert_eq!(
                recorded.other_time_steps, 0,
                "{} step {n}: action substep lengths",
                capture.name
            );
            let collidables = state
                .collidables
                .iter()
                .zip(&step.collidables_after)
                .map(|(rc, (t, l, a))| {
                    worst3(&rc.transform.cols, &t.cols)
                        .max(worst3(&[rc.linear_velocity, rc.angular_velocity], &[*l, *a]))
                })
                .max()
                .unwrap_or(0);
            let worst = [
                worst3(&state.positions, &step.after.positions),
                worst3(&state.previous, &step.after.previous),
                collidables,
            ];
            if worst != [0; 3] {
                differ.push(format!(
                    "step {n}: positions, previous, collidables differ by {worst:?} ulp"
                ));
            }
        }
        assert!(
            differ.is_empty(),
            "{}: {} of {} steps differ:\n{}",
            capture.name,
            differ.len(),
            capture.steps.len(),
            differ.join("\n")
        );
    }
}
