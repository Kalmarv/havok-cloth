//! Real-data check of the solver: every sim cloth of every distinct tagfile with cloth of both installs (see
//! tests/inputs/README.md), simulated from its rest pose for 60 frames under its own gravity and damping, with its
//! simulate operator's own configuration (substeps, solve order, stiffness model) and its collidables where the data
//! puts them; skips without the files. No game data is included in the repository.
//!
//! Without the operators that feed them, the reference-buffer constraints (local range, transition) are not applied
//! here; the rest of the solve is. Every position's `x y z` must stay finite and every pinned particle must stay put.
//! (The `w` lanes are not checked: the collidables' rotation columns store non-zero `w` lanes in the data, the contact
//! solve carries them into the particles' `w`, and nothing reads them back into `x y z`.)
#![allow(
    clippy::unwrap_used,
    reason = "a real-data test reports what it cannot read as its failure"
)]
mod inputs;

use havok_cloth::simulate::{self, ParticleData, SimState, StiffnessModel};
use havok_formats::cloth::{self, OperatorKind};
use havok_formats::{Compendium, Tagfile};
use inputs::{Game, OracleSet, Query, expect_found, resolve};
use std::collections::BTreeMap;

/// Per game: the files and the sim cloths simulated (each sim cloth with a simulate operator; every sim cloth of
/// both installs has one).
const EXPECTED: [(Game, usize, usize); 2] = [(Game::Sekiro, 149, 638), (Game::Elden, 814, 2_875)];

#[test]
fn every_sim_cloth_of_both_installs_simulates_and_stays_finite() -> Result<(), Box<dyn std::error::Error>> {
    for (game, want_files, want_sims) in EXPECTED {
        let corpus = Query::Oracle(OracleSet::Corpus, game, "havok-cloth");
        let Some(dir) = resolve(corpus) else {
            continue;
        };
        let manifest = std::fs::read_to_string(dir.join("manifest.tsv"))?;
        let mut files: BTreeMap<String, String> = BTreeMap::new();
        for line in manifest.lines().skip(1) {
            let columns: Vec<&str> = line.split('\t').collect();
            if let [_, file, compendium, ..] = columns.as_slice() {
                files.insert((*file).to_owned(), (*compendium).to_owned());
            }
        }
        expect_found(corpus, "cloth tagfiles", files.len());
        assert_eq!(files.len(), want_files, "{game:?}");
        let mut simulated = 0usize;
        for (file, compendium) in &files {
            let bytes = std::fs::read(dir.join(file))?;
            let compendium = match compendium.as_str() {
                "-" => None,
                c => Some(Compendium::read(&std::fs::read(dir.join(c))?)?),
            };
            let tag = Tagfile::read(&bytes, compendium.as_ref())?;
            let physics = cloth::read(&tag)?;
            for data in &physics.cloths {
                for op in &data.operators {
                    let OperatorKind::Simulate { sim_cloth, configs } = &op.kind else {
                        continue;
                    };
                    let sim = &data.sim_cloths[usize::try_from(*sim_cloth)?];
                    let config = &configs[0];
                    let model = if config.adapt_constraint_stiffness {
                        StiffnessModel::Adaptive
                    } else {
                        StiffnessModel::Legacy
                    };
                    let particles = ParticleData::new(sim);
                    let mut state = SimState::new(sim);
                    let pinned: Vec<_> = sim
                        .fixed_particles
                        .iter()
                        .map(|&f| state.positions[usize::from(f)])
                        .collect();
                    for _ in 0..60 {
                        simulate::prepare(&mut state, 1.0 / 60.0, config.sub_steps, model);
                        simulate::simulate(sim, config, &mut state, &particles, &[], &[], 1.0 / 60.0);
                    }
                    assert!(
                        state.positions.iter().all(|p| p.is_finite3()),
                        "{game:?} {file} {}: a position is not finite in x y z",
                        sim.name
                    );
                    for (&f, p) in sim.fixed_particles.iter().zip(&pinned) {
                        let now = state.positions[usize::from(f)];
                        assert_eq!(
                            now.0[..3],
                            p.0[..3],
                            "{game:?} {file} {}: pinned {f} moved",
                            sim.name
                        );
                    }
                    simulated += 1;
                }
            }
        }
        assert_eq!(simulated, want_sims, "{game:?}");
    }
    Ok(())
}
