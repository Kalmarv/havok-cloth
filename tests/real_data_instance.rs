//! Real-data check of the cloth instance: every cloth data of every distinct tagfile with cloth of both installs (see
//! tests/inputs/README.md) is instanced and stepped for 10 frames in each of its states, with its bones at the
//! identity; skips without the files. No game data is included in the repository.
//!
//! Every operator class the installs use runs (no refusals), and every particle's and every buffer vertex's `x y z`
//! stays finite. (The `w` lanes are not checked, as in tests/real_data.rs.)
#![allow(
    clippy::unwrap_used,
    reason = "a real-data test reports what it cannot read as its failure"
)]
mod inputs;

use havok_cloth::instance::ClothInstance;
use havok_formats::cloth;
use havok_formats::{Compendium, Tagfile};
use inputs::{Game, OracleSet, Query, expect_found, resolve};
use std::collections::BTreeMap;

/// Per game: the files and the cloth datas instanced (the census: 610 and 2,692 `hclClothData`).
const EXPECTED: [(Game, usize, usize); 2] = [(Game::Sekiro, 149, 610), (Game::Elden, 814, 2_692)];

#[test]
fn every_cloth_data_of_both_installs_runs_every_state_and_stays_finite()
-> Result<(), Box<dyn std::error::Error>> {
    for (game, want_files, want_cloths) in EXPECTED {
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
        let mut instanced = 0usize;
        let mut refused = Vec::new();
        for (file, compendium) in &files {
            let bytes = std::fs::read(dir.join(file))?;
            let compendium = match compendium.as_str() {
                "-" => None,
                c => Some(Compendium::read(&std::fs::read(dir.join(c))?)?),
            };
            let tag = Tagfile::read(&bytes, compendium.as_ref())?;
            let physics = cloth::read(&tag)?;
            for data in &physics.cloths {
                let mut instance = match ClothInstance::new(data) {
                    Ok(i) => i,
                    Err(e) => {
                        refused.push(format!("{file} {}: {e}", data.name));
                        continue;
                    }
                };
                for state in 0..data.states.len() {
                    instance.state = state;
                    for _ in 0..10 {
                        instance.step(1.0 / 30.0, &mut [])?;
                    }
                    for (s, sim) in instance.sims.iter().enumerate() {
                        assert!(
                            sim.positions.iter().all(|p| p.is_finite3()),
                            "{game:?} {file} {} state {state} sim {s}: a particle is not finite",
                            data.name
                        );
                    }
                    for (b, buffer) in instance.buffers.iter().enumerate() {
                        assert!(
                            buffer.positions.iter().all(|p| p.is_finite3()),
                            "{game:?} {file} {} state {state} buffer {b}: a vertex is not finite",
                            data.name
                        );
                    }
                }
                instanced += 1;
            }
        }
        assert!(refused.is_empty(), "{game:?}: refused {refused:#?}");
        assert_eq!(instanced, want_cloths, "{game:?}");
    }
    Ok(())
}
