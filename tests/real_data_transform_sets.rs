//! Real-data check of what a transform set is: in every distinct tagfile with cloth of both installs (see
//! tests/inputs/README.md), each cloth data has one transform set, and the file's animation container holds one
//! skeleton, whose name and bone count the transform set has. A caller fills transform i with the world transform of
//! that skeleton's bone i (tests/game_capture.rs checks the order on the game's own transforms). Skips without the
//! files. No game data is included in the repository.
#![allow(
    clippy::unwrap_used,
    reason = "a real-data test reports what it cannot read as its failure"
)]
mod inputs;

use havok_formats::animation;
use havok_formats::cloth;
use havok_formats::{Compendium, Tagfile};
use inputs::{Game, OracleSet, Query, expect_found, resolve};
use std::collections::BTreeMap;

/// Per game: the files, the files with a cloth data, and the cloth datas (each with its one transform set). Measured
/// with an independent script (a minimal TAG0 reader in Python over the same corpus): every one of Sekiro's 610 and
/// Elden Ring's 2,692 transform sets has the name and bone count of its file's one skeleton.
const EXPECTED: [(Game, usize, usize, usize); 2] =
    [(Game::Sekiro, 149, 145, 610), (Game::Elden, 814, 741, 2_692)];

#[test]
fn every_transform_set_is_the_skeleton_of_its_file() -> Result<(), Box<dyn std::error::Error>> {
    for (game, want_files, want_with_cloth, want_sets) in EXPECTED {
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
        let (mut with_cloth, mut sets) = (0usize, 0usize);
        for (file, compendium) in &files {
            let bytes = std::fs::read(dir.join(file))?;
            let compendium = match compendium.as_str() {
                "-" => None,
                c => Some(Compendium::read(&std::fs::read(dir.join(c))?)?),
            };
            let tag = Tagfile::read(&bytes, compendium.as_ref())?;
            let physics = cloth::read(&tag)?;
            if physics.cloths.is_empty() {
                continue;
            }
            with_cloth += 1;
            let skeletons = animation::container(&tag)?.skeletons;
            assert_eq!(skeletons.len(), 1, "{game:?} {file}: skeletons");
            let skeleton = &skeletons[0];
            for data in &physics.cloths {
                assert_eq!(
                    data.transform_sets.len(),
                    1,
                    "{game:?} {file} {}: transform sets",
                    data.name
                );
                let set = &data.transform_sets[0];
                assert_eq!(
                    (set.name.as_str(), set.transforms as usize),
                    (skeleton.name.as_str(), skeleton.bones.len()),
                    "{game:?} {file} {}: transform set against the skeleton",
                    data.name
                );
                sets += 1;
            }
        }
        assert_eq!((with_cloth, sets), (want_with_cloth, want_sets), "{game:?}");
    }
    Ok(())
}
