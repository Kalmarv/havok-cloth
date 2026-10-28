//! The real-data inputs this crate's tests read, each with what it is, where it comes from and whether a run that
//! requires every gate input fails without it (`primase_testdata::inputs!`; README.md here lists them). Integration
//! tests reach it with `mod inputs;`, and tests/real_data_inputs.rs checks that every query the crate's sources write
//! is declared here. Declare an input here when a test first asks for it.

primase_testdata::inputs! {
    games {
        Sekiro = "sekiro",
        Elden = "elden",
    }
    installs {}
    keys {}
    tools {}
    oracle_sets {
        Corpus = "corpus" {
            about: "the games' own files cut from their installs (here: Havok tagfiles with cloth)",
            source: "the user's own installs: files extracted as stored",
        },
    }
    declared {
        Query::Oracle(OracleSet::Corpus, Game::Sekiro, "havok-cloth") => {
            status: Live,
            gate: false,
            about: r#"every distinct tagfile with cloth (`hclClothData` in its type table) of the install: 149 files, loose, and `manifest.tsv` (sha256, file, compendium, binder, member) naming each of the 167 binder members they were found in; tests/real_data.rs simulates every sim cloth of them"#,
            source: r#"the game's own files from the user's install: every archive entry walked (named or not, nested BND4 and DCX opened), each TAG0 member whose type table has `hclClothData` taken out of its DCX envelope and deduplicated by sha256; the same set havok-formats' tests/cloth_corpus.rs reads"#,
        },
        Query::Oracle(OracleSet::Corpus, Game::Elden, "havok-cloth") => {
            status: Live,
            gate: false,
            about: r#"every distinct tagfile with cloth of the install: 814 files, loose, and `manifest.tsv` naming each of the 1,605 binder members they were found in; tests/real_data.rs simulates every sim cloth of them"#,
            source: r#"the game's own files from the user's install, found as for Sekiro"#,
        },
    }
    debug {}
}
