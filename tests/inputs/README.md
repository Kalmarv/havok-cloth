# Real-data inputs

What this crate's real-data tests read, as tests/inputs/mod.rs declares them: the keys of the configuration file
the `GAME_DATA` environment variable names (its layout is described in
[primase-testdata](https://github.com/Kalmarv/primase-testdata)), the file sets and the files of each, and the debug keys
`GAME_DATA_DEBUG` sets. A test whose input is not configured skips with a line naming its key. A `gate` input fails
instead of skipping when the configuration file sets `[run] require = 'gate'`.
The files are your own copies of game files; none are distributed.

<!-- real-data inputs: made by primase_testdata::doc::tables from tests/inputs/mod.rs -->

#### Configuration keys

| key | gate | about | source |
| --- | --- | --- | --- |
| `scratch.root` | - | where real-data tests write their outputs (the OS temporary folder when unset) | - |

#### Oracle sets (`<oracles.root>/<set>/<game>/`)

| set | about | source |
| --- | --- | --- |
| `oracles.corpus` | the games' own files cut from their installs (here: Havok tagfiles with cloth) | the user's own installs: files extracted as stored |

#### Declared files

| key | status | gate | about | source |
| --- | --- | --- | --- | --- |
| `oracles.corpus.sekiro/havok-cloth` | live | - | every distinct tagfile with cloth (`hclClothData` in its type table) of the install: 149 files, loose, and `manifest.tsv` (sha256, file, compendium, binder, member) naming each of the 167 binder members they were found in; tests/real_data.rs simulates every sim cloth of them | the game's own files from the user's install: every archive entry walked (named or not, nested BND4 and DCX opened), each TAG0 member whose type table has `hclClothData` taken out of its DCX envelope and deduplicated by sha256; the same set havok-formats' tests/cloth_corpus.rs reads |
| `oracles.corpus.elden/havok-cloth` | live | - | every distinct tagfile with cloth of the install: 814 files, loose, and `manifest.tsv` naming each of the 1,605 binder members they were found in; tests/real_data.rs simulates every sim cloth of them | the game's own files from the user's install, found as for Sekiro |

<!-- end of real-data inputs -->
