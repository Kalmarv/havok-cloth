# Contributing

## Build and test

Rust 1.99 (`rust-toolchain.toml` selects it).

```sh
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
```

All four must pass; CI runs the first three on Linux and Windows. The tests that read real game files skip without a
`GAME_DATA` configuration (see [tests/inputs/README.md](tests/inputs/README.md)). If you add a real-data input,
declare it in `tests/inputs/mod.rs` and update the table in `tests/inputs/README.md`; the `real_data_inputs` test
prints the table it expects when they differ.

## Exactness

The solver's arithmetic is part of its contract. A change that reorders a sum, fuses a multiply-add, swaps the
approximate reciprocal for a division or normalises with a different accuracy changes the last bits, and so the
comparison with recorded game motion. Write each step in the order the game computes it (`src/simd.rs` names the
primitives) and keep a test that pins the order where it matters. A test's expected value changes only when an
independent check (a closed form, a conservation law, a game capture) shows the new value.

This crate contains no Havok SDK code, and contributions must not add any: code, comments or structure taken from a
Havok SDK, leaked or licensed, will not be accepted.

## Releasing

A release is a commit that changes a version. Once every CI job has passed on main, CI's release job
(`.github/release.py`) publishes each crate whose version crates.io does not have yet, in dependency order (trusted
publishing: no token is stored anywhere), tags the commit `v<version>` (`<package>-v<version>` for a package with a
version of its own) and creates the GitHub Release with that version's CHANGELOG section as its notes.
A commit that changes no version only runs CI.

1. Bump the version when a change is worth releasing: a patch (0.x.Y) for fixes and data or manifest additions, a minor
   (0.X.0) for new API or any breaking change before 1.0. Run `cargo check` so `Cargo.lock` follows.
2. In the same commit, move the change's CHANGELOG lines under a `## <version> (<date>)` heading.
3. Push to main. Never publish by hand; a new crate's first release is the one exception (the token holder makes it,
   then the crate gets its trusted-publishing config).

## Reporting a difference from the game

Open an issue with the game, its version, the cloth (its file and sim cloth name), what you passed in (timestep, bone
transforms) and what differed. Please do **not** attach game files or recordings made from them: they are game data.
