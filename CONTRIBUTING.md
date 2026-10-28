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

## Reporting a difference from the game

Open an issue with the game, its version, the cloth (its file and sim cloth name), what you passed in (timestep, bone
transforms) and what differed. Please do **not** attach game files or recordings made from them: they are game data.
