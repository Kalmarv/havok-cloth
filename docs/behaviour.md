# Behaviour and scope

## Exactness

- Arithmetic is `f32` throughout, with one rounding per operation in the order the games' code performs it. Products and sums are never fused, and three-lane dot products add x and y before z (`src/simd.rs` names the primitives).
- The approximate reciprocal and reciprocal square root are AMD's estimate instructions (`rcpps`, `rsqrtps`), computed from their tables on every machine and target. The games' results differ between processor makers; these are the ones of the AMD processors the captures were recorded on, so the simulate step replays the captures bit for bit everywhere. The wind action's `sinf` is the C runtime's, which picks its code path by processor, so the wind replay is compared only on the capture's processor.
- Where the games compute an arcsine or arccosine, the crate uses the same polynomial.
- The same inputs give bit-identical results from run to run.
- Elden Ring's wind action takes its sines from the C runtime's single-precision `sinf`, which is what the game uses on Windows.

## Stiffness model

`SimState::stiffness_model` selects the legacy or adaptive model. It is legacy by default, matching a newly created game instance, and `prepare` reads it from the state. A simulate configuration's `adaptConstraintStiffness` controls velocity retention over substeps; it does not select the model.

## What it does not do

- It does not read files. It takes havok-formats' cloth data, and the bone transforms and timestep the caller passes.
- It does not run on the GPU or in parallel. One cloth steps on one thread.
- It does not compute landscape collision from world geometry. A caller supplies per-particle contact planes in `SimState::landscape_planes`.
- It does not reproduce Sekiro's run-time action, which adds a force to its characters' cloth that no cloth file holds. A caller can supply such a force through `simulate::Action`.
- Elden Ring's wind is reproduced only for fields sampled once for the whole cloth (kind 0, not local). `wind::global_wind` refuses other fields, and the placement of force fields in the world is the caller's.
- It does not implement what neither game's data uses: anti-pinch constraints, transfer motion, state transitions and blend operators.

## Tests

- Unit tests (`tests/solver.rs`, `tests/operators.rs` and the modules' own tests) check each step on hand-built cloths against independently computed values: closed forms in `f64`, exact-arithmetic cases, conservation of momentum, a link restored to its length, a pinned particle unmoved.
- Real-data tests simulate every sim cloth of both installs (Sekiro: 149 tagfiles, 638 sim cloths; Elden Ring: 814 tagfiles, 2,875 sim cloths), instance and step every cloth data in each state, and check that each transform set matches its file's skeleton.
- Game-capture tests replay cloth steps recorded from the running games (the frames-file layouts are documented in `tests/game_capture.rs`) and compare particles, collidables, constraint states and action forces bit for bit.
- The real-data and capture inputs are declared in `tests/inputs/mod.rs` and listed in `tests/inputs/README.md`. They are game data and are never part of the repository; without a `GAME_DATA` configuration the tests skip.
