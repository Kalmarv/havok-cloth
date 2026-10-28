# havok-cloth

A Rust library that simulates Havok cloth as Sekiro and Elden Ring run it, from the cloth data their tagfiles store
(read with [havok-formats](https://github.com/Kalmarv/havok-formats)). It is an independent implementation and
contains no Havok SDK code. Its behaviour is checked against cloth motion recorded from the running games, so that
the same inputs give the game's particle positions to the bit on the machine they were recorded on.

## What it simulates

- **Integration**: Verlet steps with the sim cloth's gravity and per-second damping, substeps, the rescaling of the
  previous positions when the timestep changes, and pinned particles stepped linearly through the substeps.
- **Constraint sets**, in each simulate configuration's order: standard, stretch, compressible and bend links, bend
  stiffness (flat and with a rest curvature, clamped or not), local range (sphere, cylinder, position only),
  transitions between animation and simulation, volume constraints, and the adaptive stiffness per substep.
- **Collision** against the sim cloth's collidables: spheres, capsules, tapered capsules, planes and convex
  geometries, moved by the bones they follow, with friction against their surface velocity; and the virtual
  collision points sampled on the cloth's triangles and edges; and the landscape, as per-particle contact planes the
  caller supplies (the games compute them from the world's collision geometry).
- **Normals** of the particles from the sim cloth's triangles.
- **Skinning and deformation**: the object-space and bone-space deformers that skin buffers and deform display
  meshes from the simulated mesh (triangle frames), with their packed and float vertex blocks.
- **Vertex operators**: copies, gathers, moving pinned particles to their skinned vertices, and the update of
  normals, tangents and bitangents (all vertices or a selection).
- **Bones driven by the cloth**: the mesh bone deformers that place bones on the simulated triangles.
- **Cloth instances**: a cloth data's buffers and transform sets at run time, stepped one cloth state at a time, and
  actions: forces the game adds on every substep.


## Example

```rust
use havok_cloth::simulate::{self, ParticleData, SimState, StiffnessModel};
use havok_formats::cloth::{self, OperatorKind};
use havok_formats::Tagfile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read("c1080_c.hkx")?;
    let physics = cloth::read(&Tagfile::read(&bytes, None)?)?;
    let data = &physics.cloths[0];
    for op in &data.operators {
        let OperatorKind::Simulate { sim_cloth, configs } = &op.kind else { continue };
        let sim = &data.sim_cloths[*sim_cloth as usize];
        let particles = ParticleData::new(sim);
        let mut state = SimState::new(sim);
        for _frame in 0..60 {
            // the bone transforms that drive the collidables, world from bone, one a bone of the transform set
            let bones = [];
            simulate::prepare(&mut state, 1.0 / 60.0, configs[0].sub_steps, StiffnessModel::Legacy);
            simulate::simulate(sim, &configs[0], &mut state, &particles, &[], &bones, &mut [], 1.0 / 60.0);
        }
        println!("{}: {:?}", sim.name, &state.positions[..4]);
    }
    Ok(())
}
```

## What it guarantees

- The arithmetic follows the games' own: `f32` throughout, one rounding per operation in the order the game's code
  performs it (products and sums are never fused, three-lane dot products add x and y before z), the approximate
  reciprocal and reciprocal square root where the game uses them, and the game's own polynomial where it computes an
  arccosine.
- Results vary in the last bits by CPU, as the games' do: the approximate reciprocal and reciprocal square root are the
  processor's estimate instructions (`rcpps`, `rsqrtps`), whose results differ between processor makers. On x86-64 the
  library issues the same instructions, so captures on one machine compare bit for bit; on other targets it uses the
  exact reciprocal (square root), which matches the game's behaviour but not its last bits.
- The same inputs give bit-identical results, run after run and in debug and release builds.
- The particle, buffer and transform indices it follows are the ones havok-formats checks when it reads the data.

## What it does not do

- It does not read files: it takes havok-formats' cloth data, and the bone transforms and timestep the caller passes.
- It does not run on the GPU or in parallel: one cloth steps on one thread (the games' task split does not change
  their results).
- It does not collide with world geometry (the games' landscape collision, which 13 Sekiro and 80 Elden Ring sim
  cloths enable) beyond the contact planes a caller provides.
- It does not implement the games' own actions: Sekiro adds a force to its characters' cloth at run time, which no
  cloth file holds. A caller supplies such forces through `simulate::Action`.
- It does not implement what neither game's data uses: anti-pinch constraints, transfer motion, state transitions,
  blend operators.

## Tests

```sh
cargo test
GAME_DATA=game-data.toml cargo test --release
```

- **Unit tests** of each step on hand-built cloths, against values computed independently (closed forms in `f64`,
  exact-arithmetic cases, conservation of momentum, a link restored to its length, a pinned particle unmoved).
- **Real data**: with a [primase-testdata](https://github.com/Kalmarv/primase-testdata) configuration naming your own
  copies of the games' cloth tagfiles (tests/inputs/README.md lists them), every sim cloth of both installs is
  simulated and checked. Without it these tests skip.
- **Game captures**: cloth recorded from the running games (each operator's inputs and outputs, frame by frame) is
  replayed and compared bit for bit. Captures are game data and are never part of the repository; the tests skip
  without them. A capture records the CPU it was made on, because of the last-bit differences above.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE).
