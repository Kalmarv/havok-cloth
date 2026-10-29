# Changelog

## 0.2.1 (2026-10-29)

- `instance`: an operator reads the triangles of the buffer it names, also where the state holds that buffer's
  vertices in a shadow buffer. A shadowed display buffer's vertex frames (`hclUpdateAllVertexFramesOperator`) were
  computed over the shadow's triangles, which neither game states, so every normal of a shadowed display buffer was
  zero (Sekiro's scarf and robes among them: 484 such operators in Sekiro, 2,533 in Elden Ring).

## 0.2.0 (2026-10-29)

- `simulate`: the stiffness model is the sim instance's (`SimState::stiffness_model`, legacy by default, as the game
  creates its instances), no longer chosen by a configuration's `adaptConstraintStiffness`, which now only lets the
  constraint sets keep the particles' velocity over substeps; `prepare` and `set_stiffness` read it from the state
  (breaking: their model argument is gone). Every set of a cloth whose configuration adapts its stiffness ran
  softened over substeps where the game runs it at full stiffness.
- `simulate`: the substep is the time step times the reciprocal of the substep count, as the game computes it (it
  differed in the last bit for three substeps).
- `volume`: a batch's target adds the frame's centre after its first column, as the game does (the moved particles
  differed in the last bit).
- `constraints`: a transition's blend divides where it took the refined reciprocal (`1/period`, `1/s`, the move
  `(bound - distance) / distance`) and associates the bound as the game does (the blended particles differed in the
  last bits).

## 0.1.0 (2026-10-29)

The first version: the simulate operator for one sim cloth, as Sekiro and Elden Ring run it.

- `simd`: four-lane `f32` arithmetic in the games' operation order, and the processor's reciprocal and reciprocal
  square root estimates (exact operations off x86-64).
- `matrix`: the cloth data's matrices and both ways the games move a point with one.
- `simulate`: Verlet integration, timestep rescaling, damping, substeps with pinned particles stepped linearly, the
  ordered or plain solve with the legacy or adaptive stiffness model, and particle normals.
- `constraints`: standard, stretch, compressible and bend links, bend stiffness (flat and rest-curvature, clamped or
  not), local range (sphere, cylinder, position only) and transitions.
- `collide`: collidables driven by bones and stepped through the substeps, the closest-point queries of spheres,
  capsules, tapered capsules, planes and convex geometries (four at a time and one at a time, as the games query them),
  and the contact solve with friction.
- `deform`: the object-space deformer (one to eight bones a vertex, packed or float local blocks, the composite
  matrices of the skin operators) and the bone-space deformer; the triangle frames of the mesh-mesh deform operators
  with their three normal-scaling modes. Recorded from Sekiro, the object-space skin and the particle move place every
  pinned particle of the player's scarf bit-exactly on 299 of 299 steps.
- `operators`: copy, gather (all and some vertices), moving particles to their reference vertices, and the vertex
  frames update (normals, tangents, bitangents).
- `buffer`: a buffer carries both of its matrices (`world_from_mesh` and `mesh_from_world`), as the game supplies them.
- `instance`: `ClothInstance` runs one cloth data's states: every operator's notification, then the state's operators
  in order, over its buffers (shadow buffers per state, particle buffers kept in step with their sim cloths) and
  transform sets. Every cloth data of both installs instances and runs every state.
- `deform`: the bone-space skin composite, and the bones the mesh bone deformers place on triangles (single and
  weighted).
- `operators`: the vertex frames update of a selection of vertices.
- `collide`: virtual collision points against the collidables that collide with them (triangle and edge fans), and
  landscape collision against per-particle contact planes the caller supplies (the game computes them from the
  world).
- `volume`: the volume constraint (weighted centre, the rotation of the polar decomposition from a symmetric
  eigen-solve warm-started from the last, particles moved towards their rest offsets in that frame).
- `simulate`: actions, the forces a game adds to a sim cloth on every substep (Sekiro adds one to its characters'
  cloth at run time: the data holds none).
- Tests: unit tests of every step against independent expectations, and every sim cloth of both installs simulated
  from its rest pose (`GAME_DATA`).

