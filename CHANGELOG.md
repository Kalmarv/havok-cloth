# Changelog

## Unreleased

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
- `simulate`: actions, the forces a game adds to a sim cloth on every substep (Sekiro adds one to its characters'
  cloth at run time: the data holds none).
- Tests: unit tests of every step against independent expectations, and every sim cloth of both installs simulated
  from its rest pose (`GAME_DATA`).
