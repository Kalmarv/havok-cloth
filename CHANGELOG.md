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
- Tests: unit tests of every step against independent expectations, and every sim cloth of both installs simulated
  from its rest pose (`GAME_DATA`).
