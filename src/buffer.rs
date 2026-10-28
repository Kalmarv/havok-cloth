//! A vertex buffer at run time: the positions, normals, tangents and bitangents operators read and write, and the
//! matrix that takes its vertices to world space.

use crate::matrix::Mat4;
use crate::simd::V4;

/// One buffer's vertices, each component one four-lane vector a vertex (an absent component is an empty list).
#[derive(Clone, Debug, PartialEq)]
pub struct Buffer {
    /// Positions.
    pub positions: Vec<V4>,
    /// Normals.
    pub normals: Vec<V4>,
    /// Tangents.
    pub tangents: Vec<V4>,
    /// Bitangents.
    pub bitangents: Vec<V4>,
    /// The matrix from the buffer's space to world space (the identity when its vertices are in world space).
    pub world_from_mesh: Mat4,
    /// Its inverse, as the game supplies it: skinning writes vertices into the buffer's space through it.
    pub mesh_from_world: Mat4,
}

impl Buffer {
    /// A buffer of `vertices` zero positions and normals, in world space.
    pub fn new(vertices: usize) -> Buffer {
        Buffer {
            positions: vec![V4::ZERO; vertices],
            normals: vec![V4::ZERO; vertices],
            tangents: Vec::new(),
            bitangents: Vec::new(),
            world_from_mesh: Mat4::IDENTITY,
            mesh_from_world: Mat4::IDENTITY,
        }
    }

    /// The world position of vertex `v` (the buffer's matrix applied as `hkMatrix4` applies it), if it has one.
    pub fn world_position(&self, v: usize) -> Option<V4> {
        self.positions
            .get(v)
            .map(|&p| self.world_from_mesh.transform_position(p))
    }

    /// The world direction of vertex `v`'s normal, if it has one.
    pub fn world_normal(&self, v: usize) -> Option<V4> {
        self.normals
            .get(v)
            .map(|&n| self.world_from_mesh.transform_direction(n))
    }
}
