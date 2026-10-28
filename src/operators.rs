//! The operators that move vertices between buffers and particles: copy, gather, and moving particles to vertices.
//! Each takes vertices from one buffer's space to another's through `out.mesh_from_world * in.world_from_mesh`
//! (an affine product), positions as points and normals as directions.

use crate::buffer::Buffer;
use crate::matrix::Mat4;
use crate::simd::V4;

/// The matrix from `input`'s space to `output`'s.
fn output_from_input(input: &Buffer, output: &Buffer) -> Mat4 {
    output.mesh_from_world.mul_affine(&input.world_from_mesh)
}

/// `hclCopyVerticesOperator`: `count` vertices from `start_in` of `input` to `start_out` of `output`, with their
/// normals when `normals` is set and both buffers have them.
pub fn copy_vertices(
    input: &Buffer,
    output: &mut Buffer,
    count: usize,
    start_in: usize,
    start_out: usize,
    normals: bool,
) {
    let m = output_from_input(input, output);
    let normals = normals && !input.normals.is_empty() && !output.normals.is_empty();
    for i in 0..count {
        let (from, to) = (start_in + i, start_out + i);
        if let (Some(&p), Some(slot)) = (input.positions.get(from), output.positions.get_mut(to)) {
            *slot = m.transform_position(p);
        }
        if normals && let (Some(&n), Some(slot)) = (input.normals.get(from), output.normals.get_mut(to)) {
            *slot = m.transform_direction(n);
        }
    }
}

/// `hclGatherAllVerticesOperator`: every output vertex from the input vertex `input_from_output` names (none for a
/// negative entry), with normals when `normals` is set and both buffers have them.
pub fn gather_all(input: &Buffer, output: &mut Buffer, input_from_output: &[i16], normals: bool) {
    let m = output_from_input(input, output);
    let normals = normals && !input.normals.is_empty() && !output.normals.is_empty();
    for (to, &from) in input_from_output.iter().enumerate() {
        let Ok(from) = usize::try_from(from) else {
            continue;
        };
        if let (Some(&p), Some(slot)) = (input.positions.get(from), output.positions.get_mut(to)) {
            *slot = m.transform_position(p);
        }
        if normals && let (Some(&n), Some(slot)) = (input.normals.get(from), output.normals.get_mut(to)) {
            *slot = m.transform_direction(n);
        }
    }
}

/// `hclGatherSomeVerticesOperator`: the `(input, output)` vertex pairs in order.
pub fn gather_some(input: &Buffer, output: &mut Buffer, pairs: &[(u16, u16)], normals: bool) {
    let m = output_from_input(input, output);
    let normals = normals && !input.normals.is_empty() && !output.normals.is_empty();
    for &(from, to) in pairs {
        let (from, to) = (usize::from(from), usize::from(to));
        if let (Some(&p), Some(slot)) = (input.positions.get(from), output.positions.get_mut(to)) {
            *slot = m.transform_position(p);
        }
        if normals && let (Some(&n), Some(slot)) = (input.normals.get(from), output.normals.get_mut(to)) {
            *slot = m.transform_direction(n);
        }
    }
}

/// `hclMoveParticlesOperator`: each paired particle to its reference vertex in world space. The particle's previous
/// position becomes its current one first, unless the sim cloth has no fixed particles (`fixed_particles` false),
/// when both are set to the vertex.
pub fn move_particles(
    reference: &Buffer,
    pairs: &[(u16, u16)],
    positions: &mut [V4],
    previous: &mut [V4],
    fixed_particles: bool,
) {
    let m = reference.world_from_mesh;
    for &(vertex, particle) in pairs {
        let (Some(&v), p) = (
            reference.positions.get(usize::from(vertex)),
            usize::from(particle),
        ) else {
            continue;
        };
        if p >= positions.len() || p >= previous.len() {
            continue;
        }
        let world = m.transform_position(v);
        previous[p] = if fixed_particles { positions[p] } else { world };
        positions[p] = world;
    }
}

/// What `hclUpdateAllVertexFramesOperator` keeps for its buffer.
#[derive(Clone, Copy, Debug)]
pub struct VertexFrames<'a> {
    /// The buffer's triangles, three vertex indices each.
    pub triangles: &'a [u16],
    /// `vertToNormalID`: per vertex, the normal it shares.
    pub vertex_to_normal_id: &'a [u16],
    /// `numUniqueNormalIDs`.
    pub unique_normal_ids: usize,
    /// `triangleFlips`: one bit per triangle, the most significant bit of a byte first.
    pub triangle_flips: &'a [u8],
    /// `referenceVertices`: per vertex, the vertex its tangent edge points to.
    pub reference_vertices: &'a [u16],
    /// `tangentEdgeCosAngle`.
    pub tangent_edge_cos: &'a [f32],
    /// `tangentEdgeSinAngle`.
    pub tangent_edge_sin: &'a [f32],
    /// `biTangentFlip`.
    pub bitangent_flip: &'a [f32],
    /// `updateNormals`.
    pub normals: bool,
    /// `updateTangents`.
    pub tangents: bool,
    /// `updateBiTangents`.
    pub bitangents: bool,
}

/// `hclUpdateAllVertexFramesOperator` on `buffer`.
///
/// Normals: every triangle's `(b - a) x (c - a)`, its sign flipped where its bit is set, is added to the shared normal
/// of each of its vertices in triangle order; each shared normal is fast-normalised and every vertex takes its own.
/// Tangents (when the buffer has them): the edge to the reference vertex, less its component along the normal,
/// normalised (fast for the vertices in whole groups of four, refined and only when not zero for the rest), turned
/// about the normal by the stored angle: `x*cos + (n x x)*sin`. Bitangents: `(n x t) * flip`.
pub fn update_vertex_frames(f: &VertexFrames<'_>, buffer: &mut Buffer) {
    if f.normals {
        let mut shared = vec![V4::ZERO; f.unique_normal_ids];
        for (t, tri) in f.triangles.as_chunks::<3>().0.iter().enumerate() {
            let [a, b, c] = tri.map(usize::from);
            let (Some(&pa), Some(&pb), Some(&pc)) = (
                buffer.positions.get(a),
                buffer.positions.get(b),
                buffer.positions.get(c),
            ) else {
                continue;
            };
            let mut n = pb.sub(pa).cross(pc.sub(pa));
            let byte = f.triangle_flips.get(t / 8).copied().unwrap_or(0);
            let flip = f32::from((byte >> (7 - t % 8)) & 1);
            n = n.mul(V4::splat(1.0 - flip * 2.0));
            for v in [a, b, c] {
                if let Some(s) = f
                    .vertex_to_normal_id
                    .get(v)
                    .and_then(|&id| shared.get_mut(usize::from(id)))
                {
                    *s = s.add(n);
                }
            }
        }
        for s in &mut shared {
            *s = s.normalized_fast();
        }
        for (v, slot) in buffer.normals.iter_mut().enumerate().take(buffer.positions.len()) {
            if let Some(&n) = f
                .vertex_to_normal_id
                .get(v)
                .and_then(|&id| shared.get(usize::from(id)))
            {
                *slot = n;
            }
        }
    }
    if !f.tangents || buffer.tangents.is_empty() {
        return;
    }
    let count = buffer.positions.len();
    let grouped = count / 4 * 4;
    let bitangents = f.bitangents && !buffer.bitangents.is_empty();
    for v in 0..count {
        let (Some(&n), Some(&p), Some(&r)) = (
            buffer.normals.get(v),
            buffer.positions.get(v),
            f.reference_vertices
                .get(v)
                .and_then(|&r| buffer.positions.get(usize::from(r))),
        ) else {
            continue;
        };
        let e = r.sub(p);
        let mut x = e.sub(n.scale(e.dot3(n)));
        if v < grouped {
            x = x.normalized_fast();
        } else {
            let l2 = x.dot3(x);
            if -l2 < 0.0 {
                x = x.scale(crate::simd::rsqrt23(l2));
            }
        }
        let y = n.cross(x);
        let (cos, sin) = (
            f.tangent_edge_cos.get(v).copied().unwrap_or(1.0),
            f.tangent_edge_sin.get(v).copied().unwrap_or(0.0),
        );
        let t = x.scale(cos).add(y.scale(sin));
        buffer.tangents[v] = t;
        if bitangents {
            let flip = f.bitangent_flip.get(v).copied().unwrap_or(1.0);
            buffer.bitangents[v] = n.cross(t).scale(flip);
        }
    }
}
