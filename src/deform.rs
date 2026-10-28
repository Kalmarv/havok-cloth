//! The object-space deformer: the skinning step of the object-space skin and mesh-mesh deform operators. Each vertex
//! has a local position (and normal, tangent) per block of sixteen vertices, and one to eight bones with weights; its
//! bones' matrices are blended by weight and the blend moves its local vectors into the output buffer.

use crate::buffer::Buffer;
use crate::matrix::Mat4;
use crate::simd::V4;
use havok_formats::cloth::{
    BoneSpaceDeformer, BoneSpaceLocalBlock, Components, MeshBoneTriangle, ObjectSpaceDeformer,
    ObjectSpaceLocalBlock, UnpackedLocalBlock,
};

/// An `hkPackedVector3` from its four stored words: each word is the high half of a 32-bit integer, converted to a
/// float and scaled by the float whose bit pattern is the fourth word's, in the high half (the shared exponent).
/// The fourth lane goes through the same steps.
pub fn unpack(words: [i16; 4]) -> V4 {
    let scale = f32::from_bits(u32::from(words[3].cast_unsigned()) << 16);
    V4(words.map(|w| ((i32::from(w) << 16) as f32) * scale))
}

/// The local vectors of a deformer's vertices, sixteen to a block: packed or as stored floats.
#[derive(Clone, Copy, Debug)]
pub enum LocalBlocks<'a> {
    /// `localPs`, `localPNs`, `localPNTs`: packed vectors.
    Packed(&'a [ObjectSpaceLocalBlock]),
    /// `localUnpackedPs`, `localUnpackedPNs`, `localUnpackedPNTs`: floats.
    Unpacked(&'a [UnpackedLocalBlock]),
}

/// One vertex's local vectors: position, normal and tangent (the last two when the block holds them).
type LocalVectors = (V4, Option<V4>, Option<V4>);

impl LocalBlocks<'_> {
    /// Vertex `i` of block `b`, or `None` past the last block.
    fn get(&self, b: usize, i: usize) -> Option<LocalVectors> {
        match self {
            LocalBlocks::Packed(blocks) => {
                let block = blocks.get(b)?;
                Some((
                    unpack(block.positions[i]),
                    block.normals.map(|n| unpack(n[i])),
                    block.tangents.map(|t| unpack(t[i])),
                ))
            }
            LocalBlocks::Unpacked(blocks) => {
                let block = blocks.get(b)?;
                Some((
                    V4(block.positions[i]),
                    block.normals.map(|n| V4(n[i])),
                    block.tangents.map(|t| V4(t[i])),
                ))
            }
        }
    }
}

/// The matrix of each bone a deformer refers to: `mesh_from_world * (bones[t] * bone_from_skin_mesh)`, both products
/// affine, where `t` runs through `subset` (all of `bones` in order when `subset` is empty).
pub fn composite_matrices(
    mesh_from_world: &Mat4,
    bones: &[Mat4],
    subset: &[u16],
    bone_from_skin_mesh: &[[f32; 16]],
) -> Vec<Mat4> {
    let bone = |i: usize| -> Mat4 {
        let t = subset.get(i).map_or(i, |&t| usize::from(t));
        bones.get(t).copied().unwrap_or(Mat4::IDENTITY)
    };
    bone_from_skin_mesh
        .iter()
        .enumerate()
        .map(|(i, &inverse_bind)| {
            mesh_from_world.mul_affine(&bone(i).mul_affine(&Mat4::from_columns(inverse_bind)))
        })
        .collect()
}

/// The matrix of each bone a bone-space deformer refers to: `mesh_from_world * bones[t]` (affine), where `t` runs
/// through `subset` (all of `bones` in order when `subset` is empty). The bone-space skin operators pass the output
/// buffer's `mesh_from_world` and the transform set; the bone-space mesh-mesh deforms pass the matrix from the input
/// buffer's space to the output's and the triangle frames.
pub fn bone_space_composite(mesh_from_world: &Mat4, bones: &[Mat4], subset: &[u16]) -> Vec<Mat4> {
    if subset.is_empty() {
        bones.iter().map(|b| mesh_from_world.mul_affine(b)).collect()
    } else {
        subset
            .iter()
            .map(|&t| mesh_from_world.mul_affine(bones.get(usize::from(t)).unwrap_or(&Mat4::IDENTITY)))
            .collect()
    }
}

/// How a mesh-mesh deform treats the length of a triangle frame's normal axis (`scaleNormalBehaviour`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormalScale {
    /// 0: made unit length (fast normalisation).
    Ignore,
    /// 1: left as the cross product of the two edges.
    Apply,
    /// 2: divided by its squared length (the processor's reciprocal estimate).
    Invert,
}

impl NormalScale {
    /// The stored value: 0, 1 or 2; any other value as 1.
    pub fn from_stored(v: u32) -> NormalScale {
        match v {
            0 => NormalScale::Ignore,
            2 => NormalScale::Invert,
            _ => NormalScale::Apply,
        }
    }
}

/// A triangle's frame: its centroid `((a + b) + c) * (1/3)` as the origin, `a - centroid` and `b - centroid` as the
/// first two axes, their cross product (scaled by `scale`) as the third; the fourth row `(0, 0, 0, 1)`.
pub fn triangle_frame(a: V4, b: V4, c: V4, scale: NormalScale) -> Mat4 {
    let centroid = a.add(b).add(c).scale(1.0 / 3.0);
    let s = a.sub(centroid);
    let t = b.sub(centroid);
    let mut n = s.cross(t);
    match scale {
        NormalScale::Ignore => n = n.normalized_fast(),
        NormalScale::Apply => {}
        NormalScale::Invert => n = n.scale(crate::simd::rcp_estimate(n.dot3(n))),
    }
    Mat4 {
        cols: [s.with_w(0.0), t.with_w(0.0), n.with_w(0.0), centroid.with_w(1.0)],
    }
}

/// The frames of the triangles of `positions` that a mesh-mesh deform reads: those `subset` names (triangle indices
/// into `triangles`, three vertex indices each), or every triangle when `subset` is empty.
pub fn triangle_frames(positions: &[V4], triangles: &[u16], subset: &[u16], scale: NormalScale) -> Vec<Mat4> {
    let frame = |t: usize| -> Mat4 {
        let v = |k: usize| {
            triangles
                .get(t * 3 + k)
                .and_then(|&i| positions.get(usize::from(i)))
                .copied()
                .unwrap_or(V4::ZERO)
        };
        triangle_frame(v(0), v(1), v(2), scale)
    };
    if subset.is_empty() {
        (0..triangles.len() / 3).map(frame).collect()
    } else {
        subset.iter().map(|&t| frame(usize::from(t))).collect()
    }
}

/// A bone matrix made orthonormal as the bone deformers do it: `y = z x x`, then `x = y x z`, each axis fast-normalised
/// (z first); the translation kept and the fourth row set to `(0, 0, 0, 1)`.
fn orthonormal_bone(m: &Mat4) -> Mat4 {
    let [x, _, z, position] = m.cols;
    let y = z.cross(x);
    let x = y.cross(z);
    let (z, y, x) = (z.normalized_fast(), y.normalized_fast(), x.normalized_fast());
    Mat4 {
        cols: [x.with_w(0.0), y.with_w(0.0), z.with_w(0.0), position.with_w(1.0)],
    }
}

/// `hclMeshBoneDeformOperator`: each bone with entries (`start_for_bone[b]..start_for_bone[b + 1]` of `pairs`) is the
/// weighted sum, from zero, of its triangles' frames times their local bone transforms (affine products), each scaled
/// by its weight, made orthonormal; bones without entries are left as they are.
pub fn mesh_bone_deform(
    positions: &[V4],
    triangles: &[u16],
    pairs: &[MeshBoneTriangle],
    start_for_bone: &[u16],
    bones: &mut [Mat4],
) {
    let frames = triangle_frames(positions, triangles, &[], NormalScale::Apply);
    for (b, bone) in bones.iter_mut().enumerate() {
        let (Some(&start), Some(&end)) = (start_for_bone.get(b), start_for_bone.get(b + 1)) else {
            break;
        };
        let entries = pairs.get(usize::from(start)..usize::from(end)).unwrap_or(&[]);
        if entries.is_empty() {
            continue;
        }
        let mut blended = Mat4 { cols: [V4::ZERO; 4] };
        for e in entries {
            let frame = frames
                .get(usize::from(e.triangle))
                .copied()
                .unwrap_or(Mat4::IDENTITY);
            let m = frame
                .mul_affine(&Mat4::from_columns(e.local_bone_transform))
                .scale(e.weight);
            blended = blended.add(&m);
        }
        *bone = orthonormal_bone(&blended);
    }
}

/// `hclSimpleMeshBoneDeformOperator`: bones placed on triangles of a buffer. For each `(bone byte offset, triangle
/// byte offset)` pair, the triangle's frame (normal axis as the cross product) times the bone's local transform (the
/// full product), made orthonormal: `y = z x x`, then `x = y x z`, each axis fast-normalised; the translation kept and
/// the fourth row set to `(0, 0, 0, 1)`. Offsets are in bytes, as stored: 64 a bone, 2 a triangle index.
pub fn simple_mesh_bone_deform(
    positions: &[V4],
    triangles: &[u16],
    pairs: &[(u16, u16)],
    local_bone_transforms: &[[f32; 16]],
    bones: &mut [Mat4],
) {
    for (&(bone_offset, triangle_offset), local) in pairs.iter().zip(local_bone_transforms) {
        let t = usize::from(triangle_offset) / 2;
        let v = |k: usize| {
            triangles
                .get(t + k)
                .and_then(|&i| positions.get(usize::from(i)))
                .copied()
                .unwrap_or(V4::ZERO)
        };
        let frame = triangle_frame(v(0), v(1), v(2), NormalScale::Apply);
        let m = frame.mul(&Mat4::from_columns(*local));
        if let Some(bone) = bones.get_mut(usize::from(bone_offset) / 64) {
            *bone = orthonormal_bone(&m);
        }
    }
}

/// The blocks a control byte selects: how many bones each vertex of the block blends (0: none, the next batch).
fn bones_per_vertex(control: u8) -> Option<usize> {
    match control {
        0 => Some(4),
        1 => Some(3),
        2 => Some(2),
        3 => Some(1),
        4 => Some(0),
        5 => Some(8),
        6 => Some(7),
        7 => Some(6),
        8 => Some(5),
        _ => None,
    }
}

/// A stored weight as a float: 8-bit weights (two to four bones) are fractions of 255, 16-bit ones (five to eight)
/// of 65535.
fn weight(stored: u16, bones: usize) -> f32 {
    if bones <= 4 {
        f32::from(stored) * (1.0 / 255.0)
    } else {
        f32::from(stored) * (1.0 / 65535.0)
    }
}

/// Runs the deformer: for each control byte in turn, the next block of its kind and the next local block. Each vertex's
/// bone matrices are scaled by their weights and summed in order (every lane), then move its local position as a
/// point and its normal and tangent as directions; the results are written whole (four lanes) at the vertex's index
/// in `out`, for the components the operator writes.
pub fn deform(
    deformer: &ObjectSpaceDeformer,
    local: LocalBlocks<'_>,
    components: Components,
    composite: &[Mat4],
    out: &mut Buffer,
) {
    let mut next_block = [0usize; 8];
    let mut next_local = 0usize;
    let matrix = |b: u16| composite.get(usize::from(b)).copied().unwrap_or(Mat4::IDENTITY);
    for &control in &deformer.control_bytes {
        let Some(n) = bones_per_vertex(control) else {
            continue;
        };
        if n == 0 {
            continue;
        }
        let Some(block) = deformer.blocks[n - 1].get(next_block[n - 1]) else {
            break;
        };
        next_block[n - 1] += 1;
        for (i, &vertex) in block.vertices.iter().enumerate() {
            let Some((position, normal, tangent)) = local.get(next_local, i) else {
                break;
            };
            let bones = &block.bones[i * n..i * n + n];
            let blended = if n == 1 {
                matrix(bones[0])
            } else {
                let weights = &block.weights[i * n..i * n + n];
                let mut acc = matrix(bones[0]).scale(weight(weights[0], n));
                for k in 1..n {
                    acc = acc.add(&matrix(bones[k]).scale(weight(weights[k], n)));
                }
                acc
            };
            let v = usize::from(vertex);
            if let Some(slot) = out.positions.get_mut(v) {
                *slot = blended.transform_position(position);
            }
            if components != Components::P
                && let (Some(nrm), Some(slot)) = (normal, out.normals.get_mut(v))
            {
                *slot = blended.transform_direction(nrm);
            }
            if components == Components::PNT
                && let (Some(tng), Some(slot)) = (tangent, out.tangents.get_mut(v))
            {
                *slot = blended.transform_direction(tng);
            }
        }
        next_local += 1;
    }
}

/// The local vectors of a bone-space deformer: one per (vertex, bone) influence, in block order.
#[derive(Clone, Copy, Debug)]
pub enum BoneSpaceLocal<'a> {
    /// `localPs`, `localPNs`, `localPNTs`: float positions, packed normals and tangents.
    Packed(&'a [BoneSpaceLocalBlock]),
    /// `localUnpackedPs`, `localUnpackedPNs`, `localUnpackedPNTs`: floats.
    Unpacked(&'a [UnpackedLocalBlock]),
}

impl BoneSpaceLocal<'_> {
    /// Influence `i` of block `b`, or `None` past the last block.
    fn get(&self, b: usize, i: usize) -> Option<LocalVectors> {
        match self {
            BoneSpaceLocal::Packed(blocks) => {
                let block = blocks.get(b)?;
                Some((
                    V4(*block.positions.get(i)?),
                    block.normals.map(|n| unpack(n[i])),
                    block.tangents.map(|t| unpack(t[i])),
                ))
            }
            BoneSpaceLocal::Unpacked(blocks) => {
                let block = blocks.get(b)?;
                Some((
                    V4(*block.positions.get(i)?),
                    block.normals.map(|n| V4(n[i])),
                    block.tangents.map(|t| V4(t[i])),
                ))
            }
        }
    }
}

/// The bone-space deformer's blocks by control byte: 0 four bones a vertex, 1 three, 2 two, 3 one, 4 the next batch.
fn bone_space_bones(control: u8) -> Option<usize> {
    match control {
        0 => Some(4),
        1 => Some(3),
        2 => Some(2),
        3 => Some(1),
        4 => Some(0),
        _ => None,
    }
}

/// Runs a bone-space deformer: each influence's local position is a weighted point (its weight in the fourth lane)
/// multiplied by its bone's matrix (`((c0*x + c1*y) + c2*z) + c3*w`); a vertex's position is the sum of its
/// influences from zero, and its normal and tangent the sums of its influences' directions. One-bone vertices take
/// their single influence as it is.
pub fn bone_space_deform(
    deformer: &BoneSpaceDeformer,
    local: BoneSpaceLocal<'_>,
    components: Components,
    composite: &[Mat4],
    out: &mut Buffer,
) {
    let mut next_block = [0usize; 4];
    let mut next_local = 0usize;
    let matrix = |b: u16| composite.get(usize::from(b)).copied().unwrap_or(Mat4::IDENTITY);
    for &control in &deformer.control_bytes {
        let Some(n) = bone_space_bones(control) else {
            continue;
        };
        if n == 0 {
            continue;
        }
        let Some(block) = deformer.blocks[n - 1].get(next_block[n - 1]) else {
            break;
        };
        next_block[n - 1] += 1;
        for (i, &vertex) in block.vertices.iter().enumerate() {
            let mut position = V4::ZERO;
            let mut normal = V4::ZERO;
            let mut tangent = V4::ZERO;
            for k in 0..n {
                let influence = i * n + k;
                let (Some((p, nrm, tng)), Some(&bone)) =
                    (local.get(next_local, influence), block.bones.get(influence))
                else {
                    continue;
                };
                let m = matrix(bone);
                let moved = (
                    m.multiply_vector(p),
                    nrm.map(|v| m.transform_direction(v)),
                    tng.map(|v| m.transform_direction(v)),
                );
                if n == 1 {
                    position = moved.0;
                    normal = moved.1.unwrap_or(V4::ZERO);
                    tangent = moved.2.unwrap_or(V4::ZERO);
                } else {
                    position = position.add(moved.0);
                    normal = normal.add(moved.1.unwrap_or(V4::ZERO));
                    tangent = tangent.add(moved.2.unwrap_or(V4::ZERO));
                }
            }
            let v = usize::from(vertex);
            if let Some(slot) = out.positions.get_mut(v) {
                *slot = position;
            }
            if components != Components::P
                && let Some(slot) = out.normals.get_mut(v)
            {
                *slot = normal;
            }
            if components == Components::PNT
                && let Some(slot) = out.tangents.get_mut(v)
            {
                *slot = tangent;
            }
        }
        next_local += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_vectors_unpack_by_their_shared_exponent() {
        // the fourth word 0x3f80 makes the scale 1.0 (0x3f800000): each word w becomes w * 65536
        assert_eq!(
            unpack([1, -2, 0, 0x3f80]),
            V4::new(65536.0, -131_072.0, 0.0, 0x3f80 as f32 * 65536.0)
        );
        // 0x3780 is 2^-16 (0x37800000): words come back as themselves
        let v = unpack([3, -4, 5, 0x3780]);
        assert_eq!((v.x(), v.y(), v.z()), (3.0, -4.0, 5.0));
    }

    #[test]
    fn weights_are_fractions_of_the_stored_range() {
        assert_eq!(weight(255, 4), 1.0);
        assert_eq!(weight(65535, 8), 1.0);
        assert_eq!(weight(0, 2), 0.0);
    }
}
