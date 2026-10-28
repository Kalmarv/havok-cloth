//! The deformers and vertex operators on hand-built data, each expectation exact by construction (powers of two and
//! translations, where every rounding is exact).
#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    reason = "tests compare exact values and report failures by panicking"
)]

use havok_cloth::buffer::Buffer;
use havok_cloth::deform::{self, BoneSpaceLocal, LocalBlocks, NormalScale};
use havok_cloth::matrix::Mat4;
use havok_cloth::operators;
use havok_cloth::simd::V4;
use havok_formats::cloth::{
    BoneSpaceBlock, BoneSpaceDeformer, Components, LocalBlock, ObjectSpaceBlock, ObjectSpaceDeformer,
};

fn translation(x: f32, y: f32, z: f32) -> Mat4 {
    let mut m = Mat4::IDENTITY;
    m.cols[3] = V4::new(x, y, z, 1.0);
    m
}

fn object_space(blocks: [Vec<ObjectSpaceBlock>; 8], control_bytes: Vec<u8>) -> ObjectSpaceDeformer {
    ObjectSpaceDeformer {
        blocks,
        control_bytes,
        start_vertex: 0,
        end_vertex: 15,
        partial_write: false,
    }
}

/// Sixteen local positions `(i, 0, 0)` with normals `(0, 1, 0)`, as stored floats.
fn unpacked_block() -> LocalBlock<[f32; 4], [f32; 4]> {
    LocalBlock {
        positions: std::array::from_fn(|i| [i as f32, 0.0, 0.0, 1.0]),
        normals: Some([[0.0, 1.0, 0.0, 0.0]; 16]),
        tangents: None,
    }
}

#[test]
fn one_bone_vertices_follow_their_bone() {
    let block = ObjectSpaceBlock {
        vertices: std::array::from_fn(|i| 15 - i as u16),
        bones: vec![0; 16],
        weights: Vec::new(),
    };
    let mut blocks: [Vec<ObjectSpaceBlock>; 8] = Default::default();
    blocks[0] = vec![block];
    // control byte 3: a one-bone block
    let d = object_space(blocks, vec![3]);
    let locals = [unpacked_block()];
    let mut out = Buffer::new(16);
    deform::deform(
        &d,
        LocalBlocks::Unpacked(&locals),
        Components::PN,
        &[translation(10.0, 20.0, 30.0)],
        &mut out,
    );
    for i in 0..16 {
        // vertex 15 - i takes local position i
        assert_eq!(out.positions[15 - i], V4::new(10.0 + i as f32, 20.0, 30.0, 1.0));
        assert_eq!(out.normals[15 - i], V4::new(0.0, 1.0, 0.0, 0.0));
    }
}

#[test]
fn two_bone_vertices_blend_their_bones_by_weight() {
    // weights 255 and 0 (8-bit): all of the first bone; then 0 and 255: all of the second
    let mut bones = Vec::new();
    let mut weights = Vec::new();
    for i in 0..16 {
        bones.extend([0, 1]);
        weights.extend(if i % 2 == 0 { [255, 0] } else { [0, 255] });
    }
    let block = ObjectSpaceBlock {
        vertices: std::array::from_fn(|i| i as u16),
        bones,
        weights,
    };
    let mut blocks: [Vec<ObjectSpaceBlock>; 8] = Default::default();
    blocks[1] = vec![block];
    // control byte 2: a two-bone block
    let d = object_space(blocks, vec![2]);
    let locals = [unpacked_block()];
    let mut out = Buffer::new(16);
    let composite = [translation(1.0, 0.0, 0.0), translation(0.0, 4.0, 0.0)];
    deform::deform(
        &d,
        LocalBlocks::Unpacked(&locals),
        Components::P,
        &composite,
        &mut out,
    );
    for i in 0..16 {
        let want = if i % 2 == 0 {
            V4::new(i as f32 + 1.0, 0.0, 0.0, 1.0)
        } else {
            V4::new(i as f32, 4.0, 0.0, 1.0)
        };
        assert_eq!(out.positions[i], want, "vertex {i}");
        // P only: normals untouched
        assert_eq!(out.normals[i], V4::ZERO);
    }
}

#[test]
fn half_and_half_sixteen_bit_weights_land_halfway() {
    // five-bone block with 16-bit weights: 0x8000 and 0x7fff of 65535 on two bones, 0 on three; the blend's x
    // translation is 2 * (32768/65535) + 0 = a value strictly between 0 and 2, and the y one 2 * 32767/65535
    let mut bones = Vec::new();
    let mut weights = Vec::new();
    for _ in 0..16 {
        bones.extend([0, 1, 2, 2, 2]);
        weights.extend([0x8000u16, 0x7fff, 0, 0, 0]);
    }
    let block = ObjectSpaceBlock {
        vertices: std::array::from_fn(|i| i as u16),
        bones,
        weights,
    };
    let mut blocks: [Vec<ObjectSpaceBlock>; 8] = Default::default();
    blocks[4] = vec![block];
    // control byte 8: a five-bone block
    let d = object_space(blocks, vec![8]);
    let locals = [LocalBlock {
        positions: [[0.0, 0.0, 0.0, 1.0]; 16],
        normals: None,
        tangents: None,
    }];
    let mut out = Buffer::new(16);
    let composite = [
        translation(2.0, 0.0, 0.0),
        translation(0.0, 2.0, 0.0),
        Mat4::IDENTITY,
    ];
    deform::deform(
        &d,
        LocalBlocks::Unpacked(&locals),
        Components::P,
        &composite,
        &mut out,
    );
    let w0 = 32768.0f32 * (1.0 / 65535.0);
    let w1 = 32767.0f32 * (1.0 / 65535.0);
    // the blended matrix's translation column: (2*w0 + 0) + 0, then +0 for the zero-weight bones
    assert_eq!(out.positions[0].x(), 2.0 * w0);
    assert_eq!(out.positions[0].y(), 2.0 * w1);
    assert!(out.positions[0].x() > 1.0 && out.positions[0].y() < 1.0);
}

#[test]
fn next_batch_control_bytes_consume_nothing() {
    let block = ObjectSpaceBlock {
        vertices: std::array::from_fn(|i| i as u16),
        bones: vec![0; 16],
        weights: Vec::new(),
    };
    let mut blocks: [Vec<ObjectSpaceBlock>; 8] = Default::default();
    blocks[0] = vec![block];
    // 4 (next batch) before the one-bone block: the block still uses local block 0
    let d = object_space(blocks, vec![4, 3]);
    let locals = [unpacked_block()];
    let mut out = Buffer::new(16);
    deform::deform(
        &d,
        LocalBlocks::Unpacked(&locals),
        Components::P,
        &[Mat4::IDENTITY],
        &mut out,
    );
    assert_eq!(out.positions[5], V4::new(5.0, 0.0, 0.0, 1.0));
}

#[test]
fn composite_matrices_chain_mesh_bone_and_inverse_bind() {
    let bones = [translation(1.0, 0.0, 0.0), translation(0.0, 1.0, 0.0)];
    let inverse_bind = [translation(0.0, 0.0, 8.0).to_columns()];
    // subset picks bone 1 for the only inverse bind matrix
    let c = deform::composite_matrices(&translation(-1.0, -1.0, -1.0), &bones, &[1], &inverse_bind);
    assert_eq!(c.len(), 1);
    assert_eq!(c[0].translation(), V4::new(-1.0, 0.0, 7.0, 1.0));
}

#[test]
fn bone_space_influences_sum_weighted_points() {
    // two bones a vertex, eight vertices: each influence's local point is stored weighted, all four lanes times its
    // weight 0.5: vertex v at x = v becomes (0.5 v, 0, 0, 0.5)
    let block = BoneSpaceBlock {
        vertices: (0..8).collect(),
        bones: (0..16).map(|k| (k % 2) as u16).collect(),
    };
    let d = BoneSpaceDeformer {
        blocks: [Vec::new(), vec![block], Vec::new(), Vec::new()],
        control_bytes: vec![2],
        start_vertex: 0,
        end_vertex: 7,
        partial_write: false,
    };
    let locals = [LocalBlock {
        positions: std::array::from_fn(|k| [0.5 * (k / 2) as f32, 0.0, 0.0, 0.5]),
        normals: Some([[0, 0x4000, 0, 0x3780]; 16]),
        tangents: None,
    }];
    let mut out = Buffer::new(8);
    let composite = [translation(2.0, 0.0, 0.0), translation(0.0, 6.0, 0.0)];
    deform::bone_space_deform(
        &d,
        BoneSpaceLocal::Packed(&locals),
        Components::PN,
        &composite,
        &mut out,
    );
    for v in 0..8 {
        // (0.5 v + 0.5*2) + 0.5 v in x, 0.5*6 in y, w 0.5 + 0.5
        assert_eq!(
            out.positions[v],
            V4::new(v as f32 + 1.0, 3.0, 0.0, 1.0),
            "vertex {v}"
        );
        // each influence's normal unpacks to (0, 16384, 0, *); translations leave directions alone; the two add
        let n = deform::unpack([0, 0x4000, 0, 0x3780]);
        assert_eq!(out.normals[v].y(), n.y() + n.y());
    }
}

#[test]
fn triangle_frames_put_the_centroid_in_the_fourth_column() {
    let p = [
        V4::new(3.0, 0.0, 0.0, 1.0),
        V4::new(0.0, 3.0, 0.0, 1.0),
        V4::new(0.0, 0.0, 0.0, 1.0),
    ];
    let f = deform::triangle_frames(&p, &[0, 1, 2], &[], NormalScale::Apply);
    let centroid = 3.0f32 * (1.0 / 3.0);
    assert_eq!(f[0].cols[3], V4::new(centroid, centroid, 0.0, 1.0));
    assert_eq!(f[0].cols[0], V4::new(3.0 - centroid, -centroid, 0.0, 0.0));
    // s x t: (3-c, -c, 0) x (-c, 3-c, 0) = (0, 0, (3-c)^2 - c^2)
    let s = 3.0 - centroid;
    assert_eq!(f[0].cols[2], V4::new(0.0, 0.0, s * s - centroid * centroid, 0.0));
    // the subset form picks triangles by index
    let g = deform::triangle_frames(&p, &[2, 2, 2, 0, 1, 2], &[1], NormalScale::Apply);
    assert_eq!(g, f);
}

#[test]
fn copy_and_gather_move_vertices_between_spaces() {
    let mut input = Buffer::new(3);
    input.positions = vec![
        V4::new(1.0, 0.0, 0.0, 1.0),
        V4::new(2.0, 0.0, 0.0, 1.0),
        V4::new(3.0, 0.0, 0.0, 1.0),
    ];
    input.normals = vec![V4::new(0.0, 1.0, 0.0, 0.0); 3];
    input.world_from_mesh = translation(10.0, 0.0, 0.0);
    let mut output = Buffer::new(3);
    output.mesh_from_world = translation(0.0, -5.0, 0.0);
    operators::copy_vertices(&input, &mut output, 2, 1, 0, true);
    assert_eq!(output.positions[0], V4::new(12.0, -5.0, 0.0, 1.0));
    assert_eq!(output.positions[1], V4::new(13.0, -5.0, 0.0, 1.0));
    assert_eq!(output.positions[2], V4::ZERO);
    assert_eq!(output.normals[0], V4::new(0.0, 1.0, 0.0, 0.0));
    let mut gathered = Buffer::new(3);
    operators::gather_all(&input, &mut gathered, &[2, -1, 0], false);
    assert_eq!(gathered.positions[0], V4::new(13.0, 0.0, 0.0, 1.0));
    assert_eq!(gathered.positions[1], V4::ZERO);
    assert_eq!(gathered.positions[2], V4::new(11.0, 0.0, 0.0, 1.0));
    assert_eq!(gathered.normals[0], V4::ZERO, "normals not gathered");
}

#[test]
fn moved_particles_keep_or_lose_their_velocity() {
    let mut reference = Buffer::new(1);
    reference.positions = vec![V4::new(1.0, 2.0, 3.0, 1.0)];
    let mut pos = vec![V4::new(9.0, 9.0, 9.0, 1.0)];
    let mut prev = vec![V4::ZERO];
    // with fixed particles: previous takes the old current position
    operators::move_particles(&reference, &[(0, 0)], &mut pos, &mut prev, true);
    assert_eq!(
        (pos[0], prev[0]),
        (V4::new(1.0, 2.0, 3.0, 1.0), V4::new(9.0, 9.0, 9.0, 1.0))
    );
    // without: both go to the vertex
    operators::move_particles(&reference, &[(0, 0)], &mut pos, &mut prev, false);
    assert_eq!(prev[0], pos[0]);
}
