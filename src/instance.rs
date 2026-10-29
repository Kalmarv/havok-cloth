//! A cloth instance: one cloth data's buffers, transform sets and sim cloths at run time, stepped one cloth state at a
//! time. A step runs the state's operators as the games do: every operator's notification first (the simulate
//! operator's substep length and damping), then each operator in the state's order.

use std::collections::BTreeSet;
use std::ops::Deref;

use crate::buffer::Buffer;
use crate::deform::{self, BoneSpaceLocal, LocalBlocks, NormalScale};
use crate::matrix::Mat4;
use crate::operators::{self, SomeVertexFrames, VertexFrames};
use crate::simd::V4;
use crate::simulate::{self, Action, ParticleData, SimState};
use havok_formats::cloth::{BufferKind, Cloth, OperatorKind};

/// Buffer types (`hclBufferDefinition::type`) whose buffers are a sim cloth's particles.
const CURRENT_POSITIONS: i32 = 1;
const PREVIOUS_POSITIONS: i32 = 2;

/// Why a cloth data cannot be instanced: an operator class this crate does not run, by name.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Refusal {
    /// An operator of a class not implemented here.
    #[error("operator {index} ({name}): class {class} is not implemented")]
    Operator {
        /// The operator's index in the cloth data.
        index: usize,
        /// Its name.
        name: String,
        /// Its class.
        class: String,
    },
    /// A buffer or transform set an operator names does not exist.
    #[error("operator {index} names a missing {what} {missing}")]
    Missing {
        /// The operator's index.
        index: usize,
        /// "buffer", "transform set" or "sim cloth".
        what: &'static str,
        /// The index it names.
        missing: u32,
    },
}

/// A cloth data at run time. `D` holds the data: a borrow (`&Cloth`) or an owner (`Arc<Cloth>`, for an instance kept
/// beside the data it was made of).
#[derive(Debug)]
pub struct ClothInstance<D: Deref<Target = Cloth> + Clone> {
    data: D,
    /// The buffers, by buffer definition. Those of the particle types mirror their sim cloth's particles, which the
    /// instance copies in before and back after each operator.
    pub buffers: Vec<Buffer>,
    /// Each buffer's triangles (three vertex indices each): the definition's, a sim cloth's for its particle buffers,
    /// and whatever the caller sets for its display buffers. An operator reads those of the buffer it names, also where
    /// the state holds that buffer's vertices in a shadow buffer ([`ClothInstance::resolve`]).
    pub triangles: Vec<Vec<u16>>,
    /// The sim cloths' states, by sim cloth.
    pub sims: Vec<SimState>,
    particles: Vec<ParticleData>,
    /// The transform sets, by definition: the caller sets the bones before a step, and bone deformers write theirs.
    /// In both games' files a transform set is the skeleton of its name in the same tagfile (its animation
    /// container's): transform i is the world transform of that skeleton's bone i.
    pub transform_sets: Vec<Vec<Mat4>>,
    /// The cloth state run by [`ClothInstance::step`].
    pub state: usize,
}

impl<D: Deref<Target = Cloth> + Clone> ClothInstance<D> {
    /// An instance of `data` at its rest pose, in its first state; refuses operator classes it does not run.
    pub fn new(data: D) -> Result<ClothInstance<D>, Refusal> {
        for (index, op) in data.operators.iter().enumerate() {
            if let OperatorKind::Other { class } = &op.kind {
                return Err(Refusal::Operator {
                    index,
                    name: op.name.clone(),
                    class: class.clone(),
                });
            }
        }
        let sims: Vec<SimState> = data.sim_cloths.iter().map(SimState::new).collect();
        let particles = data.sim_cloths.iter().map(ParticleData::new).collect();
        let mut buffers = Vec::new();
        let mut triangles = Vec::new();
        for def in &data.buffers {
            let mut b = Buffer::new(def.vertices as usize);
            b.tangents = vec![V4::ZERO; def.vertices as usize];
            b.bitangents = vec![V4::ZERO; def.vertices as usize];
            let tris = match &def.kind {
                BufferKind::Scratch { triangle_indices, .. }
                | BufferKind::Shadow { triangle_indices, .. } => triangle_indices.clone(),
                _ => match (
                    def.buffer_type,
                    usize::try_from(def.sub_type)
                        .ok()
                        .and_then(|s| data.sim_cloths.get(s)),
                ) {
                    (CURRENT_POSITIONS | PREVIOUS_POSITIONS, Some(sim)) => sim.triangle_indices.clone(),
                    _ => Vec::new(),
                },
            };
            buffers.push(b);
            triangles.push(tris);
        }
        let transform_sets = data
            .transform_sets
            .iter()
            .map(|t| vec![Mat4::IDENTITY; t.transforms as usize])
            .collect();
        let mut instance = ClothInstance {
            data,
            buffers,
            triangles,
            sims,
            particles,
            transform_sets,
            state: 0,
        };
        instance.plugs_from_sims();
        Ok(instance)
    }

    /// The cloth data.
    pub fn data(&self) -> &Cloth {
        &self.data
    }

    /// The buffer an operator of the current state reads or writes the vertices of for buffer `index`: its shadow buffer
    /// when the state gives it one. The triangles stay the named buffer's ([`triangles_of`]): in both games' files a
    /// shadowed display buffer and its shadow state none, and the caller sets the drawn mesh's on the display buffer.
    fn resolve(&self, index: u32) -> usize {
        let i = index as usize;
        self.data
            .states
            .get(self.state)
            .and_then(|s| s.used_buffers.iter().find(|u| u.buffer == index))
            .map(|u| u.shadow_buffer as usize)
            .filter(|&s| s < self.buffers.len() && s != i)
            .unwrap_or(i)
    }

    /// The sim cloth whose particles buffer `i` holds, and whether they are its previous positions.
    fn plug(&self, i: usize) -> Option<(usize, bool)> {
        let def = self.data.buffers.get(i)?;
        let sim = usize::try_from(def.sub_type)
            .ok()
            .filter(|&s| s < self.sims.len())?;
        match def.buffer_type {
            CURRENT_POSITIONS => Some((sim, false)),
            PREVIOUS_POSITIONS => Some((sim, true)),
            _ => None,
        }
    }

    /// Copies the sim cloths' particles into their buffers (identity matrices: particles are in world space).
    fn plugs_from_sims(&mut self) {
        for i in 0..self.buffers.len() {
            if let Some((s, previous)) = self.plug(i) {
                let sim = &self.sims[s];
                let b = &mut self.buffers[i];
                b.positions = if previous {
                    sim.previous.clone()
                } else {
                    sim.positions.clone()
                };
                if !previous {
                    b.normals = sim.normals.clone();
                }
                b.world_from_mesh = Mat4::IDENTITY;
                b.mesh_from_world = Mat4::IDENTITY;
            }
        }
    }

    /// Copies buffer `i` back into its sim cloth, if it holds particles.
    fn plug_to_sim(&mut self, i: usize) {
        if let Some((s, previous)) = self.plug(i) {
            let b = &self.buffers[i];
            let sim = &mut self.sims[s];
            if previous {
                sim.previous.clone_from(&b.positions);
            } else {
                sim.positions.clone_from(&b.positions);
                if sim.normals.len() == b.normals.len() {
                    sim.normals.clone_from(&b.normals);
                }
            }
        }
    }

    /// One step of the current state, `dt` long. `actions[s]` adds forces to sim cloth `s` on every substep.
    pub fn step(&mut self, dt: f32, actions: &mut [Vec<&mut dyn Action>]) -> Result<(), Refusal> {
        // a handle of its own, so the data stays readable while the instance changes
        let owner = self.data.clone();
        let data: &Cloth = &owner;
        let Some(state) = data.states.get(self.state) else {
            return Ok(());
        };
        // notifications: the simulate operators' substep length and damping
        for &opi in &state.operators {
            if let Some(op) = data.operators.get(opi as usize)
                && let OperatorKind::Simulate { sim_cloth, configs } = &op.kind
                && let (Some(sim), Some(config)) = (self.sims.get_mut(*sim_cloth as usize), configs.first())
            {
                simulate::prepare(sim, dt, config.sub_steps);
            }
        }
        for &opi in &state.operators {
            let index = opi as usize;
            let Some(op) = data.operators.get(index) else {
                continue;
            };
            self.plugs_from_sims();
            let missing_buffer = |b: u32| Refusal::Missing {
                index,
                what: "buffer",
                missing: b,
            };
            let missing_set = |t: u32| Refusal::Missing {
                index,
                what: "transform set",
                missing: t,
            };
            let mut written: Option<usize> = None;
            match &op.kind {
                OperatorKind::Simulate { sim_cloth, configs } => {
                    let s = *sim_cloth as usize;
                    let (Some(sim_data), Some(config)) = (data.sim_cloths.get(s), configs.first()) else {
                        continue;
                    };
                    let bones: &[Mat4] = sim_data
                        .collidable_transform_map
                        .as_ref()
                        .and_then(|m| self.transform_sets.get(m.transform_set as usize))
                        .map_or(&[], Vec::as_slice);
                    let no_actions: &mut [&mut dyn Action] = &mut [];
                    let acts = actions.get_mut(s).map_or(no_actions, |a| a.as_mut_slice());
                    simulate::simulate(
                        sim_data,
                        config,
                        &mut self.sims[s],
                        &self.particles[s],
                        &self.buffers,
                        bones,
                        acts,
                        dt,
                    );
                }
                OperatorKind::MoveParticles {
                    pairs,
                    sim_cloth,
                    reference_buffer,
                } => {
                    let r = self.resolve(*reference_buffer);
                    let reference = self
                        .buffers
                        .get(r)
                        .ok_or_else(|| missing_buffer(*reference_buffer))?
                        .clone();
                    let s = *sim_cloth as usize;
                    let fixed = data
                        .sim_cloths
                        .get(s)
                        .is_some_and(|d| !d.fixed_particles.is_empty());
                    if let Some(sim) = self.sims.get_mut(s) {
                        operators::move_particles(
                            &reference,
                            pairs,
                            &mut sim.positions,
                            &mut sim.previous,
                            fixed,
                        );
                    }
                }
                OperatorKind::ObjectSpaceSkin {
                    components,
                    bone_from_skin_mesh,
                    transform_subset,
                    output_buffer,
                    transform_set,
                    deformer,
                    local,
                    local_unpacked,
                } => {
                    let o = self.resolve(*output_buffer);
                    let bones = self
                        .transform_sets
                        .get(*transform_set as usize)
                        .ok_or_else(|| missing_set(*transform_set))?;
                    let out = self
                        .buffers
                        .get_mut(o)
                        .ok_or_else(|| missing_buffer(*output_buffer))?;
                    let composite = deform::composite_matrices(
                        &out.mesh_from_world,
                        bones,
                        transform_subset,
                        bone_from_skin_mesh,
                    );
                    let blocks = if local.is_empty() {
                        LocalBlocks::Unpacked(local_unpacked)
                    } else {
                        LocalBlocks::Packed(local)
                    };
                    deform::deform(deformer, blocks, *components, &composite, out);
                    written = Some(o);
                }
                OperatorKind::BoneSpaceSkin {
                    components,
                    transform_subset,
                    output_buffer,
                    transform_set,
                    deformer,
                    local,
                    local_unpacked,
                } => {
                    let o = self.resolve(*output_buffer);
                    let bones = self
                        .transform_sets
                        .get(*transform_set as usize)
                        .ok_or_else(|| missing_set(*transform_set))?;
                    let out = self
                        .buffers
                        .get_mut(o)
                        .ok_or_else(|| missing_buffer(*output_buffer))?;
                    let composite =
                        deform::bone_space_composite(&out.mesh_from_world, bones, transform_subset);
                    let blocks = if local.is_empty() {
                        BoneSpaceLocal::Unpacked(local_unpacked)
                    } else {
                        BoneSpaceLocal::Packed(local)
                    };
                    deform::bone_space_deform(deformer, blocks, *components, &composite, out);
                    written = Some(o);
                }
                OperatorKind::ObjectSpaceMeshMeshDeform {
                    components,
                    input_buffer,
                    output_buffer,
                    scale_normal_behaviour,
                    input_triangles_subset,
                    triangle_from_mesh,
                    deformer,
                    custom_skin_deform,
                    local,
                    local_unpacked,
                } => {
                    if *custom_skin_deform {
                        continue;
                    }
                    let (i, o) = (self.resolve(*input_buffer), self.resolve(*output_buffer));
                    let input = self
                        .buffers
                        .get(i)
                        .ok_or_else(|| missing_buffer(*input_buffer))?
                        .clone();
                    let frames = deform::triangle_frames(
                        &input.positions,
                        triangles_of(&self.triangles, *input_buffer),
                        input_triangles_subset,
                        NormalScale::from_stored(*scale_normal_behaviour),
                    );
                    let out = self
                        .buffers
                        .get_mut(o)
                        .ok_or_else(|| missing_buffer(*output_buffer))?;
                    let output_from_input = out.mesh_from_world.mul_affine(&input.world_from_mesh);
                    let composite =
                        deform::composite_matrices(&output_from_input, &frames, &[], triangle_from_mesh);
                    let blocks = if local.is_empty() {
                        LocalBlocks::Unpacked(local_unpacked)
                    } else {
                        LocalBlocks::Packed(local)
                    };
                    deform::deform(deformer, blocks, *components, &composite, out);
                    written = Some(o);
                }
                OperatorKind::BoneSpaceMeshMeshDeform {
                    components,
                    input_buffer,
                    output_buffer,
                    scale_normal_behaviour,
                    input_triangles_subset,
                    deformer,
                    local,
                    local_unpacked,
                } => {
                    let (i, o) = (self.resolve(*input_buffer), self.resolve(*output_buffer));
                    let input = self
                        .buffers
                        .get(i)
                        .ok_or_else(|| missing_buffer(*input_buffer))?
                        .clone();
                    let frames = deform::triangle_frames(
                        &input.positions,
                        triangles_of(&self.triangles, *input_buffer),
                        input_triangles_subset,
                        NormalScale::from_stored(*scale_normal_behaviour),
                    );
                    let out = self
                        .buffers
                        .get_mut(o)
                        .ok_or_else(|| missing_buffer(*output_buffer))?;
                    let output_from_input = out.mesh_from_world.mul_affine(&input.world_from_mesh);
                    let composite = deform::bone_space_composite(&output_from_input, &frames, &[]);
                    let blocks = if local.is_empty() {
                        BoneSpaceLocal::Unpacked(local_unpacked)
                    } else {
                        BoneSpaceLocal::Packed(local)
                    };
                    deform::bone_space_deform(deformer, blocks, *components, &composite, out);
                    written = Some(o);
                }
                OperatorKind::SimpleMeshBoneDeform {
                    input_buffer,
                    output_transform_set,
                    triangle_bone_pairs,
                    local_bone_transforms,
                } => {
                    let i = self.resolve(*input_buffer);
                    let input = self.buffers.get(i).ok_or_else(|| missing_buffer(*input_buffer))?;
                    let tris = triangles_of(&self.triangles, *input_buffer);
                    let bones = self
                        .transform_sets
                        .get_mut(*output_transform_set as usize)
                        .ok_or_else(|| missing_set(*output_transform_set))?;
                    deform::simple_mesh_bone_deform(
                        &input.positions,
                        tris,
                        triangle_bone_pairs,
                        local_bone_transforms,
                        bones,
                    );
                }
                OperatorKind::MeshBoneDeform {
                    input_buffer,
                    output_transform_set,
                    triangle_bone_pairs,
                    triangle_bone_start_for_bone,
                } => {
                    let i = self.resolve(*input_buffer);
                    let input = self.buffers.get(i).ok_or_else(|| missing_buffer(*input_buffer))?;
                    let tris = triangles_of(&self.triangles, *input_buffer);
                    let bones = self
                        .transform_sets
                        .get_mut(*output_transform_set as usize)
                        .ok_or_else(|| missing_set(*output_transform_set))?;
                    deform::mesh_bone_deform(
                        &input.positions,
                        tris,
                        triangle_bone_pairs,
                        triangle_bone_start_for_bone,
                        bones,
                    );
                }
                OperatorKind::CopyVertices {
                    input_buffer,
                    output_buffer,
                    vertices,
                    start_in,
                    start_out,
                    copy_normals,
                } => {
                    let (i, o) = (self.resolve(*input_buffer), self.resolve(*output_buffer));
                    let input = self
                        .buffers
                        .get(i)
                        .ok_or_else(|| missing_buffer(*input_buffer))?
                        .clone();
                    let out = self
                        .buffers
                        .get_mut(o)
                        .ok_or_else(|| missing_buffer(*output_buffer))?;
                    operators::copy_vertices(
                        &input,
                        out,
                        *vertices as usize,
                        *start_in as usize,
                        *start_out as usize,
                        *copy_normals,
                    );
                    written = Some(o);
                }
                OperatorKind::GatherAllVertices {
                    input_from_output,
                    input_buffer,
                    output_buffer,
                    gather_normals,
                    ..
                } => {
                    let (i, o) = (self.resolve(*input_buffer), self.resolve(*output_buffer));
                    let input = self
                        .buffers
                        .get(i)
                        .ok_or_else(|| missing_buffer(*input_buffer))?
                        .clone();
                    let out = self
                        .buffers
                        .get_mut(o)
                        .ok_or_else(|| missing_buffer(*output_buffer))?;
                    operators::gather_all(&input, out, input_from_output, *gather_normals);
                    written = Some(o);
                }
                OperatorKind::GatherSomeVertices {
                    pairs,
                    input_buffer,
                    output_buffer,
                    gather_normals,
                } => {
                    let (i, o) = (self.resolve(*input_buffer), self.resolve(*output_buffer));
                    let input = self
                        .buffers
                        .get(i)
                        .ok_or_else(|| missing_buffer(*input_buffer))?
                        .clone();
                    let out = self
                        .buffers
                        .get_mut(o)
                        .ok_or_else(|| missing_buffer(*output_buffer))?;
                    operators::gather_some(&input, out, pairs, *gather_normals);
                    written = Some(o);
                }
                OperatorKind::InputConvert {
                    user_buffer,
                    shadow_buffer,
                    ..
                } => {
                    // user buffers are kept as floats here: the conversion is a copy (no shadowing)
                    let user = self
                        .buffers
                        .get(*user_buffer as usize)
                        .ok_or_else(|| missing_buffer(*user_buffer))?
                        .clone();
                    let shadow = self
                        .buffers
                        .get_mut(*shadow_buffer as usize)
                        .ok_or_else(|| missing_buffer(*shadow_buffer))?;
                    copy_components(&user, shadow);
                    written = Some(*shadow_buffer as usize);
                }
                OperatorKind::OutputConvert {
                    user_buffer,
                    shadow_buffer,
                    ..
                } => {
                    let shadow = self
                        .buffers
                        .get(*shadow_buffer as usize)
                        .ok_or_else(|| missing_buffer(*shadow_buffer))?
                        .clone();
                    let user = self
                        .buffers
                        .get_mut(*user_buffer as usize)
                        .ok_or_else(|| missing_buffer(*user_buffer))?;
                    copy_components(&shadow, user);
                    written = Some(*user_buffer as usize);
                }
                OperatorKind::UpdateAllVertexFrames {
                    vertex_to_normal_id,
                    triangle_flips,
                    reference_vertices,
                    tangent_edge_cos,
                    tangent_edge_sin,
                    bitangent_flip,
                    buffer,
                    unique_normal_ids,
                    update_normals,
                    update_tangents,
                    update_bitangents,
                } => {
                    let b = self.resolve(*buffer);
                    let frames = VertexFrames {
                        triangles: triangles_of(&self.triangles, *buffer),
                        vertex_to_normal_id,
                        unique_normal_ids: *unique_normal_ids as usize,
                        triangle_flips,
                        reference_vertices,
                        tangent_edge_cos,
                        tangent_edge_sin,
                        bitangent_flip,
                        normals: *update_normals,
                        tangents: *update_tangents,
                        bitangents: *update_bitangents,
                    };
                    let target = self.buffers.get_mut(b).ok_or_else(|| missing_buffer(*buffer))?;
                    operators::update_vertex_frames(&frames, target);
                    written = Some(b);
                }
                OperatorKind::UpdateSomeVertexFrames {
                    involved_triangles,
                    involved_vertices,
                    selection_vertex_to_involved_vertex,
                    involved_vertex_to_normal_id,
                    triangle_flips,
                    reference_vertices,
                    tangent_edge_cos,
                    tangent_edge_sin,
                    bitangent_flip,
                    buffer,
                    unique_normal_ids,
                    update_normals,
                    update_tangents,
                    update_bitangents,
                } => {
                    let b = self.resolve(*buffer);
                    let frames = SomeVertexFrames {
                        involved_triangles,
                        involved_vertices,
                        selection_to_involved: selection_vertex_to_involved_vertex,
                        involved_to_normal_id: involved_vertex_to_normal_id,
                        unique_normal_ids: *unique_normal_ids as usize,
                        triangle_flips,
                        reference_vertices,
                        tangent_edge_cos,
                        tangent_edge_sin,
                        bitangent_flip,
                        normals: *update_normals,
                        tangents: *update_tangents,
                        bitangents: *update_bitangents,
                    };
                    let target = self.buffers.get_mut(b).ok_or_else(|| missing_buffer(*buffer))?;
                    operators::update_some_vertex_frames(&frames, target);
                    written = Some(b);
                }
                OperatorKind::Other { .. } => {}
            }
            if let Some(o) = written {
                self.plug_to_sim(o);
            }
        }
        self.plugs_from_sims();
        Ok(())
    }
}

/// The bones the cloth data's bone deformers set, as (transform set, bone) pairs: what a caller reads back after a step
/// (the cloth-driven bones); every other transform is the caller's. A simple mesh bone deformer sets the bone of each
/// pair that has a local transform (its byte offset over 64), a mesh bone deformer each bone with entries.
pub fn driven_bones(data: &Cloth) -> BTreeSet<(u32, usize)> {
    let mut out = BTreeSet::new();
    for op in &data.operators {
        match &op.kind {
            OperatorKind::SimpleMeshBoneDeform {
                output_transform_set,
                triangle_bone_pairs,
                local_bone_transforms,
                ..
            } => out.extend(
                triangle_bone_pairs
                    .iter()
                    .zip(local_bone_transforms)
                    .map(|(&(bone_offset, _), _)| (*output_transform_set, usize::from(bone_offset) / 64)),
            ),
            OperatorKind::MeshBoneDeform {
                output_transform_set,
                triangle_bone_start_for_bone,
                ..
            } => out.extend(
                triangle_bone_start_for_bone
                    .windows(2)
                    .enumerate()
                    .filter(|(_, w)| w[0] < w[1])
                    .map(|(b, _)| (*output_transform_set, b)),
            ),
            _ => {}
        }
    }
    out
}

/// Copies every component `from` has into `to`, vertex by vertex, as far as both reach.
/// The triangles of buffer `index` as an operator names it: none for an index past the buffers.
fn triangles_of(triangles: &[Vec<u16>], index: u32) -> &[u16] {
    triangles.get(index as usize).map_or(&[], Vec::as_slice)
}

fn copy_components(from: &Buffer, to: &mut Buffer) {
    for (src, dst) in [
        (&from.positions, &mut to.positions),
        (&from.normals, &mut to.normals),
        (&from.tangents, &mut to.tangents),
        (&from.bitangents, &mut to.bitangents),
    ] {
        for (d, s) in dst.iter_mut().zip(src) {
            *d = *s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use havok_formats::cloth::{MeshBoneTriangle, Operator};
    use std::sync::Arc;

    fn operator(kind: OperatorKind) -> Operator {
        Operator {
            name: String::new(),
            id: 0,
            operator_type: 0,
            used_buffers: Vec::new(),
            used_transform_sets: Vec::new(),
            kind,
        }
    }

    fn cloth(operators: Vec<Operator>) -> Cloth {
        Cloth {
            name: "test".into(),
            target_platform: 2,
            sim_cloths: Vec::new(),
            buffers: Vec::new(),
            transform_sets: Vec::new(),
            operators,
            states: Vec::new(),
            state_transitions: Vec::new(),
            actions: Vec::new(),
        }
    }

    #[test]
    fn driven_bones_are_the_bones_the_deformers_set() {
        let pair = |triangle| MeshBoneTriangle {
            local_bone_transform: [0.0; 16],
            weight: 1.0,
            triangle,
        };
        let data = cloth(vec![
            // byte offsets 64 and 192: bones 1 and 3; the third pair has no local transform, so sets nothing
            operator(OperatorKind::SimpleMeshBoneDeform {
                input_buffer: 0,
                output_transform_set: 0,
                triangle_bone_pairs: vec![(64, 0), (192, 6), (320, 12)],
                local_bone_transforms: vec![[0.0; 16]; 2],
            }),
            // entries 0..1 for bone 0, none for bone 1, 1..3 for bone 2
            operator(OperatorKind::MeshBoneDeform {
                input_buffer: 0,
                output_transform_set: 1,
                triangle_bone_pairs: vec![pair(0), pair(1), pair(2)],
                triangle_bone_start_for_bone: vec![0, 1, 1, 3],
            }),
        ]);
        let want: BTreeSet<(u32, usize)> = [(0, 1), (0, 3), (1, 0), (1, 2)].into_iter().collect();
        assert_eq!(driven_bones(&data), want);
    }

    /// A buffer definition of `vertices` vertices of class `kind`.
    fn buffer(buffer_type: i32, vertices: u32, kind: BufferKind) -> havok_formats::cloth::BufferDefinition {
        use havok_formats::cloth::{BufferElement, BufferLayout, BufferSlot};
        let element = BufferElement {
            vector_conversion: 0,
            vector_size: 0,
            slot: 0,
            slot_start: 0,
        };
        havok_formats::cloth::BufferDefinition {
            name: String::new(),
            buffer_type,
            sub_type: 0,
            vertices,
            triangles: 0,
            layout: BufferLayout {
                elements: [element; 4],
                slots: [BufferSlot { flags: 0, stride: 0 }; 4],
                num_slots: 0,
                triangle_format: 0,
            },
            kind,
        }
    }

    /// Sekiro's scarf: its display buffer (an `hclBufferDefinition` of type 4) is shadowed in its state by an
    /// `hclShadowBufferDefinition` (type 7), neither stating triangles; the state's `hclUpdateAllVertexFramesOperator`
    /// names the display buffer, whose triangles are the drawn mesh's, set by the caller. The normals it writes into the
    /// shadow are those of the mesh's triangles.
    #[test]
    fn vertex_frames_of_a_shadowed_buffer_use_the_triangles_of_the_buffer_they_name() {
        use havok_formats::cloth::{BufferAccess, BufferUsage, ClothState};
        let mut data = cloth(vec![operator(OperatorKind::UpdateAllVertexFrames {
            vertex_to_normal_id: vec![0, 1, 2],
            triangle_flips: vec![0],
            reference_vertices: Vec::new(),
            tangent_edge_cos: Vec::new(),
            tangent_edge_sin: Vec::new(),
            bitangent_flip: Vec::new(),
            buffer: 0,
            unique_normal_ids: 3,
            update_normals: true,
            update_tangents: false,
            update_bitangents: false,
        })]);
        data.buffers = vec![
            buffer(4, 3, BufferKind::Plain),
            buffer(
                7,
                3,
                BufferKind::Shadow {
                    triangle_indices: Vec::new(),
                    positions: true,
                    normals: true,
                    tangents: false,
                    bitangents: false,
                },
            ),
        ];
        data.states = vec![ClothState {
            name: String::new(),
            operators: vec![0],
            used_buffers: vec![BufferAccess {
                buffer: 0,
                usage: BufferUsage {
                    per_component_flags: [1, 2, 0, 0],
                    triangles_read: true,
                },
                shadow_buffer: 1,
            }],
            used_transform_sets: Vec::new(),
            used_sim_cloths: Vec::new(),
            dependency_graph: None,
        }];
        let mut instance = ClothInstance::new(&data).expect("nothing to refuse");
        // the drawn mesh: a triangle in the plane z = 0, wound towards +z, its vertices held in the shadow
        instance.triangles[0] = vec![0, 1, 2];
        instance.buffers[1].positions = vec![
            V4([0.0, 0.0, 0.0, 1.0]),
            V4([1.0, 0.0, 0.0, 1.0]),
            V4([0.0, 1.0, 0.0, 1.0]),
        ];
        instance.step(1.0 / 30.0, &mut []).expect("nothing to refuse");
        for (v, n) in instance.buffers[1].normals.iter().enumerate() {
            let [x, y, z, _] = n.0;
            assert!(
                x == 0.0 && y == 0.0 && (z - 1.0).abs() < 1e-3,
                "vertex {v}: normal {n:?}"
            );
        }
    }

    #[test]
    fn an_instance_may_own_its_data() {
        let data = Arc::new(cloth(Vec::new()));
        let mut owned = ClothInstance::new(Arc::clone(&data)).expect("no operators to refuse");
        let mut borrowed = ClothInstance::new(&*data).expect("no operators to refuse");
        // no states: a step runs nothing
        owned.step(1.0 / 30.0, &mut []).expect("nothing to refuse");
        borrowed.step(1.0 / 30.0, &mut []).expect("nothing to refuse");
        assert_eq!(owned.data().name, "test");
        assert_eq!(borrowed.data().name, "test");
    }
}
