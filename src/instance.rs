//! A cloth instance: one cloth data's buffers, transform sets and sim cloths at run time, stepped one cloth state at a
//! time. A step runs the state's operators as the games do: every operator's notification first (the simulate
//! operator's substep length and damping), then each operator in the state's order.

use crate::buffer::Buffer;
use crate::deform::{self, BoneSpaceLocal, LocalBlocks, NormalScale};
use crate::matrix::Mat4;
use crate::operators::{self, SomeVertexFrames, VertexFrames};
use crate::simd::V4;
use crate::simulate::{self, Action, ParticleData, SimState, StiffnessModel};
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

/// A cloth data at run time.
#[derive(Debug)]
pub struct ClothInstance<'a> {
    data: &'a Cloth,
    /// The buffers, by buffer definition. Those of the particle types mirror their sim cloth's particles, which the
    /// instance copies in before and back after each operator.
    pub buffers: Vec<Buffer>,
    /// Each buffer's triangles (three vertex indices each): the definition's, a sim cloth's for its particle buffers,
    /// and whatever the caller sets for its display buffers.
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

impl<'a> ClothInstance<'a> {
    /// An instance of `data` at its rest pose, in its first state; refuses operator classes it does not run.
    pub fn new(data: &'a Cloth) -> Result<ClothInstance<'a>, Refusal> {
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
    pub fn data(&self) -> &'a Cloth {
        self.data
    }

    /// The buffer an operator of the current state reads or writes for buffer `index`: its shadow buffer when the
    /// state gives it one.
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
        let data = self.data;
        let Some(state) = data.states.get(self.state) else {
            return Ok(());
        };
        // notifications: the simulate operators' substep length and damping
        for &opi in &state.operators {
            if let Some(op) = data.operators.get(opi as usize)
                && let OperatorKind::Simulate { sim_cloth, configs } = &op.kind
                && let (Some(sim), Some(config)) = (self.sims.get_mut(*sim_cloth as usize), configs.first())
            {
                let model = if config.adapt_constraint_stiffness {
                    StiffnessModel::Adaptive
                } else {
                    StiffnessModel::Legacy
                };
                simulate::prepare(sim, dt, config.sub_steps, model);
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
                        &self.triangles[i],
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
                        &self.triangles[i],
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
                    let tris = &self.triangles[i];
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
                    let tris = &self.triangles[i];
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
                        triangles: &self.triangles[b],
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

/// Copies every component `from` has into `to`, vertex by vertex, as far as both reach.
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
