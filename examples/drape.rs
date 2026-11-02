//! A square sheet dropped over a ball, simulated from cloth data built in code rather than read from a game file:
//! a grid of particles joined by structural, shear and bend links, colliding with a sphere and a floor plane.
//!
//! `cargo run --release --example drape -- OUT_DIR [FRAMES]` steps it at 30 frames a second (120 frames by default)
//! and writes each frame as `OUT_DIR/frame_NNN.gltf` (with its `.bin`) holding the sheet, the ball and the floor;
//! the sheet's plaid texture is `OUT_DIR/sheet.png`.

use std::path::Path;

use havok_cloth::simulate::{self, ParticleData, SimState};
use havok_formats::cloth::{
    Collidable, ConstraintKind, ConstraintSet, LandscapeCollision, Link, Particle, Pose, Shape, SimCloth,
    SimulateConfig, TransferMotion,
};
use serde_json::json;

/// Particles along each side of the sheet.
const N: usize = 72;
/// The sheet's side, metres.
const SIDE: f32 = 2.6;
/// The height the sheet starts at, metres.
const START_Y: f32 = 1.6;
/// The ball's radius, metres; it rests on the floor.
const BALL_R: f32 = 0.55;
/// The sheet's mass, kilograms, shared evenly by its particles.
const MASS: f32 = 0.6;
const FPS: f32 = 30.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let out = Path::new(args.get(1).ok_or("usage: drape OUT_DIR [FRAMES]")?);
    let frames: usize = args.get(2).map_or(Ok(120), |s| s.parse())?;
    std::fs::create_dir_all(out)?;

    let sim = sheet();
    let config = SimulateConfig {
        name: "drape".into(),
        // no listed constraint sets: every set runs in order, then the collision
        constraint_execution: Vec::new(),
        instance_collidables_used: Vec::new(),
        sub_steps: 8,
        solve_iterations: 1,
        use_all_instance_collidables: true,
        adapt_constraint_stiffness: false,
    };
    let data = ParticleData::new(&sim);
    let mut state = SimState::new(&sim);
    let indices = grid_triangles();
    // the texture repeats three times across the sheet
    let uvs: Vec<[f32; 2]> = (0..N * N)
        .map(|i| {
            [
                (i % N) as f32 / (N - 1) as f32 * 3.0,
                (i / N) as f32 / (N - 1) as f32 * 3.0,
            ]
        })
        .collect();
    std::fs::write(out.join("sheet.png"), plaid_png())?;
    let ball = sphere([0.0, BALL_R, 0.0], BALL_R, 96, 48);
    for frame in 0..frames {
        if frame > 0 {
            simulate::prepare(&mut state, 1.0 / FPS, config.sub_steps);
            simulate::simulate(&sim, &config, &mut state, &data, &[], &[], &mut [], 1.0 / FPS);
        }
        let positions: Vec<[f32; 3]> = state.positions.iter().map(|p| [p.x(), p.y(), p.z()]).collect();
        let normals = vertex_normals(&positions, &indices);
        let cloth = Mesh {
            positions,
            normals,
            indices: indices.clone(),
            uvs: Some(uvs.clone()),
        };
        write_frame(out, frame, &cloth, &ball)?;
    }
    let lowest = state
        .positions
        .iter()
        .map(|p| p.y())
        .fold(f32::INFINITY, f32::min);
    println!(
        "{frames} frames in {}; lowest particle {lowest:.3} m above the floor",
        out.display()
    );
    Ok(())
}

/// The sheet: N x N particles START_Y up, turned 25 degrees about the vertical and tipped 4 degrees so that it does
/// not land square on the ball.
fn sheet() -> SimCloth {
    let (sin_turn, cos_turn) = 25f32.to_radians().sin_cos();
    let tip = 4f32.to_radians().sin();
    let step = SIDE / (N - 1) as f32;
    let positions: Vec<[f32; 4]> = (0..N * N)
        .map(|i| {
            let (u, v) = (
                (i % N) as f32 * step - SIDE / 2.0,
                (i / N) as f32 * step - SIDE / 2.0,
            );
            [
                u * cos_turn - v * sin_turn,
                START_Y + u * tip,
                u * sin_turn + v * cos_turn,
                0.0,
            ]
        })
        .collect();
    let index = |x: usize, z: usize| u16::try_from(z * N + x).expect("fewer than 65536 particles");
    let particle_mass = MASS / (N * N) as f32;
    // A link stores the fraction of its error it corrects divided by its two particles' inverse masses: the solver
    // moves each particle by the correction times its own inverse mass.
    let link = |a: u16, b: u16, fraction: f32| {
        let (p, q) = (positions[a as usize], positions[b as usize]);
        let rest_length = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
        Link {
            particle_a: a,
            particle_b: b,
            rest_length,
            stiffness: fraction / (2.0 / particle_mass),
        }
    };
    // structural links correct all of their error, shear links half, and links two particles apart (bending) little
    let (mut structural, mut shear, mut bend) = (Vec::new(), Vec::new(), Vec::new());
    for z in 0..N {
        for x in 0..N {
            if x + 1 < N {
                structural.push(link(index(x, z), index(x + 1, z), 1.0));
            }
            if z + 1 < N {
                structural.push(link(index(x, z), index(x, z + 1), 1.0));
            }
            if x + 1 < N && z + 1 < N {
                shear.push(link(index(x, z), index(x + 1, z + 1), 0.5));
                shear.push(link(index(x + 1, z), index(x, z + 1), 0.5));
            }
            if x + 2 < N {
                bend.push(link(index(x, z), index(x + 2, z), 0.06));
            }
            if z + 2 < N {
                bend.push(link(index(x, z), index(x, z + 2), 0.06));
            }
        }
    }
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let collidable = |name: &str, shape: Shape| Collidable {
        name: name.into(),
        transform: identity,
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        user_data: 0,
        shape,
        pinch_detection_radius: 0.0,
        pinch_detection_priority: 0,
        pinch_detection_enabled: false,
        virtual_collision_point_collision: false,
        enabled: true,
    };
    let set = |kind| ConstraintSet {
        name: String::new(),
        constraint_id: None,
        kind,
    };
    SimCloth {
        name: "sheet".into(),
        gravity: [0.0, -9.8, 0.0, 0.0],
        global_damping_per_second: 0.15,
        particles: vec![
            Particle {
                mass: particle_mass,
                inv_mass: 1.0 / particle_mass,
                radius: 0.012,
                friction: 0.45
            };
            N * N
        ],
        fixed_particles: Vec::new(),
        do_normals: false,
        sim_op_ids: Vec::new(),
        poses: vec![Pose {
            name: "rest".into(),
            positions,
        }],
        constraint_sets: vec![
            set(ConstraintKind::StandardLink(structural)),
            set(ConstraintKind::StandardLink(shear)),
            set(ConstraintKind::StandardLink(bend)),
        ],
        anti_pinch_constraint_sets: Vec::new(),
        collidable_transform_map: None,
        collidables: vec![
            collidable(
                "ball",
                Shape::Sphere {
                    pos: [0.0, BALL_R, 0.0, BALL_R],
                },
            ),
            collidable(
                "floor",
                Shape::Plane {
                    equation: [0.0, 1.0, 0.0, 0.0],
                },
            ),
        ],
        max_particle_radius: 0.012,
        static_collision_masks: vec![u32::MAX; N * N],
        actions: Vec::new(),
        total_mass: MASS,
        transfer_motion: TransferMotion {
            transform_set: 0,
            transform: 0,
            transfer_translation: false,
            min_translation_speed: 0.0,
            max_translation_speed: 0.0,
            min_translation_blend: 0.0,
            max_translation_blend: 1.0,
            transfer_rotation: false,
            min_rotation_speed: 0.0,
            max_rotation_speed: 0.0,
            min_rotation_blend: 0.0,
            max_rotation_blend: 1.0,
        },
        transfer_motion_enabled: false,
        landscape_collision_enabled: false,
        landscape_collision: LandscapeCollision {
            radius: 0.0,
            stuck_particle_detection: false,
            stuck_particles_stretch_factor_sq: 9.0,
            pinch_detection: false,
            pinch_detection_priority: 0,
            pinch_detection_radius: 0.0,
            collision_tolerance: 0.2,
        },
        landscape_collidable_particles: 0,
        triangle_indices: Vec::new(),
        triangle_flips: Vec::new(),
        pinch_detection_enabled: false,
        per_particle_pinch_detection: Vec::new(),
        collidable_pinching: Vec::new(),
        min_pinched_particle: 0,
        max_pinched_particle: 0,
        max_collision_pairs: 0,
        virtual_collision_points: Default::default(),
    }
}

/// A triangle mesh for one glTF primitive.
struct Mesh {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    indices: Vec<u32>,
    uvs: Option<Vec<[f32; 2]>>,
}

/// Two triangles a cell of the N x N particle grid, facing up.
fn grid_triangles() -> Vec<u32> {
    let n = N as u32;
    (0..n - 1)
        .flat_map(|z| (0..n - 1).map(move |x| z * n + x))
        .flat_map(|i| [i, i + n, i + 1, i + 1, i + n, i + n + 1])
        .collect()
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Each vertex's normal: the sum of its triangles' area-weighted normals, normalised.
fn vertex_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0f32; 3]; positions.len()];
    for tri in indices.as_chunks::<3>().0 {
        let [a, b, c] = tri.map(|v| positions[v as usize]);
        let face = cross(sub(b, a), sub(c, a));
        for &v in tri {
            let n = &mut normals[v as usize];
            *n = [n[0] + face[0], n[1] + face[1], n[2] + face[2]];
        }
    }
    for n in &mut normals {
        let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
        *n = n.map(|c| c / length);
    }
    normals
}

/// A UV sphere around `centre`.
fn sphere(centre: [f32; 3], radius: f32, segments: u32, rings: u32) -> Mesh {
    let (mut positions, mut normals, mut indices) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..=rings {
        let theta = std::f32::consts::PI * j as f32 / rings as f32;
        for i in 0..=segments {
            let phi = std::f32::consts::TAU * i as f32 / segments as f32;
            let d = [theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin()];
            normals.push(d);
            positions.push([
                centre[0] + radius * d[0],
                centre[1] + radius * d[1],
                centre[2] + radius * d[2],
            ]);
        }
    }
    for j in 0..rings {
        for i in 0..segments {
            let a = j * (segments + 1) + i;
            let b = a + segments + 1;
            indices.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    Mesh {
        positions,
        normals,
        indices,
        uvs: None,
    }
}

/// The glTF buffer of a frame and its views and accessors, as they are added.
#[derive(Default)]
struct Buffer {
    bytes: Vec<u8>,
    views: Vec<serde_json::Value>,
    accessors: Vec<serde_json::Value>,
}

impl Buffer {
    const FLOAT: u32 = 5126;
    const UNSIGNED_INT: u32 = 5125;
    const ARRAY_BUFFER: u32 = 34962;
    const ELEMENT_ARRAY_BUFFER: u32 = 34963;

    /// Adds `values` (each `N` components) as an accessor and returns its index.
    fn floats<const C: usize>(&mut self, values: &[[f32; C]], ty: &str, bounds: bool) -> usize {
        let bytes: Vec<u8> = values.iter().flatten().flat_map(|c| c.to_le_bytes()).collect();
        let mut accessor = self.add(&bytes, values.len(), ty, Self::FLOAT, Self::ARRAY_BUFFER);
        if bounds {
            let (mut lo, mut hi) = ([f32::INFINITY; C], [f32::NEG_INFINITY; C]);
            for v in values {
                for (k, &c) in v.iter().enumerate() {
                    lo[k] = lo[k].min(c);
                    hi[k] = hi[k].max(c);
                }
            }
            accessor["min"] = json!(lo.to_vec());
            accessor["max"] = json!(hi.to_vec());
        }
        self.accessors.push(accessor);
        self.accessors.len() - 1
    }

    /// Adds triangle indices as an accessor and returns its index.
    fn indices(&mut self, indices: &[u32]) -> usize {
        let bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        let accessor = self.add(
            &bytes,
            indices.len(),
            "SCALAR",
            Self::UNSIGNED_INT,
            Self::ELEMENT_ARRAY_BUFFER,
        );
        self.accessors.push(accessor);
        self.accessors.len() - 1
    }

    fn add(
        &mut self,
        bytes: &[u8],
        count: usize,
        ty: &str,
        component: u32,
        target: u32,
    ) -> serde_json::Value {
        self.bytes.resize(self.bytes.len().next_multiple_of(4), 0);
        let offset = self.bytes.len();
        self.views
            .push(json!({"buffer": 0, "byteOffset": offset, "byteLength": bytes.len(), "target": target}));
        self.bytes.extend_from_slice(bytes);
        json!({"bufferView": self.views.len() - 1, "componentType": component, "count": count, "type": ty})
    }
}

/// Writes one frame: the sheet (double-sided), the ball and an 80 m floor, each a mesh with its own material.
fn write_frame(
    out: &Path,
    frame: usize,
    cloth: &Mesh,
    ball: &Mesh,
) -> Result<(), Box<dyn std::error::Error>> {
    let floor = Mesh {
        positions: vec![
            [-40.0, 0.0, -40.0],
            [40.0, 0.0, -40.0],
            [40.0, 0.0, 40.0],
            [-40.0, 0.0, 40.0],
        ],
        normals: vec![[0.0, 1.0, 0.0]; 4],
        indices: vec![0, 2, 1, 0, 3, 2],
        uvs: None,
    };
    let mut buffer = Buffer::default();
    let mut meshes = Vec::new();
    for (material, (name, mesh)) in [("sheet", cloth), ("ball", ball), ("floor", &floor)]
        .into_iter()
        .enumerate()
    {
        let mut attributes = json!({
            "POSITION": buffer.floats(&mesh.positions, "VEC3", true),
            "NORMAL": buffer.floats(&mesh.normals, "VEC3", false),
        });
        if let Some(uvs) = &mesh.uvs {
            attributes["TEXCOORD_0"] = json!(buffer.floats(uvs, "VEC2", false));
        }
        let indices = buffer.indices(&mesh.indices);
        meshes.push(json!({"name": name, "primitives": [{"attributes": attributes, "indices": indices, "material": material}]}));
    }
    let name = format!("frame_{frame:03}");
    let gltf = json!({
        "asset": {"version": "2.0", "generator": "havok-cloth drape example"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1, 2]}],
        "nodes": [{"mesh": 0, "name": "sheet"}, {"mesh": 1, "name": "ball"}, {"mesh": 2, "name": "floor"}],
        "meshes": meshes,
        "materials": [
            {"name": "sheet", "doubleSided": true,
             "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 0.85}},
            {"name": "ball", "pbrMetallicRoughness": {"baseColorFactor": [0.12, 0.12, 0.13, 1.0], "metallicFactor": 0.9, "roughnessFactor": 0.25}},
            {"name": "floor", "pbrMetallicRoughness": {"baseColorFactor": [0.22, 0.22, 0.23, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.6}}
        ],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "images": [{"uri": "sheet.png"}],
        "buffers": [{"uri": format!("{name}.bin"), "byteLength": buffer.bytes.len()}],
        "bufferViews": buffer.views,
        "accessors": buffer.accessors,
    });
    std::fs::write(out.join(format!("{name}.bin")), &buffer.bytes)?;
    std::fs::write(out.join(format!("{name}.gltf")), serde_json::to_string(&gltf)?)?;
    Ok(())
}

/// The sheet's texture: a 256 x 256 plaid, two repeats of a sett of red, green and dark bands with fine white and
/// yellow lines, woven as a 2/2 twill (warp and weft colours alternate along diagonals), as an RGB PNG.
fn plaid_png() -> Vec<u8> {
    const SIZE: usize = 256;
    const SETT: [([u8; 3], usize); 9] = [
        ([178, 30, 34], 40),
        ([22, 82, 52], 24),
        ([178, 30, 34], 8),
        ([60, 20, 22], 6),
        ([178, 30, 34], 8),
        ([235, 230, 220], 2),
        ([178, 30, 34], 14),
        ([230, 190, 60], 2),
        ([178, 30, 34], 24),
    ];
    let thread = |t: usize| {
        let mut t = t % SETT.iter().map(|(_, width)| width).sum::<usize>();
        for (colour, width) in SETT {
            if t < width {
                return colour;
            }
            t -= width;
        }
        unreachable!("the position is reduced modulo the sett's width")
    };
    let mut rows = Vec::with_capacity(SIZE * (1 + 3 * SIZE));
    for y in 0..SIZE {
        rows.push(0); // filter: none
        for x in 0..SIZE {
            let colour = if (x + y) / 2 % 2 == 0 {
                thread(x)
            } else {
                thread(y)
            };
            rows.extend_from_slice(&colour);
        }
    }
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(SIZE as u32).to_be_bytes());
    ihdr.extend_from_slice(&(SIZE as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit RGB, deflate, no interlace
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png_chunk(&mut png, b"IHDR", &ihdr);
    png_chunk(&mut png, b"IDAT", &zlib_stored(&rows));
    png_chunk(&mut png, b"IEND", &[]);
    png
}

/// A zlib stream holding `data` in stored (uncompressed) deflate blocks.
fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut z = vec![0x78, 0x01];
    let blocks = data.chunks(usize::from(u16::MAX));
    let count = blocks.len();
    for (i, block) in blocks.enumerate() {
        let length = u16::try_from(block.len()).expect("a stored block holds at most 65535 bytes");
        z.push(u8::from(i + 1 == count));
        z.extend_from_slice(&length.to_le_bytes());
        z.extend_from_slice(&(!length).to_le_bytes());
        z.extend_from_slice(block);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + u32::from(byte)) % 65521;
        b = (b + a) % 65521;
    }
    z.extend_from_slice(&((b << 16) | a).to_be_bytes());
    z
}

/// Appends a PNG chunk: its length, type, data and the CRC-32 of its type and data.
fn png_chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend_from_slice(
        &u32::try_from(data.len())
            .expect("a chunk under 4 GiB")
            .to_be_bytes(),
    );
    let start = png.len();
    png.extend_from_slice(kind);
    png.extend_from_slice(data);
    let mut crc = u32::MAX;
    for &byte in &png[start..] {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    png.extend_from_slice(&(!crc).to_be_bytes());
}
