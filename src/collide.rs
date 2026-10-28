//! Collision of particles with the sim cloth's collidables: the collidables' motion over a frame, the shapes'
//! closest-point queries, and the contact solve with friction.
//!
//! A shape is queried two ways: four particles at a time, and one at a time for the particles left over. The two are
//! different arithmetic (the four-wide query adds an epsilon before its reciprocal square root, and for some shapes
//! compares or normalises differently), and the game uses both, so both are here.

use crate::matrix::Mat4;
use crate::simd::{V4, max_ps, min_ps, rsqrt23};
use crate::simulate::{ParticleData, SimState};
use havok_formats::cloth::{Collidable, Shape, SimCloth};

/// `FLT_EPSILON` in every lane: what the four-wide queries add to a squared length before normalising.
const EPSILON4: V4 = V4::splat(f32::EPSILON);

/// The arcsine as the cloth code computes it: odd polynomial in `a` (or in `sqrt((1 - a)/2)` above 0.5), linear
/// below 1e-4, clamped to [-1, 1].
#[allow(
    clippy::assign_op_pattern,
    reason = "each step keeps the instruction's operand order"
)]
pub fn asin(x: f32) -> f32 {
    let linear_limit = f32::from_bits(0x38D1_B717);
    let c4 = f32::from_bits(0x3D2C_B352);
    let c3 = f32::from_bits(0x3CC6_17E3);
    let c2 = f32::from_bits(0x3D3A_3EC7);
    let c1 = f32::from_bits(0x3D99_80F6);
    let c0 = f32::from_bits(0x3E2A_AAE4);
    let a = min_ps(1.0, x.abs());
    let sign = x.to_bits() & 0x8000_0000;
    let high = a > 0.5;
    let (u, z) = if high {
        let z = 0.5 * (1.0 - a);
        (z.sqrt(), z)
    } else {
        (a, a * a)
    };
    let mut p = c4 * z;
    p = c3 + p;
    p *= z;
    p = c2 + p;
    p *= z;
    p = c1 + p;
    p *= z;
    p = c0 + p;
    p *= z;
    p *= u;
    p += u;
    if high {
        p = FRAC_PI_2 - (p + p);
    }
    if a < linear_limit {
        p = a;
    }
    f32::from_bits(p.to_bits() ^ sign)
}

/// pi / 2 as the cloth code stores it.
const FRAC_PI_2: f32 = f32::from_bits(0x3FC9_0FDB);

/// The arccosine as the cloth code computes it: `pi/2 - asin(x)`.
pub fn acos(x: f32) -> f32 {
    FRAC_PI_2 - asin(x)
}

/// A rotation quaternion: `x y z` the vector part, `w` the real part.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat(pub V4);

#[allow(
    clippy::should_implement_trait,
    reason = "the quaternion product as the cloth code computes it, not an operator"
)]
impl Quat {
    /// The quaternion of a rotation (the first three columns of `m`), by the trace when it is positive and by the
    /// largest diagonal element otherwise.
    pub fn from_rotation(m: &Mat4) -> Quat {
        let r = |row: usize, col: usize| m.cols[col].0[row];
        let trace = r(0, 0) + r(1, 1) + r(2, 2);
        let half = 0.5f32;
        let mut v = [0.0f32; 4];
        if trace > 0.0 {
            let s = (trace + 1.0).sqrt();
            let t = half / s;
            v[0] = (r(2, 1) - r(1, 2)) * t;
            v[1] = (r(0, 2) - r(2, 0)) * t;
            v[2] = (r(1, 0) - r(0, 1)) * t;
            v[3] = half * s;
        } else {
            let mut i = 0;
            if r(1, 1) > r(0, 0) {
                i = 1;
            }
            if r(2, 2) > r(i, i) {
                i = 2;
            }
            let j = (i + 1) % 3;
            let k = (j + 1) % 3;
            let s = (r(i, i) - (r(j, j) + r(k, k)) + 1.0).sqrt();
            let t = half / s;
            v[i] = half * s;
            v[3] = (r(k, j) - r(j, k)) * t;
            v[j] = (r(j, i) + r(i, j)) * t;
            v[k] = (r(k, i) + r(i, k)) * t;
        }
        Quat(V4(v))
    }

    /// The product `self * q`: vector `(cross(a, b) + b*aw) + a*bw`, real `aw*bw - dot3(a, b)`.
    pub fn mul(self, q: Quat) -> Quat {
        let (a, b) = (self.0, q.0);
        let v = a.cross(b).add_scaled(b, a.w()).add_scaled(a, b.w());
        Quat(v.with_w(a.w() * b.w() - a.dot3(b)))
    }

    /// `self` times the inverse of `q`: vector `(cross(b, a) - b*aw) + a*bw`, real the four-lane dot of `a` and `b`.
    pub fn mul_inverse(self, q: Quat) -> Quat {
        let (a, b) = (self.0, q.0);
        let v = b.cross(a).sub_scaled(b, a.w()).add_scaled(a, b.w());
        Quat(v.with_w(dot4(a, b)))
    }

    /// The rotation angle, `2 * acos(|w|)`.
    pub fn angle(self) -> f32 {
        let a = acos(self.0.w().abs());
        a + a
    }

    /// The unit rotation axis: the vector part normalised (refined), every lane negated when `w < 0`.
    pub fn axis(self) -> V4 {
        let n = self.0.normalized();
        if self.0.w() < 0.0 { n.scale(-1.0) } else { n }
    }

    /// The quaternion scaled to unit length over all four lanes (refined reciprocal square root, no zero check).
    pub fn normalized(self) -> Quat {
        Quat(self.0.scale(rsqrt23(dot4(self.0, self.0))))
    }

    /// The rotation's three columns (their `w` lanes as the conversion leaves them).
    pub fn to_rotation(self) -> [V4; 3] {
        let [x, y, z, w] = self.0.0;
        let (x2, y2, z2) = (x + x, y + y, z + z);
        let (xx2, yy2, zz2) = (x * x2, y * y2, z * z2);
        let (xy2, xz2, yz2) = (x * y2, x * z2, y * z2);
        let (wx2, wy2, wz2) = (w * x2, w * y2, w * z2);
        let one_zz = -zz2 + 1.0;
        [
            V4([one_zz - yy2, wz2 + xy2, xz2 - wy2, yz2 - wx2]),
            V4([xy2 - wz2, one_zz - xx2, yz2 + wx2, xz2 + wy2]),
            V4([xz2 + wy2, yz2 - wx2, (1.0 - xx2) - yy2, 0.0]),
        ]
    }
}

/// The four-lane dot product as the games' SSE2 code sums it, by two shuffled adds (lanes swapped in pairs, then
/// neighbours): `(x + z) + (y + w)` of the products.
pub fn dot4(a: V4, b: V4) -> f32 {
    let p = a.mul(b).0;
    (p[0] + p[2]) + (p[1] + p[3])
}

/// A matrix whose rotation columns are `r` and whose translation is `t`.
fn with_rotation(r: [V4; 3], t: V4) -> Mat4 {
    Mat4 {
        cols: [r[0], r[1], r[2], t],
    }
}

/// The inverse of a rigid transform: the transposed rotation, and the negated translation rotated by it.
fn inverse_rigid(m: &Mat4) -> Mat4 {
    let [c0, c1, c2, t] = m.cols;
    let rows = [
        V4::new(c0.x(), c1.x(), c2.x(), 0.0),
        V4::new(c0.y(), c1.y(), c2.y(), 0.0),
        V4::new(c0.z(), c1.z(), c2.z(), 0.0),
    ];
    let inv = with_rotation(rows, V4::ZERO);
    let neg = t.scale(-1.0);
    with_rotation(rows, inv.transform_direction(neg))
}

/// A collidable as the solver moves it.
#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeCollidable {
    /// Where it is: rotation columns and translation.
    pub transform: Mat4,
    /// Linear velocity.
    pub linear_velocity: V4,
    /// Angular velocity.
    pub angular_velocity: V4,
    /// Its shape, in its own space.
    pub shape: Shape,
    /// Whether particles collide with it.
    pub enabled: bool,
    /// Whether virtual collision points collide with it.
    pub virtual_collision_points: bool,
}

impl RuntimeCollidable {
    /// A collidable where the data puts it.
    pub fn new(c: &Collidable) -> RuntimeCollidable {
        RuntimeCollidable {
            transform: Mat4::from_columns(c.transform),
            linear_velocity: V4(c.linear_velocity),
            angular_velocity: V4(c.angular_velocity),
            shape: c.shape.clone(),
            enabled: c.enabled,
            virtual_collision_points: c.virtual_collision_point_collision,
        }
    }

    /// Sets the velocities that move the collidable from where it is to `target` over `dt`, leaving it where it is.
    pub fn drive_to(&mut self, target: &Mat4, dt: f32) {
        let linear = target.translation().sub(self.transform.translation());
        let goal = Quat::from_rotation(target);
        let current = Quat::from_rotation(&self.transform);
        let diff = goal.mul_inverse(current);
        let angle = diff.angle();
        self.angular_velocity = if angle > 0.0 && diff.0.dot3(diff.0) > f32::EPSILON {
            diff.axis().scale(angle / dt)
        } else {
            V4::ZERO
        };
        self.linear_velocity = linear.scale(1.0 / dt);
    }

    /// Moves the collidable forward by `dt` at its velocities: the translation directly, the rotation by a quaternion
    /// step whose real part is a polynomial approximation, then renormalised.
    pub fn step_forward(&mut self, dt: f32) {
        let t = self.transform.translation().add_scaled(self.linear_velocity, dt);
        let q = Quat::from_rotation(&self.transform);
        let half_angle = self.angular_velocity.scale(dt * 0.5);
        let pi = std::f32::consts::PI;
        let x2 = half_angle.dot3(half_angle) * (4.0 / (pi * pi));
        let x4 = x2 * x2;
        let w = ((1.0 - 0.822_948 * x2) - 0.130_529 * x4) - (0.044_408 * x2) * x4;
        let rotated = Quat(half_angle.with_w(w)).mul(q).normalized();
        self.transform = with_rotation(rotated.to_rotation(), t);
    }

    /// The velocity of the collidable's surface at world point `p`: `cross(angular, p - translation) + linear`.
    pub fn point_velocity(&self, p: V4) -> V4 {
        self.angular_velocity
            .cross(p.sub(self.transform.translation()))
            .add(self.linear_velocity)
    }
}

/// A closest point on a shape: where, the outward normal there, and the signed distance from it to the particle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    /// The point on the shape's surface.
    pub point: V4,
    /// The surface normal there.
    pub normal: V4,
    /// The particle centre's signed distance from the surface (negative inside).
    pub distance: f32,
}

/// A shape moved to world space, with what its queries precompute.
#[derive(Clone, Debug, PartialEq)]
pub enum WorldShape {
    /// A sphere.
    Sphere {
        /// Centre (`w` as the transform leaves it).
        centre: V4,
        /// Radius.
        radius: f32,
    },
    /// A capsule.
    Capsule {
        /// Start.
        start: V4,
        /// End.
        end: V4,
        /// Radius.
        radius: f32,
        /// `1 / length^2`.
        inv_length_sq: f32,
    },
    /// A tapered capsule.
    TaperedCapsule(TaperedCapsule),
    /// A plane `n . p + d = 0`.
    Plane {
        /// `n` in `x y z`, `d` in `w`.
        equation: V4,
    },
    /// A convex geometry.
    Convex(Box<Convex>),
    /// A shape this solver does not collide with.
    Unsupported,
}

/// A tapered capsule in world space with its precomputed cone.
#[derive(Clone, Debug, PartialEq)]
pub struct TaperedCapsule {
    /// The small end's centre.
    pub small: V4,
    /// The big end's centre.
    pub big: V4,
    /// The small end's radius.
    pub small_radius: f32,
    /// The big end's radius.
    pub big_radius: f32,
    /// Unit axis from small to big.
    pub axis: V4,
    /// Distance between the centres.
    pub length: f32,
    /// The cone's apex.
    pub apex: V4,
    /// Distance from the apex to the small centre.
    pub apex_distance: f32,
    /// Sine of the cone's half angle.
    pub sin_theta: f32,
    /// Cosine of the cone's half angle.
    pub cos_theta: f32,
    /// Tangent of the cone's half angle.
    pub tan_theta: f32,
}

impl TaperedCapsule {
    /// The cone through spheres at `small` and `big`.
    pub fn new(small: V4, big: V4, small_radius: f32, big_radius: f32) -> TaperedCapsule {
        let (r, rr) = (small_radius, big_radius);
        let (axis, l) = big.sub(small).normalized_with_length();
        let h = (rr * (l * l - (rr - r) * (rr - r))) / ((rr - r) * l);
        let sin_theta = (rr - r) / l;
        let d = r / sin_theta;
        let apex = small.sub_scaled(axis, d);
        let w = ((h * rr) * sin_theta).sqrt();
        let tan_theta = w / h;
        let cos_theta = h / (w * w + h * h).sqrt();
        TaperedCapsule {
            small,
            big,
            small_radius: r,
            big_radius: rr,
            axis,
            length: l,
            apex,
            apex_distance: d,
            sin_theta,
            cos_theta,
            tan_theta,
        }
    }
}

/// A convex geometry shape with its world placement.
#[derive(Clone, Debug, PartialEq)]
pub struct Convex {
    /// `tetrahedraGrid`: per grid cell, where its tetrahedra start in `cells` (and the next cell's start ends them).
    pub grid: Vec<u16>,
    /// `gridCells`: tetrahedron indices.
    pub cells: Vec<u8>,
    /// Per tetrahedron, its four face planes as a matrix (row 0 the face on the shape's surface).
    pub tetrahedra: Vec<Mat4>,
    /// World from local.
    pub world_from_local: Mat4,
    /// Local from world.
    pub local_from_world: Mat4,
    /// The local box's minimum.
    pub aabb_min: V4,
    /// The local box's maximum.
    pub aabb_max: V4,
    /// The local centroid.
    pub centroid: V4,
    /// Grid cells per unit length.
    pub inv_cell_size: V4,
    /// Cells along each axis.
    pub grid_res: u16,
}

/// Whether `p` lies in the box (`min <= p <= max` in `x y z`).
fn in_box(min: V4, max: V4, p: V4) -> bool {
    (0..3).all(|i| min.0[i] <= p.0[i] && p.0[i] <= max.0[i])
}

/// Which particles a collidable is tested against: everything, or a box around a tapered capsule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundingVolume {
    /// Every particle.
    All,
    /// The particles in this box.
    Box(V4, V4),
}

impl BoundingVolume {
    /// Whether `p` is tested.
    pub fn contains(&self, p: V4) -> bool {
        match *self {
            BoundingVolume::All => true,
            BoundingVolume::Box(min, max) => in_box(min, max, p),
        }
    }
}

impl WorldShape {
    /// `shape` placed by `transform`.
    pub fn new(shape: &Shape, transform: &Mat4) -> WorldShape {
        let place = |p: [f32; 4]| transform.transform_position_rigid(V4(p));
        match shape {
            Shape::Sphere { pos } => WorldShape::Sphere {
                centre: place(*pos),
                radius: pos[3],
            },
            Shape::Capsule {
                start, end, radius, ..
            } => {
                let (a, b) = (place(*start), place(*end));
                let (_, length) = b.sub(a).normalized_with_length();
                WorldShape::Capsule {
                    start: a,
                    end: b,
                    radius: *radius,
                    inv_length_sq: 1.0 / (length * length),
                }
            }
            Shape::TaperedCapsule {
                small,
                big,
                small_radius,
                big_radius,
                ..
            } => WorldShape::TaperedCapsule(TaperedCapsule::new(
                place(*small),
                place(*big),
                *small_radius,
                *big_radius,
            )),
            Shape::Plane { equation } => {
                let e = V4(*equation);
                let n = transform.transform_direction(e);
                WorldShape::Plane {
                    equation: n.with_w(e.w() - n.dot3(transform.translation())),
                }
            }
            Shape::ConvexGeometry {
                tetrahedra_grid,
                grid_cells,
                tetrahedra_equations,
                aabb_min,
                aabb_max,
                centroid,
                inv_cell_size,
                grid_res,
                ..
            } => WorldShape::Convex(Box::new(Convex {
                grid: tetrahedra_grid.clone(),
                cells: grid_cells.clone(),
                tetrahedra: tetrahedra_equations
                    .iter()
                    .map(|&m| Mat4::from_columns(m))
                    .collect(),
                world_from_local: *transform,
                local_from_world: inverse_rigid(transform),
                aabb_min: V4(*aabb_min),
                aabb_max: V4(*aabb_max),
                centroid: V4(*centroid),
                inv_cell_size: V4(*inv_cell_size),
                grid_res: *grid_res,
            })),
            Shape::Other { .. } => WorldShape::Unsupported,
        }
    }

    /// The particles a collidable of `shape` at `transform` is tested against, grown by the sim cloth's largest
    /// particle radius.
    pub fn bounding_volume(shape: &Shape, transform: &Mat4, max_particle_radius: f32) -> BoundingVolume {
        let Shape::TaperedCapsule {
            small,
            big,
            small_radius,
            big_radius,
            ..
        } = shape
        else {
            return BoundingVolume::All;
        };
        let s = transform.transform_position_rigid(V4(*small));
        let b = transform.transform_position_rigid(V4(*big));
        let (s_min, s_max) = (s.sub(V4::splat(*small_radius)), s.add(V4::splat(*small_radius)));
        let (b_min, b_max) = (b.sub(V4::splat(*big_radius)), b.add(V4::splat(*big_radius)));
        let e = V4::splat(max_particle_radius);
        BoundingVolume::Box(s_min.min(b_min).sub(e), s_max.max(b_max).add(e))
    }

    /// The closest point to one particle `p` of radius `radius`.
    pub fn closest(&self, p: V4, radius: f32) -> Contact {
        match self {
            WorldShape::Sphere { centre, radius: r } => sphere_point(p, *centre, *r),
            WorldShape::Capsule {
                start,
                end,
                radius: r,
                inv_length_sq,
            } => {
                let ab = end.sub(*start);
                // below 0 gives 0, above 1 gives 1, anything else (NaN too) itself
                let s = (p.sub(*start).dot3(ab) * inv_length_sq).clamp(0.0, 1.0);
                let l = start.lerp(*end, s);
                let (n, length) = p.sub(l).normalized_with_length();
                Contact {
                    point: l.add_scaled(n, *r),
                    normal: n,
                    distance: length - r,
                }
            }
            WorldShape::TaperedCapsule(t) => {
                let rel = p.sub(t.apex);
                let z = rel.dot3(t.axis);
                let (perp, perp_len) = t.axis.cross(rel.cross(t.axis)).normalized_with_length();
                let band = t.apex_distance - t.tan_theta * perp_len;
                if z <= band {
                    sphere_point(p, t.small, t.small_radius)
                } else if z >= t.length + band {
                    sphere_point(p, t.big, t.big_radius)
                } else {
                    let d = -(t.tan_theta * z - perp_len) * t.cos_theta;
                    let n = perp
                        .scale(t.cos_theta)
                        .sub_scaled(t.axis, t.sin_theta)
                        .normalized();
                    Contact {
                        point: p.add_scaled(n, -d),
                        normal: n,
                        distance: d,
                    }
                }
            }
            WorldShape::Plane { equation } => {
                let d = equation.dot4_xyz1(p);
                Contact {
                    point: p.add_scaled(*equation, -d),
                    normal: *equation,
                    distance: d,
                }
            }
            WorldShape::Convex(c) => convex_point(c, p, radius),
            WorldShape::Unsupported => Contact {
                point: p,
                normal: V4::ZERO,
                distance: f32::MAX,
            },
        }
    }

    /// The closest points to four particles at once.
    pub fn closest4(&self, p: [V4; 4], radius: [f32; 4]) -> [Contact; 4] {
        match self {
            WorldShape::Sphere { centre, radius: r } => {
                let (n, length) = batch_normalize(p.map(|x| x.sub(*centre)));
                std::array::from_fn(|i| Contact {
                    point: centre.add_scaled(n[i], *r),
                    normal: n[i],
                    distance: length[i] - r,
                })
            }
            WorldShape::Capsule {
                start,
                end,
                radius: r,
                inv_length_sq,
            } => {
                let ab = end.sub(*start);
                let l: [V4; 4] = std::array::from_fn(|i| {
                    let s = p[i].sub(*start).dot3(ab) * inv_length_sq;
                    let s = min_ps(max_ps(0.0, s), 1.0);
                    start.lerp(*end, s)
                });
                let (n, length) = batch_normalize(std::array::from_fn(|i| p[i].sub(l[i])));
                std::array::from_fn(|i| Contact {
                    point: l[i].add_scaled(n[i], *r),
                    normal: n[i],
                    distance: length[i] - r,
                })
            }
            WorldShape::TaperedCapsule(t) => {
                let rel = p.map(|x| x.sub(t.apex));
                let (perp, perp_len) = batch_normalize(rel.map(|r| t.axis.cross(r.cross(t.axis))));
                std::array::from_fn(|i| {
                    let z = rel[i].dot3(t.axis);
                    let band = t.apex_distance + (-t.tan_theta) * perp_len[i];
                    let band_max = band + t.length;
                    if z < band {
                        sphere_point(p[i], t.small, t.small_radius)
                    } else if z > band_max {
                        sphere_point(p[i], t.big, t.big_radius)
                    } else {
                        let d = (perp_len[i] - t.tan_theta * z) * t.cos_theta;
                        let n = perp[i]
                            .scale(t.cos_theta)
                            .sub_scaled(t.axis, t.sin_theta)
                            .normalized_fast();
                        Contact {
                            point: p[i].sub_scaled(n, d),
                            normal: n,
                            distance: d,
                        }
                    }
                })
            }
            WorldShape::Plane { equation } => std::array::from_fn(|i| {
                let d = equation.dot3(p[i]) + equation.w();
                Contact {
                    point: p[i].sub(equation.scale(d)),
                    normal: *equation,
                    distance: d,
                }
            }),
            WorldShape::Convex(_) | WorldShape::Unsupported => {
                std::array::from_fn(|i| self.closest(p[i], radius[i]))
            }
        }
    }
}

/// The closest point on a sphere (refined normalisation).
fn sphere_point(p: V4, centre: V4, radius: f32) -> Contact {
    let (n, length) = p.sub(centre).normalized_with_length();
    Contact {
        point: centre.add_scaled(n, radius),
        normal: n,
        distance: length - radius,
    }
}

/// Four vectors normalised together: squared lengths plus `FLT_EPSILON`, the refined reciprocal square root (0 where
/// not positive), each vector scaled; the lengths are `squared * inverse`.
fn batch_normalize(v: [V4; 4]) -> ([V4; 4], [f32; 4]) {
    let sq = v.map(|x| x.dot3(x) + EPSILON4.x());
    let inv = sq.map(|s| if s <= 0.0 { 0.0 } else { rsqrt23(s) });
    (
        std::array::from_fn(|i| v[i].scale(inv[i])),
        std::array::from_fn(|i| sq[i] * inv[i]),
    )
}

/// A float in `[0, 2^16)` to its integer part, as the cloth code does it: add `3 * 2^16` (so the float's last bit is
/// 1/64), then drop the six fractional bits of the mantissa.
fn grid_coordinate(x: f32) -> u32 {
    let shifted = x + 196_608.0;
    (shifted.to_bits() >> 6) & 0xFFFF
}

/// The closest point on a convex geometry: the particle in local space, the tetrahedra of its grid cell tested in
/// turn; the first that contains the particle (its surface face within the particle radius) gives the contact on that
/// face along the ray from the centroid.
fn convex_point(c: &Convex, world: V4, radius: f32) -> Contact {
    let no_contact = Contact {
        point: V4::ZERO,
        normal: V4::ZERO,
        distance: 256.0,
    };
    let local = c.local_from_world.transform_position_rigid(world);
    let r = V4::splat(radius);
    if !in_box(c.aabb_min.sub(r), c.aabb_max.add(r), local) {
        return no_contact;
    }
    let local = local.with_w(1.0);
    let tetrahedra: Vec<usize> = if c.grid_res > 1 {
        let rel = local.sub(c.aabb_min).mul(c.inv_cell_size);
        let rel = V4(rel.0.map(|x| if x < 0.0 { 0.0 } else { x }));
        let res = u32::from(c.grid_res);
        let id = |i: usize| grid_coordinate(rel.0[i]).min(res - 1);
        let cell = (id(2) + res * (id(1) + res * id(0))) as usize;
        let (Some(&from), Some(&to)) = (c.grid.get(cell), c.grid.get(cell + 1)) else {
            return no_contact;
        };
        (usize::from(from)..usize::from(to))
            .filter_map(|g| c.cells.get(g).map(|&t| usize::from(t)))
            .collect()
    } else {
        (0..c.tetrahedra.len()).collect()
    };
    for t in tetrahedra {
        let Some(m) = c.tetrahedra.get(t) else { continue };
        let [c0, c1, c2, c3] = m.cols;
        let d = c0
            .scale(local.x())
            .add_scaled(c1, local.y())
            .add_scaled(c2, local.z())
            .add_scaled(c3, local.w());
        let limit = [radius, 0.0, 0.0, 0.0];
        let inside: [bool; 4] = std::array::from_fn(|i| d.0[i] <= limit[i]);
        if inside.iter().all(|&b| b) {
            let direction = local.sub(c.centroid).normalized();
            let face = V4::new(c0.x(), c1.x(), c2.x(), c3.x());
            let hit = -(d.x() / face.dot3(direction));
            let point = local.add_scaled(direction, hit);
            return Contact {
                point: c.world_from_local.transform_position_rigid(point),
                normal: c.world_from_local.transform_direction(face),
                distance: d.x(),
            };
        }
        if !inside[0] {
            return no_contact;
        }
    }
    no_contact
}

/// The contact solve for one particle: out of the shape along the normal to the particle's radius, and the previous
/// position moved by the friction times the tangential displacement relative to the collidable's surface over `dt`.
#[allow(clippy::too_many_arguments, reason = "the contact and the particle it moves")]
pub fn solve_contact(
    contact: &Contact,
    collidable: &RuntimeCollidable,
    radius: f32,
    friction: f32,
    position: &mut V4,
    previous: &mut V4,
    dt: f32,
) {
    let p = *position;
    let pp = *previous;
    let n = contact.normal;
    let new_p = p.add_scaled(n, radius - p.sub(contact.point).dot3(n));
    let velocity = collidable.point_velocity(contact.point);
    let mut tangent = new_p.sub(pp);
    tangent = tangent.sub(velocity.scale(dt));
    tangent = tangent.sub_scaled(n, tangent.dot3(n));
    *previous = pp.add_scaled(tangent, friction);
    *position = new_p;
}

/// One collision pass over the sim cloth's per-instance collidables (the runtime copies in `collidables`), substep
/// length `dt`.
pub fn collide(
    sim: &SimCloth,
    collidables: &[RuntimeCollidable],
    state: &mut SimState,
    data: &ParticleData,
    dt: f32,
) {
    for (index, collidable) in collidables.iter().enumerate() {
        if !collidable.enabled {
            continue;
        }
        let particles = collected(sim, collidable, index, &state.positions);
        if particles.is_empty() {
            continue;
        }
        let shape = WorldShape::new(&collidable.shape, &collidable.transform);
        let (groups, rest) = particles.as_chunks::<4>();
        for &ix in groups {
            let radius = ix.map(|i| data.radius[i]);
            let contacts = shape.closest4(ix.map(|i| state.positions[i]), radius);
            for ((&i, contact), r) in ix.iter().zip(&contacts).zip(radius) {
                if contact.distance - r < 0.0 {
                    let SimState {
                        positions, previous, ..
                    } = state;
                    solve_contact(
                        contact,
                        collidable,
                        r,
                        data.friction[i],
                        &mut positions[i],
                        &mut previous[i],
                        dt,
                    );
                }
            }
        }
        for &i in rest {
            let r = data.radius[i];
            let contact = shape.closest(state.positions[i], r);
            if contact.distance - r < 0.0 {
                let SimState {
                    positions, previous, ..
                } = state;
                solve_contact(
                    &contact,
                    collidable,
                    r,
                    data.friction[i],
                    &mut positions[i],
                    &mut previous[i],
                    dt,
                );
            }
        }
    }
    if !state.landscape_planes.is_empty() {
        landscape(sim, state, data, dt);
    }
    // the virtual collision points, against the collidables that collide with them
    if sim.virtual_collision_points.points > 0 {
        for (index, collidable) in collidables.iter().enumerate() {
            if !collidable.enabled || !collidable.virtual_collision_points {
                continue;
            }
            let particles = collected(sim, collidable, index, &state.positions);
            if particles.is_empty() {
                continue;
            }
            let shape = WorldShape::new(&collidable.shape, &collidable.transform);
            virtual_points(sim, &shape, &particles, &mut state.positions, data);
        }
    }
}

/// The landscape collidable (after the per-instance ones): each particle whose static collision mask has bit 31 is
/// tested against its own contact plane (`x y z` the normal, `w` the offset), as the game computes them from the
/// world. Four at a time, the plane distance is `((dot3 + w) + radius)`; one at a time, `dot4xyz1 + radius`; the
/// contact point is the particle moved back along the normal by that distance. The landscape does not move.
fn landscape(sim: &SimCloth, state: &mut SimState, data: &ParticleData, dt: f32) {
    let still = RuntimeCollidable {
        transform: Mat4::IDENTITY,
        linear_velocity: V4::ZERO,
        angular_velocity: V4::ZERO,
        shape: Shape::Sphere { pos: [0.0; 4] },
        enabled: true,
        virtual_collision_points: false,
    };
    let particles: Vec<usize> = (0..state.positions.len())
        .filter(|&p| {
            sim.static_collision_masks
                .get(p)
                .is_some_and(|m| m & 0x8000_0000 != 0)
        })
        .filter(|&p| p < state.landscape_planes.len())
        .collect();
    let plane_contact = |plane: V4, p: V4, distance: f32| Contact {
        point: p.sub_scaled(plane, distance),
        normal: plane,
        distance,
    };
    let (groups, rest) = particles.as_chunks::<4>();
    for &ix in groups {
        let contacts: [Contact; 4] = ix.map(|i| {
            let plane = state.landscape_planes[i];
            let p = state.positions[i];
            plane_contact(plane, p, (plane.dot3(p) + plane.w()) + data.radius[i])
        });
        for (&i, contact) in ix.iter().zip(&contacts) {
            let r = data.radius[i];
            if contact.distance - r < 0.0 {
                let SimState {
                    positions, previous, ..
                } = state;
                solve_contact(
                    contact,
                    &still,
                    r,
                    data.friction[i],
                    &mut positions[i],
                    &mut previous[i],
                    dt,
                );
            }
        }
    }
    for &i in rest {
        let plane = state.landscape_planes[i];
        let p = state.positions[i];
        let r = data.radius[i];
        let contact = plane_contact(plane, p, plane.dot4_xyz1(p) + r);
        if contact.distance - r < 0.0 {
            let SimState {
                positions, previous, ..
            } = state;
            solve_contact(
                &contact,
                &still,
                r,
                data.friction[i],
                &mut positions[i],
                &mut previous[i],
                dt,
            );
        }
    }
}

/// The particles a collidable is tested against: those whose static collision mask has the collidable's bit
/// (`1 << min(index, 30)`) and that lie in its bounding volume.
fn collected(sim: &SimCloth, collidable: &RuntimeCollidable, index: usize, positions: &[V4]) -> Vec<usize> {
    let mask = 1u32 << index.min(30);
    let volume =
        WorldShape::bounding_volume(&collidable.shape, &collidable.transform, sim.max_particle_radius);
    (0..positions.len())
        .filter(|&p| {
            sim.static_collision_masks.get(p).is_some_and(|m| m & mask != 0) && volume.contains(positions[p])
        })
        .collect()
}

/// The virtual collision points of the collected particles against one shape (`QUERY_COLLIDABLE`). For each
/// particle with a triangle fan: each triangle's points `p0 + (p1 - p0)*u + (p2 - p0)*v` (from the particle's current
/// position, the edges fixed per triangle) are queried one at a time with the particle's radius; a point inside pushes
/// the particle out along the triangle's normal (refined, turned to face the contact normal) by its depth. Then, for
/// each particle with an edge fan, each edge's points `p0 + (p1 - p0)*u` push it out along the contact normal. The
/// previous positions are left as they are.
fn virtual_points(
    sim: &SimCloth,
    shape: &WorldShape,
    particles: &[usize],
    positions: &mut [V4],
    data: &ParticleData,
) {
    let vcp = &sim.virtual_collision_points;
    let at = |positions: &[V4], i: u16| positions.get(usize::from(i)).copied().unwrap_or(V4::ZERO);
    for &rp in particles {
        let Some(&fan_index) = vcp.triangle_fan_indices.get(rp) else {
            continue;
        };
        let Some(fan) = vcp.triangle_fans.get(usize::from(fan_index)) else {
            continue;
        };
        let radius = data.radius[rp];
        for j in 0..u16::from(fan.triangles) {
            let Some(section) = vcp.triangles.get(usize::from(fan.vcp_start + j)) else {
                continue;
            };
            let p0 = positions[rp];
            let p01 = at(positions, section.opposite[0]).sub(p0);
            let p02 = at(positions, section.opposite[1]).sub(p0);
            let Some(entry) = vcp.triangle_dictionary.get(usize::from(section.dictionary)) else {
                continue;
            };
            for k in 0..u16::from(entry.count) {
                let Some(&(u, v)) = vcp.triangle_barycentrics.get(usize::from(entry.start + k)) else {
                    continue;
                };
                let point = positions[rp].add_scaled(p01, u).add_scaled(p02, v);
                let contact = shape.closest(point, radius);
                let depth = contact.distance - radius;
                if depth < 0.0 {
                    let mut normal = p01.cross(p02).normalized();
                    if normal.dot3(contact.normal) < 0.0 {
                        normal = normal.neg3();
                    }
                    positions[rp] = positions[rp].sub_scaled(normal, depth);
                }
            }
        }
    }
    if vcp.edge_fans.is_empty() {
        return;
    }
    for &rp in particles {
        let Some(&fan_index) = vcp.edge_fan_indices.get(rp) else {
            continue;
        };
        let Some(fan) = vcp.edge_fans.get(usize::from(fan_index)) else {
            continue;
        };
        let radius = data.radius[rp];
        for j in 0..u16::from(fan.edges) {
            let Some(section) = vcp.edges.get(usize::from(fan.edge_start + j)) else {
                continue;
            };
            let p01 = at(positions, section.opposite).sub(positions[rp]);
            let Some(entry) = vcp.edge_dictionary.get(usize::from(section.dictionary)) else {
                continue;
            };
            for k in 0..u16::from(entry.count) {
                let Some(&u) = vcp.edge_barycentrics.get(usize::from(entry.start + k)) else {
                    continue;
                };
                let point = positions[rp].add_scaled(p01, u);
                let contact = shape.closest(point, radius);
                let depth = contact.distance - radius;
                if depth < 0.0 {
                    positions[rp] = positions[rp].sub_scaled(contact.normal, depth);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asin_matches_the_reference_to_its_accuracy() {
        for i in -100..=100 {
            let x = i as f32 / 100.0;
            let got = f64::from(asin(x));
            let want = f64::from(x).asin();
            assert!((got - want).abs() < 2e-6, "asin({x}) = {got}, want {want}");
            assert!((f64::from(acos(x)) - f64::from(x).acos()).abs() < 2e-6);
        }
        // linear below 1e-4, odd, clamped
        assert_eq!(asin(5e-5), 5e-5);
        assert_eq!(asin(-0.3), -asin(0.3));
        assert_eq!(asin(2.0), asin(1.0));
    }

    #[test]
    fn dot4_adds_x_to_z_and_y_to_w_first() {
        // the products are 1, 2^-24, 0 and 2^-24: (1 + 0) + (2^-24 + 2^-24) = 1 + 2^-23 is representable, where
        // (1 + 2^-24) + (0 + 2^-24) rounds 1 + 2^-24 to 1 (ties to even) and then 1 + 2^-24 to 1 again
        let tiny = 2f32.powi(-12);
        let a = V4::new(1.0, tiny, 0.0, tiny);
        assert_eq!(dot4(a, a), 1.0 + 2f32.powi(-23));
    }

    #[test]
    fn quaternion_round_trips_a_rotation() {
        // 90 degrees about z
        let m = Mat4::from_columns([
            0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let q = Quat::from_rotation(&m);
        let h = std::f32::consts::FRAC_1_SQRT_2;
        assert!((q.0.z() - h).abs() < 1e-6 && (q.0.w() - h).abs() < 1e-6, "{q:?}");
        let r = q.to_rotation();
        for (got, want) in r.iter().zip(&m.cols) {
            for k in 0..3 {
                assert!((got.0[k] - want.0[k]).abs() < 1e-6, "{r:?}");
            }
        }
        assert!((q.angle() - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
        let axis = q.axis();
        assert!((axis.z() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn quaternion_from_rotation_uses_the_largest_diagonal_when_the_trace_is_negative() {
        // 180 degrees about x: diag (1, -1, -1), trace -1
        let m = Mat4::from_columns([
            1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let q = Quat::from_rotation(&m);
        assert_eq!(q.0, V4::new(1.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn driving_a_collidable_and_stepping_reaches_the_target() {
        let shape = Shape::Sphere {
            pos: [0.0, 0.0, 0.0, 0.5],
        };
        let mut c = RuntimeCollidable {
            transform: Mat4::IDENTITY,
            linear_velocity: V4::ZERO,
            angular_velocity: V4::ZERO,
            shape,
            enabled: true,
            virtual_collision_points: false,
        };
        let a = 0.3f32;
        let target = Mat4::from_columns([
            a.cos(),
            a.sin(),
            0.0,
            0.0,
            -a.sin(),
            a.cos(),
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            2.0,
            3.0,
            1.0,
        ]);
        c.drive_to(&target, 0.5);
        assert!((c.linear_velocity.x() - 2.0).abs() < 1e-6);
        assert!(
            (c.angular_velocity.z() - 0.6).abs() < 1e-4,
            "{:?}",
            c.angular_velocity
        );
        c.step_forward(0.25);
        c.step_forward(0.25);
        let t = c.transform.translation();
        assert!((t.x() - 1.0).abs() < 1e-6 && (t.y() - 2.0).abs() < 1e-6 && (t.z() - 3.0).abs() < 1e-6);
        // the polynomial step is an approximation of the rotation: within 0.2% of the angle
        let q = Quat::from_rotation(&c.transform);
        assert!((q.angle() - a).abs() < a * 0.003, "{}", q.angle());
    }

    fn contact_dist(shape: &WorldShape, p: V4) -> f32 {
        shape.closest(p, 0.0).distance
    }

    #[test]
    fn closest_points_on_each_shape() {
        let sphere = WorldShape::new(
            &Shape::Sphere {
                pos: [1.0, 0.0, 0.0, 0.5],
            },
            &Mat4::IDENTITY,
        );
        assert!((contact_dist(&sphere, V4::new(3.0, 0.0, 0.0, 0.0)) - 1.5).abs() < 1e-6);
        let capsule = WorldShape::new(
            &Shape::Capsule {
                start: [0.0, 0.0, 0.0, 0.0],
                end: [0.0, 2.0, 0.0, 0.0],
                dir: [0.0, 1.0, 0.0, 0.0],
                radius: 0.25,
                cap_len_sqrd_inv: 0.25,
            },
            &Mat4::IDENTITY,
        );
        let c = capsule.closest(V4::new(1.0, 1.0, 0.0, 0.0), 0.0);
        assert!((c.distance - 0.75).abs() < 1e-6 && (c.normal.x() - 1.0).abs() < 1e-6);
        // beyond the end: the end cap
        assert!((contact_dist(&capsule, V4::new(0.0, 4.0, 0.0, 0.0)) - 1.75).abs() < 1e-6);
        let plane = WorldShape::new(
            &Shape::Plane {
                equation: [0.0, 1.0, 0.0, -1.0],
            },
            &Mat4::IDENTITY,
        );
        assert_eq!(contact_dist(&plane, V4::new(5.0, 3.0, -2.0, 0.0)), 2.0);
        let c = plane.closest(V4::new(5.0, 3.0, -2.0, 0.0), 0.0);
        assert_eq!((c.point.x(), c.point.y(), c.point.z()), (5.0, 1.0, -2.0));
    }

    #[test]
    fn the_four_wide_queries_agree_with_the_single_ones_to_rounding() {
        let shapes = [
            WorldShape::new(
                &Shape::Sphere {
                    pos: [1.0, 0.0, 0.0, 0.5],
                },
                &Mat4::IDENTITY,
            ),
            WorldShape::new(
                &Shape::Capsule {
                    start: [0.0, 0.0, 0.0, 0.0],
                    end: [0.0, 2.0, 0.0, 0.0],
                    dir: [0.0, 1.0, 0.0, 0.0],
                    radius: 0.25,
                    cap_len_sqrd_inv: 0.25,
                },
                &Mat4::IDENTITY,
            ),
            WorldShape::TaperedCapsule(TaperedCapsule::new(
                V4::new(0.0, 0.0, 0.0, 0.0),
                V4::new(0.0, 1.0, 0.0, 0.0),
                0.1,
                0.3,
            )),
            WorldShape::new(
                &Shape::Plane {
                    equation: [0.0, 1.0, 0.0, -1.0],
                },
                &Mat4::IDENTITY,
            ),
        ];
        let p = [
            V4::new(0.3, -0.2, 0.1, 1.0),
            V4::new(0.2, 0.5, -0.1, 1.0),
            V4::new(-0.4, 1.4, 0.2, 1.0),
            V4::new(0.05, 0.9, 0.0, 1.0),
        ];
        for shape in &shapes {
            let four = shape.closest4(p, [0.0; 4]);
            for i in 0..4 {
                let one = shape.closest(p[i], 0.0);
                assert!(
                    (four[i].distance - one.distance).abs() < 1e-5,
                    "{shape:?} {i}: {four:?} {one:?}"
                );
            }
        }
    }

    #[test]
    fn a_resting_contact_moves_the_particle_out_and_keeps_friction_off_a_still_collidable() {
        let collidable = RuntimeCollidable {
            transform: Mat4::IDENTITY,
            linear_velocity: V4::ZERO,
            angular_velocity: V4::ZERO,
            shape: Shape::Plane {
                equation: [0.0, 1.0, 0.0, 0.0],
            },
            enabled: true,
            virtual_collision_points: false,
        };
        let shape = WorldShape::new(&collidable.shape, &collidable.transform);
        let mut p = V4::new(0.0, -0.5, 0.0, 0.0);
        let mut pp = V4::new(-1.0, -0.5, 0.0, 0.0);
        let c = shape.closest(p, 0.1);
        solve_contact(&c, &collidable, 0.1, 0.0, &mut p, &mut pp, 0.1);
        assert!((p.y() - 0.1).abs() < 1e-6);
        assert_eq!(pp, V4::new(-1.0, -0.5, 0.0, 0.0));
        // full friction: the previous position takes the whole tangential displacement
        let mut p = V4::new(0.0, -0.5, 0.0, 0.0);
        let mut pp = V4::new(-1.0, -0.5, 0.0, 0.0);
        solve_contact(&c, &collidable, 0.1, 1.0, &mut p, &mut pp, 0.1);
        assert!((pp.x() - 0.0).abs() < 1e-6, "{pp:?}");
    }

    #[test]
    fn grid_coordinates_round_to_sixty_fourths_then_truncate() {
        assert_eq!(grid_coordinate(0.0), 0);
        assert_eq!(grid_coordinate(2.99), 2);
        // 2.995 rounds to 3 at a precision of 1/64
        assert_eq!(grid_coordinate(2.995), 3);
        assert_eq!(grid_coordinate(3.5), 3);
    }
}
