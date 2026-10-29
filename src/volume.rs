//! The volume constraint (`hclVolumeConstraintMx`): a set of particles keeps its shape about its weighted centre. Each
//! application finds the frame that best matches the particles to their rest offsets (the rotational part of a polar
//! decomposition, from a symmetric eigen-solve warm-started from the last one) and moves the constrained particles
//! towards their rest offsets in that frame.

use crate::simd::{V4, rcp23, sqrt23};
use havok_formats::cloth::{VolumeApplyBatch, VolumeApplySingle, VolumeFrameBatch, VolumeFrameSingle};

/// The smallest positive `f32` step from 1 (`FLT_EPSILON`).
const EPS: f32 = f32::EPSILON;

/// A 3x3 matrix as three four-lane columns (only `x y z` are used).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3 {
    /// The columns.
    pub cols: [V4; 3],
}

impl Mat3 {
    /// The identity.
    pub const IDENTITY: Mat3 = Mat3 {
        cols: [
            V4::new(1.0, 0.0, 0.0, 0.0),
            V4::new(0.0, 1.0, 0.0, 0.0),
            V4::new(0.0, 0.0, 1.0, 0.0),
        ],
    };

    /// Element at `row`, `col`.
    pub fn at(&self, row: usize, col: usize) -> f32 {
        self.cols[col].0[row]
    }

    fn set(&mut self, row: usize, col: usize, v: f32) {
        self.cols[col].0[row] = v;
    }

    /// A vector moved by the columns: `(c0*x + c1*y) + c2*z`.
    pub fn rotate(&self, v: V4) -> V4 {
        let [c0, c1, c2] = self.cols;
        c0.scale(v.x()).add_scaled(c1, v.y()).add_scaled(c2, v.z())
    }

    /// `self * b`: each of `b`'s columns through [`Mat3::rotate`].
    pub fn mul(&self, b: &Mat3) -> Mat3 {
        Mat3 {
            cols: b.cols.map(|c| self.rotate(c)),
        }
    }

    /// The transpose (`w` lanes zero).
    pub fn transpose(&self) -> Mat3 {
        let c = |i: usize| V4::new(self.at(i, 0), self.at(i, 1), self.at(i, 2), 0.0);
        Mat3 {
            cols: [c(0), c(1), c(2)],
        }
    }

    /// The column-wise sum.
    pub fn add(&self, b: &Mat3) -> Mat3 {
        Mat3 {
            cols: [0, 1, 2].map(|i| self.cols[i].add(b.cols[i])),
        }
    }

    /// The inverse of a symmetric matrix: the cofactor columns `c1 x c2`, `c2 x c0`, `c0 x c1` times the refined
    /// reciprocal of the determinant `dot3(c0, c1 x c2)`; zero when `|det|` is not above
    /// `(eps * |c0|_1) * (|c1|_1 * |c2|_1)` (the columns' L1 norms).
    pub fn inverse_symmetric(&self) -> Mat3 {
        let [c0, c1, c2] = self.cols;
        let r0 = c1.cross(c2);
        let r1 = c2.cross(c0);
        let r2 = c0.cross(c1);
        let det = c0.dot3(r0);
        let l1 = |c: V4| (c.x().abs() + c.y().abs()) + c.z().abs();
        let limit = (EPS * l1(c0)) * (l1(c1) * l1(c2));
        if det.abs() <= limit {
            return Mat3 { cols: [V4::ZERO; 3] };
        }
        let inv = rcp23(det);
        Mat3 {
            cols: [r0.scale(inv), r1.scale(inv), r2.scale(inv)],
        }
    }

    /// The diagonal matrix of `d`'s `x y z`.
    pub fn diagonal(d: V4) -> Mat3 {
        Mat3 {
            cols: [
                V4::new(d.x(), 0.0, 0.0, 0.0),
                V4::new(0.0, d.y(), 0.0, 0.0),
                V4::new(0.0, 0.0, d.z(), 0.0),
            ],
        }
    }

    /// The eigenvalues and eigenvectors of a symmetric matrix by cyclic Jacobi rotations, starting from the
    /// eigenvectors `vectors` (updated in place, their columns normalised at the end): at most `max_iter` rotations,
    /// stopping once twice the sum of the squared off-diagonal entries falls below `(eps^2) * |self|_F^2`.
    pub fn diagonalize_symmetric_warm(&self, vectors: &mut Mat3, max_iter: u32) -> V4 {
        let mut m = vectors.transpose().mul(self).mul(vectors);
        let frob = (self.cols[0].dot3(self.cols[0]) + self.cols[1].dot3(self.cols[1]))
            + self.cols[2].dot3(self.cols[2]);
        let eps_sqr = (EPS * EPS) * frob;
        let mut iter = 0;
        loop {
            let (m01, m02, m12) = (m.at(0, 1), m.at(0, 2), m.at(1, 2));
            let off = 2.0 * ((m01 * m01 + m02 * m02) + m12 * m12);
            if off < eps_sqr || iter >= max_iter {
                break;
            }
            // the largest off-diagonal entry, first of equals
            let (mut p, mut q, mut largest) = (0, 1, m01.abs());
            if m02.abs() > largest {
                (p, q, largest) = (0, 2, m02.abs());
            }
            if m12.abs() > largest {
                (p, q) = (1, 2);
            }
            let mpq = m.at(p, q);
            let (c, s) = if mpq != 0.0 {
                let tau = (m.at(q, q) - m.at(p, p)) / (2.0 * mpq);
                let root = (1.0 + tau * tau).sqrt();
                let t = if tau >= 0.0 {
                    1.0 / (tau + root)
                } else {
                    1.0 / (tau - root)
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                (c, t * c)
            } else {
                (1.0, 0.0)
            };
            let mut jacobi = Mat3::IDENTITY;
            jacobi.set(p, p, c);
            jacobi.set(p, q, s);
            jacobi.set(q, p, -s);
            jacobi.set(q, q, c);
            m = m.mul(&jacobi);
            m = jacobi.transpose().mul(&m);
            *vectors = vectors.mul(&jacobi);
            iter += 1;
        }
        for c in &mut vectors.cols {
            *c = c.normalized();
        }
        V4::new(m.at(0, 0), m.at(1, 1), m.at(2, 2), m.at(2, 2))
    }
}

/// A volume constraint's state on one sim cloth instance: the frame of its last application and the eigenvectors
/// that warm-start the next.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeState {
    /// The frame's rotation (not orthonormal for planar sets).
    pub rotation: Mat3,
    /// The frame's origin, the weighted centre.
    pub translation: V4,
    /// The eigenvectors of the last solve.
    pub eigenvectors: Mat3,
}

impl Default for VolumeState {
    fn default() -> Self {
        VolumeState {
            rotation: Mat3::IDENTITY,
            translation: V4::ZERO,
            eigenvectors: Mat3::IDENTITY,
        }
    }
}

/// The weighted centre of a batch list: per lane of four, `acc + weight*position`, the four sums added as
/// `(a0 + a1) + (a2 + a3)`; then the singles' weighted sum added.
fn centre(positions: &[V4], batches: &[VolumeFrameBatch], singles: &[VolumeFrameSingle]) -> V4 {
    let mut t = V4::ZERO;
    if !batches.is_empty() {
        let mut acc = [V4::ZERO; 4];
        for b in batches {
            for i in 0..16 {
                let p = positions
                    .get(usize::from(b.particles[i]))
                    .copied()
                    .unwrap_or(V4::ZERO);
                acc[i % 4] = acc[i % 4].add(p.scale(b.weights[i]));
            }
        }
        t = acc[0].add(acc[1]).add(acc[2].add(acc[3]));
    }
    if !singles.is_empty() {
        let mut sum = V4::ZERO;
        for s in singles {
            let p = positions
                .get(usize::from(s.particle))
                .copied()
                .unwrap_or(V4::ZERO);
            sum = sum.add(p.scale(s.weight));
        }
        t = t.add(sum);
    }
    t
}

/// The weighted correlation of the particles' offsets from `com` with their rest offsets: column j sums
/// `weight * (p - com) * f[j]`, each accumulator starting at `eps` in every lane.
fn correlation(
    positions: &[V4],
    com: V4,
    batches: &[VolumeFrameBatch],
    singles: &[VolumeFrameSingle],
) -> Mat3 {
    let mut out = Mat3 { cols: [V4::ZERO; 3] };
    if !batches.is_empty() {
        let mut acc = [[V4::splat(EPS); 4]; 3];
        for b in batches {
            for i in 0..16 {
                let p = positions
                    .get(usize::from(b.particles[i]))
                    .copied()
                    .unwrap_or(V4::ZERO)
                    .sub(com);
                let f = b.frame_vectors[i];
                for (j, a) in acc.iter_mut().enumerate() {
                    a[i % 4] = a[i % 4].add(p.scale(f[j]).scale(b.weights[i]));
                }
            }
        }
        let sum = |a: [V4; 4]| a[0].add(a[1]).add(a[2].add(a[3]));
        out = out.add(&Mat3 {
            cols: [sum(acc[0]), sum(acc[1]), sum(acc[2])],
        });
    }
    if !singles.is_empty() {
        let mut m = [V4::splat(EPS); 3];
        for s in singles {
            let p = positions
                .get(usize::from(s.particle))
                .copied()
                .unwrap_or(V4::ZERO)
                .sub(com);
            for (j, c) in m.iter_mut().enumerate() {
                *c = c.add(p.scale(s.frame_vector[j]).scale(s.weight));
            }
        }
        out = out.add(&Mat3 { cols: m });
    }
    out
}

/// One application of a volume constraint at `stiffness` (nothing at or below zero).
pub fn apply(
    positions: &mut [V4],
    state: &mut VolumeState,
    frame_batches: &[VolumeFrameBatch],
    frame_singles: &[VolumeFrameSingle],
    apply_batches: &[VolumeApplyBatch],
    apply_singles: &[VolumeApplySingle],
    stiffness: f32,
) {
    if stiffness <= 0.0 {
        return;
    }
    state.translation = centre(positions, frame_batches, frame_singles);
    let a = correlation(positions, state.translation, frame_batches, frame_singles);
    // the rotation of the polar decomposition a = R S: R = a (V sqrt(D) V^T)^-1, with V D V^T = a^T a
    let ata = a.transpose().mul(&a);
    let values = ata.diagonalize_symmetric_warm(&mut state.eigenvectors, 20);
    let root = V4(values.0.map(|v| sqrt23(v.abs()))).add(V4::splat(EPS));
    let s = state
        .eigenvectors
        .mul(&Mat3::diagonal(root).mul(&state.eigenvectors.transpose()));
    state.rotation = a.mul(&s.inverse_symmetric());
    let [c0, c1, c2] = state.rotation.cols;
    let t = state.translation;
    for b in apply_batches {
        for i in 0..16 {
            let p = usize::from(b.particles[i]);
            let Some(&x) = positions.get(p) else { continue };
            // a batch's target adds the centre after the first column: `((c0*x + t) + c1*y) + c2*z`
            let f = V4(b.frame_vectors[i]);
            let target = c0.scale(f.x()).add(t).add_scaled(c1, f.y()).add_scaled(c2, f.z());
            let k = b.stiffnesses[i] * stiffness;
            positions[p] = x.add(target.sub(x).scale(k));
        }
    }
    for s in apply_singles {
        let p = usize::from(s.particle);
        let Some(&x) = positions.get(p) else { continue };
        // a single's target is the frame's transform of the point: `((c0*x + c1*y) + c2*z) + t`
        let target = state.rotation.rotate(V4(s.frame_vector)).add(t);
        let k = s.stiffness * stiffness;
        positions[p] = x.add(target.sub(x).scale(k));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_diagonal_matrix_is_its_own_eigen_decomposition() {
        let m = Mat3::diagonal(V4::new(4.0, 2.0, 1.0, 0.0));
        let mut v = Mat3::IDENTITY;
        let values = m.diagonalize_symmetric_warm(&mut v, 20);
        assert_eq!((values.x(), values.y(), values.z()), (4.0, 2.0, 1.0));
        assert_eq!(v, Mat3::IDENTITY);
    }

    #[test]
    fn a_symmetric_inverse_undoes_the_matrix() {
        // a symmetric matrix with off-diagonal terms: the product with its inverse is the identity to the refined
        // reciprocal's accuracy (a few ulp)
        let m = Mat3 {
            cols: [
                V4::new(4.0, 1.0, 0.5, 0.0),
                V4::new(1.0, 3.0, -0.25, 0.0),
                V4::new(0.5, -0.25, 2.0, 0.0),
            ],
        };
        let product = m.mul(&m.inverse_symmetric());
        for (c, w) in product.cols.iter().zip(Mat3::IDENTITY.cols) {
            for k in 0..3 {
                assert!((c.0[k] - w.0[k]).abs() < 4.0 * EPS, "{product:?}");
            }
        }
        // singular: zero
        let singular = Mat3::diagonal(V4::new(1.0, 1.0, 0.0, 0.0));
        assert_eq!(singular.inverse_symmetric(), Mat3 { cols: [V4::ZERO; 3] });
    }

    #[test]
    fn the_transpose_swaps_rows_and_columns() {
        let m = Mat3 {
            cols: [
                V4::new(1.0, 2.0, 3.0, 0.0),
                V4::new(4.0, 5.0, 6.0, 0.0),
                V4::new(7.0, 8.0, 9.0, 0.0),
            ],
        };
        let t = m.transpose();
        assert_eq!(t.cols[0], V4::new(1.0, 4.0, 7.0, 0.0));
        assert_eq!(t.transpose(), m);
    }
}

#[cfg(test)]
mod rigid_tests {
    use super::*;

    /// Rest offsets of a non-planar set (a tetrahedron about its centre) with equal weights.
    fn rest() -> Vec<[f32; 4]> {
        vec![
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, -2.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
        ]
    }

    #[test]
    fn a_rigidly_moved_set_gives_back_its_rotation_and_centre_and_stays_put() {
        // rest offsets turned 90 degrees about z ((x, y) -> (-y, x)) and moved by (10, 20, 30)
        let offsets = rest();
        let positions_rest: Vec<V4> = offsets
            .iter()
            .map(|f| V4::new(-f[1] + 10.0, f[0] + 20.0, f[2] + 30.0, 1.0))
            .collect();
        let mut positions = positions_rest.clone();
        let w = 1.0 / offsets.len() as f32;
        let frames: Vec<VolumeFrameSingle> = offsets
            .iter()
            .enumerate()
            .map(|(i, f)| VolumeFrameSingle {
                frame_vector: *f,
                particle: i as u16,
                weight: w,
            })
            .collect();
        let applies: Vec<VolumeApplySingle> = offsets
            .iter()
            .enumerate()
            .map(|(i, f)| VolumeApplySingle {
                frame_vector: *f,
                particle: i as u16,
                stiffness: 1.0,
            })
            .collect();
        let mut state = VolumeState::default();
        apply(&mut positions, &mut state, &[], &frames, &[], &applies, 1.0);
        let want = [
            V4::new(0.0, 1.0, 0.0, 0.0),
            V4::new(-1.0, 0.0, 0.0, 0.0),
            V4::new(0.0, 0.0, 1.0, 0.0),
        ];
        for (c, w) in state.rotation.cols.iter().zip(want) {
            for k in 0..3 {
                assert!(
                    (c.0[k] - w.0[k]).abs() < 1e-4,
                    "rotation {:?} not {:?}",
                    state.rotation,
                    want
                );
            }
        }
        for k in 0..3 {
            assert!((state.translation.0[k] - [10.0, 20.0, 30.0][k]).abs() < 1e-4);
        }
        for (p, q) in positions.iter().zip(&positions_rest) {
            assert!(p.sub(*q).length3() < 1e-3, "{p:?} moved from {q:?}");
        }
    }
}
