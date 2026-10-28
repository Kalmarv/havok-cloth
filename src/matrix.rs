//! 4x4 matrices as the cloth data stores them (four columns of four floats) and the two ways the cloth code moves a
//! point with one. The two differ only in where the translation is added, which changes the last bit.

use crate::simd::V4;

/// A matrix of four columns (`hkMatrix4`, or an `hkTransform` whose first three columns are the rotation and whose
/// fourth is the translation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
    /// The columns.
    pub cols: [V4; 4],
}

impl Mat4 {
    /// The identity.
    pub const IDENTITY: Mat4 = Mat4 {
        cols: [
            V4::new(1.0, 0.0, 0.0, 0.0),
            V4::new(0.0, 1.0, 0.0, 0.0),
            V4::new(0.0, 0.0, 1.0, 0.0),
            V4::new(0.0, 0.0, 0.0, 1.0),
        ],
    };

    /// A matrix from its 16 floats as stored: four columns of four.
    pub fn from_columns(m: [f32; 16]) -> Mat4 {
        let c = |i: usize| V4([m[i * 4], m[i * 4 + 1], m[i * 4 + 2], m[i * 4 + 3]]);
        Mat4 {
            cols: [c(0), c(1), c(2), c(3)],
        }
    }

    /// The 16 floats, four columns of four.
    pub fn to_columns(self) -> [f32; 16] {
        let mut out = [0.0; 16];
        for (chunk, col) in out.as_chunks_mut::<4>().0.iter_mut().zip(self.cols) {
            chunk.copy_from_slice(&col.0);
        }
        out
    }

    /// The translation column.
    pub fn translation(self) -> V4 {
        self.cols[3]
    }

    /// A direction moved by the first three columns, as `hkMatrix4` does it: `(c0*x + c1*y) + c2*z`.
    pub fn transform_direction(&self, d: V4) -> V4 {
        let [c0, c1, c2, _] = self.cols;
        c0.scale(d.x()).add_scaled(c1, d.y()).add_scaled(c2, d.z())
    }

    /// A point moved as `hkMatrix4` does it: `((c3 + c0*x) + c1*y) + c2*z`.
    pub fn transform_position(&self, p: V4) -> V4 {
        let [c0, c1, c2, c3] = self.cols;
        c3.add_scaled(c0, p.x())
            .add_scaled(c1, p.y())
            .add_scaled(c2, p.z())
    }

    /// A point moved as an `hkTransform` does it: the rotated direction `(c0*x + c1*y) + c2*z`, then `+ c3`.
    pub fn transform_position_rigid(&self, p: V4) -> V4 {
        self.transform_direction(p).add(self.cols[3])
    }

    /// `self * b` for affine matrices: each of `b`'s first three columns as a direction, its fourth as a point.
    pub fn mul_affine(&self, b: &Mat4) -> Mat4 {
        Mat4 {
            cols: [
                self.transform_direction(b.cols[0]),
                self.transform_direction(b.cols[1]),
                self.transform_direction(b.cols[2]),
                self.transform_position(b.cols[3]),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rotation_z_90_and_move() -> Mat4 {
        Mat4::from_columns([
            0.0, 1.0, 0.0, 0.0, // x goes to y
            -1.0, 0.0, 0.0, 0.0, // y goes to -x
            0.0, 0.0, 1.0, 0.0, //
            10.0, 20.0, 30.0, 1.0,
        ])
    }

    #[test]
    fn columns_round_trip() {
        let m = rotation_z_90_and_move();
        assert_eq!(Mat4::from_columns(m.to_columns()), m);
        assert_eq!(m.translation(), V4::new(10.0, 20.0, 30.0, 1.0));
    }

    #[test]
    fn exact_transforms_agree_on_exact_values() {
        let m = rotation_z_90_and_move();
        let p = V4::new(1.0, 2.0, 3.0, 1.0);
        assert_eq!(m.transform_direction(p), V4::new(-2.0, 1.0, 3.0, 0.0));
        assert_eq!(m.transform_position(p), V4::new(8.0, 21.0, 33.0, 1.0));
        assert_eq!(m.transform_position_rigid(p), V4::new(8.0, 21.0, 33.0, 1.0));
    }

    #[test]
    fn the_two_position_orders_round_differently() {
        // x picks up 1 from each of the first two columns and 2^24 from the translation: added first, the
        // translation swallows each 1 in turn ((2^24 + 1) + 1 = 2^24); added last, (1 + 1) + 2^24 = 2^24 + 2
        let m = Mat4 {
            cols: [
                V4::new(1.0, 0.0, 0.0, 0.0),
                V4::new(1.0, 0.0, 0.0, 0.0),
                V4::new(0.0, 0.0, 1.0, 0.0),
                V4::new(16_777_216.0, 0.0, 0.0, 1.0),
            ],
        };
        let p = V4::new(1.0, 1.0, 0.0, 0.0);
        assert_eq!(m.transform_position(p).x(), 16_777_216.0);
        assert_eq!(m.transform_position_rigid(p).x(), 16_777_218.0);
    }

    #[test]
    fn affine_product_composes() {
        let m = rotation_z_90_and_move();
        let mm = m.mul_affine(&m);
        let p = V4::new(1.0, 2.0, 3.0, 1.0);
        assert_eq!(
            mm.transform_position(p),
            m.transform_position(m.transform_position(p))
        );
        assert_eq!(Mat4::IDENTITY.mul_affine(&m), m);
    }
}
