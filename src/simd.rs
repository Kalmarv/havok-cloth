//! Four-lane `f32` arithmetic that rounds as the games' cloth code does.
//!
//! The games run their cloth on SSE2 vector instructions: every operation rounds once, a "multiply-add" is a product
//! rounded and then a sum rounded (no fused multiply-add), and a three-lane dot product adds the x and y products first
//! and the z product after. Each function here names the exact sequence it computes; the order matters, because
//! reordering a sum changes its last bit.
//!
//! The approximate reciprocal and reciprocal square root the solver uses are the CPU's own estimate instructions
//! (`rcpps`, `rsqrtps`). Their results differ in the last bits between processor makers, and the game's cloth differs
//! with them. On x86-64 this module issues the same instructions, so on one machine the results equal the game's bit
//! for bit; on other targets it computes the exact reciprocal (square root) instead, which matches the game's behaviour
//! but not its last bits.

/// Four `f32` lanes, `x y z w`. A position or direction keeps its `w` lane as the operations leave it; the cloth
/// code carries it through rather than clearing it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V4(pub [f32; 4]);

/// The lane-wise minimum as `minps` computes it: `a` when `a < b`, else `b` (so a NaN in either gives `b`).
#[inline]
pub fn min_ps(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// The lane-wise maximum as `maxps` computes it: `a` when `a > b`, else `b`.
#[inline]
pub fn max_ps(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// The CPU's reciprocal estimate of `x` (`rcpss`, about 12 bits) on x86-64; `1 / x` elsewhere.
#[inline]
pub fn rcp_estimate(x: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    #[allow(
        unsafe_code,
        reason = "the estimate instruction itself; SSE is part of every x86-64 CPU"
    )]
    {
        use std::arch::x86_64::{_mm_cvtss_f32, _mm_rcp_ss, _mm_set_ss};
        // SAFETY: the intrinsics need SSE, which the x86-64 baseline always has; they touch no memory.
        unsafe { _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(x))) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        1.0 / x
    }
}

/// The CPU's reciprocal square root estimate of `x` (`rsqrtss`, about 12 bits) on x86-64; `1 / sqrt(x)` elsewhere.
#[inline]
pub fn rsqrt_estimate(x: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    #[allow(
        unsafe_code,
        reason = "the estimate instruction itself; SSE is part of every x86-64 CPU"
    )]
    {
        use std::arch::x86_64::{_mm_cvtss_f32, _mm_rsqrt_ss, _mm_set_ss};
        // SAFETY: the intrinsics need SSE, which the x86-64 baseline always has; they touch no memory.
        unsafe { _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x))) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        1.0 / x.sqrt()
    }
}

/// The refined ("23-bit") reciprocal: the estimate `r` and one Newton step, `r * (2 - x*r)`.
#[inline]
pub fn rcp23(x: f32) -> f32 {
    let r = rcp_estimate(x);
    r * (2.0 - x * r)
}

/// The refined ("23-bit") reciprocal square root: the estimate `e` and one Newton step, `(0.5*e) * (3 - (x*e)*e)`.
#[inline]
pub fn rsqrt23(x: f32) -> f32 {
    let e = rsqrt_estimate(x);
    (0.5 * e) * (3.0 - (x * e) * e)
}

/// `rcp23(x)`, or 0 where `x` is 0.
#[inline]
pub fn rcp23_or_zero(x: f32) -> f32 {
    if x == 0.0 { 0.0 } else { rcp23(x) }
}

/// The refined square root, `x * rsqrt23(x)`, or 0 where `x <= 0`.
#[inline]
pub fn sqrt23(x: f32) -> f32 {
    if x <= 0.0 { 0.0 } else { x * rsqrt23(x) }
}

#[allow(
    clippy::should_implement_trait,
    reason = "lane-wise instruction steps named as such, not operators"
)]
impl V4 {
    /// All lanes zero.
    pub const ZERO: V4 = V4([0.0; 4]);

    /// `x y z w`.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> V4 {
        V4([x, y, z, w])
    }

    /// `s` in every lane.
    #[inline]
    pub const fn splat(s: f32) -> V4 {
        V4([s; 4])
    }

    /// Lane `x`.
    #[inline]
    pub fn x(self) -> f32 {
        self.0[0]
    }

    /// Lane `y`.
    #[inline]
    pub fn y(self) -> f32 {
        self.0[1]
    }

    /// Lane `z`.
    #[inline]
    pub fn z(self) -> f32 {
        self.0[2]
    }

    /// Lane `w`.
    #[inline]
    pub fn w(self) -> f32 {
        self.0[3]
    }

    /// `self` with lane `w` set to `w`.
    #[inline]
    pub fn with_w(self, w: f32) -> V4 {
        V4([self.0[0], self.0[1], self.0[2], w])
    }

    /// Lane-wise `f(a, b)`.
    #[inline]
    fn zip(self, o: V4, f: impl Fn(f32, f32) -> f32) -> V4 {
        let [a, b, c, d] = self.0;
        let [e, g, h, i] = o.0;
        V4([f(a, e), f(b, g), f(c, h), f(d, i)])
    }

    /// Lane-wise `self + o`.
    #[inline]
    pub fn add(self, o: V4) -> V4 {
        self.zip(o, |a, b| a + b)
    }

    /// Lane-wise `self - o`.
    #[inline]
    pub fn sub(self, o: V4) -> V4 {
        self.zip(o, |a, b| a - b)
    }

    /// Lane-wise `self * o`.
    #[inline]
    pub fn mul(self, o: V4) -> V4 {
        self.zip(o, |a, b| a * b)
    }

    /// Every lane times `s`.
    #[inline]
    pub fn scale(self, s: f32) -> V4 {
        self.mul(V4::splat(s))
    }

    /// `self + v*s`: the product rounded, then the sum.
    #[inline]
    pub fn add_scaled(self, v: V4, s: f32) -> V4 {
        self.add(v.scale(s))
    }

    /// `self - v*s`: the product rounded, then the difference.
    #[inline]
    pub fn sub_scaled(self, v: V4, s: f32) -> V4 {
        self.sub(v.scale(s))
    }

    /// `self + v*w` lane-wise: the products rounded, then the sums.
    #[inline]
    pub fn add_mul(self, v: V4, w: V4) -> V4 {
        self.add(v.mul(w))
    }

    /// Lane-wise minimum (`minps`).
    #[inline]
    pub fn min(self, o: V4) -> V4 {
        self.zip(o, min_ps)
    }

    /// Lane-wise maximum (`maxps`).
    #[inline]
    pub fn max(self, o: V4) -> V4 {
        self.zip(o, max_ps)
    }

    /// `x y z` negated, `w` kept.
    #[inline]
    pub fn neg3(self) -> V4 {
        let [x, y, z, w] = self.0;
        V4([-x, -y, -z, w])
    }

    /// The dot product of `x y z`: `(x*x' + y*y') + z*z'`, each product rounded.
    #[inline]
    pub fn dot3(self, o: V4) -> f32 {
        let p = self.mul(o).0;
        (p[0] + p[1]) + p[2]
    }

    /// The plane distance `self . (p.xyz, 1)` as the SSE2 build sums it: `(x*px + y*py) + (z*pz + w)`.
    #[inline]
    pub fn dot4_xyz1(self, p: V4) -> f32 {
        let [x, y, z, w] = self.0;
        let [a, b, c, _] = p.0;
        (x * a + y * b) + (z * c + w)
    }

    /// The cross product of `x y z`: lane i is `self_i * o_(i+1) - o_i * self_(i+1)` rotated into place, so
    /// `x = self.y*o.z - o.y*self.z` and so on; `w = self.w*o.w - o.w*self.w`.
    #[inline]
    #[allow(
        clippy::eq_op,
        reason = "the w lane of the instruction sequence: 0, or NaN from an infinite w"
    )]
    pub fn cross(self, o: V4) -> V4 {
        let [ax, ay, az, aw] = self.0;
        let [bx, by, bz, bw] = o.0;
        V4([
            ay * bz - by * az,
            az * bx - bz * ax,
            ax * by - bx * ay,
            aw * bw - bw * aw,
        ])
    }

    /// `self + (b - self)*t`.
    #[inline]
    pub fn lerp(self, b: V4, t: f32) -> V4 {
        self.add_scaled(b.sub(self), t)
    }

    /// The fast normalisation: `l2 = dot3(self, self)`, `inv` the raw reciprocal square root estimate of `l2` (0 where
    /// `l2 <= 0`), all four lanes times `inv`; returns the vector and `l2 * inv` as its length.
    #[inline]
    pub fn normalized_fast_with_length(self) -> (V4, f32) {
        let l2 = self.dot3(self);
        let inv = if l2 <= 0.0 { 0.0 } else { rsqrt_estimate(l2) };
        (self.scale(inv), l2 * inv)
    }

    /// The fast normalisation without the length.
    #[inline]
    pub fn normalized_fast(self) -> V4 {
        self.normalized_fast_with_length().0
    }

    /// The refined normalisation: as [`V4::normalized_fast_with_length`] with the refined reciprocal square root.
    #[inline]
    pub fn normalized_with_length(self) -> (V4, f32) {
        let l2 = self.dot3(self);
        let inv = if l2 <= 0.0 { 0.0 } else { rsqrt23(l2) };
        (self.scale(inv), l2 * inv)
    }

    /// The refined normalisation without the length.
    #[inline]
    pub fn normalized(self) -> V4 {
        self.normalized_with_length().0
    }

    /// The refined length of `x y z`: `sqrt23(dot3(self, self))`.
    #[inline]
    pub fn length3(self) -> f32 {
        sqrt23(self.dot3(self))
    }

    /// Whether every lane is finite.
    pub fn is_finite(self) -> bool {
        self.0.iter().all(|x| x.is_finite())
    }

    /// Whether `x y z` are finite. The `w` lane of a particle position carries what the operations leave in it (the
    /// contact solve adds the difference between a particle's and a shape point's `w`) and no `x y z` result reads it.
    pub fn is_finite3(self) -> bool {
        self.0[..3].iter().all(|x| x.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot3_adds_x_and_y_before_z() {
        // with x*x' = 1, y*y' = 2^-24 and z*z' = 2^-24, (1 + 2^-24) rounds to 1 and 1 + 2^-24 to 1 again;
        // the other order, 1 + (2^-24 + 2^-24) = 1 + 2^-23, is representable
        let tiny = 2f32.powi(-12);
        let a = V4::new(1.0, tiny, tiny, 0.0);
        assert_eq!(a.dot3(a), 1.0);
        let other_order = 1.0 + (tiny * tiny + tiny * tiny);
        assert_ne!(other_order, 1.0);
    }

    #[test]
    fn plane_distance_pairs_the_sums() {
        // (x*px + y*py) + (z*pz + w): 2^24 + 1 rounds to 2^24 (ties to even), 3 - 1 = 2, so the sum is 2^24 + 2;
        // summed left to right, ((2^24 + 1) + 3) - 1 rounds twice to 2^24 + 4
        let plane = V4::new(1.0, 1.0, 1.0, -1.0);
        let p = V4::new(16_777_216.0, 1.0, 3.0, 0.0);
        let left_to_right = ((16_777_216.0f32 + 1.0) + 3.0) - 1.0;
        assert_eq!(plane.dot4_xyz1(p), 16_777_218.0);
        assert_eq!(left_to_right, 16_777_220.0);
    }

    #[test]
    fn cross_matches_the_textbook_product_on_exact_values() {
        let a = V4::new(1.0, 2.0, 3.0, 0.0);
        let b = V4::new(4.0, 5.0, 6.0, 0.0);
        assert_eq!(a.cross(b), V4::new(-3.0, 6.0, -3.0, 0.0));
        assert_eq!(b.cross(a), V4::new(3.0, -6.0, 3.0, 0.0));
    }

    #[test]
    fn min_and_max_return_the_second_operand_on_nan_and_ties() {
        assert_eq!(min_ps(f32::NAN, 0.0), 0.0);
        assert!(min_ps(0.0, f32::NAN).is_nan());
        assert_eq!(max_ps(-0.0, 0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(max_ps(0.0, -0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(min_ps(-1.0, 0.0), -1.0);
    }

    #[test]
    fn estimates_are_within_their_accuracy() {
        for x in [1e-6f32, 0.01, 0.5, 1.0, 2.0, 3.0, 1234.5, 1e6] {
            let exact = 1.0 / f64::from(x).sqrt();
            let e = f64::from(rsqrt_estimate(x));
            assert!(
                ((e - exact) / exact).abs() < 1.0 / 2048.0,
                "rsqrt estimate of {x}"
            );
            let r = f64::from(rsqrt23(x));
            assert!(((r - exact) / exact).abs() < 2e-7, "refined rsqrt of {x}");
            let exact = 1.0 / f64::from(x);
            let e = f64::from(rcp_estimate(x));
            assert!(((e - exact) / exact).abs() < 1.0 / 2048.0, "rcp estimate of {x}");
            let r = f64::from(rcp23(x));
            assert!(((r - exact) / exact).abs() < 2e-7, "refined rcp of {x}");
        }
    }

    #[test]
    fn normalisation_of_zero_is_zero_with_zero_length() {
        assert_eq!(V4::ZERO.normalized_fast_with_length(), (V4::ZERO, 0.0));
        assert_eq!(V4::ZERO.normalized_with_length(), (V4::ZERO, 0.0));
        assert_eq!(sqrt23(0.0), 0.0);
        assert_eq!(rcp23_or_zero(0.0), 0.0);
    }

    #[test]
    fn normalisation_gives_unit_length_to_its_accuracy() {
        let v = V4::new(3.0, -4.0, 12.0, 7.0);
        let (n, l) = v.normalized_with_length();
        assert!((l - 13.0).abs() < 13.0 * 2e-7);
        assert!((n.dot3(n) - 1.0).abs() < 1e-6);
        // the w lane is scaled too
        assert!((n.w() - 7.0 / 13.0).abs() < 1e-6);
        let (n, l) = v.normalized_fast_with_length();
        assert!((l - 13.0).abs() < 13.0 / 2048.0);
        assert!((n.dot3(n) - 1.0).abs() < 1.0 / 512.0);
    }

    #[test]
    fn lerp_is_start_plus_scaled_difference() {
        let a = V4::new(1.0, 2.0, 3.0, 4.0);
        let b = V4::new(3.0, 2.0, -1.0, 0.0);
        assert_eq!(a.lerp(b, 0.5), V4::new(2.0, 2.0, 1.0, 2.0));
        assert_eq!(a.lerp(b, 0.0), a);
        assert_eq!(a.lerp(b, 1.0), b);
    }
}
