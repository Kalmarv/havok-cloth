//! Writes `src/estimates.bin`, the tables [`havok_cloth::simd::rcp_estimate`] and
//! [`havok_cloth::simd::rsqrt_estimate`] read, from this CPU's own `rcpss` and `rsqrtss`. Run it on an AMD processor
//! (the captures' Zen 5, or a Zen 2: their tables agree on every input the captures exercise):
//!
//! ```text
//! cargo run --example estimate_tables
//! ```
//!
//! Each instruction's result for a normal input `1.m` (exponent 127, or 128 for the square root's odd half) depends
//! only on the top 12 bits of `m`, is a 12-bit significand with exponent 126, and moves with the input's exponent
//! exactly; so 4,096 entries per table hold every result. The file is three tables of 4,096 little-endian u16, each
//! entry the result's top 12 significand bits: `rcpss` of `1.m`, `rsqrtss` of `1.m`, `rsqrtss` of `2 * 1.m`.

#[cfg(target_arch = "x86_64")]
#[allow(
    unsafe_code,
    reason = "the estimate instructions themselves; SSE is part of every x86-64 CPU"
)]
fn main() {
    use std::arch::x86_64::{_mm_cvtss_f32, _mm_rcp_ss, _mm_rsqrt_ss, _mm_set_ss};
    // SAFETY: the intrinsics need SSE, which the x86-64 baseline always has; they touch no memory.
    let rcp = |x: f32| unsafe { _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(x))) };
    // SAFETY: as above.
    let rsqrt = |x: f32| unsafe { _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x))) };
    let mut out = Vec::with_capacity(3 * 4096 * 2);
    for (f, exponent) in [
        (&rcp as &dyn Fn(f32) -> f32, 127u32),
        (&rsqrt, 127),
        (&rsqrt, 128),
    ] {
        for top in 0..4096u32 {
            let r = f(f32::from_bits((exponent << 23) | (top << 11))).to_bits();
            assert_eq!(r >> 23, 126, "an estimate of exponent other than 126");
            assert_eq!(r & 0x7ff, 0, "an estimate of more than 12 significand bits");
            out.extend_from_slice(&u16::try_from((r >> 11) & 0xfff).unwrap_or(0).to_le_bytes());
        }
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/estimates.bin");
    if let Err(e) = std::fs::write(&path, out) {
        eprintln!("{}: {e}", path.display());
        std::process::exit(1);
    }
    println!("wrote {}", path.display());
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("the tables come from an x86-64 CPU's own estimate instructions");
    std::process::exit(1);
}
