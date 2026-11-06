//! The reciprocal and reciprocal square root estimates are AMD's on every machine: a few results pinned, and, on an AMD processor, the estimates equal the CPU's `rcpss` and
//! `rsqrtss` on every significand table entry at every exponent and on the special inputs.
#![allow(clippy::unwrap_used)]

use havok_cloth::simd::{rcp_estimate, rsqrt_estimate};

#[test]
fn the_estimates_are_amds_on_every_machine() {
    // an AMD Zen 2's own rcpss and rsqrtss of these
    for (x, rcp, rsqrt) in [
        (1.0f32, 0x3f7f_f000u32, 0x3f7f_f800u32),
        (3.0, 0x3eaa_a800, 0x3f13_c800),
        (0.7, 0x3fb6_e000, 0x3f99_0000),
        (1234.5, 0x3a54_5800, 0x3ce9_2800),
    ] {
        assert_eq!(rcp_estimate(x).to_bits(), rcp, "rcp {x}");
        assert_eq!(rsqrt_estimate(x).to_bits(), rsqrt, "rsqrt {x}");
    }
    assert_eq!(rcp_estimate(0.0), f32::INFINITY);
    assert_eq!(rcp_estimate(-f32::from_bits(1)), f32::NEG_INFINITY);
    assert_eq!(rcp_estimate(f32::INFINITY).to_bits(), 0);
    assert_eq!(rcp_estimate(f32::MAX).to_bits(), 0);
    assert_eq!(rsqrt_estimate(-0.0), f32::NEG_INFINITY);
    assert_eq!(rsqrt_estimate(-1.0).to_bits(), 0xffc0_0000);
    assert_eq!(rsqrt_estimate(f32::INFINITY).to_bits(), 0);
    assert!(rsqrt_estimate(f32::NAN).is_nan());
}

#[cfg(target_arch = "x86_64")]
#[test]
#[allow(
    unsafe_code,
    reason = "the estimate instructions themselves; SSE is part of every x86-64 CPU"
)]
fn on_an_amd_processor_the_estimates_equal_its_instructions() {
    use std::arch::x86_64::{__cpuid, _mm_cvtss_f32, _mm_rcp_ss, _mm_rsqrt_ss, _mm_set_ss};
    let leaf = __cpuid(0);
    let vendor: Vec<u8> = [leaf.ebx, leaf.edx, leaf.ecx]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    if vendor != b"AuthenticAMD" {
        eprintln!(
            "not an AMD processor ({}): the instructions are not compared",
            String::from_utf8_lossy(&vendor)
        );
        return;
    }
    // SAFETY: the intrinsics need SSE, which the x86-64 baseline always has; they touch no memory.
    let rcp = |x: f32| unsafe { _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(x))) };
    // SAFETY: as above.
    let rsqrt = |x: f32| unsafe { _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x))) };
    let mut inputs: Vec<u32> = vec![
        0,
        1,
        0x7f_ffff,
        0x7f80_0000,
        0x7fc0_0000,
        0x7f80_0001,
        0x7f7f_ffff,
    ];
    for exponent in 1..255u32 {
        for top in 0..4096u32 {
            // the first, a middle and the last significand of the entry
            for low in [0, 0x400, 0x7ff] {
                inputs.push((exponent << 23) | (top << 11) | low);
            }
        }
    }
    let mut differ = Vec::new();
    for b in inputs.iter().flat_map(|&b| [b, b | 0x8000_0000]) {
        let x = f32::from_bits(b);
        if rcp(x).to_bits() != rcp_estimate(x).to_bits() || rsqrt(x).to_bits() != rsqrt_estimate(x).to_bits()
        {
            differ.push(format!("{b:#010x}"));
        }
    }
    assert!(
        differ.is_empty(),
        "{} inputs differ: {:?}",
        differ.len(),
        &differ[..differ.len().min(10)]
    );
}
