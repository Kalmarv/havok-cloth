//! Havok cloth simulation as Sekiro and Elden Ring run it, from the cloth data `havok-formats` reads.
//!
//! This is an independent implementation: it contains no Havok SDK code. Its behaviour is checked against cloth
//! motion recorded from the running games, operator by operator and frame by frame.
//!
//! Results vary in the last bits by CPU, as the games' do: the solver uses the processor's approximate reciprocal
//! and reciprocal square root instructions, whose estimates differ between processor makers (see [`simd`]).
#![deny(clippy::disallowed_methods, clippy::disallowed_types)]

pub mod buffer;
pub mod collide;
pub mod constraints;
pub mod deform;
pub mod matrix;
pub mod operators;
pub mod simd;
pub mod simulate;
