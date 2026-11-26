//! Base32768: binary data as UTF-16-friendly text, 15 bits per character.
//!
//! With the `safe-only` feature, the crate is built from safe Rust alone
//! (`forbid(unsafe_code)`): the SIMD and pointer-based fast paths are not
//! compiled, and a simpler implementation provides the same API and results.
#![cfg_attr(feature = "safe-only", forbid(unsafe_code))]

use thiserror::Error;

use base32768_table as table;

#[cfg(not(feature = "safe-only"))]
mod fast;
#[cfg(any(feature = "safe-only", test))]
mod safe;
#[cfg(not(feature = "safe-only"))]
mod simd;

#[cfg(not(feature = "safe-only"))]
pub use fast::{decode, decode_into, encode, encode_into};
#[cfg(feature = "safe-only")]
pub use safe::{decode, decode_into, encode, encode_into};

#[derive(Error, Debug)]
pub enum DecodeError {
    #[error("Unrecognised Base32768 character: {0}")]
    UnrecognizedCharacter(char),

    #[error("Secondary character found before end of input at position {0}")]
    UnexpectedSecondaryCharacter(usize),

    #[error("Padding mismatch")]
    PaddingMismatch,
}

/// `table::DECODE` value for a code point that is not a Base32768 character.
const INVALID: u32 = 0xFFFF;
/// Set in `table::DECODE` values of 7-bit characters (and of `INVALID`).
const FLAG7: u32 = 0x8000;

/// Number of characters `encode` produces for `len` input bytes:
/// `ceil(len * 8 / 15)`, computed without overflowing `len * 8`.
#[inline]
fn encoded_chars(len: usize) -> usize {
    len / 15 * 8 + (len % 15 * 8).div_ceil(15)
}

/// Output capacity `decode` needs for `len` bytes of input:
/// `floor(len / 2 * 15 / 8) + 1`, computed without overflowing. Every character
/// is at least 2 bytes and carries at most 15 bits; the result is below `len`.
#[inline]
fn decoded_capacity(len: usize) -> usize {
    let chars = len / 2;
    chars / 8 * 15 + chars % 8 * 15 / 8 + 1
}
