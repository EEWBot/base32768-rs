use thiserror::Error;

use base32768_table as table;

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

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// Packed UTF-8 (`len << 24 | le_bytes`) of the character for the 15-bit value `z`.
#[inline(always)]
fn z15_utf8(z: u32) -> u32 {
    table::Z15_UTF8[z as usize & 0x7FFF]
}

/// Packed UTF-8 of the character for the 7-bit value `z`.
#[inline(always)]
fn z7_utf8(z: u32) -> u32 {
    table::Z7_UTF8[z as usize & 0x7F]
}

/// Writes a packed character as 4 bytes and returns the new position. Only the
/// first `len` bytes are meaningful; the rest are overwritten by the next
/// character or left in the spare capacity.
///
/// # Safety
/// `dst + pos .. dst + pos + 4` must be writable.
#[inline(always)]
unsafe fn put(dst: *mut u8, pos: usize, packed: u32) -> usize {
    unsafe {
        dst.add(pos)
            .cast::<[u8; 4]>()
            .write_unaligned(packed.to_le_bytes())
    };
    pos + (packed >> 24) as usize
}

pub fn encode(b: &[u8]) -> String {
    let num_chars = (b.len() * 8).div_ceil(15);
    // 3 bytes per character at most, plus 1 for the spare byte `put` writes.
    let mut out: Vec<u8> = Vec::with_capacity(num_chars * 3 + 1);
    let dst = out.as_mut_ptr();
    let mut pos = 0usize;

    // Fast path: 15 bytes -> 8 characters.
    let (chunks, remainder) = b.as_chunks::<15>();
    for chunk in chunks {
        let hi = u64::from_be_bytes(chunk[0..8].try_into().unwrap());
        let lo = u64::from_be_bytes(chunk[7..15].try_into().unwrap());

        let c = [
            z15_utf8((hi >> 49) as u32),
            z15_utf8((hi >> 34) as u32 & 0x7FFF),
            z15_utf8((hi >> 19) as u32 & 0x7FFF),
            z15_utf8((hi >> 4) as u32 & 0x7FFF),
            // `lo` starts at byte 7, so it already holds `hi`'s low 4 bits.
            z15_utf8((lo >> 45) as u32 & 0x7FFF),
            z15_utf8((lo >> 30) as u32 & 0x7FFF),
            z15_utf8((lo >> 15) as u32 & 0x7FFF),
            z15_utf8(lo as u32 & 0x7FFF),
        ];

        // SAFETY: at most 8 * 3 + 1 bytes per chunk; covered by the capacity.
        unsafe {
            // Common case (~95% for random data): eight 3-byte characters, so
            // every store has a fixed offset and none waits on the previous
            // character's length. Lengths are 2 (0b10) or 3 (0b11), so the AND
            // of all of them is 3 only if every one is 3.
            if (c[0] & c[1] & c[2] & c[3] & c[4] & c[5] & c[6] & c[7]) >> 24 == 3 {
                for (i, &packed) in c.iter().enumerate() {
                    put(dst, pos + 3 * i, packed);
                }
                pos += 24;
            } else {
                for &packed in &c {
                    pos = put(dst, pos, packed);
                }
            }
        }
    }

    // Remaining 0..=14 bytes.
    let mut acc = 0u64;
    let mut bits = 0u32;
    for &byte in remainder {
        acc = (acc << 8) | byte as u64;
        bits += 8;
        if bits >= 15 {
            bits -= 15;
            // SAFETY: still within the `num_chars` characters reserved.
            pos = unsafe { put(dst, pos, z15_utf8((acc >> bits) as u32 & 0x7FFF)) };
            acc &= (1 << bits) - 1;
        }
    }

    // Pad the final partial character with 1 bits.
    if bits > 0 {
        let packed = if bits <= 7 {
            let pad = 7 - bits;
            z7_utf8(((acc << pad) | ((1 << pad) - 1)) as u32)
        } else {
            let pad = 15 - bits;
            z15_utf8(((acc << pad) | ((1 << pad) - 1)) as u32)
        };
        // SAFETY: this is the last of the `num_chars` characters reserved.
        pos = unsafe { put(dst, pos, packed) };
    }

    debug_assert!(pos < out.capacity());
    // SAFETY: `pos` bytes were initialised by `put`; every character is a
    // well-formed 2 or 3 byte UTF-8 sequence taken from the tables.
    unsafe {
        out.set_len(pos);
        String::from_utf8_unchecked(out)
    }
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// Looks up a code point decoded from a 2 or 3 byte sequence (`cp <= 0xFFFF`).
/// Returns `INVALID`, `FLAG7 | z` or `z`.
#[inline(always)]
fn lookup(cp: u32) -> u32 {
    // The table spans all of U+0000..=U+FFFF, so the mask removes the bounds check.
    table::DECODE[cp as usize & 0xFFFF] as u32
}

/// Decodes a character known to be a 3 byte UTF-8 sequence.
///
/// # Safety
/// `p .. p + 3` must be readable.
#[inline(always)]
unsafe fn read_3byte(p: *const u8) -> u32 {
    let (b0, b1, b2) = unsafe { (*p as u32, *p.add(1) as u32, *p.add(2) as u32) };
    lookup(((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F))
}

/// Decodes one character at `p` assuming it is a 2 or 3 byte UTF-8 sequence.
/// Returns `(value, byte length)`; `value` has `FLAG7` set for anything that is
/// not a 15-bit character (including invalid lead bytes).
///
/// # Safety
/// `p .. p + 3` must be readable.
#[inline(always)]
unsafe fn read_z15(p: *const u8) -> (u32, usize) {
    let (b0, b1, b2) = unsafe { (*p as u32, *p.add(1) as u32, *p.add(2) as u32) };
    let three = b0 >= 0xE0;
    let cp = if three {
        ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F)
    } else {
        ((b0 & 0x1F) << 6) | (b1 & 0x3F)
    };
    // Lead bytes outside 0xC0..=0xEF are ASCII or 4-byte sequences.
    let bad_lead = b0.wrapping_sub(0xC0) >= 0x30;
    let v = lookup(cp) | if bad_lead { FLAG7 } else { 0 };
    (v, 2 + three as usize)
}

pub fn decode(s: &str) -> Result<Vec<u8>, DecodeError> {
    decode_impl::<true>(s)
}

/// `USE_SIMD = false` forces the scalar fast path (used by tests to cover it
/// on machines where the SIMD path would be taken).
#[inline(always)]
fn decode_impl<const USE_SIMD: bool>(s: &str) -> Result<Vec<u8>, DecodeError> {
    let bytes = s.as_bytes();
    let len = bytes.len();
    // Every character is at least 2 bytes and carries at most 15 bits.
    let mut out: Vec<u8> = Vec::with_capacity(len / 2 * 15 / 8 + 1);

    let mut pos = 0usize;
    let mut char_index = 0usize;

    // Fast path: 8 characters (120 bits) -> 15 bytes. Bails out to the exact
    // path below on anything other than eight 15-bit characters.
    if len >= 24 {
        let dst = out.as_mut_ptr();
        // SAFETY: `out` has the capacity `decode_fast` relies on.
        let (p, written, chars) = unsafe { decode_fast_dispatch::<USE_SIMD>(bytes, dst) };
        // SAFETY: `written` bytes were initialised by `decode_fast`.
        unsafe { out.set_len(written) };
        pos = p;
        char_index = chars;
    }

    // Exact path: the remaining characters, the 7-bit final character and all
    // error reporting.
    let mut acc = 0u64;
    let mut bits = 0u32;

    while pos < len {
        let b0 = bytes[pos] as u32;
        let (v, width) = match b0 {
            0xC0..=0xDF => (
                lookup(((b0 & 0x1F) << 6) | (bytes[pos + 1] as u32 & 0x3F)),
                2,
            ),
            0xE0..=0xEF => (
                lookup(
                    ((b0 & 0x0F) << 12)
                        | ((bytes[pos + 1] as u32 & 0x3F) << 6)
                        | (bytes[pos + 2] as u32 & 0x3F),
                ),
                3,
            ),
            _ => (INVALID, 0),
        };

        if v == INVALID {
            let chr = s[pos..].chars().next().unwrap();
            return Err(DecodeError::UnrecognizedCharacter(chr));
        }

        let num_bits = if v & FLAG7 != 0 { 7 } else { 15 };
        pos += width;
        if num_bits == 7 && pos != len {
            return Err(DecodeError::UnexpectedSecondaryCharacter(char_index));
        }

        acc = (acc << num_bits) | (v & 0x7FFF) as u64;
        bits += num_bits;
        while bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
        acc &= (1 << bits) - 1;
        char_index += 1;
    }

    if acc != (1 << bits) - 1 {
        return Err(DecodeError::PaddingMismatch);
    }

    Ok(out)
}

/// Reads 8 characters of any width (2 or 3 bytes) one after another.
///
/// # Safety
/// `p .. p + 24` must be readable.
#[inline(always)]
unsafe fn read_8_chained(p: *const u8) -> ([u32; 8], usize) {
    let mut v = [0u32; 8];
    let mut o = 0usize;
    for slot in &mut v {
        let (value, n) = unsafe { read_z15(p.add(o)) };
        *slot = value;
        o += n;
    }
    (v, o)
}

/// Reads 8 characters, taking the fixed-offset route when all are 3 bytes.
///
/// # Safety
/// `p .. p + 24` must be readable; with `SIMD`, the CPU must support the
/// instructions `simd::code_points_8x3` uses.
#[inline(always)]
unsafe fn read_8<const SIMD: bool>(p: *const u8) -> ([u32; 8], usize) {
    unsafe {
        if SIMD {
            if let Some(cps) = simd::code_points_8x3(p) {
                return (cps.map(|cp| lookup(cp as u32)), 24);
            }
        } else {
            // Common case (~95% of blocks for random data, since 99.3% of
            // 15-bit characters are 3 bytes): eight 3-byte characters at
            // fixed offsets, so the loads and lookups do not depend on each
            // other.
            let l: [u8; 8] = core::array::from_fn(|i| *p.add(3 * i));
            let and = l.iter().fold(0xFF, |a, &b| a & b);
            let or = l.iter().fold(0, |a, &b| a | b);
            if and >= 0xE0 && or <= 0xEF {
                return (core::array::from_fn(|i| read_3byte(p.add(3 * i))), 24);
            }
        }
        read_8_chained(p)
    }
}

/// Decodes whole 8-character blocks from the start of `bytes` into `dst` until
/// fewer than 24 bytes remain or a block is not eight 15-bit characters.
/// Returns `(bytes consumed, bytes written, characters consumed)`.
///
/// # Safety
/// `dst` must have room for `bytes.len() / 2 * 15 / 8` bytes; with `SIMD`, see
/// [`read_8`].
#[inline(always)]
unsafe fn decode_fast<const SIMD: bool>(bytes: &[u8], dst: *mut u8) -> (usize, usize, usize) {
    let src = bytes.as_ptr();
    let len = bytes.len();
    let mut pos = 0usize;
    let mut written = 0usize;

    while pos + 24 <= len {
        // SAFETY: `pos + 24 <= len`.
        let (v, n) = unsafe { read_8::<SIMD>(src.add(pos)) };
        if v.iter().fold(0, |a, &b| a | b) & FLAG7 != 0 {
            break;
        }

        let [v0, v1, v2, v3, v4, v5, v6, v7] = v.map(u64::from);
        let w0 = (v0 << 49) | (v1 << 34) | (v2 << 19) | (v3 << 4) | (v4 >> 11);
        // Starts at output byte 7, like `lo` in `encode`.
        let w1 = (v3 << 60) | (v4 << 45) | (v5 << 30) | (v6 << 15) | v7;

        // SAFETY: 16 input bytes or more were consumed for these 15 output
        // bytes, so `written + 15 <= len / 2 * 15 / 8`.
        unsafe {
            let d = dst.add(written);
            d.cast::<[u8; 8]>().write_unaligned(w0.to_be_bytes());
            d.add(7).cast::<[u8; 8]>().write_unaligned(w1.to_be_bytes());
        }

        written += 15;
        pos += n;
    }

    (pos, written, written / 15 * 8)
}

/// Runs [`decode_fast`] with the best instruction set available.
///
/// # Safety
/// See [`decode_fast`].
#[inline(always)]
unsafe fn decode_fast_dispatch<const USE_SIMD: bool>(
    bytes: &[u8],
    dst: *mut u8,
) -> (usize, usize, usize) {
    #[cfg(target_arch = "x86_64")]
    if USE_SIMD && std::is_x86_feature_detected!("ssse3") {
        #[target_feature(enable = "ssse3")]
        unsafe fn decode_fast_ssse3(bytes: &[u8], dst: *mut u8) -> (usize, usize, usize) {
            unsafe { decode_fast::<true>(bytes, dst) }
        }
        // SAFETY: SSSE3 support was just checked.
        return unsafe { decode_fast_ssse3(bytes, dst) };
    }

    #[cfg(target_arch = "aarch64")]
    if USE_SIMD {
        // SAFETY: NEON is part of the aarch64 baseline.
        return unsafe { decode_fast::<true>(bytes, dst) };
    }

    unsafe { decode_fast::<false>(bytes, dst) }
}

/// Code points of eight 3-byte UTF-8 characters at `p .. p + 24`, or `None` if
/// any of the eight lead bytes is not a 3-byte lead (0xE0..=0xEF).
mod simd {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "ssse3")]
    #[inline]
    pub unsafe fn code_points_8x3(p: *const u8) -> Option<[u16; 8]> {
        use core::arch::x86_64::*;

        // Gathers byte `k` of each character into the low byte of a u16 lane:
        // characters 0..=4 come from `lo` (bytes 0..16), 5..=7 from `hi` (bytes 8..24).
        #[target_feature(enable = "ssse3")]
        #[inline]
        fn gather<const K: i8>(lo: __m128i, hi: __m128i) -> __m128i {
            let from_lo = _mm_setr_epi8(
                K,
                -1,
                K + 3,
                -1,
                K + 6,
                -1,
                K + 9,
                -1,
                K + 12,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            );
            let from_hi = _mm_setr_epi8(
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                K + 7,
                -1,
                K + 10,
                -1,
                K + 13,
                -1,
            );
            _mm_or_si128(_mm_shuffle_epi8(lo, from_lo), _mm_shuffle_epi8(hi, from_hi))
        }

        // SAFETY: the caller guarantees `p .. p + 24` is readable.
        let (lo, hi) = unsafe { (_mm_loadu_si128(p.cast()), _mm_loadu_si128(p.add(8).cast())) };

        let b0 = gather::<0>(lo, hi);
        let lead = _mm_and_si128(b0, _mm_set1_epi16(0xF0));
        let is_3byte = _mm_cmpeq_epi16(lead, _mm_set1_epi16(0xE0));
        if _mm_movemask_epi8(is_3byte) != 0xFFFF {
            return None;
        }
        let b1 = gather::<1>(lo, hi);
        let b2 = gather::<2>(lo, hi);

        let cp = _mm_or_si128(
            _mm_or_si128(
                _mm_slli_epi16::<12>(_mm_and_si128(b0, _mm_set1_epi16(0x0F))),
                _mm_slli_epi16::<6>(_mm_and_si128(b1, _mm_set1_epi16(0x3F))),
            ),
            _mm_and_si128(b2, _mm_set1_epi16(0x3F)),
        );
        let mut out = [0u16; 8];
        // SAFETY: `out` is 16 bytes.
        unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), cp) };
        Some(out)
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    #[inline]
    pub unsafe fn code_points_8x3(p: *const u8) -> Option<[u16; 8]> {
        use core::arch::aarch64::*;

        /// `tbl` indices gathering byte `K` of each character into the low byte
        /// of a u16 lane. The table is bytes 0..16 followed by bytes 8..24, so
        /// offsets 16..24 are read from the second half; 0xFF yields zero.
        const fn indices<const K: usize>() -> [u8; 16] {
            let mut idx = [0xFF; 16];
            let mut i = 0;
            while i < 8 {
                let off = 3 * i + K;
                idx[2 * i] = if off < 16 { off as u8 } else { (off + 8) as u8 };
                i += 1;
            }
            idx
        }
        const I0: [u8; 16] = indices::<0>();
        const I1: [u8; 16] = indices::<1>();
        const I2: [u8; 16] = indices::<2>();

        // SAFETY: the caller guarantees `p .. p + 24` is readable.
        let table = unsafe { uint8x16x2_t(vld1q_u8(p), vld1q_u8(p.add(8))) };
        // SAFETY: the index arrays are 16 bytes.
        let gather = |idx: &[u8; 16]| unsafe {
            vreinterpretq_u16_u8(vqtbl2q_u8(table, vld1q_u8(idx.as_ptr())))
        };

        let b0 = gather(&I0);
        let is_3byte = vceqq_u16(vandq_u16(b0, vdupq_n_u16(0xF0)), vdupq_n_u16(0xE0));
        if vminvq_u16(is_3byte) == 0 {
            return None;
        }
        let b1 = gather(&I1);
        let b2 = gather(&I2);

        let cp = vorrq_u16(
            vorrq_u16(
                vshlq_n_u16::<12>(vandq_u16(b0, vdupq_n_u16(0x0F))),
                vshlq_n_u16::<6>(vandq_u16(b1, vdupq_n_u16(0x3F))),
            ),
            vandq_u16(b2, vdupq_n_u16(0x3F)),
        );

        let mut out = [0u16; 8];
        // SAFETY: `out` is 16 bytes.
        unsafe { vst1q_u16(out.as_mut_ptr(), cp) };
        Some(out)
    }

    /// Never called: `SIMD` is only enabled on the architectures above.
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[inline(always)]
    pub unsafe fn code_points_8x3(_p: *const u8) -> Option<[u16; 8]> {
        None
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_decode() {
        let encoded_s = "媒腻㐤┖ꈳ埳";
        assert_eq!(
            crate::decode(encoded_s).unwrap(),
            [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
        );
    }

    #[test]
    fn test_encode() {
        let bin = [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100];
        assert_eq!(crate::encode(&bin), "媒腻㐤┖ꈳ埳");
    }

    #[test]
    fn roundtrip_all_lengths() {
        let data: Vec<u8> = (0..400u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
            .collect();
        for n in 0..data.len() {
            let encoded = crate::encode(&data[..n]);
            assert_eq!(crate::decode(&encoded).unwrap(), &data[..n], "length {n}");
        }
    }

    #[test]
    fn fast_path_mixed_widths() {
        // Zero bytes encode to 2-byte characters (values 0..224 are in the
        // U+04A0..U+07DF blocks), so runs of zeros put 2-byte characters at
        // varying places inside 8-character blocks.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for round in 0..200 {
            let data: Vec<u8> = (0..300 + round)
                .map(|i| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    if (i / 7 + round) % 5 == 0 {
                        0
                    } else {
                        (state >> 56) as u8
                    }
                })
                .collect();
            let encoded = crate::encode(&data);
            assert_eq!(crate::decode(&encoded).unwrap(), data, "round {round}");
            assert_eq!(
                crate::decode_impl::<false>(&encoded).unwrap(),
                data,
                "round {round}"
            );
        }
    }

    #[test]
    fn bad_inputs() {
        use crate::DecodeError::*;
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/test-resources/bad");
        for name in ["bad-padding.txt", "bad0.txt", "not-base32768-char.txt"] {
            let s = std::fs::read_to_string(format!("{dir}/{name}")).unwrap();
            assert!(crate::decode(&s).is_err(), "{name}");
        }

        // Errors inside what would otherwise be a fast-path block.
        let valid = crate::encode(&[0x55; 60]); // 32 chars
        let chars: Vec<char> = valid.chars().collect();
        for i in 0..chars.len() {
            let mut c = chars.clone();
            c[i] = 'A';
            let s: String = c.iter().collect();
            assert!(
                matches!(crate::decode(&s), Err(UnrecognizedCharacter('A'))),
                "{i}"
            );

            let mut c = chars.clone();
            c[i] = '😀';
            let s: String = c.iter().collect();
            assert!(
                matches!(crate::decode(&s), Err(UnrecognizedCharacter('😀'))),
                "{i}"
            );

            if i + 1 < chars.len() {
                let mut c = chars.clone();
                c[i] = 'ƀ'; // 7-bit character
                let s: String = c.iter().collect();
                assert!(
                    matches!(crate::decode(&s), Err(UnexpectedSecondaryCharacter(p)) if p == i),
                    "{i}"
                );
            }
        }
    }
}
