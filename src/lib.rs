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

/// Returns `x` unchanged. When AVX-512 is enabled at compile time (e.g.
/// `-C target-cpu=native` on Zen 4/5 or Sapphire Rapids), LLVM turns the eight
/// table lookups in `encode` into a `vpgatherqd` plus `vpextrd` stores, which
/// is 20-57% slower than plain loads. An empty `asm!` makes each value opaque,
/// so it stays in a general-purpose register and the gather is never formed.
/// Other targets compile exactly as without it.
#[inline(always)]
fn scalar_barrier(#[allow(unused_mut)] mut x: u32) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    // SAFETY: an empty asm block with one register operand has no effect.
    unsafe {
        core::arch::asm!("/* {0:e} */", inout(reg) x, options(pure, nomem, nostack, preserves_flags));
    }
    x
}

/// Number of characters `encode` produces for `len` input bytes:
/// `ceil(len * 8 / 15)`, computed without overflowing `len * 8`.
#[inline]
fn encoded_chars(len: usize) -> usize {
    len / 15 * 8 + (len % 15 * 8).div_ceil(15)
}

/// Output capacity `encode` needs: 3 bytes per character at most, plus 1 for
/// the spare byte `put` writes.
///
/// # Panics
/// If the size does not fit in `usize` (only possible on 32-bit or smaller
/// targets). Wrapping here would under-allocate the buffer `encode` writes to
/// through raw pointers.
#[inline]
fn encoded_capacity(len: usize) -> usize {
    encoded_chars(len)
        .checked_mul(3)
        .and_then(|n| n.checked_add(1))
        .expect("base32768: encoded output size overflows usize")
}

/// Output capacity `decode` needs for `len` bytes of input:
/// `floor(len / 2 * 15 / 8) + 1`, computed without overflowing. Every character
/// is at least 2 bytes and carries at most 15 bits; the result is below `len`.
#[inline]
fn decoded_capacity(len: usize) -> usize {
    let chars = len / 2;
    chars / 8 * 15 + chars % 8 * 15 / 8 + 1
}

pub fn encode(b: &[u8]) -> String {
    let mut out: Vec<u8> = Vec::with_capacity(encoded_capacity(b.len()));
    let dst = out.as_mut_ptr();
    let mut pos = 0usize;

    // Fast path: 15 bytes -> 8 characters.
    let (chunks, remainder) = b.as_chunks::<15>();
    for chunk in chunks {
        let hi = u64::from_be_bytes(chunk[0..8].try_into().unwrap());
        let lo = u64::from_be_bytes(chunk[7..15].try_into().unwrap());

        let c = [
            scalar_barrier(z15_utf8((hi >> 49) as u32)),
            scalar_barrier(z15_utf8((hi >> 34) as u32 & 0x7FFF)),
            scalar_barrier(z15_utf8((hi >> 19) as u32 & 0x7FFF)),
            scalar_barrier(z15_utf8((hi >> 4) as u32 & 0x7FFF)),
            // `lo` starts at byte 7, so it already holds `hi`'s low 4 bits.
            scalar_barrier(z15_utf8((lo >> 45) as u32 & 0x7FFF)),
            scalar_barrier(z15_utf8((lo >> 30) as u32 & 0x7FFF)),
            scalar_barrier(z15_utf8((lo >> 15) as u32 & 0x7FFF)),
            scalar_barrier(z15_utf8(lo as u32 & 0x7FFF)),
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
            // SAFETY: still within the `encoded_chars` characters reserved.
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
        // SAFETY: this is the last of the `encoded_chars` characters reserved.
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
    decode_impl::<LEVEL_AVX512>(s)
}

// The widest fast path `decode_impl` may use. Each level falls back to the one
// below when the CPU lacks the instructions; tests run every level so that
// each path is covered on machines where a wider one would be taken.
/// Scalar code only.
#[cfg_attr(not(test), allow(dead_code))]
const LEVEL_SCALAR: u8 = 0;
/// SSSE3 on x86-64, NEON on little-endian aarch64.
#[cfg_attr(
    not(any(
        test,
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_endian = "little")
    )),
    allow(dead_code)
)]
const LEVEL_SIMD: u8 = 1;
/// AVX-512 (BW, VBMI, BITALG) on x86-64.
const LEVEL_AVX512: u8 = 2;

#[inline(always)]
fn decode_impl<const LEVEL: u8>(s: &str) -> Result<Vec<u8>, DecodeError> {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut out: Vec<u8> = Vec::with_capacity(decoded_capacity(len));

    let mut pos = 0usize;
    let mut char_index = 0usize;

    // Fast path: 8 characters (120 bits) -> 15 bytes. Bails out to the exact
    // path below on anything other than eight 15-bit characters.
    if len >= 24 {
        let dst = out.as_mut_ptr();
        // SAFETY: `out` has the capacity `decode_fast` relies on.
        let (p, written, chars) = unsafe { decode_fast_dispatch::<LEVEL>(bytes, dst) };
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
/// `dst` must have room for `decoded_capacity(bytes.len()) - 1` bytes; with `SIMD`, see
/// [`read_8`]; with `AVX512`, the CPU must support AVX-512 BW, VBMI and BITALG.
#[inline(always)]
unsafe fn decode_fast<const SIMD: bool, const AVX512: bool>(
    bytes: &[u8],
    dst: *mut u8,
) -> (usize, usize, usize) {
    let src = bytes.as_ptr();
    let len = bytes.len();
    let mut pos = 0usize;
    let mut written = 0usize;
    // After the AVX-512 path rejects a window, the 8-character path takes the
    // next few blocks (past the character it stopped at) before it is retried.
    #[cfg(target_arch = "x86_64")]
    let mut skip = 0u32;

    while pos + 24 <= len {
        #[cfg(target_arch = "x86_64")]
        if AVX512 {
            if skip == 0 && pos + 96 <= len {
                // SAFETY: `pos + 96 <= len`. Those 96 input bytes account for
                // 48 * 15 / 8 = 90 bytes of `dst` capacity (2 bytes per
                // character), more than the 60 bytes written.
                if unsafe { simd::decode32_avx512(src.add(pos), dst.add(written)) } {
                    pos += 96;
                    written += 60;
                    continue;
                }
                skip = 4;
            }
            skip = skip.saturating_sub(1);
        }
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
        // bytes, so `written + 15 < decoded_capacity(len)`.
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
unsafe fn decode_fast_dispatch<const LEVEL: u8>(
    bytes: &[u8],
    dst: *mut u8,
) -> (usize, usize, usize) {
    #[cfg(target_arch = "x86_64")]
    if LEVEL >= LEVEL_AVX512
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vbmi")
        && std::is_x86_feature_detected!("avx512bitalg")
    {
        #[target_feature(enable = "ssse3,avx512f,avx512bw,avx512vbmi,avx512bitalg")]
        unsafe fn decode_fast_avx512(bytes: &[u8], dst: *mut u8) -> (usize, usize, usize) {
            unsafe { decode_fast::<true, true>(bytes, dst) }
        }
        // SAFETY: the AVX-512 features were just checked (they imply SSSE3).
        return unsafe { decode_fast_avx512(bytes, dst) };
    }

    #[cfg(target_arch = "x86_64")]
    if LEVEL >= LEVEL_SIMD && std::is_x86_feature_detected!("ssse3") {
        #[target_feature(enable = "ssse3")]
        unsafe fn decode_fast_ssse3(bytes: &[u8], dst: *mut u8) -> (usize, usize, usize) {
            unsafe { decode_fast::<true, false>(bytes, dst) }
        }
        // SAFETY: SSSE3 support was just checked.
        return unsafe { decode_fast_ssse3(bytes, dst) };
    }

    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if LEVEL >= LEVEL_SIMD {
        // SAFETY: NEON is part of the aarch64 baseline.
        return unsafe { decode_fast::<true, false>(bytes, dst) };
    }

    unsafe { decode_fast::<false, false>(bytes, dst) }
}

/// Code points of eight 3-byte UTF-8 characters at `p .. p + 24`, or `None` if
/// any of the eight lead bytes is not a 3-byte lead (0xE0..=0xEF).
mod simd {
    /// Decodes 32 three-byte 15-bit characters at `p .. p + 96` into 60 bytes
    /// at `d`. Returns false, having written nothing, if any character is not
    /// a 3-byte 15-bit character.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512bitalg")]
    #[inline]
    pub unsafe fn decode32_avx512(p: *const u8, d: *mut u8) -> bool {
        use crate::table;
        use core::arch::x86_64::*;

        /// vpermi2b indices putting byte `K` of character i in the low byte of u16 lane i.
        const fn char_bytes<const K: usize>() -> [u8; 64] {
            let mut m = [0u8; 64];
            let mut i = 0;
            while i < 32 {
                m[2 * i] = (3 * i + K) as u8;
                i += 1;
            }
            m
        }
        /// vpermb indices: output byte j = byte (3 - t % 4) of dword t / 4 in
        /// 128-bit lane g, where g = j / 15, t = j % 15 (big-endian dwords,
        /// skipping the unused 16th byte of each lane).
        const fn pack_order() -> [u8; 64] {
            let mut m = [0u8; 64];
            let mut j = 0;
            while j < 60 {
                let (g, t) = (j / 15, j % 15);
                m[j] = (16 * g + 4 * (t / 4) + 3 - t % 4) as u8;
                j += 1;
            }
            m
        }
        const fn shifts(v: [u32; 4]) -> [u32; 16] {
            let mut m = [0u32; 16];
            let mut i = 0;
            while i < 16 {
                m[i] = v[i % 4];
                i += 1;
            }
            m
        }
        static I0: [u8; 64] = char_bytes::<0>();
        static I1: [u8; 64] = char_bytes::<1>();
        static I2: [u8; 64] = char_bytes::<2>();
        static PACK: [u8; 64] = pack_order();
        static SHL: [u32; 16] = shifts([2, 4, 6, 8]);
        static SHR: [u32; 16] = shifts([28, 26, 24, 32]);
        static DESC: [u16; 64] = table::PAGE_DESC;
        static HMASK: [u16; 32] = table::HALF_MASK;
        static HPREFIX: [u16; 32] = table::HALF_PREFIX;
        // SAFETY (every use): `p` points to a 64-byte array or into the
        // readable input.
        let ld = |p: *const u8| unsafe { _mm512_loadu_si512(p.cast()) };
        let w = |x: i16| _mm512_set1_epi16(x);

        // SAFETY: the caller guarantees `p .. p + 96` is readable; the masked
        // load does not touch bytes 96..128.
        let (a, b) = unsafe {
            (
                ld(p),
                _mm512_maskz_loadu_epi8(0xFFFF_FFFF, p.add(64).cast()),
            )
        };
        const EVEN: u64 = 0x5555_5555_5555_5555;
        let b0 = _mm512_maskz_permutex2var_epi8(EVEN, a, ld(I0.as_ptr()), b);
        if _mm512_cmpeq_epi16_mask(_mm512_and_si512(b0, w(0xF0)), w(0xE0)) != u32::MAX {
            return false;
        }
        let b1 = _mm512_maskz_permutex2var_epi8(EVEN, a, ld(I1.as_ptr()), b);
        let b2 = _mm512_maskz_permutex2var_epi8(EVEN, a, ld(I2.as_ptr()), b);
        let cp = _mm512_or_si512(
            _mm512_or_si512(
                _mm512_slli_epi16::<12>(b0),
                _mm512_slli_epi16::<6>(_mm512_and_si512(b1, w(0x3F))),
            ),
            _mm512_and_si512(b2, w(0x3F)),
        );

        // Page + rank: z = 32 * (blocks before the page + blocks before the
        // half + 15-bit blocks below this one in the half) + cp % 32.
        let desc = _mm512_permutex2var_epi16(
            ld(DESC.as_ptr().cast()),
            _mm512_srli_epi16::<10>(cp),
            ld(DESC[32..].as_ptr().cast()),
        );
        // vpermw uses only the low 5 index bits: pattern << 1 | half.
        let half = _mm512_ternarylogic_epi32::<0xF8>(desc, _mm512_srli_epi16::<9>(cp), w(1));
        let mask = _mm512_permutexvar_epi16(half, ld(HMASK.as_ptr().cast()));
        let hprefix = _mm512_permutexvar_epi16(half, ld(HPREFIX.as_ptr().cast()));
        let bit = _mm512_sllv_epi16(w(1), _mm512_and_si512(_mm512_srli_epi16::<5>(cp), w(15)));
        if _mm512_test_epi16_mask(mask, bit) != u32::MAX {
            return false;
        }
        let rank = _mm512_popcnt_epi16(_mm512_and_si512(mask, _mm512_sub_epi16(bit, w(1))));
        let blocks = _mm512_add_epi16(
            _mm512_add_epi16(_mm512_srli_epi16::<5>(desc), hprefix),
            rank,
        );
        let z = _mm512_or_si512(_mm512_slli_epi16::<5>(blocks), _mm512_and_si512(cp, w(31)));

        // Pack: dword i = z[2i] << 15 | z[2i+1] (madd by -32768, -1, negated),
        // then per 128-bit lane w_i = p_i << (2i+2) | p_{i+1} >> (28-2i).
        let m = _mm512_madd_epi16(z, _mm512_set1_epi32(((-1i32) << 16) | 0x8000));
        let pr = _mm512_sub_epi32(_mm512_setzero_si512(), m);
        let words = _mm512_or_si512(
            _mm512_sllv_epi32(pr, ld(SHL.as_ptr().cast())),
            _mm512_srlv_epi32(_mm512_bsrli_epi128::<4>(pr), ld(SHR.as_ptr().cast())),
        );
        let out = _mm512_permutexvar_epi8(ld(PACK.as_ptr()), words);
        #[cfg(test)]
        AVX512_BLOCKS.with(|n| n.set(n.get() + 1));
        // SAFETY: the caller guarantees 60 writable bytes at `d`.
        unsafe { _mm512_mask_storeu_epi8(d.cast(), (1 << 60) - 1, out) };
        true
    }

    #[cfg(all(test, target_arch = "x86_64"))]
    thread_local! {
        /// Number of 32-character blocks the AVX-512 path decoded on this
        /// thread. Thread-local because tests run in parallel threads: a test
        /// sees only its own calls.
        pub static AVX512_BLOCKS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
    }

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

    // `vzip` of the low and high bytes builds u16 lanes, which assumes
    // little-endian lanes.
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[target_feature(enable = "neon")]
    #[inline]
    pub unsafe fn code_points_8x3(p: *const u8) -> Option<[u16; 8]> {
        use core::arch::aarch64::*;

        // One `ld3` reads the 24 bytes without overlap and de-interleaves them:
        // `b0`, `b1`, `b2` hold byte 0, 1, 2 of each character. Written as
        // `asm!` because LLVM splits the `vld3_u8` intrinsic into plain loads
        // plus per-lane moves here (it merges them with the loads of the
        // fallback path), which is far slower than `ld3`.
        let (b0, b1, b2): (uint8x8_t, uint8x8_t, uint8x8_t);
        // SAFETY: the caller guarantees `p .. p + 24` is readable, and `ld3`
        // with three 8-byte registers reads exactly those 24 bytes.
        unsafe {
            core::arch::asm!(
                "ld3 {{v0.8b, v1.8b, v2.8b}}, [{p}]",
                p = in(reg) p,
                out("v0") b0,
                out("v1") b1,
                out("v2") b2,
                options(pure, readonly, nostack, preserves_flags),
            );
        }

        let is_3byte = vceq_u8(vand_u8(b0, vdup_n_u8(0xF0)), vdup_n_u8(0xE0));
        if vminv_u8(is_3byte) == 0 {
            return None;
        }

        // 1110aaaa 10bbbbbb 10cccccc -> high byte aaaabbbb, low byte bbcccccc.
        let hi = vsli_n_u8::<4>(vshr_n_u8::<2>(b1), b0);
        let lo = vsli_n_u8::<6>(b2, b1);
        let cp = vcombine_u8(vzip1_u8(lo, hi), vzip2_u8(lo, hi));

        let mut out = [0u16; 8];
        // SAFETY: `out` is 16 bytes.
        unsafe { vst1q_u8(out.as_mut_ptr().cast(), cp) };
        Some(out)
    }

    /// Never called: `SIMD` is only enabled on the architectures above.
    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_endian = "little")
    )))]
    #[inline(always)]
    pub unsafe fn code_points_8x3(_p: *const u8) -> Option<[u16; 8]> {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::{DecodeError, LEVEL_AVX512, LEVEL_SCALAR, LEVEL_SIMD, decode_impl};

    /// Decodes with every fast path level, checking that they all agree.
    fn decode_every_level(s: &str) -> Result<Vec<u8>, DecodeError> {
        let scalar = decode_impl::<LEVEL_SCALAR>(s);
        for (level, r) in [
            ("SIMD", decode_impl::<LEVEL_SIMD>(s)),
            ("AVX-512", decode_impl::<LEVEL_AVX512>(s)),
        ] {
            assert_eq!(format!("{r:?}"), format!("{scalar:?}"), "{level}");
        }
        scalar
    }

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
            assert_eq!(
                decode_every_level(&encoded).unwrap(),
                &data[..n],
                "length {n}"
            );
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
            assert_eq!(decode_every_level(&encoded).unwrap(), data, "round {round}");
        }
    }

    #[test]
    fn bad_inputs() {
        use crate::DecodeError::*;
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/test-resources/bad");
        for name in ["bad-padding.txt", "bad0.txt", "not-base32768-char.txt"] {
            let s = std::fs::read_to_string(format!("{dir}/{name}")).unwrap();
            assert!(decode_every_level(&s).is_err(), "{name}");
        }

        // Errors inside what would otherwise be a fast-path block.
        let valid = crate::encode(&[0x55; 60]); // 32 chars
        let chars: Vec<char> = valid.chars().collect();
        for i in 0..chars.len() {
            let mut c = chars.clone();
            c[i] = 'A';
            let s: String = c.iter().collect();
            assert!(
                matches!(decode_every_level(&s), Err(UnrecognizedCharacter('A'))),
                "{i}"
            );

            let mut c = chars.clone();
            c[i] = '😀';
            let s: String = c.iter().collect();
            assert!(
                matches!(decode_every_level(&s), Err(UnrecognizedCharacter('😀'))),
                "{i}"
            );

            if i + 1 < chars.len() {
                let mut c = chars.clone();
                c[i] = 'ƀ'; // 7-bit character
                let s: String = c.iter().collect();
                assert!(
                    matches!(decode_every_level(&s), Err(UnexpectedSecondaryCharacter(p)) if p == i),
                    "{i}"
                );
            }
        }
    }

    /// Every 15-bit value (2- and 3-byte characters) with the input bit
    /// stream shifted by 0..15 leading bytes, which changes where each value's
    /// bits fall within the encoded 15-bit groups. (Memory addresses and SIMD
    /// lanes are covered by the tests below.)
    #[test]
    fn every_value_every_bit_offset() {
        let mut all = vec![0u8; 32768 * 15 / 8];
        for z in 0..32768usize {
            for k in 0..15 {
                if z >> (14 - k) & 1 == 1 {
                    let bit = z * 15 + k;
                    all[bit / 8] |= 0x80 >> (bit % 8);
                }
            }
        }
        for shift in 0..15 {
            let mut data = vec![0xA5u8; shift];
            data.extend_from_slice(&all);
            let encoded = crate::encode(&data);
            assert_eq!(decode_every_level(&encoded).unwrap(), data, "shift {shift}");
        }
    }

    /// A 3-byte character outside the repertoire at every position of a long
    /// input, so it lands in every lane of the SIMD blocks.
    #[test]
    fn invalid_3byte_char_every_position() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let data: Vec<u8> = (0..3000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 56) as u8
            })
            .collect();
        let valid = crate::encode(&data);
        for (i, c) in valid.char_indices() {
            let s = format!("{}\u{3000}{}", &valid[..i], &valid[i + c.len_utf8()..]);
            assert!(
                matches!(
                    decode_every_level(&s),
                    Err(DecodeError::UnrecognizedCharacter('\u{3000}'))
                ),
                "{i}"
            );
        }
    }

    /// The AVX-512 path must actually run where the CPU supports it.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_path_runs() {
        if !(std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vbmi")
            && std::is_x86_feature_detected!("avx512bitalg"))
        {
            eprintln!("AVX-512 BW/VBMI/BITALG not available; skipped");
            return;
        }
        let data: Vec<u8> = (0..3000).map(|i| (i * 7) as u8).collect();
        let encoded = crate::encode(&data);
        let blocks = || crate::simd::AVX512_BLOCKS.with(|n| n.get());
        let before = blocks();
        assert_eq!(decode_impl::<LEVEL_AVX512>(&encoded).unwrap(), data);
        assert!(blocks() > before);
    }

    /// Pseudo-random bytes (xorshift64).
    fn random_bytes(n: usize, mut state: u64) -> Vec<u8> {
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 56) as u8
            })
            .collect()
    }

    /// All 3-byte UTF-8 characters that encode a 15-bit value.
    fn three_byte_z15_chars() -> Vec<char> {
        (0x800..=0xFFFFu32)
            .filter(|&cp| crate::table::DECODE[cp as usize] < 0x8000)
            .map(|cp| char::from_u32(cp).unwrap())
            .collect()
    }

    /// The input starting at every offset 0..64 from an allocation, so the
    /// SIMD loads see every address alignment up to 64 bytes.
    #[test]
    fn input_address_offsets() {
        let data = random_bytes(1000, 0x2545_F491_4F6C_DD1D);
        let encoded = crate::encode(&data);
        for offset in 0..64 {
            let mut buf = "x".repeat(offset);
            buf.push_str(&encoded);
            assert_eq!(
                decode_every_level(&buf[offset..]).unwrap(),
                data,
                "{offset}"
            );
        }
    }

    /// Every 3-byte 15-bit character in every lane of the 8- and 32-character
    /// SIMD blocks: the characters are decoded in sequence, preceded by 0..32
    /// filler characters, so character i lands in lane (i + shift) % 32.
    #[test]
    fn every_3byte_char_in_every_lane() {
        let chars = three_byte_z15_chars();
        let filler = chars[0];
        for shift in 0..32 {
            let mut s: String = std::iter::repeat_n(filler, shift).collect();
            s.extend(&chars);
            // A multiple of 8 characters is a whole number of bytes with no padding.
            let n = shift + chars.len();
            s.extend(std::iter::repeat_n(filler, n.next_multiple_of(8) - n));
            let decoded = decode_every_level(&s).unwrap();
            assert_eq!(crate::encode(&decoded), s, "shift {shift}");
        }
    }

    /// Every 3-byte code point outside the repertoire, in lane cp % 32 of an
    /// otherwise valid 64-character input.
    #[test]
    fn every_invalid_3byte_code_point() {
        let valid: Vec<char> = crate::encode(&random_bytes(120, 7)).chars().collect();
        assert_eq!(valid.len(), 64);
        for cp in 0x800..=0xFFFFu32 {
            let Some(c) = char::from_u32(cp) else {
                continue; // surrogates
            };
            if crate::table::DECODE[cp as usize] < 0x8000 {
                continue;
            }
            let mut chars = valid.clone();
            chars[cp as usize % 32] = c;
            let s: String = chars.into_iter().collect();
            assert!(
                matches!(
                    decode_every_level(&s),
                    Err(DecodeError::UnrecognizedCharacter(e)) if e == c
                ),
                "U+{cp:04X}"
            );
        }
    }

    /// The page + rank tables used by the AVX-512 path, evaluated in scalar
    /// code, agree with `DECODE` for every code point U+0000..=U+FFFF.
    #[test]
    fn rank_tables_match_decode_table() {
        use crate::table::{DECODE, HALF_MASK, HALF_PREFIX, PAGE_DESC};
        for cp in 0..=0xFFFFu32 {
            let desc = PAGE_DESC[cp as usize >> 10] as u32;
            let half = ((desc | (cp >> 9 & 1)) & 31) as usize;
            let mask = HALF_MASK[half] as u32;
            let bit = 1 << (cp >> 5 & 15);
            let expected = DECODE[cp as usize];
            if mask & bit == 0 {
                assert!(
                    expected >= 0x8000,
                    "U+{cp:04X} should be a 15-bit character"
                );
                continue;
            }
            let rank = (mask & (bit - 1)).count_ones();
            let z = ((desc >> 5) + HALF_PREFIX[half] as u32 + rank) << 5 | (cp & 31);
            assert_eq!(z, expected as u32, "U+{cp:04X}");
        }
    }

    /// Inputs that end right before an inaccessible page: any read past the
    /// end of the input faults.
    #[cfg(any(windows, all(target_os = "linux", target_pointer_width = "64")))]
    #[test]
    fn reads_stay_within_input() {
        // 64 KiB covers both the page size and Windows' allocation granularity.
        const SIZE: usize = 1 << 16;
        let base = unsafe { guard_page::alloc(2 * SIZE) };
        assert!(!base.is_null());
        assert!(unsafe { guard_page::protect_none(base.add(SIZE), SIZE) });
        let end = unsafe { base.add(SIZE) };

        let decode_at_end = |s: &str| {
            // SAFETY: `s` fits in the accessible first half.
            let copy = unsafe {
                let dst = end.sub(s.len());
                std::ptr::copy_nonoverlapping(s.as_ptr(), dst, s.len());
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(dst, s.len()))
            };
            decode_every_level(copy)
        };

        let random = random_bytes(600, 11);
        // Zero bytes encode to 2-byte characters, which take the fallback paths.
        let mixed: Vec<u8> = random
            .iter()
            .enumerate()
            .map(|(i, &b)| if i % 23 < 4 { 0 } else { b })
            .collect();
        for data in [&random, &mixed] {
            for n in 0..=data.len() {
                let encoded = crate::encode(&data[..n]);
                assert_eq!(decode_at_end(&encoded).unwrap(), &data[..n], "{n}");
                let bad = format!("{encoded}\u{3000}");
                assert!(decode_at_end(&bad).is_err(), "{n}");
            }
        }
        unsafe { guard_page::free(base, 2 * SIZE) };
    }

    #[cfg(windows)]
    mod guard_page {
        use core::ffi::c_void;
        unsafe extern "system" {
            fn VirtualAlloc(addr: *mut c_void, size: usize, ty: u32, protect: u32) -> *mut c_void;
            fn VirtualProtect(addr: *mut c_void, size: usize, protect: u32, old: *mut u32) -> i32;
            fn VirtualFree(addr: *mut c_void, size: usize, ty: u32) -> i32;
        }
        const MEM_COMMIT_RESERVE: u32 = 0x3000;
        const MEM_RELEASE: u32 = 0x8000;
        const PAGE_NOACCESS: u32 = 0x01;
        const PAGE_READWRITE: u32 = 0x04;

        pub unsafe fn alloc(size: usize) -> *mut u8 {
            unsafe {
                VirtualAlloc(
                    core::ptr::null_mut(),
                    size,
                    MEM_COMMIT_RESERVE,
                    PAGE_READWRITE,
                )
                .cast()
            }
        }
        pub unsafe fn protect_none(p: *mut u8, size: usize) -> bool {
            let mut old = 0;
            unsafe { VirtualProtect(p.cast(), size, PAGE_NOACCESS, &mut old) != 0 }
        }
        pub unsafe fn free(p: *mut u8, _size: usize) {
            unsafe { VirtualFree(p.cast(), 0, MEM_RELEASE) };
        }
    }

    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    mod guard_page {
        use core::ffi::c_void;
        unsafe extern "C" {
            fn mmap(
                addr: *mut c_void,
                len: usize,
                prot: i32,
                flags: i32,
                fd: i32,
                off: i64,
            ) -> *mut c_void;
            fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
            fn munmap(addr: *mut c_void, len: usize) -> i32;
        }
        const PROT_NONE: i32 = 0;
        const PROT_READ_WRITE: i32 = 3;
        const MAP_PRIVATE_ANONYMOUS: i32 = 0x22;

        pub unsafe fn alloc(size: usize) -> *mut u8 {
            let p = unsafe {
                mmap(
                    core::ptr::null_mut(),
                    size,
                    PROT_READ_WRITE,
                    MAP_PRIVATE_ANONYMOUS,
                    -1,
                    0,
                )
            };
            // MAP_FAILED is (void *)-1.
            if p as usize == usize::MAX {
                core::ptr::null_mut()
            } else {
                p.cast()
            }
        }
        pub unsafe fn protect_none(p: *mut u8, size: usize) -> bool {
            unsafe { mprotect(p.cast(), size, PROT_NONE) == 0 }
        }
        pub unsafe fn free(p: *mut u8, size: usize) {
            unsafe { munmap(p.cast(), size) };
        }
    }

    /// The capacity helpers must equal the exact formulas evaluated without
    /// overflow, for every `usize` including lengths where `len * 8` or
    /// `len / 2 * 15` would wrap (>= 512 MiB on 32-bit targets).
    #[test]
    fn capacity_no_overflow() {
        let max = usize::MAX;
        let mut lens: Vec<usize> = (0..1000).collect();
        for k in [8usize, 15, 16] {
            let edge = max / k;
            lens.extend(edge.saturating_sub(40)..=edge.saturating_add(40));
        }
        lens.extend(max - 40..=max);
        lens.extend([1 << 29, (1 << 29) + 1, 1 << 31, isize::MAX as usize]);

        for len in lens {
            let l = len as u128;
            let chars = (l * 8).div_ceil(15);
            assert_eq!(
                crate::encoded_chars(len) as u128,
                chars,
                "encoded_chars({len})"
            );
            let enc = chars * 3 + 1;
            if enc <= max as u128 {
                assert_eq!(
                    crate::encoded_capacity(len) as u128,
                    enc,
                    "encoded_capacity({len})"
                );
            } else {
                assert!(std::panic::catch_unwind(|| crate::encoded_capacity(len)).is_err());
            }
            assert_eq!(
                crate::decoded_capacity(len) as u128,
                l / 2 * 15 / 8 + 1,
                "decoded_capacity({len})"
            );
        }
    }
}
