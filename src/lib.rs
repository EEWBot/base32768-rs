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

/// Encodes `b` as a new `String`.
pub fn encode(b: &[u8]) -> String {
    encode_impl::<LEVEL_AVX512>(b)
}

/// Appends the encoding of `b` to `out`, reusing its allocation when it has
/// room for about `b.len() * 8 / 5 + 4` more bytes.
pub fn encode_into(b: &[u8], out: &mut String) {
    // SAFETY: `encode_append` appends only well-formed UTF-8.
    encode_append::<LEVEL_AVX512>(b, unsafe { out.as_mut_vec() });
}

/// [`encode_append`] into a new `String`.
fn encode_impl<const LEVEL: u8>(b: &[u8]) -> String {
    // Sized up front: `reserve` on an empty `Vec` takes the slower growth path
    // (3-6 ns more for short inputs).
    let mut out = Vec::with_capacity(encoded_capacity(b.len()));
    encode_append::<LEVEL>(b, &mut out);
    // SAFETY: `encode_append` appends only well-formed UTF-8.
    unsafe { String::from_utf8_unchecked(out) }
}

/// Appends the encoding of `b` to `out` as UTF-8, with fast paths up to
/// `LEVEL` (see [`LEVEL_SCALAR`]).
#[inline(always)]
fn encode_append<const LEVEL: u8>(b: &[u8], out: &mut Vec<u8>) {
    let start = out.len();
    out.reserve(encoded_capacity(b.len()));
    // SAFETY: `start <= capacity`; the reserved bytes follow.
    let dst = unsafe { out.as_mut_ptr().add(start) };
    #[allow(unused_mut)]
    let mut pos = 0usize;
    #[allow(unused_mut)]
    let mut done = 0usize;

    // AVX-512: 60 bytes -> 32 characters. Shorter inputs skip the feature
    // detection.
    #[cfg(target_arch = "x86_64")]
    if LEVEL >= LEVEL_AVX512
        && b.len() >= 60
        && std::is_x86_feature_detected!("popcnt")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vbmi")
        && std::is_x86_feature_detected!("avx512vbmi2")
    {
        #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2,popcnt")]
        unsafe fn encode_avx512(b: &[u8], dst: *mut u8) -> (usize, usize) {
            let (mut done, mut pos) = (0, 0);
            // Two blocks per iteration (1.02-1.08x when 2-byte characters
            // take the slower store paths).
            // SAFETY (all calls): 60 bytes are readable at each block's
            // input offset, and each 60 input bytes reserve 32 * 3 = 96 output
            // bytes, the most a block writes.
            while done + 120 <= b.len() {
                pos += unsafe { simd::encode32_avx512(b.as_ptr().add(done), dst.add(pos)) };
                pos += unsafe { simd::encode32_avx512(b.as_ptr().add(done + 60), dst.add(pos)) };
                done += 120;
            }
            if done + 60 <= b.len() {
                pos += unsafe { simd::encode32_avx512(b.as_ptr().add(done), dst.add(pos)) };
                done += 60;
            }
            (done, pos)
        }
        // SAFETY: the AVX-512 features were just checked; see above for the
        // buffer sizes.
        (done, pos) = unsafe { encode_avx512(b, dst) };
    }

    // AVX2: 60 bytes -> 32 characters. After the AVX-512 path fewer than 60
    // bytes remain, so this runs only on CPUs without it.
    #[cfg(target_arch = "x86_64")]
    if LEVEL >= LEVEL_SIMD && b.len() - done >= 61 && std::is_x86_feature_detected!("avx2") {
        #[target_feature(enable = "avx2")]
        unsafe fn encode_avx2(
            b: &[u8],
            dst: *mut u8,
            mut done: usize,
            mut pos: usize,
        ) -> (usize, usize) {
            // At least 61 bytes: the last 16-byte load ends at done + 61, and
            // the byte after the block reserves at least 3 + 1 output bytes
            // beyond its 96, covering the 100 bytes the block may write.
            while done + 61 <= b.len() {
                // SAFETY: see above.
                pos += unsafe { simd::encode32_avx2(b.as_ptr().add(done), dst.add(pos)) };
                done += 60;
            }
            (done, pos)
        }
        // SAFETY: AVX2 was just checked.
        (done, pos) = unsafe { encode_avx2(b, dst, done, pos) };
    }

    // Fast path: 15 bytes -> 8 characters.
    let (chunks, remainder) = b[done..].as_chunks::<15>();
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

    debug_assert!(start + pos < out.capacity());
    // SAFETY: `pos` bytes after `start` were initialised by `put`; every
    // character is a well-formed 2 or 3 byte UTF-8 sequence from the tables.
    unsafe { out.set_len(start + pos) };
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

/// Decodes a character known to be a 2 byte UTF-8 sequence.
///
/// # Safety
/// `p .. p + 2` must be readable.
#[inline(always)]
unsafe fn read_2byte(p: *const u8) -> u32 {
    let (b0, b1) = unsafe { (*p as u32, *p.add(1) as u32) };
    lookup(((b0 & 0x1F) << 6) | (b1 & 0x3F))
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

/// Decodes `s` into a new `Vec`.
pub fn decode(s: &str) -> Result<Vec<u8>, DecodeError> {
    decode_impl::<LEVEL_AVX512>(s)
}

/// Appends the decoding of `s` to `out`, reusing its allocation when it has
/// room for about `s.len() * 15 / 16 + 1` more bytes. On error `out` is left
/// as it was (its capacity may have grown).
pub fn decode_into(s: &str, out: &mut Vec<u8>) -> Result<(), DecodeError> {
    decode_append::<LEVEL_AVX512>(s, out)
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
/// AVX-512 (BW, VBMI, VBMI2, BITALG) on x86-64.
const LEVEL_AVX512: u8 = 2;

/// [`decode_append`] into a new `Vec`.
#[inline(always)]
fn decode_impl<const LEVEL: u8>(s: &str) -> Result<Vec<u8>, DecodeError> {
    // Sized up front, as in `encode_impl`.
    let mut out = Vec::with_capacity(decoded_capacity(s.len()));
    decode_append::<LEVEL>(s, &mut out)?;
    Ok(out)
}

/// Appends the decoding of `s` to `out`; on error, truncates `out` back to
/// its original length.
#[inline(always)]
fn decode_append<const LEVEL: u8>(s: &str, out: &mut Vec<u8>) -> Result<(), DecodeError> {
    let start = out.len();
    let r = decode_append_unchecked::<LEVEL>(s, out);
    if r.is_err() {
        out.truncate(start);
    }
    r
}

/// Appends the decoding of `s` to `out`, leaving a partial result on error.
#[inline(always)]
fn decode_append_unchecked<const LEVEL: u8>(s: &str, out: &mut Vec<u8>) -> Result<(), DecodeError> {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let start = out.len();
    out.reserve(decoded_capacity(len));

    let mut pos = 0usize;
    let mut char_index = 0usize;

    // Fast path: 8 characters (120 bits) -> 15 bytes. Bails out to the exact
    // path below on anything other than eight 15-bit characters.
    if len >= 24 {
        // SAFETY: `start <= capacity`; the reserved bytes follow.
        let dst = unsafe { out.as_mut_ptr().add(start) };
        // SAFETY: `out` has the capacity `decode_fast` relies on after `start`.
        let (p, written, chars) = unsafe { decode_fast_dispatch::<LEVEL>(bytes, dst) };
        // SAFETY: `written` bytes after `start` were initialised by `decode_fast`.
        unsafe { out.set_len(start + written) };
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

    Ok(())
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
            // All 2-byte (runs of zero bytes in the input data).
            if let Some(cps) = simd::code_points_8x2(p) {
                return (cps.map(|cp| lookup(cp as u32)), 16);
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
            // All 2-byte (runs of zero bytes in the input data).
            let l: [u8; 8] = core::array::from_fn(|i| *p.add(2 * i));
            if l.iter().all(|&b| b & 0xE0 == 0xC0) {
                return (core::array::from_fn(|i| read_2byte(p.add(2 * i))), 16);
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
/// [`read_8`]; with `AVX512`, the CPU must support AVX-512 BW, VBMI, VBMI2 and BITALG.
/// `DENSE` (with `AVX512`) adds a path for windows of 2-byte characters; see
/// [`decode_fast_dispatch`] for why it is a separate instance.
#[inline(always)]
unsafe fn decode_fast<const SIMD: bool, const AVX512: bool, const DENSE: bool>(
    bytes: &[u8],
    dst: *mut u8,
) -> (usize, usize, usize) {
    let src = bytes.as_ptr();
    let len = bytes.len();
    let mut pos = 0usize;
    let mut written = 0usize;

    while pos + 24 <= len {
        #[cfg(target_arch = "x86_64")]
        if DENSE && AVX512 && pos + 96 <= len && unsafe { *src.add(pos) } < 0xE0 {
            // A 2-byte (or invalid) lead: 32 two-byte characters.
            // SAFETY: `pos + 96 <= len`. 64 input bytes account for 60 bytes of
            // `dst` capacity, the most written.
            let groups = unsafe { simd::decode32_2byte_avx512(src.add(pos), dst.add(written)) };
            // A whole window keeps the next position independent of the
            // vector computation, as below.
            if groups == 4 {
                pos += 64;
                written += 60;
                continue;
            }
            // The 8-character path below takes the rejected block.
            pos += 16 * groups;
            written += 15 * groups;
        } else if AVX512 && pos + 96 <= len {
            // SAFETY: `pos + 96 <= len`. Those 96 input bytes account for
            // 48 * 15 / 8 = 90 bytes of `dst` capacity (2 bytes per
            // character), more than the 60 bytes written.
            let groups = unsafe { simd::decode32_avx512(src.add(pos), dst.add(written)) };
            // A separate branch for a whole window keeps the next position
            // independent of the vector computation, so windows overlap.
            if groups == 4 {
                pos += 96;
                written += 60;
                continue;
            }
            core::hint::cold_path();
            // Keep the leading groups that were decoded; the 8-character path
            // below takes the block with the rejected character, and the next
            // iteration retries AVX-512. `pos + 24 <= len` still holds since
            // `groups <= 3`.
            pos += 24 * groups;
            written += 15 * groups;
        }
        #[cfg(target_arch = "x86_64")]
        if AVX512 && !DENSE && pos + 64 <= len {
            // SAFETY: pos is a UTF-8 boundary with 64 readable bytes.
            // Consuming at least 32 bytes reserves room for 30 decoded bytes.
            let n = unsafe { simd::decode16_mixed_avx512(src.add(pos), dst.add(written)) };
            if n != 0 {
                pos += n;
                written += 30;
                continue;
            }
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
    // Inputs shorter than one AVX-512 window (96 bytes) skip the feature
    // detection; `decode_fast` would not use AVX-512 for them anyway.
    #[cfg(target_arch = "x86_64")]
    if LEVEL >= LEVEL_AVX512
        && bytes.len() >= 96
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vbmi")
        && std::is_x86_feature_detected!("avx512bitalg")
        && std::is_x86_feature_detected!("avx512vbmi2")
    {
        // Two separate functions: in one, the 2-byte path's constants take
        // zmm registers from the 3-byte loop, which then reloads constants
        // from memory on every window (2-4% slower on random data). The input
        // is classified once by its first 64 bytes; either function decodes
        // any input correctly.
        #[target_feature(enable = "ssse3,avx512f,avx512bw,avx512vbmi,avx512vbmi2,avx512bitalg")]
        #[inline(never)]
        unsafe fn decode_fast_avx512(bytes: &[u8], dst: *mut u8) -> (usize, usize, usize) {
            unsafe { decode_fast::<true, true, false>(bytes, dst) }
        }
        #[target_feature(enable = "ssse3,avx512f,avx512bw,avx512vbmi,avx512vbmi2,avx512bitalg")]
        #[inline(never)]
        unsafe fn decode_fast_avx512_dense(bytes: &[u8], dst: *mut u8) -> (usize, usize, usize) {
            unsafe { decode_fast::<true, true, true>(bytes, dst) }
        }
        /// At least 24 of the 64 bytes at `p` are 2-byte UTF-8 leads (runs of
        /// zero bytes in the input data).
        #[target_feature(enable = "avx512f,avx512bw")]
        unsafe fn two_byte_dense(p: *const u8) -> bool {
            use core::arch::x86_64::*;
            // SAFETY: the caller guarantees 64 readable bytes.
            let v = unsafe { _mm512_loadu_si512(p.cast()) };
            let leads = _mm512_cmpeq_epi8_mask(
                _mm512_and_si512(v, _mm512_set1_epi8(0xE0u8 as i8)),
                _mm512_set1_epi8(0xC0u8 as i8),
            );
            leads.count_ones() >= 24
        }
        // SAFETY: the AVX-512 features were just checked (they imply SSSE3);
        // 64 bytes are readable. Shorter inputs skip the classification.
        unsafe {
            if bytes.len() >= 1024 && two_byte_dense(bytes.as_ptr()) {
                return decode_fast_avx512_dense(bytes, dst);
            }
            return decode_fast_avx512(bytes, dst);
        }
    }

    #[cfg(target_arch = "x86_64")]
    if LEVEL >= LEVEL_SIMD && std::is_x86_feature_detected!("ssse3") {
        #[target_feature(enable = "ssse3")]
        unsafe fn decode_fast_ssse3(bytes: &[u8], dst: *mut u8) -> (usize, usize, usize) {
            unsafe { decode_fast::<true, false, false>(bytes, dst) }
        }
        // SAFETY: SSSE3 support was just checked.
        return unsafe { decode_fast_ssse3(bytes, dst) };
    }

    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if LEVEL >= LEVEL_SIMD {
        // SAFETY: NEON is part of the aarch64 baseline.
        return unsafe { decode_fast::<true, false, false>(bytes, dst) };
    }

    unsafe { decode_fast::<false, false, false>(bytes, dst) }
}

/// Code points of eight 3-byte UTF-8 characters at `p .. p + 24`, or `None` if
/// any of the eight lead bytes is not a 3-byte lead (0xE0..=0xEF).
mod simd {
    /// Encodes the 60 bytes at `src` (61 are read) as 32 characters at `d`
    /// and returns the number of bytes (64..=96). Up to 100 bytes may be
    /// written.
    ///
    /// Each half of 16 characters gets its 15-bit values in u16 lanes, using
    /// multiplies in place of the variable shifts AVX2 lacks. Code points are
    /// z + 32 * offset, the offset coming from 8 tables of 16 bytes that
    /// both halves share.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    pub unsafe fn encode32_avx2(src: *const u8, d: *mut u8) -> usize {
        use crate::table;
        use core::arch::x86_64::*;

        /// u16 lane l of each 128-bit half (15 input bytes): bytes j + 1, j
        /// (x) or byte j + 2 in the high byte (y), where j = 15 * l / 8.
        const fn unpack(y: bool) -> [u8; 32] {
            let mut m = [0x80u8; 32];
            let mut i = 0;
            while i < 16 {
                let (h, l) = (i / 8, i % 8);
                let j = 15 * l / 8;
                if y {
                    m[16 * h + 2 * l + 1] = (j + 2) as u8;
                } else {
                    m[16 * h + 2 * l] = (j + 1) as u8;
                    m[16 * h + 2 * l + 1] = j as u8;
                }
                i += 1;
            }
            m
        }
        /// 2^(15 i mod 8): the left shift that drops the previous value's bits.
        const fn factors() -> [u16; 16] {
            let mut m = [0u16; 16];
            let mut i = 0;
            while i < 16 {
                m[i] = 1 << (15 * (i % 8) % 8);
                i += 1;
            }
            m
        }
        /// Output bytes `from..from + 16` of each 128-bit half (8 characters,
        /// 24 bytes): character k's bytes are A[2k], A[2k + 1] and B[2k].
        const fn utf8(b: bool, from: usize) -> [u8; 32] {
            let mut m = [0x80u8; 32];
            let mut h = 0;
            while h < 2 {
                let mut o = 0;
                while o < 16 && from + o < 24 {
                    let (k, part) = ((from + o) / 3, (from + o) % 3);
                    if b && part == 2 {
                        m[16 * h + o] = (2 * k) as u8;
                    } else if !b && part < 2 {
                        m[16 * h + o] = (2 * k + part) as u8;
                    }
                    o += 1;
                }
                h += 1;
            }
            m
        }
        static UNPACK_X: [u8; 32] = unpack(false);
        static UNPACK_Y: [u8; 32] = unpack(true);
        static FACTORS: [u16; 16] = factors();
        static OUT_A0: [u8; 32] = utf8(false, 0);
        static OUT_B0: [u8; 32] = utf8(true, 0);
        static OUT_A1: [u8; 32] = utf8(false, 16);
        static OUT_B1: [u8; 32] = utf8(true, 16);
        static LOW: [u8; 128] = table::ENCODE_AVX2;
        /// Per 4 characters in u32 lanes: keep bytes 0, 1 and, unless the
        /// character is 2 bytes (mask bit set), byte 2.
        const fn compress() -> [[u8; 16]; 16] {
            let mut t = [[0x80u8; 16]; 16];
            let mut m = 0;
            while m < 16 {
                let (mut o, mut k) = (0, 0);
                while k < 4 {
                    t[m][o] = (4 * k) as u8;
                    t[m][o + 1] = (4 * k + 1) as u8;
                    o += 2;
                    if m >> k & 1 == 0 {
                        t[m][o] = (4 * k + 2) as u8;
                        o += 1;
                    }
                    k += 1;
                }
                m += 1;
            }
            t
        }
        static COMPRESS: [[u8; 16]; 16] = compress();

        // SAFETY: the caller guarantees 61 readable bytes at `src` and 100
        // writable bytes at `d`.
        unsafe {
            let ld = |p: *const u8| _mm256_loadu_si256(p.cast());
            let k = |c: u16| _mm256_set1_epi16(c as i16);
            let opaque = |mut v: __m256i| {
                core::arch::asm!("/* {0} */", inout(ymm_reg) v, options(pure, nomem, nostack, preserves_flags));
                v
            };
            // Hide the powers of two: LLVM otherwise turns the multiplies
            // into 32-bit variable shifts plus packs and a vpermq.
            let f = opaque(ld(FACTORS.as_ptr().cast()));
            let unpack = |p: *const u8| {
                let v = _mm256_loadu2_m128i(p.add(15).cast(), p.cast());
                let x = _mm256_shuffle_epi8(v, ld(UNPACK_X.as_ptr()));
                let y = _mm256_shuffle_epi8(v, ld(UNPACK_Y.as_ptr()));
                // LLVM widens the mulhi intrinsic to 32 bits (vpmovzxwd, packs, vpermq).
                let lo: __m256i;
                core::arch::asm!("vpmulhuw {r}, {y}, {f}", r = lateout(ymm_reg) lo, y = in(ymm_reg) y, f = in(ymm_reg) f, options(pure, nomem, nostack, preserves_flags));
                // Opaque so that LLVM keeps the signed compares as single vpcmpgtw.
                opaque(_mm256_srli_epi16::<1>(_mm256_or_si256(
                    _mm256_mullo_epi16(x, f),
                    lo,
                )))
            };
            let za = unpack(src);
            let zb = unpack(src.add(30));

            // z = (x << r | y >> (8 - r)) >> 1 with r = 15 l mod 8: the
            // multiplies by 2^r stand in for per-lane shifts.

            // Shared table lookup for all 32 characters: block index bits
            // 0..=6 packed to bytes (in-lane: low half A, high half B).
            let idx = |z: __m256i| _mm256_and_si256(_mm256_srli_epi16::<5>(z), k(127));
            let indices = _mm256_packus_epi16(idx(za), idx(zb));
            let lo4 = _mm256_and_si256(indices, _mm256_set1_epi8(0x0F));
            let t = |n: usize| {
                let tab =
                    _mm256_broadcastsi128_si256(_mm_loadu_si128(LOW.as_ptr().add(16 * n).cast()));
                _mm256_shuffle_epi8(tab, lo4)
            };
            // Index bits 4, 5, 6 to each byte's top bit (bits from the
            // neighbouring byte only reach lower bits).
            let m4 = _mm256_slli_epi16::<3>(indices);
            let m5 = _mm256_slli_epi16::<2>(indices);
            let m6 = _mm256_slli_epi16::<1>(indices);
            let r01 = _mm256_blendv_epi8(t(0), t(1), m4);
            let r23 = _mm256_blendv_epi8(t(2), t(3), m4);
            let r45 = _mm256_blendv_epi8(t(4), t(5), m4);
            let r67 = _mm256_blendv_epi8(t(6), t(7), m4);
            let r03 = _mm256_blendv_epi8(r01, r23, m5);
            let r47 = _mm256_blendv_epi8(r45, r67, m5);
            let offsets = _mm256_blendv_epi8(r03, r47, m6);
            let zero = _mm256_setzero_si256();

            let code_points = |z: __m256i, lut8: __m256i| {
                let gt = |c: u16| _mm256_cmpgt_epi16(z, k(c));
                let lut = _mm256_add_epi16(lut8, _mm256_and_si256(gt(0xD3F), k(256)));
                // Outside the tables: 306, 307 or 309.
                let m2 = gt(0x795F);
                let default = _mm256_sub_epi16(
                    _mm256_sub_epi16(_mm256_sub_epi16(k(306), gt(0x275F)), m2),
                    m2,
                );
                let use_lut = _mm256_or_si256(_mm256_cmpgt_epi16(k(0xDC0), z), gt(0x7DDF));
                let off = _mm256_blendv_epi8(default, lut, use_lut);
                _mm256_add_epi16(z, _mm256_slli_epi16::<5>(off))
            };
            let cpa = code_points(za, _mm256_unpacklo_epi8(offsets, zero));
            let cpb = code_points(zb, _mm256_unpackhi_epi8(offsets, zero));

            // 16 three-byte characters -> 48 bytes.
            let store3 = |cp: __m256i, d: *mut u8| {
                // A = lead byte | second byte << 8, B = third byte.
                let a = _mm256_or_si256(
                    _mm256_or_si256(_mm256_srli_epi16::<12>(cp), k(0x80E0)),
                    _mm256_and_si256(_mm256_slli_epi16::<2>(cp), k(0x3F00)),
                );
                let b = _mm256_or_si256(_mm256_and_si256(cp, k(0x3F)), k(0x80));
                let o0 = _mm256_or_si256(
                    _mm256_shuffle_epi8(a, ld(OUT_A0.as_ptr())),
                    _mm256_shuffle_epi8(b, ld(OUT_B0.as_ptr())),
                );
                let o1 = _mm256_or_si256(
                    _mm256_shuffle_epi8(a, ld(OUT_A1.as_ptr())),
                    _mm256_shuffle_epi8(b, ld(OUT_B1.as_ptr())),
                );
                _mm_storeu_si128(d.cast(), _mm256_castsi256_si128(o0));
                _mm_storel_epi64(d.add(16).cast(), _mm256_castsi256_si128(o1));
                _mm_storeu_si128(d.add(24).cast(), _mm256_extracti128_si256::<1>(o0));
                _mm_storel_epi64(d.add(40).cast(), _mm256_extracti128_si256::<1>(o1));
            };

            let two_a = _mm256_cmpgt_epi16(k(224), za);
            let two_b = _mm256_cmpgt_epi16(k(224), zb);
            let any = _mm256_or_si256(two_a, two_b);
            if _mm256_testz_si256(any, any) != 0 {
                #[cfg(test)]
                count_avx2_half(0, 2);
                store3(cpa, d);
                store3(cpb, d.add(48));
                return 96;
            }
            core::hint::cold_path();
            let mut pos = 0;
            for (cp, two) in [(cpa, two_a), (cpb, two_b)] {
                let m = _mm256_movemask_epi8(two) as u32;
                #[cfg(test)]
                count_avx2_half(
                    match m {
                        0 => 0,
                        u32::MAX => 1,
                        _ => 2,
                    },
                    1,
                );
                if m == 0 {
                    store3(cp, d.add(pos));
                    pos += 48;
                } else if m == u32::MAX {
                    // 16 two-byte characters: UTF-8 fits each u16 lane.
                    let b0 = _mm256_or_si256(_mm256_srli_epi16::<6>(cp), k(0xC0));
                    let b1 = _mm256_or_si256(_mm256_and_si256(cp, k(0x3F)), k(0x80));
                    _mm256_storeu_si256(
                        d.add(pos).cast(),
                        _mm256_or_si256(b0, _mm256_slli_epi16::<8>(b1)),
                    );
                    pos += 32;
                } else {
                    // Mixed: each character's bytes in a u32 lane, then
                    // compress 4 characters at a time.
                    let a3 = _mm256_or_si256(
                        _mm256_or_si256(_mm256_srli_epi16::<12>(cp), k(0x80E0)),
                        _mm256_and_si256(_mm256_slli_epi16::<2>(cp), k(0x3F00)),
                    );
                    let low6 = _mm256_or_si256(_mm256_and_si256(cp, k(0x3F)), k(0x80));
                    let a2 = _mm256_or_si256(
                        _mm256_or_si256(_mm256_srli_epi16::<6>(cp), k(0xC0)),
                        _mm256_slli_epi16::<8>(low6),
                    );
                    let a = _mm256_blendv_epi8(a3, a2, two);
                    let c0 = _mm256_unpacklo_epi16(a, low6);
                    let c1 = _mm256_unpackhi_epi16(a, low6);
                    // One bit per character.
                    let mut bits = m & 0x5555_5555;
                    bits = (bits | bits >> 1) & 0x3333_3333;
                    bits = (bits | bits >> 2) & 0x0F0F_0F0F;
                    bits = (bits | bits >> 4) & 0x00FF_00FF;
                    bits = (bits | bits >> 8) & 0xFFFF;
                    let quarters = [
                        _mm256_castsi256_si128(c0),
                        _mm256_castsi256_si128(c1),
                        _mm256_extracti128_si256::<1>(c0),
                        _mm256_extracti128_si256::<1>(c1),
                    ];
                    for (q, v) in quarters.into_iter().enumerate() {
                        let n = (bits >> (4 * q) & 15) as usize;
                        let mask = _mm_loadu_si128(COMPRESS[n].as_ptr().cast());
                        _mm_storeu_si128(d.add(pos).cast(), _mm_shuffle_epi8(v, mask));
                        pos += 12 - n.count_ones() as usize;
                    }
                }
            }
            pos
        }
    }

    /// Encodes the 60 bytes at `src` as 32 characters at `d`, with table
    /// lookups only in registers, and returns the number of bytes produced
    /// (64..=96). May write beyond the returned length, but never beyond the
    /// 96 bytes at `d`.
    ///
    /// The code point of a 15-bit value z is
    /// `z + 0x2000 * (z >= 0xD40) + 32 * low`, where `low` depends only on
    /// the block `z >> 5`: byte `(ENCODE_START_BIASED[z >> 8] + (z >> 5)) % 128`
    /// of `ENCODE_OFFSET` (see build.rs).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2,popcnt")]
    #[inline]
    pub unsafe fn encode32_avx512(src: *const u8, d: *mut u8) -> usize {
        use crate::table;
        use core::arch::x86_64::*;

        /// Each qword spans four 15-bit values. Reverse its eight input bytes.
        const fn unpack_order() -> [u8; 64] {
            let mut m = [0u8; 64];
            let mut i = 0;
            while i < 64 {
                m[i] = (60 * (i / 8) / 8 + 7 - i % 8) as u8;
                i += 1;
            }
            m
        }
        /// Extract the low and high bytes of each value. Alternate qwords
        /// start four bits into the first byte.
        const fn unpack_bits() -> [u8; 64] {
            let mut m = [0u8; 64];
            let mut i = 0;
            while i < 64 {
                m[i] = (49 - 4 * (i / 8 % 2) - 15 * (i / 2 % 4) + 8 * (i % 2)) as u8;
                i += 1;
            }
            m
        }
        /// Extract bits 12..15 and 6..11 of each code point into adjacent bytes.
        const fn utf8_bits() -> [u8; 64] {
            let mut m = [0u8; 64];
            let mut i = 0;
            while i < 64 {
                m[i] = (16 * (i / 2 % 4) + if i % 2 == 0 { 12 } else { 6 }) as u8;
                i += 1;
            }
            m
        }
        /// vpermt2b indices interleaving (b0, b1) from the first table and b2
        /// from the second into 3 bytes per character, for characters
        /// 16 * half .. 16 * half + 16.
        const fn utf8_idx(half: usize) -> [u8; 64] {
            let mut m = [0u8; 64];
            let mut t = 0;
            while t < 48 {
                let k = 16 * half + t / 3;
                m[t] = match t % 3 {
                    0 => 2 * k,
                    1 => 2 * k + 1,
                    _ => 64 + 2 * k,
                } as u8;
                t += 1;
            }
            m
        }
        static ORDER: [u8; 64] = unpack_order();
        static BITS: [u8; 64] = unpack_bits();
        static UTF8_BITS: [u8; 64] = utf8_bits();
        static OUT0: [u8; 64] = utf8_idx(0);
        static OUT1: [u8; 64] = utf8_idx(1);
        /// (b1, b2) of each of 32 characters.
        const fn utf8_idx2() -> [u8; 64] {
            let mut m = [0u8; 64];
            let mut k = 0;
            while k < 32 {
                m[2 * k] = (2 * k + 1) as u8;
                m[2 * k + 1] = (64 + 2 * k) as u8;
                k += 1;
            }
            m
        }
        static OUT2: [u8; 64] = utf8_idx2();
        static START: [u8; 128] = table::ENCODE_START_BIASED;
        static OFF: [u8; 128] = table::ENCODE_OFFSET;
        // SAFETY: each load reads 64 bytes from a table at least that long.
        let ld = |p: *const u8| unsafe { _mm512_loadu_si512(p.cast()) };
        let w = |x: i16| _mm512_set1_epi16(x);
        const ODD: u64 = 0xAAAA_AAAA_AAAA_AAAA;
        const M48: u64 = (1 << 48) - 1;

        // 60 bytes -> 32 x 15-bit values. The masked load touches only 60 bytes.
        // SAFETY: the caller guarantees 60 readable bytes at `src`.
        let v = unsafe { _mm512_maskz_loadu_epi8((1 << 60) - 1, src.cast()) };
        let input = _mm512_permutexvar_epi8(ld(ORDER.as_ptr()), v);
        let z = _mm512_and_si512(
            _mm512_multishift_epi64_epi8(ld(BITS.as_ptr()), input),
            w(0x7FFF),
        );

        // z -> code point, in the high byte of each u16 lane: vpermi2b uses
        // the low 7 bits of each index byte, z >> 8 (the unit) for the start,
        // then start + (z >> 5) % 256 for `low`. The low bytes are ignored and
        // zeroed.
        let hi = _mm512_cmpge_epu16_mask(z, w(0x0D40));
        let z_adj = _mm512_mask_add_epi16(z, hi, z, w(0x2000));
        let start = _mm512_permutex2var_epi8(ld(START.as_ptr()), z, ld(START[64..].as_ptr()));
        let idx = _mm512_add_epi8(start, _mm512_slli_epi16::<3>(z));
        let low_hi =
            _mm512_maskz_permutex2var_epi8(ODD, ld(OFF.as_ptr()), idx, ld(OFF[64..].as_ptr()));
        let cp = _mm512_add_epi16(z_adj, _mm512_srli_epi16::<3>(low_hi));

        // UTF-8 as 3 bytes per character; 2-byte characters get a leading 0,
        // the only zero byte (every other byte has its top bit set).
        // Only the first seven 32-character blocks use two bytes. Their
        // code points also have zero in bits 12..15.
        let two = _mm512_cmplt_epu16_mask(z, w(224));
        let headers = _mm512_mask_blend_epi16(two, w(0x80E0u16 as i16), w(0xC000u16 as i16));
        let bits = _mm512_multishift_epi64_epi8(ld(UTF8_BITS.as_ptr()), cp);
        let pair = _mm512_ternarylogic_epi32::<0xF8>(headers, bits, w(0x3F0F));
        let b2 = _mm512_or_si512(_mm512_and_si512(cp, w(0x3F)), w(0x80));
        let out0 = _mm512_permutex2var_epi8(pair, ld(OUT0.as_ptr()), b2);
        let out1 = _mm512_permutex2var_epi8(pair, ld(OUT1.as_ptr()), b2);

        #[cfg(test)]
        AVX512_ENCODE_BLOCKS.with(|n| n.set(n.get() + 1));

        // SAFETY (stores below): the caller guarantees 96 writable bytes at
        // `d`; each half writes at most 48, right after the previous one.
        if two == 0 {
            unsafe {
                _mm512_mask_storeu_epi8(d.cast(), M48, out0);
                _mm512_mask_storeu_epi8(d.add(48).cast(), M48, out1);
            }
            return 96;
        }
        core::hint::cold_path();
        // All 2-byte (runs of zero bytes in the input data): a fixed shuffle
        // drops the leading zeros.
        if two == u32::MAX {
            let out = _mm512_permutex2var_epi8(pair, ld(OUT2.as_ptr()), b2);
            // SAFETY: 64 of the 96 bytes the caller guarantees.
            unsafe { _mm512_storeu_si512(d.cast(), out) };
            return 64;
        }
        // For a mixed block, each half may write all 48 bytes: the next
        // half or block overwrites the spare bytes, within the reserved 96.
        let compress = |out| {
            let keep = _mm512_test_epi8_mask(out, out);
            _mm512_maskz_compress_epi8(keep, out)
        };
        let n0 = 48 - (two as u16).count_ones() as usize;
        unsafe {
            _mm512_mask_storeu_epi8(d.cast(), M48, compress(out0));
            _mm512_mask_storeu_epi8(d.add(n0).cast(), M48, compress(out1));
        }
        96 - two.count_ones() as usize
    }

    /// vpermb indices for packing decoded 15-bit values: output byte j =
    /// byte (3 - t % 4) of dword t / 4 in 128-bit lane g, where g = j / 15,
    /// t = j % 15 (big-endian dwords, skipping the unused 16th byte of each
    /// lane).
    #[cfg(target_arch = "x86_64")]
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

    /// Per-dword shift counts for packing, repeated in every 128-bit lane.
    #[cfg(target_arch = "x86_64")]
    const fn pack_shifts(v: [u32; 4]) -> [u32; 16] {
        let mut m = [0u32; 16];
        let mut i = 0;
        while i < 16 {
            m[i] = v[i % 4];
            i += 1;
        }
        m
    }

    /// Pack 32 negated 15-bit values into the low 60 bytes. Each signed
    /// multiply-add produces z[2i] << 15 | z[2i+1], a 30-bit pair p. Dword i
    /// of each 128-bit lane becomes p_i << s_i | p_{i+1} >> (30 - s_i) with
    /// s = 2, 4, 6, 8 (one vpshldvd on p_{i+1} << 2; dword 3 shifts in
    /// zeros). PACK drops the last byte in each group.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
    #[inline]
    unsafe fn pack_neg_z(neg_z: core::arch::x86_64::__m512i) -> core::arch::x86_64::__m512i {
        use core::arch::x86_64::*;
        static SHL: [u32; 16] = pack_shifts([2, 4, 6, 8]);
        static PACK: [u8; 64] = pack_order();
        // SAFETY: the two tables each contain 64 bytes.
        let ld = |p: *const u8| unsafe { _mm512_loadu_si512(p.cast()) };
        let pairs = _mm512_madd_epi16(neg_z, _mm512_set1_epi32(((-1i32) << 16) | 0x8000));
        let next = _mm512_slli_epi32::<2>(_mm512_bsrli_epi128::<4>(pairs));
        let words = _mm512_shldv_epi32(pairs, next, ld(SHL.as_ptr().cast()));
        _mm512_permutexvar_epi8(ld(PACK.as_ptr()), words)
    }

    /// Decodes up to 32 three-byte 15-bit characters at `p .. p + 96`.
    /// Returns how many leading 8-character groups (0..=4) hold only 3-byte
    /// 15-bit characters; those groups are written to `d` as 15 bytes each,
    /// and nothing else is written.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2,avx512bitalg")]
    #[inline]
    pub unsafe fn decode32_avx512(p: *const u8, d: *mut u8) -> usize {
        use crate::table;
        use core::arch::x86_64::*;

        /// vpermi2b indices: bytes 0 (high byte) and 1 (low byte) of
        /// character i in u16 lane i, or byte 2 in the low byte.
        const fn char_bytes(third: bool) -> [u8; 64] {
            let mut m = [0u8; 64];
            let mut i = 0;
            while i < 32 {
                if third {
                    m[2 * i] = (3 * i + 2) as u8;
                } else {
                    m[2 * i] = (3 * i + 1) as u8;
                    m[2 * i + 1] = (3 * i) as u8;
                }
                i += 1;
            }
            m
        }
        static I01: [u8; 64] = char_bytes(false);
        static I2: [u8; 64] = char_bytes(true);
        static HDESC: [u8; 128] = table::HALF_DESC_BIASED;
        static MISSING: [u16; 32] = table::MISSING_MASK;
        static END_OFFSET: [u16; 32] = table::END_OFFSET_BIASED;
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
        // u16 lane i: b0 << 8 | b1 of character i, and b2. Lanes from a
        // character that is not 3 bytes long on are misaligned and decode to
        // garbage, but they are never stored (see `valid`).
        let b0b1 = _mm512_permutex2var_epi8(a, ld(I01.as_ptr()), b);
        let lead_ok = _mm512_cmpeq_epi16_mask(
            _mm512_and_si512(b0b1, w(0xF000u16 as i16)),
            w(0xE000u16 as i16),
        );
        let b2 = _mm512_permutex2var_epi8(a, ld(I2.as_ptr()), b);
        // t = 64 * b0 + b1 (vpmaddubsw), so t >> 3 = (cp >> 9) + 0x710 and,
        // with the low 6 bits of b2, t << 6 gives cp + 0x2000 (mod 2^16). The
        // tables absorb both offsets, and the half page lookup does not wait
        // for b2. vpermi2b uses the low 7 bits of t >> 3, vpermw the low 5
        // bits of the entry index (its high byte is garbage).
        let t = _mm512_maddubs_epi16(b0b1, w(0x4001));
        let entry = _mm512_permutex2var_epi8(
            ld(HDESC.as_ptr()),
            _mm512_srli_epi16::<3>(t),
            ld(HDESC[64..].as_ptr()),
        );
        let cp = _mm512_ternarylogic_epi32::<0xF8>(_mm512_slli_epi16::<6>(t), b2, w(0x3F));

        // The double shift uses bits 0..3 of the count, the block number
        // within the half page. Bit zero of the result marks a missing block.
        let mask = _mm512_permutexvar_epi16(entry, ld(MISSING.as_ptr().cast()));
        let offset = _mm512_permutexvar_epi16(entry, ld(END_OFFSET.as_ptr().cast()));
        let missing = _mm512_shrdv_epi16(mask, _mm512_setzero_si512(), _mm512_srli_epi16::<5>(cp));
        let valid = lead_ok & _mm512_testn_epi16_mask(missing, w(1));
        // 8-character groups before the first rejected lane. With none, skip
        // the rest of the work (common when 2-byte characters are dense).
        let groups = (valid.trailing_ones() / 8) as usize;
        if groups == 0 {
            return 0;
        }
        // Count missing blocks back from the end of the half page. Keep
        // -z so the signed madd coefficients produce a positive 30-bit pair.
        let holes = _mm512_popcnt_epi16(missing);
        let neg_z = _mm512_sub_epi16(_mm512_sub_epi16(offset, cp), _mm512_slli_epi16::<5>(holes));
        let out = unsafe { pack_neg_z(neg_z) };
        #[cfg(test)]
        AVX512_BLOCKS.with(|n| n.set(n.get() + 1));

        if groups == 4 {
            // SAFETY: the caller guarantees 60 writable bytes at `d`.
            unsafe { _mm512_mask_storeu_epi8(d.cast(), (1 << 60) - 1, out) };
            return 4;
        }
        core::hint::cold_path();
        // SAFETY: as above; at most 45 bytes are written.
        unsafe { _mm512_mask_storeu_epi8(d.cast(), (1u64 << (15 * groups)) - 1, out) };
        groups
    }

    /// Decode 16 mixed-width characters from 64 readable bytes, or leave
    /// the caller at the original position for its usual error handling.
    ///
    /// # Safety
    /// `p` must be at a character boundary in a valid UTF-8 string with at
    /// least 64 readable bytes, and `d` must have 30 writable bytes.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
    #[inline]
    pub unsafe fn decode16_mixed_avx512(p: *const u8, d: *mut u8) -> usize {
        use core::arch::x86_64::*;
        // Source offsets and the next character's index, repeated once
        // per output dword. Subtract 1, 2, 3 to read a character backwards.
        const INDICES: ([u8; 64], [u8; 64]) = {
            let mut seq = [0; 64];
            let mut next = [0; 64];
            let mut i = 0;
            while i < 64 {
                seq[i] = i as u8;
                next[i] = (i / 4 + 1) as u8;
                i += 1;
            }
            (seq, next)
        };
        static BLOCK: [u16; 2049] = crate::table::DECODE_MIXED_BLOCK;
        let ld = |p: *const u8| unsafe { _mm512_loadu_si512(p.cast()) };

        // SAFETY: the caller guarantees 64 readable input bytes.
        let v = ld(p);
        // A valid UTF-8 string containing only bytes in 0x80..0xEF has
        // no ASCII or four-byte leads. The last character may be incomplete.
        if _mm512_cmpge_epu8_mask(
            _mm512_add_epi8(v, _mm512_set1_epi8(-128)),
            _mm512_set1_epi8(0x70),
        ) != 0
        {
            return 0;
        }
        let leads = _mm512_test_epi8_mask(v, _mm512_set1_epi8(0x40));
        let starts = _mm512_maskz_compress_epi8(leads, ld(INDICES.0.as_ptr()));
        let ends = _mm512_permutexvar_epi8(ld(INDICES.1.as_ptr()), starts);
        let idx = _mm512_add_epi8(
            ends,
            _mm512_set1_epi32(i32::from_le_bytes([0xff, 0xfe, 0xfd, 0xfc])),
        );
        let bytes = _mm512_permutexvar_epi8(idx, v);
        let w = |x| _mm512_set1_epi32(x);
        // The final byte and its predecessor supply bits 0..11. A
        // continuation in the penultimate position identifies a three-byte
        // character, whose lead supplies the remaining four bits.
        let three = _mm512_testn_epi32_mask(bytes, w(0x4000));
        let high = _mm512_maskz_and_epi32(three, _mm512_srli_epi32::<4>(bytes), w(0xf000));
        let mid = _mm512_ternarylogic_epi32::<0xf8>(high, _mm512_srli_epi32::<2>(bytes), w(0xfc0));
        let cp = _mm512_ternarylogic_epi32::<0xf8>(mid, bytes, w(0x3f));
        // SAFETY: cp is 16 bits. The final u16 of BLOCK pads the last
        // dword gather. Subsequent operations use only its low 16 bits.
        let base = unsafe {
            _mm512_i32gather_epi32::<2>(_mm512_srli_epi32::<5>(cp), BLOCK.as_ptr().cast())
        };
        if _mm512_test_epi32_mask(base, w(0x8000)) != 0 {
            return 0;
        }
        let z = _mm512_ternarylogic_epi32::<0xf8>(base, cp, w(31));
        let z = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(z));
        let out = unsafe { pack_neg_z(_mm512_sub_epi16(_mm512_setzero_si512(), z)) };
        // SAFETY: only the 30 packed bytes from the low 16 lanes are stored.
        unsafe { _mm512_mask_storeu_epi8(d.cast(), (1 << 30) - 1, out) };
        _mm_cvtsi128_si32(_mm512_extracti32x4_epi32::<1>(starts)) as u8 as usize
    }

    /// Decodes up to 32 two-byte 15-bit characters at `p .. p + 64`. Returns
    /// how many leading 8-character groups (0..=4) hold only such
    /// characters; those groups are written to `d` as 15 bytes each, and
    /// nothing else is written.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
    #[inline]
    pub unsafe fn decode32_2byte_avx512(p: *const u8, d: *mut u8) -> usize {
        use crate::table;
        use core::arch::x86_64::*;

        // Each value fits in eight bits. Its set bits occupy two output
        // bytes of their own, except the eighth value which occupies one.
        const fn pack_bytes() -> [u8; 64] {
            let mut order = [0; 64];
            let mut j = 0;
            while j < 60 {
                let i = j % 15;
                order[j] = (16 * (j / 15) + if i == 14 { 14 } else { i ^ 1 }) as u8;
                j += 1;
            }
            order
        }
        static PACK: [u8; 64] = pack_bytes();
        static SHIFT: [[u16; 8]; 4] = [[1, 2, 3, 4, 5, 6, 7, 0]; 4];
        static BLOCK: [u8; 64] = table::DECODE_2BYTE_BLOCK;
        // SAFETY (every use): `p` points to a 64-byte array or to the 64
        // readable input bytes.
        let ld = |p: *const u8| unsafe { _mm512_loadu_si512(p.cast()) };
        let w = |x: i16| _mm512_set1_epi16(x);

        // u16 lane i: continuation byte << 8 | lead byte of character i.
        // Lanes from a character that is not 2 bytes long on are misaligned
        // and decode to garbage, but they are never stored (see `valid`).
        let v = ld(p);
        const EVEN: u64 = 0x5555_5555_5555_5555;
        // Lead byte 0b110xxxxx: (lead ^ 0xC0) & 0xE0 == 0.
        let lead_bits = _mm512_and_si512(_mm512_xor_si512(v, w(0xC0)), w(0xE0));
        let lead_ok = _mm512_testn_epi16_mask(lead_bits, lead_bits);
        // cp >> 5 = (lead & 0x1F) << 1 | bit 5 of the continuation byte (bit
        // 13 of the lane): the low byte of (v << 1) | (v << 2) >> 15
        // (vpshldw), of which vpermb uses the low 6 bits.
        let index = _mm512_shldi_epi16::<1>(v, _mm512_slli_epi16::<2>(v));
        // The z of the block's first character (0..=192), or 0xFF; the high
        // bytes are zeroed.
        let base = _mm512_maskz_permutexvar_epi8(EVEN, index, ld(BLOCK.as_ptr()));
        let valid = lead_ok & _mm512_testn_epi16_mask(base, w(1));
        let groups = (valid.trailing_ones() / 8) as usize;
        if groups == 0 {
            return 0;
        }
        // z = base | the continuation byte's low 5 bits (cp % 32).
        let z = _mm512_ternarylogic_epi32::<0xF8>(base, _mm512_srli_epi16::<8>(v), w(0x1F));

        let words = _mm512_sllv_epi16(z, ld(SHIFT.as_ptr().cast()));
        let out = _mm512_permutexvar_epi8(ld(PACK.as_ptr()), words);
        #[cfg(test)]
        AVX512_2BYTE_BLOCKS.with(|n| n.set(n.get() + 1));
        // SAFETY: the caller guarantees 60 writable bytes at `d`.
        unsafe { _mm512_mask_storeu_epi8(d.cast(), (1u64 << (15 * groups)) - 1, out) };
        groups
    }

    #[cfg(all(test, target_arch = "x86_64"))]
    thread_local! {
        /// Number of 32-character blocks the AVX-512 path decoded on this
        /// thread. Thread-local because tests run in parallel threads: a test
        /// sees only its own calls.
        pub static AVX512_BLOCKS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
        /// Number of 32-character blocks the 2-byte AVX-512 decode path
        /// processed on this thread.
        pub static AVX512_2BYTE_BLOCKS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
        /// Number of 60-byte blocks the AVX-512 path encoded on this thread.
        pub static AVX512_ENCODE_BLOCKS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
        /// Number of 16-character halves the AVX2 encode path stored as all
        /// 3-byte, all 2-byte and mixed characters on this thread.
        pub static AVX2_ENCODE_HALVES: core::cell::Cell<[usize; 3]> = const { core::cell::Cell::new([0; 3]) };
    }

    #[cfg(all(test, target_arch = "x86_64"))]
    fn count_avx2_half(kind: usize, n: usize) {
        AVX2_ENCODE_HALVES.with(|c| {
            let mut v = c.get();
            v[kind] += n;
            c.set(v);
        });
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

    /// Code points of eight 2-byte characters at `p .. p + 16`, or `None` if
    /// any lead byte is not a 2-byte lead (0xC0..=0xDF). Needs only SSE2,
    /// which every x86-64 CPU has.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub unsafe fn code_points_8x2(p: *const u8) -> Option<[u16; 8]> {
        use core::arch::x86_64::*;
        // SAFETY: the caller guarantees `p .. p + 16` is readable; `out` is 16 bytes.
        unsafe {
            let v = _mm_loadu_si128(p.cast());
            let lead = _mm_and_si128(v, _mm_set1_epi16(0xE0));
            if _mm_movemask_epi8(_mm_cmpeq_epi16(lead, _mm_set1_epi16(0xC0))) != 0xFFFF {
                return None;
            }
            let cp = _mm_or_si128(
                _mm_slli_epi16::<6>(_mm_and_si128(v, _mm_set1_epi16(0x1F))),
                _mm_and_si128(_mm_srli_epi16::<8>(v), _mm_set1_epi16(0x3F)),
            );
            let mut out = [0u16; 8];
            _mm_storeu_si128(out.as_mut_ptr().cast(), cp);
            Some(out)
        }
    }

    /// See the x86-64 version. Little-endian only: the u16 lanes hold the
    /// lead byte in their low half.
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[target_feature(enable = "neon")]
    #[inline]
    pub unsafe fn code_points_8x2(p: *const u8) -> Option<[u16; 8]> {
        use core::arch::aarch64::*;
        // SAFETY: the caller guarantees `p .. p + 16` is readable; `out` is 16 bytes.
        unsafe {
            let v = vreinterpretq_u16_u8(vld1q_u8(p));
            let lead = vandq_u16(v, vdupq_n_u16(0xE0));
            if vminvq_u16(vceqq_u16(lead, vdupq_n_u16(0xC0))) == 0 {
                return None;
            }
            let cp = vorrq_u16(
                vshlq_n_u16::<6>(vandq_u16(v, vdupq_n_u16(0x1F))),
                vandq_u16(vshrq_n_u16::<8>(v), vdupq_n_u16(0x3F)),
            );
            let mut out = [0u16; 8];
            vst1q_u16(out.as_mut_ptr(), cp);
            Some(out)
        }
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

    /// Never called, as above.
    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_endian = "little")
    )))]
    #[inline(always)]
    pub unsafe fn code_points_8x2(_p: *const u8) -> Option<[u16; 8]> {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::{DecodeError, LEVEL_AVX512, LEVEL_SCALAR, LEVEL_SIMD, decode_impl, encode_impl};

    /// Encodes with every fast path level, checking that they all agree.
    fn encode_every_level(b: &[u8]) -> String {
        let scalar = encode_impl::<LEVEL_SCALAR>(b);
        assert_eq!(encode_impl::<LEVEL_SIMD>(b), scalar, "SIMD");
        assert_eq!(encode_impl::<LEVEL_AVX512>(b), scalar, "AVX-512");
        scalar
    }

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

    /// `encode_into` and `decode_into` append to existing contents, reuse a
    /// buffer with enough capacity, and leave the buffer unchanged on error.
    #[test]
    fn into_appends_and_reuses() {
        use crate::{decode_into, encode_into};
        let data = random_bytes(5000, 0x9E37_79B9_7F4A_7C15);
        let mut s = String::from("prefix");
        let mut bytes = vec![1u8, 2, 3];
        for n in [0, 1, 14, 15, 16, 29, 30, 59, 60, 61, 95, 96, 97, 1000, 5000] {
            let d = &data[..n];
            let expected = crate::encode(d);

            s.truncate(6);
            encode_into(d, &mut s);
            assert_eq!(&s[6..], expected, "encode {n}");
            assert_eq!(&s[..6], "prefix");

            bytes.truncate(3);
            decode_into(&expected, &mut bytes).unwrap();
            assert_eq!(&bytes[3..], d, "decode {n}");
            assert_eq!(&bytes[..3], [1, 2, 3]);

            // A buffer that is already large enough is not reallocated.
            let (sp, bp) = (s.as_ptr(), bytes.as_ptr());
            s.truncate(6);
            encode_into(d, &mut s);
            bytes.truncate(3);
            decode_into(&expected, &mut bytes).unwrap();
            assert_eq!((s.as_ptr(), bytes.as_ptr()), (sp, bp), "realloc {n}");
        }

        // Errors found after the fast path has written output (4999 bytes end
        // in a 7-bit character, so a second copy after it is an error).
        let good = crate::encode(&data[..4999]);
        for bad in [
            format!("{good}\u{3000}"),
            format!("{good}{good}"),
            format!("{}x", &good[..good.char_indices().last().unwrap().0]),
        ] {
            bytes.truncate(3);
            assert!(decode_into(&bad, &mut bytes).is_err());
            assert_eq!(bytes, [1, 2, 3]);
        }
    }

    #[test]
    fn roundtrip_all_lengths() {
        let data: Vec<u8> = (0..400u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
            .collect();
        for n in 0..data.len() {
            let encoded = encode_every_level(&data[..n]);
            assert_eq!(
                decode_every_level(&encoded).unwrap(),
                &data[..n],
                "length {n}"
            );
        }
    }

    /// Blocks of eight 2-byte characters: all-zero and mostly-zero data, and
    /// a 7-bit or invalid character at each position of a 2-byte run.
    #[test]
    fn two_byte_blocks() {
        for n in 0..400 {
            let zeros = vec![0u8; n];
            let encoded = encode_every_level(&zeros);
            assert_eq!(decode_every_level(&encoded).unwrap(), zeros, "zeros {n}");
            let sparse: Vec<u8> = (0..n)
                .map(|i| if i % 37 == 5 { (i * 151) as u8 | 1 } else { 0 })
                .collect();
            let encoded = encode_every_level(&sparse);
            assert_eq!(decode_every_level(&encoded).unwrap(), sparse, "sparse {n}");
        }

        let zeros: Vec<char> = encode_every_level(&[0; 150]).chars().collect();
        assert_eq!(zeros.len(), 80);
        for i in 0..zeros.len() {
            // U+0180 is a 7-bit character, U+0100 is not in the repertoire;
            // both are 2 bytes, so the block still looks all 2-byte.
            for (c, ok) in [('\u{180}', false), ('\u{100}', false), ('\u{4A1}', true)] {
                let mut chars = zeros.clone();
                chars[i] = c;
                let s: String = chars.into_iter().collect();
                let r = decode_every_level(&s);
                assert_eq!(r.is_ok(), ok || (c == '\u{180}' && i == 79), "{i} {c:?}");
            }
        }
    }

    /// `simd::code_points_8x2` against the code points of every 2-byte
    /// character in every lane, and rejection of a 1- or 3-byte character.
    /// (A wrong code point would only send decode to the slow path, which
    /// the round trip tests cannot see.)
    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_endian = "little")
    ))]
    #[test]
    fn code_points_8x2_every_char() {
        let chars: Vec<char> = (0x80..0x800u32)
            .map(|cp| char::from_u32(cp).unwrap())
            .collect();
        for (k, w) in chars.windows(8).enumerate().step_by(3) {
            let s: String = w.iter().collect();
            // SAFETY: `s` is 16 bytes.
            let cps = unsafe { crate::simd::code_points_8x2(s.as_ptr()) };
            assert_eq!(cps, Some(core::array::from_fn(|i| w[i] as u16)), "{k}");
        }
        for (i, c) in [(0, 'a'), (3, '\u{3000}'), (7, '\u{800}')] {
            let mut w = ['\u{4A0}'; 8];
            w[i] = c;
            let s: String = w.iter().collect::<String>() + "xxxxxxxx";
            // SAFETY: `s` is at least 16 bytes.
            assert_eq!(
                unsafe { crate::simd::code_points_8x2(s.as_ptr()) },
                None,
                "{c:?}"
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
            let encoded = encode_every_level(&data);
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
        let valid = encode_every_level(&[0x55; 60]); // 32 chars
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
            let encoded = encode_every_level(&data);
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
        let valid = encode_every_level(&data);
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
            && std::is_x86_feature_detected!("avx512vbmi2")
            && std::is_x86_feature_detected!("avx512bitalg"))
        {
            eprintln!("AVX-512 BW/VBMI/VBMI2/BITALG not available; skipped");
            return;
        }
        let data: Vec<u8> = (0..3000).map(|i| (i * 7) as u8).collect();
        let encoded = encode_every_level(&data);
        let blocks = || crate::simd::AVX512_BLOCKS.with(|n| n.get());
        let before = blocks();
        assert_eq!(decode_impl::<LEVEL_AVX512>(&encoded).unwrap(), data);
        assert!(blocks() > before);
    }

    /// The 2-byte AVX-512 decode path runs for inputs of 2-byte characters
    /// (runs of zero bytes), but not for short ones.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_2byte_path_runs() {
        if !(std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vbmi")
            && std::is_x86_feature_detected!("avx512vbmi2")
            && std::is_x86_feature_detected!("avx512bitalg"))
        {
            eprintln!("AVX-512 BW/VBMI/VBMI2/BITALG not available; skipped");
            return;
        }
        let blocks = || crate::simd::AVX512_2BYTE_BLOCKS.with(|n| n.get());
        // 500 bytes encode to 534 bytes, below the 1 KiB the path needs.
        for (len, used) in [(3000, true), (500, false)] {
            let data = vec![0u8; len];
            let encoded = encode_every_level(&data);
            let before = blocks();
            assert_eq!(decode_impl::<LEVEL_AVX512>(&encoded).unwrap(), data);
            assert_eq!(blocks() > before, used, "{len}");
        }
    }

    /// All 16-character width patterns, including the first character (whose
    /// unused third-byte index wraps) and a rejected final repertoire block.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn mixed_decode_width_masks() {
        if !(std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vbmi")
            && std::is_x86_feature_detected!("avx512vbmi2"))
        {
            return;
        }
        let filler = "㐀".repeat(16);
        for widths in 0..=u16::MAX {
            let values: [u16; 16] = core::array::from_fn(|i| {
                if widths & (1 << i) == 0 {
                    (13 * i) as u16
                } else {
                    (32767 - 173 * i) as u16
                }
            });
            let expected = pack_z15(&values);
            let mut input = encode_impl::<LEVEL_SCALAR>(&expected);
            let consumed = input.len();
            input.push_str(&filler);
            let mut out = [0xA5; 31];
            let n = unsafe { crate::simd::decode16_mixed_avx512(input.as_ptr(), out.as_mut_ptr()) };
            assert_eq!(n, consumed);
            assert_eq!(&out[..30], expected);
            assert_eq!(out[30], 0xA5);
        }
        for c in ['A', 'ƀ', '￿', '𐀀'] {
            for i in 0..16 {
                let mut input = "Ҡ㐀".repeat(16);
                let start = input.char_indices().nth(i).unwrap().0;
                let end = input.char_indices().nth(i + 1).unwrap().0;
                input.replace_range(start..end, &c.to_string());
                let mut out = [0xA5; 30];
                let n =
                    unsafe { crate::simd::decode16_mixed_avx512(input.as_ptr(), out.as_mut_ptr()) };
                assert_eq!(n, 0);
                assert_eq!(out, [0xA5; 30]);
            }
        }
    }

    /// Inputs of 2-byte characters with an invalid, 7-bit or 3-byte
    /// character in every position: every level agrees, on the 2-byte AVX-512
    /// path too.
    #[test]
    fn two_byte_runs_bad_char_every_position() {
        let mut data = vec![0u8; 3000];
        for i in (0..3000).step_by(397) {
            data[i] = 0xFF;
        }
        let s = encode_every_level(&data);
        assert_eq!(decode_every_level(&s).unwrap(), data);
        let starts: Vec<usize> = s.char_indices().map(|(i, _)| i).collect();
        for bad in ["\u{180}", "A", "\u{3000}", "\u{7FF}"] {
            for &i in &starts {
                let c = s[i..].chars().next().unwrap();
                let t = format!("{}{bad}{}", &s[..i], &s[i + c.len_utf8()..]);
                let _ = decode_every_level(&t);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_encode_path_runs() {
        if !(std::is_x86_feature_detected!("popcnt")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vbmi")
            && std::is_x86_feature_detected!("avx512vbmi2"))
        {
            eprintln!("AVX-512 BW/VBMI/VBMI2 and POPCNT required; skipped");
            return;
        }
        let data: Vec<u8> = (0..3000).map(|i| (i * 7) as u8).collect();
        let blocks = || crate::simd::AVX512_ENCODE_BLOCKS.with(|n| n.get());
        let before = blocks();
        let encoded = encode_impl::<LEVEL_AVX512>(&data);
        assert_eq!(blocks() - before, 3000 / 60);
        assert_eq!(encoded, encode_impl::<LEVEL_SCALAR>(&data));
    }

    /// Every output path of the AVX2 encode block, in both halves: all
    /// 3-byte, all 2-byte, and mixed with every 4-character pattern of 2-byte
    /// characters in every quarter.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_encode_paths() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 not available; skipped");
            return;
        }
        // Bit i set: character i is 2-byte.
        let mut patterns = vec![0u16, 0xFFFF, 0x1234, 0xFEDC, 0x8001, 0x7FFE];
        patterns.extend((1..16).map(|n| n * 0x1111));
        let mut seed = 0u32;
        let mut half = |pattern: u16| -> Vec<u16> {
            (0..16)
                .map(|i| {
                    seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    let r = (seed >> 8) as u16;
                    if pattern >> i & 1 == 1 {
                        r % 224
                    } else {
                        224 + r % (32768 - 224)
                    }
                })
                .collect()
        };
        let halves = || crate::simd::AVX2_ENCODE_HALVES.with(|n| n.get());
        let before = halves();
        for &pa in &patterns {
            for &pb in &patterns {
                let mut values = half(pa);
                values.extend(half(pb));
                let mut data = pack_z15(&values);
                // One more byte so that the block takes the AVX2 path.
                data.push(0xA5);
                let encoded = encode_every_level(&data);
                assert_eq!(
                    decode_every_level(&encoded).unwrap(),
                    data,
                    "{pa:#x} {pb:#x}"
                );
            }
        }
        let after = halves();
        for kind in 0..3 {
            assert!(
                after[kind] > before[kind],
                "kind {kind}: {before:?} {after:?}"
            );
        }
    }

    /// Packs 15-bit values into bytes, most significant bit first.
    fn pack_z15(values: &[u16]) -> Vec<u8> {
        let mut out = vec![0u8; (values.len() * 15).div_ceil(8)];
        for (i, &z) in values.iter().enumerate() {
            for k in 0..15 {
                if z >> (14 - k) & 1 == 1 {
                    let bit = i * 15 + k;
                    out[bit / 8] |= 0x80 >> (bit % 8);
                }
            }
        }
        out
    }

    /// Every 15-bit value in every lane of the 32-character encode block,
    /// and a 2-byte character (z < 224) at every position of a block of
    /// 3-byte characters, alone or next to a second one.
    #[test]
    fn encode_every_value_every_lane() {
        for lane in 0..32u16 {
            let mut values: Vec<u16> = (0..lane).map(|i| 0x7FFF - i).chain(0..32768).collect();
            // Whole 32-character blocks, plus one byte because the AVX2 path
            // needs 61 bytes for a block, so no value is left to the scalar tail.
            values.resize(values.len().next_multiple_of(32), 0x7FFF);
            let mut data = pack_z15(&values);
            data.push(0xA5);
            let encoded = encode_every_level(&data);
            assert_eq!(decode_every_level(&encoded).unwrap(), data, "lane {lane}");
        }
        let three: Vec<u16> = (0..64).map(|i| 1000 + 97 * i).collect();
        for i in 0..64 {
            for j in [None, Some((i + 1) % 64), Some((i + 17) % 64)] {
                let mut values = three.clone();
                values[i] = i as u16;
                if let Some(j) = j {
                    values[j] = 223;
                }
                let mut data = pack_z15(&values);
                // Both blocks of 32 take the AVX2 path too.
                data.push(0xA5);
                let encoded = encode_every_level(&data);
                assert_eq!(decode_every_level(&encoded).unwrap(), data, "{i} {j:?}");
            }
        }
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
        let encoded = encode_every_level(&data);
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
            // Whole 32-character blocks, so every character takes the widest
            // path; a multiple of 8 characters is also a whole number of bytes
            // with no padding.
            let n = shift + chars.len();
            s.extend(std::iter::repeat_n(filler, n.next_multiple_of(32) - n));
            let decoded = decode_every_level(&s).unwrap();
            assert_eq!(encode_every_level(&decoded), s, "shift {shift}");
        }
    }

    /// Every 3-byte code point outside the repertoire, in lane cp % 32 of an
    /// otherwise valid 64-character input.
    #[test]
    fn every_invalid_3byte_code_point() {
        let valid: Vec<char> = encode_every_level(&random_bytes(120, 7)).chars().collect();
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

    /// The half page tables used by the AVX-512 decode path, evaluated in
    /// scalar code as there, agree with `DECODE` for every 3-byte code point.
    #[test]
    fn half_page_tables_match_decode_table() {
        use crate::table::{DECODE, END_OFFSET_BIASED, HALF_DESC_BIASED, MISSING_MASK};
        for cp in 0x800..=0xFFFFu32 {
            let t = 64 * (0xE0 | cp >> 12) + (0x80 | (cp >> 6 & 0x3F));
            let entry = HALF_DESC_BIASED[(t >> 3 & 127) as usize] as usize;
            let cp_adj = (t << 6 | (cp & 0x3F)) & 0xFFFF;
            // vpshrdvw with zero: missing blocks from this one through the
            // end of the half page.
            let missing = MISSING_MASK[entry] as u32 >> (cp_adj >> 5 & 15);
            let expected = DECODE[cp as usize];
            if missing & 1 != 0 {
                assert!(
                    expected >= 0x8000,
                    "U+{cp:04X} should not be a 15-bit character"
                );
                continue;
            }
            let z = cp_adj
                .wrapping_sub(END_OFFSET_BIASED[entry] as u32)
                .wrapping_add(32 * missing.count_ones())
                & 0xFFFF;
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
        let encode_at_end = |b: &[u8]| {
            // SAFETY: `b` fits in the accessible first half.
            let copy = unsafe {
                let dst = end.sub(b.len());
                std::ptr::copy_nonoverlapping(b.as_ptr(), dst, b.len());
                std::slice::from_raw_parts(dst, b.len())
            };
            encode_every_level(copy)
        };

        let random = random_bytes(600, 11);
        // Zero bytes encode to 2-byte characters, which take the fallback paths.
        let mixed: Vec<u8> = random
            .iter()
            .enumerate()
            .map(|(i, &b)| if i % 23 < 4 { 0 } else { b })
            .collect();
        // Runs of zeros give blocks of eight 2-byte characters.
        let sparse: Vec<u8> = random
            .iter()
            .enumerate()
            .map(|(i, &b)| if i % 64 == 0 { b } else { 0 })
            .collect();
        for data in [&random, &mixed, &sparse, &vec![0; 600]] {
            for n in 0..=data.len() {
                let encoded = encode_at_end(&data[..n]);
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
