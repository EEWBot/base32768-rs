//! The default implementation: scalar code with unchecked buffer writes and
//! SIMD fast paths (SSSE3, AVX2 and AVX-512 on x86-64, NEON on aarch64).

use crate::{DecodeError, FLAG7, INVALID, decoded_capacity, encoded_chars, simd, table};

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

/// At least 24 of the 32 15-bit values in the 60 bytes are below 224
/// (2-byte characters).
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[inline]
fn two_byte_dense_values(head: &[u8]) -> bool {
    let (chunks, _) = head.as_chunks::<15>();
    let count: usize = chunks
        .iter()
        .map(|c| {
            let hi = u64::from_be_bytes(c[0..8].try_into().unwrap());
            let lo = u64::from_be_bytes(c[7..15].try_into().unwrap());
            [
                hi >> 49,
                hi >> 34,
                hi >> 19,
                hi >> 4,
                lo >> 45,
                lo >> 30,
                lo >> 15,
                lo,
            ]
            .iter()
            .filter(|&&z| z & 0x7FFF < 224)
            .count()
        })
        .sum();
    count >= 24
}

/// The scalar code for 15 bytes -> 8 characters at `dst + pos`.
///
/// # Safety
/// `dst + pos .. dst + pos + 25` must be writable.
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[inline(always)]
unsafe fn encode_chunk(chunk: &[u8; 15], dst: *mut u8, mut pos: usize) -> usize {
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
    pos
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

    // NEON: inputs of mostly 2-byte characters (runs of zero bytes), 30
    // bytes -> 16 characters per window. Other inputs take only the scalar
    // path, which is faster for 3-byte characters than NEON.
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if LEVEL >= LEVEL_SIMD && b.len() >= 1024 && two_byte_dense_values(&b[..60]) {
        // In its own function, like the 2-byte AVX-512 decode path, so that
        // the scalar loop below compiles as without it.
        #[inline(never)]
        unsafe fn encode_neon_dense(b: &[u8], dst: *mut u8) -> (usize, usize) {
            let (mut done, mut pos) = (0, 0);
            while done + 32 <= b.len() {
                // SAFETY: 32 bytes are readable at `done`, and the 30 bytes
                // encoded reserve 16 * 3 = 48 output bytes.
                if unsafe { simd::encode16_2byte_short_neon(b.as_ptr().add(done), dst.add(pos)) } {
                    pos += 32;
                    done += 30;
                    continue;
                }
                // Some 3-byte characters: the scalar code, 15 bytes at a time.
                for chunk in b[done..done + 30].as_chunks::<15>().0 {
                    // SAFETY: as in the scalar path.
                    pos = unsafe { encode_chunk(chunk, dst, pos) };
                }
                done += 30;
            }
            (done, pos)
        }
        // SAFETY: see above; `dst` has the capacity for all of `b`.
        (done, pos) = unsafe { encode_neon_dense(b, dst) };
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
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        if SIMD && pos + 48 <= len {
            // 16 three-byte characters, or (in the dense instance only) 16
            // two-byte ones after a 2-byte lead.
            // SAFETY: `pos + 48 <= len`. Each group consumes at least 16
            // input bytes, which account for its 15 bytes of `dst` capacity;
            // the 16th byte stored is within the capacity of the rest of
            // the 48 bytes.
            let two = DENSE && unsafe { *src.add(pos) } < 0xE0;
            // Bytes taken: 16 characters in 32 (2-byte), 47 or 48 bytes, or
            // the first 8 in 16 or 24.
            let n = unsafe {
                if two {
                    16 * simd::decode16_2byte_neon(src.add(pos), dst.add(written))
                } else {
                    // Inputs of 2-byte characters: no 47-byte windows.
                    simd::decode16_neon(src.add(pos), dst.add(written), !DENSE)
                }
            };
            // As for AVX-512, a whole window keeps the next position
            // independent of the vector computation.
            if n >= 32 {
                pos += n;
                written += 30;
                continue;
            }
            // The 8-character path below takes the rejected group.
            if n != 0 {
                pos += n;
                written += 15;
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
/// At least 24 of the 64 bytes are 2-byte UTF-8 leads (runs of zero bytes
/// in the input data).
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[inline]
fn two_byte_dense(head: &[u8]) -> bool {
    head.iter().filter(|&&b| b & 0xE0 == 0xC0).count() >= 24
}

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
            let (pos, written) = if bytes.len() >= 4096
                && std::is_x86_feature_detected!("bmi2")
                && std::is_x86_feature_detected!("popcnt")
            {
                simd::decode32_windows(bytes, dst)
            } else {
                (0, 0)
            };
            let remaining = &bytes[pos..];
            let tail_dst = dst.add(written);
            let (p, w, c) = if remaining.len() >= 1024 && two_byte_dense(remaining.as_ptr()) {
                decode_fast_avx512_dense(remaining, tail_dst)
            } else {
                decode_fast_avx512(remaining, tail_dst)
            };
            return (pos + p, written + w, written / 15 * 8 + c);
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
        // As for AVX-512, two functions: with the 2-byte path in the same
        // function, the 3-byte loop is 10-20% slower on random data.
        #[inline(never)]
        unsafe fn decode_fast_neon(bytes: &[u8], dst: *mut u8) -> (usize, usize, usize) {
            unsafe { decode_fast::<true, false, false>(bytes, dst) }
        }
        #[inline(never)]
        unsafe fn decode_fast_neon_dense(bytes: &[u8], dst: *mut u8) -> (usize, usize, usize) {
            unsafe { decode_fast::<true, false, true>(bytes, dst) }
        }
        // SAFETY: NEON is part of the aarch64 baseline.
        return unsafe {
            if bytes.len() >= 1024 && two_byte_dense(&bytes[..64]) {
                decode_fast_neon_dense(bytes, dst)
            } else {
                decode_fast_neon(bytes, dst)
            }
        };
    }

    unsafe { decode_fast::<false, false, false>(bytes, dst) }
}

#[cfg(test)]
mod tests {
    use super::{LEVEL_AVX512, LEVEL_SCALAR, LEVEL_SIMD, decode_impl, encode_impl};
    use crate::DecodeError;

    /// Encodes with every fast path level, checking that they all agree with
    /// each other and with the safe-only implementation.
    fn encode_every_level(b: &[u8]) -> String {
        let scalar = encode_impl::<LEVEL_SCALAR>(b);
        assert_eq!(encode_impl::<LEVEL_SIMD>(b), scalar, "SIMD");
        assert_eq!(encode_impl::<LEVEL_AVX512>(b), scalar, "AVX-512");
        assert_eq!(crate::safe::encode(b), scalar, "safe-only");
        scalar
    }

    /// Decodes with every fast path level, checking that they all agree with
    /// each other and with the safe-only implementation (also on errors).
    fn decode_every_level(s: &str) -> Result<Vec<u8>, DecodeError> {
        let scalar = decode_impl::<LEVEL_SCALAR>(s);
        for (level, r) in [
            ("SIMD", decode_impl::<LEVEL_SIMD>(s)),
            ("AVX-512", decode_impl::<LEVEL_AVX512>(s)),
            ("safe-only", crate::safe::decode(s)),
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
            let mut safe = String::from("prefix");
            crate::safe::encode_into(d, &mut safe);
            assert_eq!(s, safe, "safe-only encode {n}");

            bytes.truncate(3);
            decode_into(&expected, &mut bytes).unwrap();
            assert_eq!(&bytes[3..], d, "decode {n}");
            assert_eq!(&bytes[..3], [1, 2, 3]);
            let mut safe = vec![1u8, 2, 3];
            crate::safe::decode_into(&expected, &mut safe).unwrap();
            assert_eq!(bytes, safe, "safe-only decode {n}");

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
            let err = decode_into(&bad, &mut bytes).unwrap_err();
            assert_eq!(bytes, [1, 2, 3]);
            let mut safe = vec![1u8, 2, 3];
            let safe_err = crate::safe::decode_into(&bad, &mut safe).unwrap_err();
            assert_eq!(format!("{err:?}"), format!("{safe_err:?}"));
            assert_eq!(safe, [1, 2, 3]);
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

    /// Enumerate the widths of each half independently. Changing the first
    /// half also moves the second half relative to the two overlapping loads.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn decode32_mixed_boundaries() {
        if !(std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vbmi")
            && std::is_x86_feature_detected!("avx512vbmi2")
            && std::is_x86_feature_detected!("avx512bitalg")
            && std::is_x86_feature_detected!("bmi2")
            && std::is_x86_feature_detected!("popcnt"))
        {
            return;
        }
        for half in 0..=u16::MAX {
            // Multiplication by an odd number permutes all 16-bit masks.
            let widths = u32::from(half) | (u32::from(half.wrapping_mul(40503)) << 16);
            let values: [u16; 32] = core::array::from_fn(|i| {
                let seed = (u32::from(half) + 1999 * i as u32) as u16;
                if widths & (1u32 << i) != 0 {
                    seed % 224
                } else {
                    224 + seed % (32768 - 224)
                }
            });
            let expected = pack_z15(&values);
            let mut input = encode_impl::<LEVEL_SCALAR>(&expected);
            let consumed = input.len();
            input.push_str(&"㐀".repeat(16));
            let mut out = [0xA5; 144];
            // SAFETY: valid UTF-8, at most 144 bytes, and a larger output
            // buffer than the worst-case binary length.
            let result =
                unsafe { crate::simd::decode32_windows(input.as_bytes(), out.as_mut_ptr()) };
            assert_eq!(result, (consumed, 60));
            assert_eq!(&out[..60], expected);
            assert_eq!(out[60], 0xA5);
        }

        // The fast loop has already committed an output prefix when an
        // invalid or secondary character appears in the middle of a window.
        let data: Vec<u8> = (0..6000)
            .map(|i| if i % 16 < 2 { 0 } else { (i * 37) as u8 })
            .collect();
        let input = encode_impl::<LEVEL_SCALAR>(&data);
        for (pos, c) in input.char_indices().skip(64).take(96) {
            for bad in ['A', 'ƀ', '￿', '𐀀'] {
                let mut damaged = input.clone();
                damaged.replace_range(pos..pos + c.len_utf8(), &bad.to_string());
                assert!(decode_every_level(&damaged).is_err());
                let mut dst = vec![1, 2, 3];
                assert!(crate::decode_into(&damaged, &mut dst).is_err());
                assert_eq!(dst, [1, 2, 3]);
            }
        }
    }

    /// The mixed-width windows look up 2-byte characters in the half page
    /// tables (below U+0800, which build.rs does not check): every 2-byte
    /// code point in every lane decodes as in scalar code.
    #[test]
    fn mixed_windows_every_2byte_code_point() {
        let data: Vec<u8> = (0..4000)
            .map(|i| if i % 16 < 2 { 0 } else { (i * 37) as u8 })
            .collect();
        let input = encode_impl::<LEVEL_SCALAR>(&data);
        assert!(input.len() >= 4096 + 96);
        let chars: Vec<(usize, char)> = input.char_indices().collect();
        for cp in 0x80..0x800u32 {
            let c = char::from_u32(cp).unwrap().to_string();
            for lane in 0..32 {
                let (pos, old) = chars[1024 + 32 * (cp as usize % 7) + lane];
                let mut s = input.clone();
                s.replace_range(pos..pos + old.len_utf8(), &c);
                assert_eq!(
                    format!("{:?}", decode_impl::<LEVEL_AVX512>(&s)),
                    format!("{:?}", decode_impl::<LEVEL_SCALAR>(&s)),
                    "U+{cp:04X} lane {lane}"
                );
            }
        }
    }

    /// Windows with exactly one 2-byte character, in every lane, followed
    /// by a 2- or 3-byte character at byte 95 (the next window's first).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn decode32_one_short_every_lane() {
        if !(std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vbmi")
            && std::is_x86_feature_detected!("avx512vbmi2")
            && std::is_x86_feature_detected!("avx512bitalg")
            && std::is_x86_feature_detected!("bmi2")
            && std::is_x86_feature_detected!("popcnt"))
        {
            return;
        }
        for k in 0..32 {
            for next in [5u16, 3000] {
                let values: [u16; 32] = core::array::from_fn(|i| {
                    if i == k {
                        100 + i as u16
                    } else {
                        300 + 999 * i as u16
                    }
                });
                let expected = pack_z15(&values);
                let head = encode_impl::<LEVEL_SCALAR>(&expected);
                let next = encode_impl::<LEVEL_SCALAR>(&pack_z15(&[next; 8]));
                let next = next.chars().next().unwrap();
                let input = format!("{head}{next}{}", "㐀".repeat(16));
                let mut out = [0xA5; 144];
                // SAFETY: valid UTF-8 and a large enough output buffer.
                let result =
                    unsafe { crate::simd::decode32_windows(input.as_bytes(), out.as_mut_ptr()) };
                assert_eq!(result, (head.len(), 60), "{k} {next}");
                assert_eq!(&out[..60], expected, "{k} {next}");
                // A 7-bit or invalid character as the short one.
                for bad in ['ƀ', '߿'] {
                    let pos = head.char_indices().nth(k).unwrap().0;
                    let mut damaged = head.clone();
                    damaged.replace_range(pos..pos + 2, &bad.to_string());
                    let input = format!("{damaged}{next}{}", "㐀".repeat(16));
                    let result = unsafe {
                        crate::simd::decode32_windows(input.as_bytes(), out.as_mut_ptr())
                    };
                    assert_eq!(result, (0, 0), "{k} {bad:?}");
                }
            }
        }
    }

    /// `decode32_windows` on the 32 values and what follows them, with
    /// "\u{3400}" padding so that no second window starts. Returns the
    /// result and the first 60 bytes written.
    #[cfg(target_arch = "x86_64")]
    fn one_window(values: &[u16; 32], next: &str) -> ((usize, usize), Vec<u8>) {
        let head = encode_impl::<LEVEL_SCALAR>(&pack_z15(values));
        let input = format!("{head}{next}{}", "\u{3400}".repeat(16));
        let mut out = [0xA5; 144];
        // SAFETY: valid UTF-8 and a large enough output buffer.
        let result = unsafe { crate::simd::decode32_windows(input.as_bytes(), out.as_mut_ptr()) };
        (result, out[..60].to_vec())
    }

    #[cfg(target_arch = "x86_64")]
    fn avx512_decode_available() -> bool {
        std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vbmi")
            && std::is_x86_feature_detected!("avx512vbmi2")
            && std::is_x86_feature_detected!("avx512bitalg")
            && std::is_x86_feature_detected!("bmi2")
            && std::is_x86_feature_detected!("popcnt")
    }

    /// Several 2-byte characters whose leads still leave `aligned` a run of
    /// low ones (every lead from the first short character on is off the
    /// fixed positions): the 95-byte path must reject them, and the general
    /// path decodes them.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn decode32_several_short_contiguous_aligned() {
        if !avx512_decode_available() {
            return;
        }
        for k in 0..32 {
            for others in [
                vec![k + 1],
                vec![k + 2],
                vec![31],
                vec![k + 1, k + 2],
                vec![k + 3, 30],
            ] {
                let mut shorts = vec![k];
                shorts.extend(others.into_iter().filter(|&j| j > k && j < 32));
                shorts.dedup();
                if shorts.len() < 2 {
                    continue;
                }
                let values: [u16; 32] = core::array::from_fn(|i| {
                    if shorts.contains(&i) {
                        7 * i as u16
                    } else {
                        300 + 999 * i as u16
                    }
                });
                let len = encode_impl::<LEVEL_SCALAR>(&pack_z15(&values)).len();
                assert_eq!(len, 96 - shorts.len());
                let count = || crate::simd::AVX512_ONE_SHORT.with(|n| n.get());
                let before = count();
                let (result, out) = one_window(&values, "");
                assert_eq!(count(), before, "{shorts:?} took the 95-byte path");
                assert_eq!(result, (len, 60), "{shorts:?}");
                assert_eq!(out, pack_z15(&values), "{shorts:?}");
            }
        }
    }

    /// One 2-byte character, then a character at byte 95 that is ASCII, a
    /// 4-byte character or a 7-bit (final) character: the window takes its
    /// 95 bytes and leaves that character to the code after it.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn decode32_one_short_then_other_characters() {
        if !avx512_decode_available() {
            return;
        }
        for k in [0, 1, 15, 16, 30, 31] {
            let values: [u16; 32] =
                core::array::from_fn(|i| if i == k { 100 } else { 300 + 999 * i as u16 });
            for next in ["A", "\u{10000}", "\u{180}"] {
                let count = || crate::simd::AVX512_ONE_SHORT.with(|n| n.get());
                let before = count();
                let (result, out) = one_window(&values, next);
                assert_eq!(count(), before + 1, "{k} {next:?}");
                assert_eq!(result, (95, 60), "{k} {next:?}");
                assert_eq!(out, pack_z15(&values), "{k} {next:?}");
            }
            // Whole inputs through the dispatch (at least 4096 bytes), the
            // window after 43 windows of 3-byte characters: results and
            // errors as in scalar code.
            let mut all: Vec<u16> = (0..43 * 32).map(|i| 300 + (i % 30000) as u16).collect();
            all.extend_from_slice(&values);
            let head = encode_impl::<LEVEL_SCALAR>(&pack_z15(&all));
            assert!(head.len() >= 4096);
            for next in ["A", "\u{10000}", "\u{180}", "\u{3400}\u{180}"] {
                let _ = decode_every_level(&format!("{head}{next}"));
            }
        }
    }

    /// Windows that alternate between one 2-byte character (95 bytes) and
    /// none (96 bytes), through the dispatch: the 95-byte path runs.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn decode32_alternating_95_and_96() {
        if !avx512_decode_available() {
            return;
        }
        let values: Vec<u16> = (0..64 * 32)
            .map(|i| {
                let (block, lane) = (i / 32, i % 32);
                if block % 2 == 0 && lane == block / 2 % 32 {
                    50
                } else {
                    300 + (i * 7 % 30000) as u16
                }
            })
            .collect();
        let data = pack_z15(&values);
        let encoded = encode_every_level(&data);
        assert!(encoded.len() >= 4096);
        let count = || crate::simd::AVX512_ONE_SHORT.with(|n| n.get());
        let before = count();
        assert_eq!(decode_every_level(&encoded).unwrap(), data);
        assert!(count() - before >= 32, "{}", count() - before);
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

    /// Long inputs of mostly 2-byte characters (values below 224, not only
    /// zeros), with values of 224 and above (3-byte characters, including
    /// ones that differ from a 2-byte value in one bit) at every position:
    /// every level agrees.
    #[test]
    fn two_byte_values_with_3byte_at_every_position() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let base: Vec<u16> = (0..1600).map(|_| (next() % 224) as u16).collect();
        let encoded = encode_every_level(&pack_z15(&base));
        assert_eq!(decode_every_level(&encoded).unwrap(), pack_z15(&base));
        for i in 0..64 {
            for big in [224, 255, 256, 1 << (i % 15), 0x7FFF] {
                let mut values = base.clone();
                values[100 + i] = big.max(224);
                let data = pack_z15(&values);
                assert_eq!(
                    decode_every_level(&encode_every_level(&data)).unwrap(),
                    data
                );
            }
        }
    }

    /// The NEON decode paths run: 3-byte windows for random data, and the
    /// 2-byte path for long inputs of 2-byte characters only.
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[test]
    fn neon_decode_paths_run() {
        let count = |k: usize| crate::simd::NEON_DECODE_WINDOWS.with(|n| n[k].get());
        let random = random_bytes(3000, 0x1234_5678);
        let encoded = encode_every_level(&random);
        let before = count(0);
        assert_eq!(decode_impl::<LEVEL_SIMD>(&encoded).unwrap(), random);
        assert!(count(0) > before);
        // 500 bytes encode to 534 bytes, below the 1 KiB the path needs.
        for (len, used) in [(3000, true), (500, false)] {
            let zeros = vec![0u8; len];
            let encoded = encode_every_level(&zeros);
            let before = count(1);
            assert_eq!(decode_impl::<LEVEL_SIMD>(&encoded).unwrap(), zeros);
            assert_eq!(count(1) > before, used, "{len}");
        }
    }

    /// The 47-byte NEON window: one 2-byte character in each of the 16
    /// lanes, followed by ASCII, a 4-byte, a 7-bit or a 3-byte character;
    /// a 7-bit or invalid character as the short one; and two 2-byte
    /// characters at every pair of lanes, which the path must leave to the
    /// others. Every level agrees with scalar code.
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[test]
    fn neon_one_short_windows() {
        let count = || crate::simd::NEON_DECODE_WINDOWS.with(|n| n[2].get());
        for k in 0..16 {
            let values: [u16; 16] = core::array::from_fn(|i| {
                if i == k {
                    100 + i as u16
                } else {
                    300 + 999 * i as u16
                }
            });
            let head = encode_impl::<LEVEL_SCALAR>(&pack_z15(&values));
            assert_eq!(head.len(), 47);
            for next in ["A", "\u{10000}", "\u{180}", "\u{3400}\u{3400}"] {
                let before = count();
                let _ = decode_every_level(&format!("{head}{next}"));
                assert!(count() > before, "{k} {next:?}");
            }
            let pos = head.char_indices().nth(k).unwrap().0;
            for bad in ['\u{180}', '\u{7FF}'] {
                let mut damaged = head.clone();
                damaged.replace_range(pos..pos + 2, &bad.to_string());
                let before = count();
                assert!(decode_every_level(&format!("{damaged}\u{3400}")).is_err());
                assert_eq!(count(), before, "{k} {bad:?}");
            }
            for j in k + 1..16 {
                let mut values = values;
                values[j] = 7 * j as u16;
                let head = encode_impl::<LEVEL_SCALAR>(&pack_z15(&values));
                let before = count();
                let data = decode_every_level(&format!("{head}{head}")).unwrap();
                assert_eq!(&data[..30], pack_z15(&values));
                assert_eq!(count(), before, "{k} {j}");
            }
        }
    }

    /// The NEON 2-byte encode path runs for long inputs of 2-byte
    /// characters only.
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[test]
    fn neon_encode_path_runs() {
        let windows = || crate::simd::NEON_ENCODE_WINDOWS.with(|n| n.get());
        // 500 bytes are below the 1 KiB the path needs.
        for (len, used) in [(3000, true), (500, false)] {
            let zeros = vec![0u8; len];
            let before = windows();
            let encoded = encode_impl::<LEVEL_SIMD>(&zeros);
            assert_eq!(windows() > before, used, "{len}");
            assert_eq!(decode_impl::<LEVEL_SIMD>(&encoded).unwrap(), zeros);
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
        // Exercise the long-input decoder with its last 96-byte window
        // at different distances from the guard page.
        let mut long = random_bytes(5000, 17);
        for i in (0..long.len() - 1).step_by(16) {
            long[i..i + 2].fill(0);
        }
        for n in 4900..=5000 {
            let text = encode_at_end(&long[..n]);
            assert_eq!(decode_at_end(&text).unwrap(), &long[..n]);
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
                    super::encoded_capacity(len) as u128,
                    enc,
                    "encoded_capacity({len})"
                );
            } else {
                assert!(std::panic::catch_unwind(|| super::encoded_capacity(len)).is_err());
            }
            assert_eq!(
                crate::decoded_capacity(len) as u128,
                l / 2 * 15 / 8 + 1,
                "decoded_capacity({len})"
            );
        }
    }
}
