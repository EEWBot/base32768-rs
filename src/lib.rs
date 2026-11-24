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

const BLOCK_MASK: u32 = (1 << table::BLOCK_BITS) - 1;

/// `table::DECODE_BLOCK` value for a code point that is not a Base32768 character.
const INVALID: u32 = 0xFFFF;
/// Set in `table::DECODE_BLOCK` values of 7-bit characters (and of `INVALID`).
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

        // SAFETY: at most 8 * 3 + 1 bytes per chunk; covered by the capacity.
        unsafe {
            pos = put(dst, pos, z15_utf8((hi >> 49) as u32));
            pos = put(dst, pos, z15_utf8((hi >> 34) as u32 & 0x7FFF));
            pos = put(dst, pos, z15_utf8((hi >> 19) as u32 & 0x7FFF));
            pos = put(dst, pos, z15_utf8((hi >> 4) as u32 & 0x7FFF));
            // `lo` starts at byte 7, so it already holds `hi`'s low 4 bits.
            pos = put(dst, pos, z15_utf8((lo >> 45) as u32 & 0x7FFF));
            pos = put(dst, pos, z15_utf8((lo >> 30) as u32 & 0x7FFF));
            pos = put(dst, pos, z15_utf8((lo >> 15) as u32 & 0x7FFF));
            pos = put(dst, pos, z15_utf8(lo as u32 & 0x7FFF));
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
    // The table spans all of U+0000..=U+FFFF, so the mask removes the bounds
    // check. Block bases are multiples of 32, so OR-ing in the offset equals
    // adding it, and INVALID (0xFFFF) stays INVALID.
    let base = table::DECODE_BLOCK[(cp >> table::BLOCK_BITS) as usize & 0x7FF] as u32;
    base | (cp & BLOCK_MASK)
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
    let bytes = s.as_bytes();
    let len = bytes.len();
    // Every character is at least 2 bytes and carries at most 15 bits.
    let mut out: Vec<u8> = Vec::with_capacity(len / 2 * 15 / 8 + 1);

    let mut pos = 0usize;
    let mut char_index = 0usize;

    // Fast path: 8 characters (120 bits) -> 15 bytes, while 8 characters of up
    // to 3 bytes each are guaranteed to be readable. Bails out to the exact
    // path below on anything other than eight 15-bit characters.
    {
        let src = bytes.as_ptr();
        let dst = out.as_mut_ptr();
        let mut written = 0usize;

        while pos + 24 <= len {
            // SAFETY: each read covers at most 3 bytes and advances by at most 3,
            // so all reads stay within `pos .. pos + 24 <= len`.
            let (v0, v1, v2, v3, v4, v5, v6, v7, next) = unsafe {
                let p = src.add(pos);

                // Common case (~95% of blocks for random data, since 99.3% of
                // 15-bit characters are 3 bytes): eight 3-byte characters at
                // fixed offsets, so the loads and lookups do not depend on
                // each other.
                let (l0, l1, l2, l3) = (*p, *p.add(3), *p.add(6), *p.add(9));
                let (l4, l5, l6, l7) = (*p.add(12), *p.add(15), *p.add(18), *p.add(21));
                let and = l0 & l1 & l2 & l3 & l4 & l5 & l6 & l7;
                let or = l0 | l1 | l2 | l3 | l4 | l5 | l6 | l7;
                if and >= 0xE0 && or <= 0xEF {
                    (
                        read_3byte(p),
                        read_3byte(p.add(3)),
                        read_3byte(p.add(6)),
                        read_3byte(p.add(9)),
                        read_3byte(p.add(12)),
                        read_3byte(p.add(15)),
                        read_3byte(p.add(18)),
                        read_3byte(p.add(21)),
                        pos + 24,
                    )
                } else {
                    let (v0, n0) = read_z15(p);
                    let (v1, n1) = read_z15(p.add(n0));
                    let o = n0 + n1;
                    let (v2, n2) = read_z15(p.add(o));
                    let o = o + n2;
                    let (v3, n3) = read_z15(p.add(o));
                    let o = o + n3;
                    let (v4, n4) = read_z15(p.add(o));
                    let o = o + n4;
                    let (v5, n5) = read_z15(p.add(o));
                    let o = o + n5;
                    let (v6, n6) = read_z15(p.add(o));
                    let o = o + n6;
                    let (v7, n7) = read_z15(p.add(o));
                    (v0, v1, v2, v3, v4, v5, v6, v7, pos + o + n7)
                }
            };

            if (v0 | v1 | v2 | v3 | v4 | v5 | v6 | v7) & FLAG7 != 0 {
                break;
            }

            let (v0, v1, v2, v3) = (v0 as u64, v1 as u64, v2 as u64, v3 as u64);
            let (v4, v5, v6, v7) = (v4 as u64, v5 as u64, v6 as u64, v7 as u64);
            let w0 = (v0 << 49) | (v1 << 34) | (v2 << 19) | (v3 << 4) | (v4 >> 11);
            // Starts at output byte 7, like `lo` in `encode`.
            let w1 = (v3 << 60) | (v4 << 45) | (v5 << 30) | (v6 << 15) | v7;

            // SAFETY: 16 input bytes or more were consumed for these 15 output
            // bytes, so `written + 15 <= len / 2 * 15 / 8 <= capacity`.
            unsafe {
                let d = dst.add(written);
                d.cast::<[u8; 8]>().write_unaligned(w0.to_be_bytes());
                d.add(7).cast::<[u8; 8]>().write_unaligned(w1.to_be_bytes());
            }

            written += 15;
            pos = next;
            char_index += 8;
        }

        // SAFETY: `written` bytes were initialised above.
        unsafe { out.set_len(written) };
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
