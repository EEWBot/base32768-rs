//! The `safe-only` implementation: safe Rust alone, with the same results
//! and errors as the default one. Without the feature it is built only for
//! tests, which check the default implementation against it.
#![forbid(unsafe_code)]

use crate::{DecodeError, FLAG7, INVALID, decoded_capacity, encoded_chars, table};

/// The character for the 15-bit value `z`.
#[inline(always)]
fn z15_char(z: u64) -> char {
    table::Z15_CHAR[z as usize & 0x7FFF]
}

/// Encodes `b` as a new `String`.
pub fn encode(b: &[u8]) -> String {
    let mut out = String::with_capacity(encoded_chars(b.len()).saturating_mul(3));
    encode_into(b, &mut out);
    out
}

/// Appends the encoding of `b` to `out`, reusing its allocation when it has
/// room for about `b.len() * 8 / 5` more bytes.
pub fn encode_into(b: &[u8], out: &mut String) {
    out.reserve(encoded_chars(b.len()).saturating_mul(3));

    // 15 bytes -> 8 characters.
    let (chunks, remainder) = b.as_chunks::<15>();
    for chunk in chunks {
        let hi = u64::from_be_bytes(chunk[0..8].try_into().unwrap());
        // `lo` starts at byte 7, so it already holds `hi`'s low 4 bits.
        let lo = u64::from_be_bytes(chunk[7..15].try_into().unwrap());
        for z in [
            hi >> 49,
            hi >> 34,
            hi >> 19,
            hi >> 4,
            lo >> 45,
            lo >> 30,
            lo >> 15,
            lo,
        ] {
            out.push(z15_char(z));
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
            out.push(z15_char(acc >> bits));
            acc &= (1 << bits) - 1;
        }
    }

    // Pad the final partial character with 1 bits.
    if bits > 0 {
        out.push(if bits <= 7 {
            let pad = 7 - bits;
            table::Z7_CHAR[((acc << pad) | ((1 << pad) - 1)) as usize & 0x7F]
        } else {
            let pad = 15 - bits;
            z15_char((acc << pad) | ((1 << pad) - 1))
        });
    }
}

/// Decodes `s` into a new `Vec`.
pub fn decode(s: &str) -> Result<Vec<u8>, DecodeError> {
    let mut out = Vec::with_capacity(decoded_capacity(s.len()));
    decode_append(s, &mut out)?;
    Ok(out)
}

/// Appends the decoding of `s` to `out`, reusing its allocation when it has
/// room for about `s.len() * 15 / 16 + 1` more bytes. On error `out` is left
/// as it was (its capacity may have grown).
pub fn decode_into(s: &str, out: &mut Vec<u8>) -> Result<(), DecodeError> {
    let start = out.len();
    let r = decode_append(s, out);
    if r.is_err() {
        out.truncate(start);
    }
    r
}

/// Appends the decoding of `s` to `out`, leaving a partial result on error.
fn decode_append(s: &str, out: &mut Vec<u8>) -> Result<(), DecodeError> {
    out.reserve(decoded_capacity(s.len()));
    let mut acc = 0u64;
    let mut bits = 0u32;

    for (index, (pos, c)) in s.char_indices().enumerate() {
        // `INVALID` for characters outside the Basic Multilingual Plane.
        let v = table::DECODE.get(c as usize).map_or(INVALID, |&v| v as u32);
        if v == INVALID {
            return Err(DecodeError::UnrecognizedCharacter(c));
        }
        let num_bits = if v & FLAG7 != 0 {
            // A 7-bit character may only end the input.
            if pos + c.len_utf8() != s.len() {
                return Err(DecodeError::UnexpectedSecondaryCharacter(index));
            }
            7
        } else {
            15
        };

        acc = (acc << num_bits) | (v & 0x7FFF) as u64;
        bits += num_bits;
        while bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
        acc &= (1 << bits) - 1;
    }

    if acc != (1 << bits) - 1 {
        return Err(DecodeError::PaddingMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{decode, decode_into, encode, encode_into};
    use crate::DecodeError;

    #[test]
    fn round_trip_and_known_values() {
        let bin = [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100];
        assert_eq!(encode(&bin), "媒腻㐤┖ꈳ埳");
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let data: Vec<u8> = (0..2000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 32) as u8
            })
            .collect();
        for n in 0..=200 {
            assert_eq!(decode(&encode(&data[..n])).unwrap(), &data[..n], "{n}");
        }
        assert_eq!(decode(&encode(&data)).unwrap(), data);
    }

    #[test]
    fn errors() {
        assert!(matches!(
            decode("媒A"),
            Err(DecodeError::UnrecognizedCharacter('A'))
        ));
        assert!(matches!(
            decode("媒𐀀"),
            Err(DecodeError::UnrecognizedCharacter('𐀀'))
        ));
        // "ƀ" is a 7-bit character, allowed only at the end.
        assert!(matches!(
            decode("媒ƀ媒"),
            Err(DecodeError::UnexpectedSecondaryCharacter(1))
        ));
        // 15 bits in one character, none of them padding.
        assert!(matches!(
            decode(&encode(&[0, 0]).chars().next().unwrap().to_string()),
            Err(DecodeError::PaddingMismatch)
        ));
    }

    #[test]
    fn into_appends_and_rolls_back() {
        let mut s = String::from("prefix");
        encode_into(&[1, 2, 3], &mut s);
        assert_eq!(s, format!("prefix{}", encode(&[1, 2, 3])));
        let mut v = vec![9u8];
        decode_into(&encode(&[1, 2, 3]), &mut v).unwrap();
        assert_eq!(v, [9, 1, 2, 3]);
        assert!(decode_into("媒腻㐤A", &mut v).is_err());
        assert_eq!(v, [9, 1, 2, 3]);
    }
}
