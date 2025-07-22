use thiserror::Error;

const BITS_PER_CHAR: u8 = 15;
const BITS_PER_BYTE: u8 = 8;

mod table;

#[derive(Error, Debug)]
pub enum DecodeError {
    #[error("Unrecognised Base32768 character: {0}")]
    UnrecognizedCharacter(char),

    #[error("Secondary character found before end of input at position {0}")]
    UnexpectedSecondaryCharacter(usize),

    #[error("Padding mismatch")]
    PaddingMismatch,
}

pub fn encode(b: &[u8]) -> String {
    let mut s = String::new();
    let mut z = 0u16;
    let mut numZBits = 0;

    for uint8 in b {
        for j in itertools::rev(0..BITS_PER_BYTE) {
            let bit = (uint8 >> j) & 1;

            z = (z << 1) + (bit as u16);
            numZBits += 1;

            if numZBits == BITS_PER_CHAR {
                s.push(table::Z15_REPERTOIRE[z as usize]);
                z = 0;
                numZBits = 0;
            }
        }
    }

    if numZBits != 0 {
        while !(numZBits == 7 || numZBits == 15) {
            z = (z << 1) + 1;
            numZBits += 1;
        }

        match numZBits {
            7 => s.push(table::Z7_REPERTOIRE[z as usize]),
            15 => s.push(table::Z15_REPERTOIRE[z as usize]),
            _ => unreachable!(),
        }
    }

    s
}

pub fn decode(s: &str) -> Result<Vec<u8>, DecodeError> {
    let length = s.chars().count();

    let mut uint8Array: Vec<u8> =
        Vec::with_capacity(length * BITS_PER_CHAR as usize / BITS_PER_BYTE as usize);

    let mut uint8 = 0u8;
    let mut numUint8Bits = 0;

    for (i, chr) in s.chars().enumerate() {
        let Some((numZBits, z)) = table::DECODE_LOOKUP_TABLE.get(&chr) else {
            return Err(DecodeError::UnrecognizedCharacter(chr));
        };

        if *numZBits != BITS_PER_CHAR && i != length - 1 {
            return Err(DecodeError::UnexpectedSecondaryCharacter(i));
        }

        for j in itertools::rev(0..*numZBits) {
            let bit = (z >> j) & 1;

            uint8 = (uint8 << 1) + (bit as u8);
            numUint8Bits += 1;

            if numUint8Bits == BITS_PER_BYTE {
                uint8Array.push(uint8);
                uint8 = 0;
                numUint8Bits = 0;
            }
        }
    }

    if uint8 != ((1 << numUint8Bits) - 1) {
        return Err(DecodeError::PaddingMismatch);
    }

    Ok(uint8Array)
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
}
