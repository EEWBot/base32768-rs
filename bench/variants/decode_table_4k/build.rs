use std::fmt::Write;

const Z15_CHAR_RANGES: &str = "ҠҿԀԟڀڿݠޟ߀ߟကဟႠႿᄀᅟᆀᆟᇠሿበቿዠዿጠጿᎠᏟᐠᙟᚠᛟកសᠠᡟᣀᣟᦀᦟ᧠᧿ᨠᨿᯀᯟᰀᰟᴀᴟ⇠⇿⋀⋟⍀⏟␀␟─❟➀➿⠀⥿⦠⦿⨠⩟⪀⪿⫠⭟ⰀⰟⲀⳟⴀⴟⵀⵟ⺠⻟㇀㇟㐀䶟䷀龿ꀀꑿ꒠꒿ꔀꗿꙀꙟꚠꛟ꜀ꝟꞀꞟꡀꡟ";
const Z7_CHAR_RANGES: &str = "ƀƟɀʟ";

/// Every repertoire range starts and ends on a 32-codepoint boundary, so the
/// repertoires can be described as a list of 32-character blocks.
const BLOCK_BITS: u32 = 5;
const BLOCK_SIZE: u32 = 1 << BLOCK_BITS;

/// Returns the first code point of each 32-character block, in repertoire order.
fn ranges_to_blocks(ranges: &str) -> Vec<u32> {
    let ranges: Vec<u32> = ranges.chars().map(u32::from).collect();
    assert!(ranges.len().is_multiple_of(2));

    let mut blocks = vec![];
    for &[first, last] in ranges.as_chunks::<2>().0 {
        assert_eq!(
            first % BLOCK_SIZE,
            0,
            "range start U+{first:04X} not block aligned"
        );
        assert_eq!(
            (last + 1) % BLOCK_SIZE,
            0,
            "range end U+{last:04X} not block aligned"
        );
        blocks.extend((first..=last).step_by(BLOCK_SIZE as usize));
    }
    blocks
}

/// Packs the UTF-8 form of `cp` as little-endian bytes in the low 24 bits and
/// its byte length in the top 8 bits.
fn packed_utf8(cp: u32) -> u32 {
    let mut buf = [0u8; 4];
    let len = char::from_u32(cp).unwrap().encode_utf8(&mut buf).len();
    assert!(
        len == 2 || len == 3,
        "U+{cp:04X} must be 2 or 3 bytes in UTF-8"
    );
    u32::from_le_bytes(buf) | (len as u32) << 24
}

fn write_array<T: std::fmt::LowerHex>(
    out: &mut String,
    doc: &str,
    name: &str,
    ty: &str,
    values: &[T],
) {
    writeln!(out, "{doc}").unwrap();
    let item = if values.len() > 4096 {
        "static"
    } else {
        "const"
    };
    writeln!(out, "pub {item} {name}: [{ty}; {}] = [", values.len()).unwrap();
    for chunk in values.chunks(16) {
        let line: Vec<String> = chunk.iter().map(|v| format!("{v:#x}")).collect();
        writeln!(out, "    {},", line.join(", ")).unwrap();
    }
    writeln!(out, "];").unwrap();
}

fn main() {
    let z15_blocks = ranges_to_blocks(Z15_CHAR_RANGES);
    let z7_blocks = ranges_to_blocks(Z7_CHAR_RANGES);
    assert_eq!(z15_blocks.len(), (1 << 15) / BLOCK_SIZE as usize);
    assert_eq!(z7_blocks.len(), (1 << 7) / BLOCK_SIZE as usize);

    // Decode table indexed by code point, covering every 2 or 3 byte UTF-8
    // sequence (U+0000..=U+FFFF) so lookups need no bounds check.
    // 0xFFFF = not a Base32768 character, 0x8000 | z = 7-bit, z = 15-bit.
    let chars = |blocks: &[u32]| -> Vec<u32> {
        blocks.iter().flat_map(|&cp| cp..cp + BLOCK_SIZE).collect()
    };
    let mut decode = vec![0xFFFFu16; 0x10000];
    for (z, cp) in chars(&z15_blocks).into_iter().enumerate() {
        decode[cp as usize] = z as u16;
    }
    for (z, cp) in chars(&z7_blocks).into_iter().enumerate() {
        decode[cp as usize] = 0x8000 | z as u16;
    }

    // The same per 32-code-point block: every block is entirely in one
    // repertoire or none, and its z values start at a multiple of 32, so
    // DECODE[cp] = DECODE_BLOCK[cp >> 5] | cp % 32 (0xFFFF stays 0xFFFF).
    let decode_block: Vec<u16> = decode.iter().step_by(BLOCK_SIZE as usize).copied().collect();
    for (cp, &d) in decode.iter().enumerate() {
        assert_eq!(decode_block[cp >> BLOCK_BITS] | (cp as u16 & 31), d, "U+{cp:04X}");
    }

    // Page + rank tables for the AVX-512 decode path. A page is 1024 code
    // points (32 blocks) and its 32-bit bitmap marks the 15-bit blocks. The z
    // of a code point is 32 * (number of 15-bit blocks below its block) +
    // cp % 32, which holds because the blocks are in ascending order. Only 16
    // distinct bitmaps occur, so each page stores the number of blocks before
    // it and a pattern id; each pattern is split into two 16-block halves.
    assert!(z15_blocks.is_sorted());
    let mut bitmaps = [0u32; 64];
    for &cp in &z15_blocks {
        bitmaps[(cp >> 10) as usize] |= 1 << ((cp >> BLOCK_BITS) & 31);
    }
    let mut patterns: Vec<u32> = vec![];
    let mut page_desc = vec![0u16; 64];
    let mut prefix = 0u32;
    for (page, &bitmap) in bitmaps.iter().enumerate() {
        let id = match patterns.iter().position(|&x| x == bitmap) {
            Some(i) => i,
            None => {
                patterns.push(bitmap);
                patterns.len() - 1
            }
        };
        page_desc[page] = (prefix << 5 | (id as u32) << 1) as u16;
        prefix += bitmap.count_ones();
    }
    // The pattern id and half bit must fit the 5-bit `vpermw` index.
    assert!(patterns.len() <= 16, "{} bitmap patterns", patterns.len());
    let mut half_mask = vec![0u16; 32];
    let mut half_prefix = vec![0u16; 32];
    for (id, &bitmap) in patterns.iter().enumerate() {
        half_mask[2 * id] = bitmap as u16;
        half_mask[2 * id + 1] = (bitmap >> 16) as u16;
        half_prefix[2 * id + 1] = (bitmap & 0xFFFF).count_ones() as u16;
    }

    // Select tables for the AVX-512 encode path. z is split into 32 groups of
    // 1024 values (32 blocks). Block b of group g starts at code point
    // ((base_g + b + delta_p[b]) << 5), where delta_p counts the 32-code-point
    // blocks skipped so far inside the group and p is one of a few patterns.
    let mut encode_desc = vec![0u16; 32];
    let mut encode_patterns: Vec<[u8; 32]> = vec![];
    for g in 0..32 {
        let blocks = &z15_blocks[g * 32..g * 32 + 32];
        let base = blocks[0] >> BLOCK_BITS;
        let mut delta = [0u8; 32];
        for (b, &cp) in blocks.iter().enumerate() {
            let d = (cp >> BLOCK_BITS) - base - b as u32;
            delta[b] = u8::try_from(d).unwrap();
        }
        let id = match encode_patterns.iter().position(|x| *x == delta) {
            Some(i) => i,
            None => {
                encode_patterns.push(delta);
                encode_patterns.len() - 1
            }
        };
        encode_desc[g] = u16::try_from(base << 3 | id as u32).unwrap();
    }
    assert!(
        encode_patterns.len() <= 8,
        "{} encode patterns",
        encode_patterns.len()
    );
    let mut encode_delta = vec![0u8; 256];
    for (id, pat) in encode_patterns.iter().enumerate() {
        encode_delta[id * 32..id * 32 + 32].copy_from_slice(pat);
    }

    // AVX2 encode: cp = z + 32 * D(z), where D(z) = low + (z >= 0xD40) * 256.
    // For z < 0xDC0 or z >= 0x7DE0, low is byte z >> 5 & 127 of this table
    // (8 tables of 16 for vpshufb). Both ranges fit because they share no
    // index: 0..=109 for z < 0xDC0, 111..=127 for z >= 0x7DE0. Elsewhere low
    // is 50, 51 or 53.
    let z15_cps = chars(&z15_blocks);
    let d_of = |z: u32| -> u32 { (z15_cps[z as usize] >> BLOCK_BITS) - (z >> BLOCK_BITS) };
    let high = |z: u32| -> u32 { if z >= 0xD40 { 256 } else { 0 } };
    let mut encode_avx2 = vec![0u8; 128];
    for t in 0..4u32 {
        for i in 0..32u32 {
            let z = match t {
                0..=2 => t << 10 | i << 5,
                _ if i < 14 => 0xC00 | i << 5,
                _ if i >= 15 => 0x7C00 | i << 5,
                _ => continue,
            };
            encode_avx2[(t * 32 + i) as usize] = u8::try_from(d_of(z) - high(z)).unwrap();
        }
    }
    for z in 0..0x8000u32 {
        let low = if !(0xDC0..0x7DE0).contains(&z) {
            encode_avx2[((z >> 10).min(3) * 32 + (z >> 5 & 31)) as usize] as u32
        } else {
            50 + (z >= 0x2760) as u32 + 2 * (z >= 0x7960) as u32
        };
        assert_eq!(
            z + ((low + high(z)) << BLOCK_BITS),
            z15_cps[z as usize],
            "z {z:#x}"
        );
    }

    // Encode tables: packed UTF-8 for every value.
    let expand =
        |blocks: &[u32]| -> Vec<u32> { chars(blocks).into_iter().map(packed_utf8).collect() };
    let z15_utf8 = expand(&z15_blocks);
    let z7_utf8 = expand(&z7_blocks);

    // Large tables are emitted as `static` so each exists exactly once in the binary.
    let mut out = String::new();
    write_array(
        &mut out,
        "/// Packed UTF-8 (`len << 24 | le_bytes`) of the character for each 15-bit value.",
        "Z15_UTF8",
        "u32",
        &z15_utf8,
    );
    write_array(
        &mut out,
        "/// Packed UTF-8 (`len << 24 | le_bytes`) of the character for each 7-bit value.",
        "Z7_UTF8",
        "u32",
        &z7_utf8,
    );
    write_array(
        &mut out,
        "/// Code point (U+0000..=U+FFFF) -> 0xFFFF (invalid), `0x8000 | z` (7-bit) or `z` (15-bit).",
        "DECODE",
        "u16",
        &decode,
    );
    write_array(
        &mut out,
        "/// `DECODE` per 32-code-point block: `DECODE[cp] = DECODE_BLOCK[cp >> 5] | cp % 32`.",
        "DECODE_BLOCK",
        "u16",
        &decode_block,
    );

    write_array(
        &mut out,
        "/// Per 1024-code-point page: `15-bit blocks before the page << 5 | pattern << 1`.",
        "PAGE_DESC",
        "u16",
        &page_desc,
    );
    write_array(
        &mut out,
        "/// 15-bit block bitmap of a half page (16 blocks), indexed by `pattern << 1 | half`.",
        "HALF_MASK",
        "u16",
        &half_mask,
    );
    write_array(
        &mut out,
        "/// 15-bit blocks in the first half of the page, indexed like `HALF_MASK`.",
        "HALF_PREFIX",
        "u16",
        &half_prefix,
    );

    write_array(
        &mut out,
        "/// Per 1024-value group: `first block (cp >> 5) << 3 | pattern`.",
        "ENCODE_DESC",
        "u16",
        &encode_desc,
    );
    write_array(
        &mut out,
        "/// Skipped blocks before block b of a group, indexed by `pattern << 5 | b`.",
        "ENCODE_DELTA",
        "u8",
        &encode_delta,
    );

    write_array(
        &mut out,
        "/// AVX2 encode offsets (see build.rs).",
        "ENCODE_AVX2",
        "u8",
        &encode_avx2,
    );

    std::fs::write(
        format!("{}/table.rs", std::env::var("OUT_DIR").unwrap()),
        out,
    )
    .unwrap();
    println!("cargo::rerun-if-changed=build.rs");
}
