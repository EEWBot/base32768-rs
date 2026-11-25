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

    // AVX-512 encode: cp = z + 0x2000 * (z >= 0xD40) + 32 * low, where low
    // (as in the AVX2 table) depends only on the block z >> 5. The 8 lows of
    // each 1024 / 8 = 128 units of 8 blocks (u = z >> 8) form 22 distinct
    // patterns, overlapped into one string of at most 128 bytes. A unit's
    // start in it is stored biased by -8u (mod 128), so that adding the
    // block's low 8 bits ((z >> 5) % 256 = 8 (u % 32) + b) gives start + b.
    let unit_lows = |u: u32| -> Vec<u8> {
        (0..8)
            .map(|b| u8::try_from(d_of(u << 8 | b << 5) - high(u << 8 | b << 5)).unwrap())
            .collect()
    };
    let mut strings: Vec<Vec<u8>> = vec![];
    for u in 0..128 {
        let p = unit_lows(u);
        if !strings.contains(&p) {
            strings.push(p);
        }
    }
    // Greedy shortest common superstring: merge the pair with the longest
    // overlap until one string is left.
    let overlap = |a: &[u8], b: &[u8]| -> usize {
        (0..=a.len().min(b.len()))
            .rev()
            .find(|&k| a[a.len() - k..] == b[..k])
            .unwrap()
    };
    while strings.len() > 1 {
        let mut best = (0, 1, 0);
        for i in 0..strings.len() {
            for j in 0..strings.len() {
                if i != j {
                    let k = overlap(&strings[i], &strings[j]);
                    if k >= best.2 {
                        best = (i, j, k);
                    }
                }
            }
        }
        let (i, j, k) = best;
        let mut merged = strings[i].clone();
        merged.extend_from_slice(&strings[j][k..]);
        strings.remove(i.max(j));
        strings.remove(i.min(j));
        strings.push(merged);
    }
    let mut encode_offset = strings.pop().unwrap();
    assert!(encode_offset.len() <= 128, "{} bytes", encode_offset.len());
    let mut encode_start = vec![0u8; 128];
    for u in 0..128u32 {
        let p = unit_lows(u);
        let start = encode_offset.windows(8).position(|w| w == &p[..]).unwrap() as u32;
        encode_start[u as usize] = (start.wrapping_sub(8 * u) & 127) as u8;
    }
    encode_offset.resize(128, 0);
    for z in 0..0x8000u32 {
        let idx = (encode_start[(z >> 8) as usize] as u32 + (z >> 5 & 0xFF)) & 127;
        let low = encode_offset[idx as usize] as u32;
        let cp = (z + 0x2000 * (z >= 0xD40) as u32 + (low << BLOCK_BITS)) & 0xFFFF;
        assert_eq!(cp, z15_cps[z as usize], "z {z:#x}");
    }

    // AVX-512 decode: z = cp - 32 * (blocks below cp's block that are not
    // 15-bit), which holds because the 15-bit blocks are in ascending order.
    // Per half page h (512 code points, 16 blocks): the bitmap of its 15-bit
    // blocks and 32 * the non-15-bit blocks before it. The 128 half pages
    // share 27 distinct entries (all halves without 15-bit blocks share one),
    // so the SIMD code looks up an entry index and then the entry.
    assert!(z15_blocks.is_sorted());
    let is15 = |blk: u32| decode[(blk << BLOCK_BITS) as usize] < 0x8000;
    let mut entries: Vec<(u16, u16)> = vec![(0, 0)];
    let mut half_desc = [0u8; 128];
    let mut missing = 0u32;
    for h in 0..128u32 {
        let mask = (0..16).fold(0u16, |m, i| m | (is15(h * 16 + i) as u16) << i);
        let e = if mask == 0 {
            (0, 0)
        } else {
            (mask, (missing << BLOCK_BITS) as u16)
        };
        missing += 16 - mask.count_ones();
        let i = entries.iter().position(|&x| x == e).unwrap_or_else(|| {
            entries.push(e);
            entries.len() - 1
        });
        half_desc[h as usize] = i as u8;
    }
    // Entry indices are `vpermw` indices.
    assert!(entries.len() <= 32, "{} half page entries", entries.len());
    // The SIMD code computes t = 64 * b0 + b1 from the first two UTF-8 bytes
    // with `vpmaddubsw`: t >> 3 is (cp >> 9) + 0x710 and t << 6 is
    // cp + 0x2000 (mod 2^16). The tables absorb both offsets.
    let mut half_desc_biased = vec![0u8; 128];
    for h in 0..128 {
        half_desc_biased[(h + 0x10) & 127] = half_desc[h];
    }
    let mut valid_mask = vec![0u16; 32];
    let mut base_offset_biased = vec![0u16; 32];
    for (i, &(mask, offset)) in entries.iter().enumerate() {
        valid_mask[i] = mask;
        base_offset_biased[i] = offset.wrapping_add(0x2000);
    }
    for cp in 0x800..0x10000u32 {
        let (b0, b1) = (0xE0 | cp >> 12, 0x80 | (cp >> 6 & 0x3F));
        let t = 64 * b0 + b1;
        let i = half_desc_biased[(t >> 3 & 127) as usize] as usize;
        let cp_adj = (t << 6 | (cp & 0x3F)) & 0xFFFF;
        let bit = 1u32 << (cp_adj >> 5 & 15);
        let mask = valid_mask[i] as u32;
        assert_eq!(mask & bit != 0, decode[cp as usize] < 0x8000, "U+{cp:04X}");
        if mask & bit != 0 {
            let holes = (!mask & (bit - 1)).count_ones();
            let z = cp_adj.wrapping_sub(base_offset_biased[i] as u32 + (holes << 5)) & 0xFFFF;
            assert_eq!(z, decode[cp as usize] as u32, "U+{cp:04X}");
        }
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
        "/// AVX2 encode offsets (see build.rs).",
        "ENCODE_AVX2",
        "u8",
        &encode_avx2,
    );

    write_array(
        &mut out,
        "/// AVX-512 encode: per 8-block unit, start in `ENCODE_OFFSET` - 8 * unit (mod 128).",
        "ENCODE_START_BIASED",
        "u8",
        &encode_start,
    );
    write_array(
        &mut out,
        "/// AVX-512 encode: block offsets (see build.rs), shared between units.",
        "ENCODE_OFFSET",
        "u8",
        &encode_offset,
    );
    write_array(
        &mut out,
        "/// AVX-512 decode: entry index of half page `(h + 16) % 128`.",
        "HALF_DESC_BIASED",
        "u8",
        &half_desc_biased,
    );
    write_array(
        &mut out,
        "/// AVX-512 decode: bitmap of a half page's 15-bit blocks, per entry.",
        "VALID_MASK",
        "u16",
        &valid_mask,
    );
    write_array(
        &mut out,
        "/// AVX-512 decode: 32 * non-15-bit blocks before the half page, + 0x2000.",
        "BASE_OFFSET_BIASED",
        "u16",
        &base_offset_biased,
    );

    std::fs::write(
        format!("{}/table.rs", std::env::var("OUT_DIR").unwrap()),
        out,
    )
    .unwrap();
    println!("cargo::rerun-if-changed=build.rs");
}
