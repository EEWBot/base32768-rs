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

/// Overlaps the 8-byte patterns of the 128 units into one string of at most
/// 128 bytes (greedy shortest common superstring) and returns it, padded to
/// 128 bytes, with each unit's start biased by -8u (mod 128).
fn overlap_units(unit: &dyn Fn(u32) -> Vec<u8>) -> (Vec<u8>, Vec<u8>) {
    let mut strings: Vec<Vec<u8>> = vec![];
    for u in 0..128 {
        let p = unit(u);
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
    let mut string = strings.pop().unwrap();
    assert!(string.len() <= 128, "{} bytes", string.len());
    let mut start = vec![0u8; 128];
    for u in 0..128u32 {
        let p = unit(u);
        let s = string.windows(8).position(|w| w == &p[..]).unwrap() as u32;
        start[u as usize] = (s.wrapping_sub(8 * u) & 127) as u8;
    }
    string.resize(128, 0);
    (string, start)
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
    let (encode_offset, encode_start) = overlap_units(&unit_lows);
    for z in 0..0x8000u32 {
        let idx = (encode_start[(z >> 8) as usize] as u32 + (z >> 5 & 0xFF)) & 127;
        let low = encode_offset[idx as usize] as u32;
        let cp = (z + 0x2000 * (z >= 0xD40) as u32 + (low << BLOCK_BITS)) & 0xFFFF;
        assert_eq!(cp, z15_cps[z as usize], "z {z:#x}");
    }
    // AVX-512 encode: each slot of the string instead holds the ID of the
    // full offset cp - z (including the 0x2000 above 0xD40) of the blocks that
    // use it. Those blocks share one offset, and the IDs index a dictionary
    // of at most 64 offsets (one two-table vpermi2w).
    let mut encode_offset_ids = vec![u8::MAX; 128];
    let mut encode_offset_dict: Vec<u16> = vec![];
    for z in (0..0x8000u32).step_by(BLOCK_SIZE as usize) {
        let offset = (z15_cps[z as usize].wrapping_sub(z) & 0xFFFF) as u16;
        let id = encode_offset_dict
            .iter()
            .position(|&x| x == offset)
            .unwrap_or_else(|| {
                encode_offset_dict.push(offset);
                encode_offset_dict.len() - 1
            }) as u8;
        let slot = ((encode_start[(z >> 8) as usize] as u32 + (z >> 5 & 0xFF)) & 127) as usize;
        assert!(
            encode_offset_ids[slot] == u8::MAX || encode_offset_ids[slot] == id,
            "slot {slot}"
        );
        encode_offset_ids[slot] = id;
    }
    assert!(
        encode_offset_dict.len() <= 64,
        "{} offsets",
        encode_offset_dict.len()
    );
    encode_offset_dict.resize(64, 0);
    for z in 0..0x8000u32 {
        let slot = (encode_start[(z >> 8) as usize] as u32 + (z >> 5 & 0xFF)) & 127;
        let offset = encode_offset_dict[(encode_offset_ids[slot as usize] & 63) as usize];
        let cp = (z + offset as u32) & 0xFFFF;
        assert_eq!(cp, z15_cps[z as usize], "z {z:#x}");
    }
    // NEON encode: L = (D + 32) % 256 as an i8, with D = (cp - z) / 32,
    // overlapped per unit as above. 2-byte characters have D in 37..=56 and
    // 3-byte ones in 121..=323, so cp = z + 32 * L - 0x400 for z < 224 and
    // z + 32 * L + 0x1C00 otherwise.
    let unit_l =
        |u: u32| -> Vec<u8> { (0..8).map(|b| (d_of(u << 8 | b << 5) + 32) as u8).collect() };
    let (neon_encode_l, neon_encode_start) = overlap_units(&unit_l);
    for z in 0..0x8000u32 {
        let idx = (neon_encode_start[(z >> 8) as usize] as u32 + (z >> 5 & 0xFF)) & 127;
        let l = neon_encode_l[idx as usize] as i8 as i32;
        let adj = if z < 224 { -0x400 } else { 0x1C00 };
        let cp = (z as i32 + 32 * l + adj) as u32 & 0xFFFF;
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
    // Counting missing blocks from the current block through the end of
    // its half page lets decode use the shifted mask for both membership
    // (bit zero) and rank (population count).
    // Padded to 32 `vpermw` entries with half pages that have no 15-bit
    // blocks.
    entries.resize(32, (0, 0));
    let missing_mask: Vec<_> = entries.iter().map(|&(mask, _)| !mask).collect();
    // 32 * non-15-bit blocks through the end of the half page, + 0x2000.
    let end_offset: Vec<_> = entries
        .iter()
        .map(|&(mask, offset)| {
            offset
                .wrapping_add(0x2000)
                .wrapping_add(32 * (16 - mask.count_ones()) as u16)
        })
        .collect();
    for cp in 0x800..0x10000u32 {
        let (b0, b1) = (0xE0 | cp >> 12, 0x80 | (cp >> 6 & 0x3F));
        let t = 64 * b0 + b1;
        let i = half_desc_biased[(t >> 3 & 127) as usize] as usize;
        let cp_adj = (t << 6 | (cp & 0x3F)) & 0xFFFF;
        let suffix = missing_mask[i] >> (cp_adj >> 5 & 15);
        assert_eq!(suffix & 1 == 0, decode[cp as usize] < 0x8000, "U+{cp:04X}");
        if suffix & 1 == 0 {
            let z = cp_adj
                .wrapping_sub(end_offset[i] as u32)
                .wrapping_add(32 * suffix.count_ones())
                & 0xFFFF;
            assert_eq!(z, decode[cp as usize] as u32, "U+{cp:04X}");
        }
    }

    // NEON decode: rank queries per 256-code-point page (8 blocks), so that
    // masks and popcounts are bytes. Entry 2e + k is page k (low or high) of
    // half page entry e: its missing blocks, and the non-15-bit blocks below
    // the page's end minus 224. With D = (cp - z) / 32, D - 224 fits in an
    // i8 for every 3-byte 15-bit character, so z = cp - 0x1C00 - 32 * (end -
    // popcount(missing >> block)).
    let neon_half_desc: Vec<u8> = half_desc.iter().map(|&e| 2 * e).collect();
    let mut neon_missing = vec![0u8; 64];
    let mut neon_end = vec![0u8; 64];
    for (e, &(mask, offset)) in entries.iter().enumerate() {
        let missing = !mask;
        for k in 0..2 {
            let below = missing & (0xFFFF >> (8 - 8 * k));
            let end = (offset >> BLOCK_BITS) as i32 + below.count_ones() as i32;
            neon_missing[2 * e + k] = (missing >> (8 * k)) as u8;
            neon_end[2 * e + k] = (end - 224) as u8;
        }
    }
    for cp in 0x800..0x10000u32 {
        let (hi, lo) = (cp >> 8, cp & 0xFF);
        let page = (neon_half_desc[(hi >> 1) as usize] | (hi & 1) as u8) as usize;
        let suffix = neon_missing[page] >> (lo >> 5);
        let z = decode[cp as usize] as u32;
        assert_eq!(suffix & 1 == 0, z < 0x8000, "U+{cp:04X}");
        if suffix & 1 == 0 {
            assert!(
                (-128..128).contains(&((cp - z) as i32 / 32 - 224)),
                "U+{cp:04X}"
            );
            let delta = neon_end[page].wrapping_sub(suffix.count_ones() as u8) as i8;
            let neon_z = (cp as i32 - 0x1C00 - 32 * delta as i32) as u32 & 0xFFFF;
            assert_eq!(neon_z, z, "U+{cp:04X}");
        }
    }
    // For 2-byte characters, D + 32 (69..=88) fits in an i8 and equals
    // D - 224 mod 256, so the same tables give z = cp + 0x400 - 32 * delta.
    for cp in 0x80..0x800u32 {
        let (hi, lo) = (cp >> 8, cp & 0xFF);
        let page = (neon_half_desc[(hi >> 1) as usize] | (hi & 1) as u8) as usize;
        let suffix = neon_missing[page] >> (lo >> 5);
        let z = decode[cp as usize] as u32;
        assert_eq!(suffix & 1 == 0, z < 0x8000, "U+{cp:04X}");
        if suffix & 1 == 0 {
            assert!(
                (-128..128).contains(&((cp as i32 - z as i32) / 32 + 32)),
                "U+{cp:04X}"
            );
            let delta = neon_end[page].wrapping_sub(suffix.count_ones() as u8) as i8;
            let neon_z = (cp as i32 + 0x400 - 32 * delta as i32) as u32 & 0xFFFF;
            assert_eq!(neon_z, z, "U+{cp:04X}");
        }
    }

    // AVX-512 decode of 2-byte characters: the 224 2-byte 15-bit characters
    // (z < 224) are the first 7 blocks, all below U+0800. Indexed by cp >> 5
    // (a 2-byte code point has 11 bits): 32 * the block's index (the z of its
    // first character), or 0xFF (odd, unlike every valid entry).
    let mut decode_2byte_block = vec![0xFFu8; 64];
    for (k, &blk) in z15_blocks.iter().enumerate().take(7) {
        assert!(blk < 0x800);
        decode_2byte_block[(blk >> BLOCK_BITS) as usize] = (k as u8) << 5;
    }
    assert!(z15_blocks[7] >= 0x800);

    // Encode tables: packed UTF-8 for every value.
    let expand =
        |blocks: &[u32]| -> Vec<u32> { chars(blocks).into_iter().map(packed_utf8).collect() };
    let z15_utf8 = expand(&z15_blocks);
    let z7_utf8 = expand(&z7_blocks);

    // AVX-512 encode of blocks of 2-byte characters (z < 224, the first 7
    // blocks): the UTF-8 bytes (b0, b1) of a block's first character as a
    // little-endian u16. Within a block only the low 5 bits of b1 change,
    // and they equal those of z.
    let mut encode_2byte_prefix = vec![0u16; 8];
    for (i, prefix) in encode_2byte_prefix.iter_mut().take(7).enumerate() {
        *prefix = z15_utf8[i << BLOCK_BITS] as u16;
    }
    for z in 0..224 {
        let packed = encode_2byte_prefix[z >> BLOCK_BITS] as u32 | ((z as u32 & 31) << 8);
        assert_eq!(packed | 2 << 24, z15_utf8[z], "z {z}");
    }

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
    // The same characters as `char`s, for the safe-only implementation.
    for (name, bits, cps) in [
        ("Z15_CHAR", 15, &z15_cps),
        ("Z7_CHAR", 7, &chars(&z7_blocks)),
    ] {
        writeln!(out, "/// The character for each {bits}-bit value.").unwrap();
        writeln!(out, "pub static {name}: [char; {}] = [", cps.len()).unwrap();
        for chunk in cps.chunks(16) {
            let line: Vec<String> = chunk.iter().map(|cp| format!("'\\u{{{cp:x}}}'")).collect();
            writeln!(out, "    {},", line.join(", ")).unwrap();
        }
        writeln!(out, "];").unwrap();
    }
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
        "/// AVX-512 encode: per 8-block unit, start in `ENCODE_OFFSET_IDS` - 8 * unit (mod 128).",
        "ENCODE_START_BIASED",
        "u8",
        &encode_start,
    );
    write_array(
        &mut out,
        "/// AVX-512 encode: per slot, an `ENCODE_OFFSET_DICT` index (see build.rs).",
        "ENCODE_OFFSET_IDS",
        "u8",
        &encode_offset_ids,
    );
    write_array(
        &mut out,
        "/// AVX-512 encode: full code point offsets cp - z.",
        "ENCODE_OFFSET_DICT",
        "u16",
        &encode_offset_dict,
    );
    write_array(
        &mut out,
        "/// AVX-512 encode: UTF-8 (b0, b1) of the first character of each 2-byte block.",
        "ENCODE_2BYTE_PREFIX",
        "u16",
        &encode_2byte_prefix,
    );
    write_array(
        &mut out,
        "/// NEON encode: per 8-block unit, start in `NEON_ENCODE_L` - 8 * unit (mod 128).",
        "NEON_ENCODE_START",
        "u8",
        &neon_encode_start,
    );
    write_array(
        &mut out,
        "/// NEON encode: (D + 32) % 256 per block (see build.rs), shared between units.",
        "NEON_ENCODE_L",
        "u8",
        &neon_encode_l,
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
        "/// AVX-512 decode: 32 * the 2-byte 15-bit block index (0..=192) of cp >> 5, or 0xFF.",
        "DECODE_2BYTE_BLOCK",
        "u8",
        &decode_2byte_block,
    );
    write_array(
        &mut out,
        "/// NEON decode: 2 * the half page entry index of each half page.",
        "NEON_HALF_DESC",
        "u8",
        &neon_half_desc,
    );
    write_array(
        &mut out,
        "/// NEON decode: missing 15-bit blocks of each 256-code-point page entry.",
        "NEON_MISSING",
        "u8",
        &neon_missing,
    );
    write_array(
        &mut out,
        "/// NEON decode: non-15-bit blocks below each page entry's end, minus 224.",
        "NEON_END",
        "u8",
        &neon_end,
    );

    write_array(
        &mut out,
        "/// Missing 15-bit blocks per half-page entry.",
        "MISSING_MASK",
        "u16",
        &missing_mask,
    );
    write_array(
        &mut out,
        "/// Biased offset through the end of a half page.",
        "END_OFFSET_BIASED",
        "u16",
        &end_offset,
    );
    // A dword gather reads each 16-bit entry and the one after it.
    let mut mixed_block: Vec<_> = decode.iter().step_by(32).copied().collect();
    mixed_block.push(0xFFFF);
    write_array(
        &mut out,
        "/// First value per 32 code points, plus gather padding.",
        "DECODE_MIXED_BLOCK",
        "u16",
        &mixed_block,
    );

    std::fs::write(
        format!("{}/table.rs", std::env::var("OUT_DIR").unwrap()),
        out,
    )
    .unwrap();
    println!("cargo::rerun-if-changed=build.rs");
}
