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

    // Decode table indexed by `code_point >> 5`.
    // 0xFFFF = not a Base32768 character, 0x8000 | z = 7-bit block, z = 15-bit block.
    // Covers every code point of a 2 or 3 byte UTF-8 sequence (U+0000..=U+FFFF),
    // so lookups need no bounds check.
    let max_cp = z15_blocks.iter().chain(&z7_blocks).max().unwrap() + BLOCK_SIZE - 1;
    assert!(max_cp <= 0xFFFF);
    let mut decode = vec![0xFFFFu16; 0x10000 >> BLOCK_BITS];
    for (i, &cp) in z15_blocks.iter().enumerate() {
        decode[(cp >> BLOCK_BITS) as usize] = (i as u32 * BLOCK_SIZE) as u16;
    }
    for (i, &cp) in z7_blocks.iter().enumerate() {
        decode[(cp >> BLOCK_BITS) as usize] = 0x8000 | (i as u32 * BLOCK_SIZE) as u16;
    }

    // Full per-value tables: one load per character instead of block + shift.
    let expand = |blocks: &[u32]| -> Vec<u32> {
        blocks
            .iter()
            .flat_map(|&cp| (cp..cp + BLOCK_SIZE).map(packed_utf8))
            .collect()
    };
    let z15_utf8 = expand(&z15_blocks);
    let z7_utf8 = expand(&z7_blocks);

    let mut out = String::new();
    writeln!(out, "pub const BLOCK_BITS: u32 = {BLOCK_BITS};").unwrap();
    // `static` so the 128 KB table exists exactly once in the binary.
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
        "/// `code_point >> 5` -> 0xFFFF (invalid), `0x8000 | z` (7-bit) or `z` (15-bit), where z is the block's first value.",
        "DECODE_BLOCK",
        "u16",
        &decode,
    );

    std::fs::write(
        format!("{}/table.rs", std::env::var("OUT_DIR").unwrap()),
        out,
    )
    .unwrap();
    println!("cargo::rerun-if-changed=build.rs");
}
