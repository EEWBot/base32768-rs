use std::collections::HashMap;

use const_gen::*;

const Z15_CHAR_RANGES: &'static str = "ҠҿԀԟڀڿݠޟ߀ߟကဟႠႿᄀᅟᆀᆟᇠሿበቿዠዿጠጿᎠᏟᐠᙟᚠᛟកសᠠᡟᣀᣟᦀᦟ᧠᧿ᨠᨿᯀᯟᰀᰟᴀᴟ⇠⇿⋀⋟⍀⏟␀␟─❟➀➿⠀⥿⦠⦿⨠⩟⪀⪿⫠⭟ⰀⰟⲀⳟⴀⴟⵀⵟ⺠⻟㇀㇟㐀䶟䷀龿ꀀꑿ꒠꒿ꔀꗿꙀꙟꚠꛟ꜀ꝟꞀꞟꡀꡟ";
const Z7_CHAR_RANGES: &'static str = "ƀƟɀʟ";

fn ranges_to_encode_repertoire(ranges: &'static str) -> Vec<char> {
    let ranges: Vec<_> = ranges.chars().collect();

    if ranges.len() % 2 != 0 {
        panic!();
    }

    let mut repertoire = vec![];

    ranges.chunks_exact(2).for_each(|pair| {
        repertoire.extend(pair[0]..=pair[1]);
    });

    repertoire
}

fn main() {
    let mut decode_lookup_table: HashMap<char, (u8, u16)> = HashMap::new();

    let z15_repertoire = ranges_to_encode_repertoire(Z15_CHAR_RANGES);

    z15_repertoire.iter().enumerate().for_each(|(z, chr)| {
        decode_lookup_table.insert(*chr, (15, z as u16));
    });

    let z7_repertoire = ranges_to_encode_repertoire(Z7_CHAR_RANGES);

    z7_repertoire.iter().enumerate().for_each(|(z, chr)| {
        decode_lookup_table.insert(*chr, (7, z as u16));
    });

    let const_declarations = [
        const_declaration!(pub Z15_REPERTOIRE = z15_repertoire),
        const_declaration!(pub Z7_REPERTOIRE = z7_repertoire),
        const_declaration!(pub DECODE_LOOKUP_TABLE = decode_lookup_table),
    ]
    .join("\n");

    std::fs::write(
        format!("{}/table.rs", std::env::var("OUT_DIR").unwrap()),
        const_declarations,
    )
    .unwrap();
    println!("cargo::rerun-if-changed=build.rs");
}
