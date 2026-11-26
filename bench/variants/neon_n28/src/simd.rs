//! SIMD kernels of the fast paths.

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
            let tab = _mm256_broadcastsi128_si256(_mm_loadu_si128(LOW.as_ptr().add(16 * n).cast()));
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
/// The code point of a 15-bit value z is `z + offset`, where `offset`
/// depends only on the block `z >> 5`: entry
/// `ENCODE_OFFSET_IDS[(ENCODE_START_BIASED[z >> 8] + (z >> 5)) % 128]` of
/// `ENCODE_OFFSET_DICT` (see build.rs).
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
    static START: [u8; 128] = table::ENCODE_START_BIASED;
    static IDS: [u8; 128] = table::ENCODE_OFFSET_IDS;
    static DICT: [u16; 64] = table::ENCODE_OFFSET_DICT;
    static PREFIX: [u16; 8] = table::ENCODE_2BYTE_PREFIX;
    // SAFETY: each load reads 64 bytes from a table at least that long.
    let ld = |p: *const u8| unsafe { _mm512_loadu_si512(p.cast()) };
    let w = |x: i16| _mm512_set1_epi16(x);
    const M48: u64 = (1 << 48) - 1;

    // 60 bytes -> 32 x 15-bit values. The masked load touches only 60 bytes.
    // SAFETY: the caller guarantees 60 readable bytes at `src`.
    let v = unsafe { _mm512_maskz_loadu_epi8((1 << 60) - 1, src.cast()) };
    let input = _mm512_permutexvar_epi8(ld(ORDER.as_ptr()), v);
    let z = _mm512_and_si512(
        _mm512_multishift_epi64_epi8(ld(BITS.as_ptr()), input),
        w(0x7FFF),
    );

    // All 2-byte (runs of zero bytes in the input data): the first 7 blocks
    // of z. Each character is its block's (b0, b1) with the low 5 bits of z
    // in b1, which needs no code point.
    let two = _mm512_cmplt_epu16_mask(z, w(224));
    if two == u32::MAX {
        // SAFETY: PREFIX has 16 bytes.
        let lane = unsafe { _mm_loadu_si128(PREFIX.as_ptr().cast()) };
        let prefix =
            _mm512_permutexvar_epi16(_mm512_srli_epi16::<5>(z), _mm512_broadcast_i32x4(lane));
        let out = _mm512_ternarylogic_epi32::<0xF8>(prefix, _mm512_slli_epi16::<8>(z), w(0x1F00));
        // SAFETY: 64 of the 96 bytes the caller guarantees.
        unsafe { _mm512_storeu_si512(d.cast(), out) };
        #[cfg(test)]
        AVX512_ENCODE_BLOCKS.with(|n| n.set(n.get() + 1));
        return 64;
    }

    // z -> code point, from the high byte of each u16 lane: vpermi2b uses
    // the low 7 bits of each index byte, z >> 8 (the unit) for the start,
    // then start + (z >> 5) % 256 for the offset ID. vpermi2w uses its low
    // 6 bits. The low bytes are ignored.
    let start = _mm512_permutex2var_epi8(ld(START.as_ptr()), z, ld(START[64..].as_ptr()));
    let idx = _mm512_add_epi8(start, _mm512_slli_epi16::<3>(z));
    let ids = _mm512_permutex2var_epi8(ld(IDS.as_ptr()), idx, ld(IDS[64..].as_ptr()));
    let offset = _mm512_permutex2var_epi16(
        ld(DICT.as_ptr().cast()),
        _mm512_srli_epi16::<8>(ids),
        ld(DICT[32..].as_ptr().cast()),
    );
    let cp = _mm512_add_epi16(z, offset);

    // UTF-8 as 3 bytes per character; 2-byte characters get a leading 0,
    // the only zero byte (every other byte has its top bit set).
    // 2-byte code points have zero in bits 12..15.
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

/// Repertoire lookup for triples of UTF-8 bytes. The mixed-width path
/// supplies (0xFF, lead, continuation) for a two-byte character.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2,avx512bitalg")]
#[inline]
unsafe fn lookup_neg_z(
    b0b1: core::arch::x86_64::__m512i,
    b2: core::arch::x86_64::__m512i,
) -> (core::arch::x86_64::__m512i, u32) {
    use crate::table;
    use core::arch::x86_64::*;
    static HDESC: [u8; 128] = table::HALF_DESC_BIASED;
    static MISSING: [u16; 32] = table::MISSING_MASK;
    static END_OFFSET: [u16; 32] = table::END_OFFSET_BIASED;
    // SAFETY: each load reads 64 bytes within its table.
    let ld = |p: *const u8| unsafe { _mm512_loadu_si512(p.cast()) };
    let w = |x: i16| _mm512_set1_epi16(x);
    // For t = 64*b0 + b1, (t >> 3) % 128 = ((cp >> 9) + 0x10) % 128.
    // With the low 6 bits of b2, t << 6 gives cp + 0x2000 (mod 2^16). The
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
    let valid = _mm512_testn_epi16_mask(missing, w(1));
    // Count missing blocks back from the end of the half page. Keep
    // -z so the signed madd coefficients produce a positive 30-bit pair.
    let holes = _mm512_popcnt_epi16(missing);
    let neg_z = _mm512_sub_epi16(_mm512_sub_epi16(offset, cp), _mm512_slli_epi16::<5>(holes));
    (neg_z, valid)
}

/// Decodes up to 32 three-byte 15-bit characters at `p .. p + 96`.
/// Returns how many leading 8-character groups (0..=4) hold only 3-byte
/// 15-bit characters; those groups are written to `d` as 15 bytes each,
/// and nothing else is written.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2,avx512bitalg")]
#[inline]
pub unsafe fn decode32_avx512(p: *const u8, d: *mut u8) -> usize {
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
    let (neg_z, repertoire_ok) = unsafe { lookup_neg_z(b0b1, b2) };
    let valid = lead_ok & repertoire_ok;
    // 8-character groups before the first rejected lane. With none, skip
    // the rest of the work (common when 2-byte characters are dense).
    let groups = (valid.trailing_ones() / 8) as usize;
    if groups == 0 {
        return 0;
    }
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

/// Decode whole 32-symbol windows and return the committed UTF-8 and
/// binary byte counts. The caller resumes at the first uncommitted symbol.
///
/// # Safety
/// `bytes` must be valid UTF-8 starting on a character boundary. `dst`
/// must have room for the maximum decoded length, as in `decode_fast`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2,avx512bitalg,bmi2,popcnt")]
pub unsafe fn decode32_windows(bytes: &[u8], dst: *mut u8) -> (usize, usize) {
    use core::arch::x86_64::*;

    const SEQ: [u8; 64] = {
        let mut seq = [0; 64];
        let mut i = 0;
        while i < 64 {
            seq[i] = i as u8;
            i += 1;
        }
        seq
    };
    // End positions of the 32 symbols, duplicated in both bytes of each
    // word. Symbols 16..31 select the second list of character boundaries.
    const PICK: [u8; 64] = {
        let mut pick = [0; 64];
        let mut i = 0;
        while i < 64 {
            pick[i] = (i / 2 + 1 + if i >= 32 { 64 } else { 0 }) as u8;
            i += 1;
        }
        pick
    };
    // A two-source permutation sees p[0..64] at indices 0..64 and
    // p[32..96] at 64..128. Prefer the first source in the overlap.
    const fn index(j: usize) -> u8 {
        (j + if j >= 64 { 32 } else { 0 }) as u8
    }
    const FIXED: ([u8; 64], [u8; 64]) = {
        let mut first = [0; 64];
        let mut last = [0; 64];
        let mut i = 0;
        while i < 32 {
            first[2 * i] = index(3 * i + 1);
            first[2 * i + 1] = index(3 * i);
            last[2 * i] = index(3 * i + 2);
            i += 1;
        }
        (first, last)
    };
    // SAFETY: all tables have 64 bytes. Both input loads fit in the
    // 96-byte window checked in the loop condition.
    let ld = |p: *const u8| unsafe { _mm512_loadu_si512(p.cast()) };
    let w = |x: i16| _mm512_set1_epi16(x);
    let (mut pos, mut written) = (0, 0);
    while bytes.len() - pos >= 96 {
        let p = unsafe { bytes.as_ptr().add(pos) };
        let a = ld(p);
        let b = unsafe { ld(p.add(32)) };
        if bytes[pos] < 0xE0 && _mm512_cmpge_epu8_mask(a, _mm512_set1_epi8(0xE0u8 as i8)) == 0 {
            let start = pos;
            while bytes.len() - pos >= 96
                && unsafe { decode32_2byte_avx512(bytes.as_ptr().add(pos), dst.add(written)) } == 4
            {
                pos += 64;
                written += 60;
            }
            if pos == start {
                break;
            }
            continue;
        }
        let fixed = _mm512_permutex2var_epi8(a, ld(FIXED.0.as_ptr()), b);
        let aligned = _mm512_cmpeq_epi16_mask(
            _mm512_and_si512(fixed, w(0xF000u16 as i16)),
            w(0xE000u16 as i16),
        );
        let (b0b1, b2, advance) = if aligned == u32::MAX {
            (
                fixed,
                _mm512_permutex2var_epi8(a, ld(FIXED.1.as_ptr()), b),
                96,
            )
        } else {
            // With valid UTF-8, all bytes in 0x80..0xEF implies only
            // two- and three-byte characters. On rejection the caller
            // resumes here and reports the original character and position.
            let high_bit = _mm512_set1_epi8(-128);
            let max = _mm512_max_epu8(_mm512_xor_si512(a, high_bit), _mm512_xor_si512(b, high_bit));
            if _mm512_cmpge_epu8_mask(max, _mm512_set1_epi8(0x70)) != 0 {
                break;
            }
            let lead_bit = _mm512_set1_epi8(0x40);
            let leads_a = _mm512_test_epi8_mask(a, lead_bit);
            let leads_b = _mm512_test_epi8_mask(b, lead_bit);
            let before_b = (leads_a as u32).count_ones();
            let starts_a = _mm512_maskz_compress_epi8(leads_a, ld(SEQ.as_ptr()));
            // Padding 128 denotes a symbol ending exactly at byte 96.
            let starts_b = _mm512_mask_compress_epi8(
                high_bit,
                leads_b,
                _mm512_add_epi8(ld(SEQ.as_ptr()), lead_bit),
            );
            let pick = _mm512_mask_sub_epi8(
                ld(PICK.as_ptr()),
                0xFFFF_FFFF_0000_0000,
                ld(PICK.as_ptr()),
                _mm512_set1_epi8(before_b as i8),
            );
            let ends = _mm512_permutex2var_epi8(starts_a, pick, starts_b);
            let preceding =
                _mm512_permutex2var_epi8(a, _mm512_add_epi8(ends, w(0xFDFEu16 as i16)), b);
            let two = _mm512_test_epi16_mask(preceding, w(0x40));
            // For a two-byte symbol, 64*0xFF + lead has the same low
            // ten bits as 64*0xE0 + (lead - 0x40), a synthetic three-byte
            // prefix. Those ten bits suffice for both repertoire lookups.
            let prefix = _mm512_mask_mov_epi16(
                preceding,
                two,
                _mm512_or_si512(preceding, w(0xFF00u16 as i16)),
            );
            let last = _mm512_permutex2var_epi8(a, _mm512_add_epi8(ends, _mm512_set1_epi8(-1)), b);
            let next_lead = _pdep_u64(1u64 << (32 - before_b), leads_b);
            (prefix, last, 32 + next_lead.trailing_zeros() as usize)
        };
        let (neg_z, valid) = unsafe { lookup_neg_z(b0b1, b2) };
        if valid != u32::MAX {
            break;
        }
        let out = unsafe { pack_neg_z(neg_z) };
        // 32 symbols consume at least 64 input bytes, reserving 60 output bytes.
        unsafe { _mm512_mask_storeu_epi8(dst.add(written).cast(), (1 << 60) - 1, out) };
        #[cfg(test)]
        AVX512_BLOCKS.with(|n| n.set(n.get() + 1));
        pos += advance;
        written += 60;
    }
    (pos, written)
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
    let base =
        unsafe { _mm512_i32gather_epi32::<2>(_mm512_srli_epi32::<5>(cp), BLOCK.as_ptr().cast()) };
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

#[cfg(all(test, target_arch = "aarch64", target_endian = "little"))]
thread_local! {
    /// Number of 16-character windows the NEON 2-byte encode path wrote on
    /// this thread.
    pub static NEON_ENCODE_WINDOWS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
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

/// Encodes the 30 bytes at `src` (31 are read) as 16 characters at `d` if
/// all of them are 2-byte characters, and returns whether it did (nothing
/// is written otherwise). Works on bytes only: the low 8 bits of value i of
/// each 15-byte half are byte 2i << (7 - i) | byte 2i + 1 >> (i + 1), and
/// the value is below 224 if its top 7 bits are zero and its low 8 bits are
/// below 224.
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn encode16_2byte_short_neon(src: *const u8, d: *mut u8) -> bool {
    use crate::table;
    use core::arch::aarch64::*;
    /// The bits of a 15-byte half that hold the top 7 bits of a value. Byte
    /// 15 of the first load is byte 0 of the second, checked there.
    const fn short_mask() -> [u8; 16] {
        let mut m = [0u8; 16];
        let mut i = 0;
        while i < 8 {
            let mut k = 0;
            while k < 7 {
                let bit = 15 * i + k;
                m[bit / 8] |= 0x80 >> (bit % 8);
                k += 1;
            }
            i += 1;
        }
        m
    }
    static SHORT: [u8; 16] = short_mask();
    static LEFT: [i8; 16] = [7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0];
    static RIGHT: [i8; 16] = [
        -1, -2, -3, -4, -5, -6, -7, -8, -1, -2, -3, -4, -5, -6, -7, -8,
    ];
    static PREFIX: [u16; 8] = table::ENCODE_2BYTE_PREFIX;
    let dup = vdupq_n_u8;

    // SAFETY: the caller guarantees 31 readable bytes at `src`; the tables
    // are 16 bytes.
    let (a, b, short, left, right, prefix) = unsafe {
        (
            vld1q_u8(src),
            vld1q_u8(src.add(15)),
            vld1q_u8(SHORT.as_ptr()),
            vld1q_s8(LEFT.as_ptr()),
            vld1q_s8(RIGHT.as_ptr()),
            vld1q_u8(PREFIX.as_ptr().cast()),
        )
    };
    // Lanes 0..8 from the first half, 8..16 from the second.
    let even = vuzp1q_u8(a, b);
    let odd = vuzp2q_u8(a, b);
    let z8 = vorrq_u8(vshlq_u8(even, left), vshlq_u8(odd, right));
    let bad = vorrq_u8(vandq_u8(vorrq_u8(a, b), short), vcgeq_u8(z8, dup(224)));
    if vmaxvq_u8(bad) != 0 {
        return false;
    }
    let pair = vshlq_n_u8::<1>(vshrq_n_u8::<5>(z8));
    let b0 = vqtbl1q_u8(prefix, pair);
    let b1 = vorrq_u8(
        vqtbl1q_u8(prefix, vorrq_u8(pair, dup(1))),
        vandq_u8(z8, dup(31)),
    );
    // SAFETY: the caller guarantees 32 writable bytes at `d`.
    unsafe { vst2q_u8(d, uint8x16x2_t(b0, b1)) };
    #[cfg(test)]
    NEON_ENCODE_WINDOWS.with(|n| n.set(n.get() + 1));
    true
}

/// Encodes the 30 bytes at `src` (32 are read) as 16 characters at `d` if
/// all of them are 2-byte characters, and returns 32 (the bytes written).
/// Otherwise writes nothing, stores the 16 values at `zs` and returns 0.
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn encode16_2byte_neon(src: *const u8, d: *mut u8, zs: *mut u16) -> usize {
    use crate::table;
    use core::arch::aarch64::*;

    /// tbl indices of u16 lane i of half h (15 bytes from byte 15h): bytes
    /// (j, j + 1) as a big-endian u16, or byte j + 2 in the high byte (y),
    /// where j = 15h + 15i / 8. 0xFF reads as zero.
    const fn unpack(h: usize, y: bool) -> [u8; 16] {
        let mut m = [0xFFu8; 16];
        let mut i = 0;
        while i < 8 {
            let j = 15 * h + 15 * i / 8;
            if y {
                m[2 * i + 1] = (j + 2) as u8;
            } else {
                m[2 * i] = (j + 1) as u8;
                m[2 * i + 1] = j as u8;
            }
            i += 1;
        }
        m
    }
    /// Shifts of x (bit offset 15i % 8 of value i in byte j), and of y.
    const fn shifts(y: bool) -> [i16; 8] {
        let mut s = [0i16; 8];
        let mut i = 0;
        while i < 8 {
            s[i] = (15 * i % 8) as i16 - if y { 16 } else { 0 };
            i += 1;
        }
        s
    }
    static X0: [u8; 16] = unpack(0, false);
    static Y0: [u8; 16] = unpack(0, true);
    static X1: [u8; 16] = unpack(1, false);
    static Y1: [u8; 16] = unpack(1, true);
    static SHX: [i16; 8] = shifts(false);
    static SHY: [i16; 8] = shifts(true);
    static PREFIX: [u16; 8] = table::ENCODE_2BYTE_PREFIX;
    let dup = vdupq_n_u8;

    // SAFETY: the caller guarantees 32 readable bytes at `src`; the tables
    // are read within their lengths.
    let (input, shx, shy, prefix) = unsafe {
        (
            vld1q_u8_x2(src),
            vld1q_s16(SHX.as_ptr()),
            vld1q_s16(SHY.as_ptr()),
            vld1q_u8(PREFIX.as_ptr().cast()),
        )
    };
    // z = (x << r | y >> (16 - r)) >> 1: bits r..r + 15 of bytes j..j + 2.
    let unpack_half = |xi: &[u8; 16], yi: &[u8; 16]| {
        // SAFETY: both index tables are 16 bytes.
        let (xi, yi) = unsafe { (vld1q_u8(xi.as_ptr()), vld1q_u8(yi.as_ptr())) };
        let x = vreinterpretq_u16_u8(vqtbl2q_u8(input, xi));
        let y = vreinterpretq_u16_u8(vqtbl2q_u8(input, yi));
        vshrq_n_u16::<1>(vorrq_u16(vshlq_u16(x, shx), vshlq_u16(y, shy)))
    };
    let z0 = unpack_half(&X0, &Y0);
    let z1 = unpack_half(&X1, &Y1);

    // 2-byte characters are the first 7 blocks.
    let limit = vdupq_n_u16(224);
    let two = vcombine_u8(
        vmovn_u16(vcltq_u16(z0, limit)),
        vmovn_u16(vcltq_u16(z1, limit)),
    );
    if vminvq_u8(two) == 0 {
        // SAFETY: the caller guarantees 16 writable values at `zs`.
        unsafe {
            vst1q_u16(zs, z0);
            vst1q_u16(zs.add(8), z1);
        }
        return 0;
    }
    // (b0, b1) of the block's first character from the 16-byte
    // `ENCODE_2BYTE_PREFIX`, with the low 5 bits of z in b1 (as for AVX-512).
    let z8 = vcombine_u8(vmovn_u16(z0), vmovn_u16(z1));
    let pair = vshlq_n_u8::<1>(vshrq_n_u8::<5>(z8));
    let b0 = vqtbl1q_u8(prefix, pair);
    let b1 = vorrq_u8(
        vqtbl1q_u8(prefix, vorrq_u8(pair, dup(1))),
        vandq_u8(z8, dup(31)),
    );
    // SAFETY: the caller guarantees 32 writable bytes at `d`.
    unsafe { vst2q_u8(d, uint8x16x2_t(b0, b1)) };
    #[cfg(test)]
    NEON_ENCODE_WINDOWS.with(|n| n.set(n.get() + 1));
    32
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
