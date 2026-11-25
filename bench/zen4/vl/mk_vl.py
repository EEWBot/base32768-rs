"""Builds the AVX-512VL (256-bit) variant: HEAD lib.rs with the AVX-512 encode
and decode kernels replaced by 16-character YMM versions."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding='utf-8').read().replace('\r\n', '\n')


def rep(a, b, count=1):
    global s
    assert s.count(a) == count, (s.count(a), a)
    s = s.replace(a, b)


# ---- encode loop: 30 bytes -> 16 characters, two per iteration.
rep('''    if LEVEL >= LEVEL_AVX512
        && b.len() >= 60
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vbmi")
        && std::is_x86_feature_detected!("avx512vbmi2")
    {
        #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2")]''', '''    if LEVEL >= LEVEL_AVX512
        && b.len() >= 30
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vl")
        && std::is_x86_feature_detected!("avx512vbmi")
        && std::is_x86_feature_detected!("avx512vbmi2")
    {
        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2")]''')
rep('''            while done + 120 <= b.len() {
                pos += unsafe { simd::encode32_avx512(b.as_ptr().add(done), dst.add(pos)) };
                pos += unsafe { simd::encode32_avx512(b.as_ptr().add(done + 60), dst.add(pos)) };
                done += 120;
            }
            if done + 60 <= b.len() {
                pos += unsafe { simd::encode32_avx512(b.as_ptr().add(done), dst.add(pos)) };
                done += 60;
            }''', '''            while done + 60 <= b.len() {
                pos += unsafe { simd::encode16_avx512vl(b.as_ptr().add(done), dst.add(pos)) };
                pos += unsafe { simd::encode16_avx512vl(b.as_ptr().add(done + 30), dst.add(pos)) };
                done += 60;
            }
            if done + 30 <= b.len() {
                pos += unsafe { simd::encode16_avx512vl(b.as_ptr().add(done), dst.add(pos)) };
                done += 30;
            }''')

# ---- decode loop: 48 bytes -> 30 bytes.
rep('''        if AVX512 && pos + 96 <= len {
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
            }''', '''        if AVX512 && pos + 48 <= len {
            // SAFETY: `pos + 48 <= len`. Those 48 input bytes account for
            // 24 * 15 / 8 = 45 bytes of `dst` capacity, more than the 30
            // bytes written.
            let groups = unsafe { simd::decode16_avx512vl(src.add(pos), dst.add(written)) };
            if groups == 2 {
                pos += 48;
                written += 30;
                continue;
            }''')
rep('''            // `groups <= 3`.''', '''            // `groups <= 1`.''')
rep('''        && bytes.len() >= 96
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vbmi")
        && std::is_x86_feature_detected!("avx512bitalg")
    {
        #[target_feature(enable = "ssse3,avx512f,avx512bw,avx512vbmi,avx512bitalg")]''', '''        && bytes.len() >= 48
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vl")
        && std::is_x86_feature_detected!("avx512vbmi")
        && std::is_x86_feature_detected!("avx512bitalg")
    {
        #[target_feature(enable = "ssse3,avx512f,avx512bw,avx512vl,avx512vbmi,avx512bitalg")]''')

# ---- kernels.
rep('''    #[cfg(all(test, target_arch = "x86_64"))]
    thread_local! {''', r'''    /// Encodes the 30 bytes at `src` as 16 characters at `d` with 256-bit
    /// AVX-512VL and returns the number of bytes written (32..=48). Writes
    /// nothing past them. Code points come from the 128-byte `ENCODE_AVX2`
    /// offset table (two vpermi2b) like the AVX2 path.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2")]
    #[inline]
    pub unsafe fn encode16_avx512vl(src: *const u8, d: *mut u8) -> usize {
        use crate::table;
        use core::arch::x86_64::*;

        const fn unpack_idx(second: bool) -> [u8; 32] {
            let mut m = [0u8; 32];
            let mut i = 0;
            while i < 16 {
                let j = 15 * i / 8 + second as usize;
                m[2 * i] = (j + 1) as u8;
                m[2 * i + 1] = j as u8;
                i += 1;
            }
            m
        }
        const fn unpack_shift(right: bool) -> [u16; 16] {
            let mut m = [0u16; 16];
            let mut i = 0;
            while i < 16 {
                let o = (15 * i % 8) as u16;
                m[i] = if right { 8 - o } else { o };
                i += 1;
            }
            m
        }
        /// vpermt2b indices: (b0, b1) from `pair`, b2 from the second table;
        /// output bytes 32 * part .. 32 * part + 32 of the 48.
        const fn utf8_idx(part: usize) -> [u8; 32] {
            let mut m = [0u8; 32];
            let mut i = 0;
            while i < 32 {
                let t = 32 * part + i;
                let k = t / 3;
                m[i] = if t >= 48 {
                    0
                } else {
                    match t % 3 {
                        0 => 2 * k,
                        1 => 2 * k + 1,
                        _ => 32 + 2 * k,
                    }
                } as u8;
                i += 1;
            }
            m
        }
        const fn utf8_idx2() -> [u8; 32] {
            let mut m = [0u8; 32];
            let mut k = 0;
            while k < 16 {
                m[2 * k] = (2 * k + 1) as u8;
                m[2 * k + 1] = (32 + 2 * k) as u8;
                k += 1;
            }
            m
        }
        static IDX_A: [u8; 32] = unpack_idx(false);
        static IDX_B: [u8; 32] = unpack_idx(true);
        static SHL: [u16; 16] = unpack_shift(false);
        static SHR: [u16; 16] = unpack_shift(true);
        static OUT0: [u8; 32] = utf8_idx(0);
        static OUT1: [u8; 32] = utf8_idx(1);
        static OUT2: [u8; 32] = utf8_idx2();
        static LUT: [u8; 128] = table::ENCODE_AVX2;
        // SAFETY: each load reads 32 bytes from a table at least that long.
        let ld = |p: *const u8| unsafe { _mm256_loadu_si256(p.cast()) };
        let w = |x: i16| _mm256_set1_epi16(x);
        const EVEN: u32 = 0x5555_5555;

        // SAFETY: the caller guarantees 30 readable bytes at `src`.
        let v = unsafe { _mm256_maskz_loadu_epi8((1 << 30) - 1, src.cast()) };
        let x = _mm256_permutexvar_epi8(ld(IDX_A.as_ptr()), v);
        let y = _mm256_permutexvar_epi8(ld(IDX_B.as_ptr()), v);
        let z = _mm256_srli_epi16::<1>(_mm256_or_si256(
            _mm256_sllv_epi16(x, ld(SHL.as_ptr().cast())),
            _mm256_srlv_epi16(y, ld(SHR.as_ptr().cast())),
        ));

        // Offset in blocks: byte (z >> 5) & 127 of the table, one vpermi2b
        // per 64-byte half, picked by bit 6 of the index.
        let idx = _mm256_srli_epi16::<5>(z);
        let lo = _mm256_maskz_permutex2var_epi8(EVEN, ld(LUT.as_ptr()), idx, ld(LUT[32..].as_ptr()));
        let hi = _mm256_maskz_permutex2var_epi8(EVEN, ld(LUT[64..].as_ptr()), idx, ld(LUT[96..].as_ptr()));
        let lut = _mm256_mask_blend_epi16(_mm256_test_epi16_mask(idx, w(64)), lo, hi);
        let lut = _mm256_mask_add_epi16(lut, _mm256_cmpgt_epu16_mask(z, w(0xD3F)), lut, w(256));
        // Outside the table: 306, 307 or 309.
        let d1 = _mm256_mask_add_epi16(w(306), _mm256_cmpgt_epu16_mask(z, w(0x275F)), w(306), w(1));
        let default = _mm256_mask_add_epi16(d1, _mm256_cmpgt_epu16_mask(z, w(0x795F)), d1, w(2));
        let use_lut = _mm256_cmplt_epu16_mask(z, w(0xDC0)) | _mm256_cmpgt_epu16_mask(z, w(0x7DDF));
        let off = _mm256_mask_blend_epi16(use_lut, default, lut);
        let cp = _mm256_add_epi16(z, _mm256_slli_epi16::<5>(off));

        let two = _mm256_cmplt_epu16_mask(cp, w(0x800));
        let b0 =
            _mm256_maskz_mov_epi16(!two, _mm256_or_si256(_mm256_srli_epi16::<12>(cp), w(0xE0)));
        let b1 = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16::<6>(cp), w(0x3F)),
            _mm256_mask_blend_epi16(two, w(0x80), w(0xC0)),
        );
        let b2 = _mm256_or_si256(_mm256_and_si256(cp, w(0x3F)), w(0x80));
        let pair = _mm256_or_si256(b0, _mm256_slli_epi16::<8>(b1));

        #[cfg(test)]
        AVX512_ENCODE_BLOCKS.with(|n| n.set(n.get() + 1));

        // SAFETY (stores below): the caller guarantees 48 writable bytes at `d`.
        if two == 0 {
            let out0 = _mm256_permutex2var_epi8(pair, ld(OUT0.as_ptr()), b2);
            let out1 = _mm256_permutex2var_epi8(pair, ld(OUT1.as_ptr()), b2);
            unsafe {
                _mm256_storeu_si256(d.cast(), out0);
                _mm_storeu_si128(d.add(32).cast(), _mm256_castsi256_si128(out1));
            }
            return 48;
        }
        core::hint::cold_path();
        if two == u16::MAX {
            let out = _mm256_permutex2var_epi8(pair, ld(OUT2.as_ptr()), b2);
            unsafe { _mm256_storeu_si256(d.cast(), out) };
            return 32;
        }
        let out0 = _mm256_permutex2var_epi8(pair, ld(OUT0.as_ptr()), b2);
        let out1 = _mm256_permutex2var_epi8(pair, ld(OUT1.as_ptr()), b2);
        let m0 = _mm256_test_epi8_mask(out0, out0);
        let m1 = _mm256_test_epi8_mask(out1, out1) & 0xFFFF;
        let (n0, n1) = (m0.count_ones() as usize, m1.count_ones() as usize);
        unsafe {
            _mm256_mask_storeu_epi8(d.cast(), ((1u64 << n0) - 1) as u32, _mm256_maskz_compress_epi8(m0, out0));
            _mm256_mask_storeu_epi8(d.add(n0).cast(), (1 << n1) - 1, _mm256_maskz_compress_epi8(m1, out1));
        }
        n0 + n1
    }

    /// Decodes up to 16 three-byte 15-bit characters at `p .. p + 48` with
    /// 256-bit AVX-512VL. Returns how many leading 8-character groups (0..=2)
    /// hold only 3-byte 15-bit characters; those are written to `d` as 15
    /// bytes each, and nothing else is written.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512bitalg")]
    #[inline]
    pub unsafe fn decode16_avx512vl(p: *const u8, d: *mut u8) -> usize {
        use crate::table;
        use core::arch::x86_64::*;

        const fn char_bytes<const K: usize>() -> [u8; 32] {
            let mut m = [0u8; 32];
            let mut i = 0;
            while i < 16 {
                m[2 * i] = (3 * i + K) as u8;
                i += 1;
            }
            m
        }
        const fn pack_order() -> [u8; 32] {
            let mut m = [0u8; 32];
            let mut j = 0;
            while j < 30 {
                let (g, t) = (j / 15, j % 15);
                m[j] = (16 * g + 4 * (t / 4) + 3 - t % 4) as u8;
                j += 1;
            }
            m
        }
        const fn shifts(v: [u32; 4]) -> [u32; 8] {
            let mut m = [0u32; 8];
            let mut i = 0;
            while i < 8 {
                m[i] = v[i % 4];
                i += 1;
            }
            m
        }
        static I0: [u8; 32] = char_bytes::<0>();
        static I1: [u8; 32] = char_bytes::<1>();
        static I2: [u8; 32] = char_bytes::<2>();
        static PACK: [u8; 32] = pack_order();
        static SHL: [u32; 8] = shifts([2, 4, 6, 8]);
        static SHR: [u32; 8] = shifts([28, 26, 24, 32]);
        static DESC: [u16; 64] = table::PAGE_DESC;
        static HMASK: [u16; 32] = table::HALF_MASK;
        static HPREFIX: [u16; 32] = table::HALF_PREFIX;
        // SAFETY (every use): `p` points into a table at least 32 bytes long.
        let ld = |p: *const u8| unsafe { _mm256_loadu_si256(p.cast()) };
        let w = |x: i16| _mm256_set1_epi16(x);
        const EVEN: u32 = 0x5555_5555;

        // SAFETY: the caller guarantees `p .. p + 48` is readable.
        let (a, b) = unsafe { (ld(p), _mm256_maskz_loadu_epi8(0xFFFF, p.add(32).cast())) };
        let b0 = _mm256_maskz_permutex2var_epi8(EVEN, a, ld(I0.as_ptr()), b);
        let lead_ok = _mm256_cmpeq_epi16_mask(_mm256_and_si256(b0, w(0xF0)), w(0xE0));
        let b1 = _mm256_maskz_permutex2var_epi8(EVEN, a, ld(I1.as_ptr()), b);
        let b2 = _mm256_maskz_permutex2var_epi8(EVEN, a, ld(I2.as_ptr()), b);
        let cp = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_slli_epi16::<12>(b0),
                _mm256_slli_epi16::<6>(_mm256_and_si256(b1, w(0x3F))),
            ),
            _mm256_and_si256(b2, w(0x3F)),
        );

        // 64-entry page table: one vpermi2w per 32 entries, picked by bit 5.
        let page = _mm256_srli_epi16::<10>(cp);
        let d_lo = _mm256_permutex2var_epi16(ld(DESC.as_ptr().cast()), page, ld(DESC[16..].as_ptr().cast()));
        let d_hi = _mm256_permutex2var_epi16(ld(DESC[32..].as_ptr().cast()), page, ld(DESC[48..].as_ptr().cast()));
        let desc = _mm256_mask_blend_epi16(_mm256_test_epi16_mask(page, w(32)), d_lo, d_hi);
        let half = _mm256_ternarylogic_epi32::<0xF8>(desc, _mm256_srli_epi16::<9>(cp), w(1));
        let mask = _mm256_permutex2var_epi16(ld(HMASK.as_ptr().cast()), half, ld(HMASK[16..].as_ptr().cast()));
        let hprefix =
            _mm256_permutex2var_epi16(ld(HPREFIX.as_ptr().cast()), half, ld(HPREFIX[16..].as_ptr().cast()));
        let bit = _mm256_sllv_epi16(w(1), _mm256_and_si256(_mm256_srli_epi16::<5>(cp), w(15)));
        let valid = lead_ok & _mm256_test_epi16_mask(mask, bit);
        let groups = (valid.trailing_ones() / 8) as usize;
        if groups == 0 {
            return 0;
        }
        let rank = _mm256_popcnt_epi16(_mm256_and_si256(mask, _mm256_sub_epi16(bit, w(1))));
        let blocks = _mm256_add_epi16(
            _mm256_add_epi16(_mm256_srli_epi16::<5>(desc), hprefix),
            rank,
        );
        let z = _mm256_or_si256(_mm256_slli_epi16::<5>(blocks), _mm256_and_si256(cp, w(31)));

        let m = _mm256_madd_epi16(z, _mm256_set1_epi32(((-1i32) << 16) | 0x8000));
        let pr = _mm256_sub_epi32(_mm256_setzero_si256(), m);
        let words = _mm256_or_si256(
            _mm256_sllv_epi32(pr, ld(SHL.as_ptr().cast())),
            _mm256_srlv_epi32(_mm256_bsrli_epi128::<4>(pr), ld(SHR.as_ptr().cast())),
        );
        let out = _mm256_permutexvar_epi8(ld(PACK.as_ptr()), words);
        #[cfg(test)]
        AVX512_BLOCKS.with(|n| n.set(n.get() + 1));

        if groups == 2 {
            // SAFETY: the caller guarantees 30 writable bytes at `d`.
            unsafe { _mm256_mask_storeu_epi8(d.cast(), (1 << 30) - 1, out) };
            return 2;
        }
        core::hint::cold_path();
        unsafe { _mm256_mask_storeu_epi8(d.cast(), (1 << 15) - 1, out) };
        1
    }

    #[cfg(all(test, target_arch = "x86_64"))]
    thread_local! {''')

# The old 512-bit kernels stay but are unused.
rep('''    pub unsafe fn encode32_avx512(''', '''    #[allow(dead_code)]
    pub unsafe fn encode32_avx512(''')
rep('''    pub unsafe fn decode32_avx512(''', '''    #[allow(dead_code)]
    pub unsafe fn decode32_avx512(''')
rep('''        assert_eq!(blocks() - before, 3000 / 60);''', '''        assert_eq!(blocks() - before, 3000 / 30);''')

open(dst, 'w', encoding='utf-8', newline='\n').write(s)
