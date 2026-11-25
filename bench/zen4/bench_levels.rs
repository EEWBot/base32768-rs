
/// Benchmark entry points that force a level (appended by bench/zen4/run.sh).
#[doc(hidden)]
pub mod bench_levels {
    use crate::{DecodeError, LEVEL_AVX512, LEVEL_SCALAR, LEVEL_SIMD, decode_impl, encode_impl};
    pub fn encode_scalar(b: &[u8]) -> String { encode_impl::<LEVEL_SCALAR>(b) }
    pub fn encode_simd(b: &[u8]) -> String { encode_impl::<LEVEL_SIMD>(b) }
    pub fn encode_avx512(b: &[u8]) -> String { encode_impl::<LEVEL_AVX512>(b) }
    pub fn decode_scalar(s: &str) -> Result<Vec<u8>, DecodeError> { decode_impl::<LEVEL_SCALAR>(s) }
    pub fn decode_simd(s: &str) -> Result<Vec<u8>, DecodeError> { decode_impl::<LEVEL_SIMD>(s) }
    pub fn decode_avx512(s: &str) -> Result<Vec<u8>, DecodeError> { decode_impl::<LEVEL_AVX512>(s) }
}
