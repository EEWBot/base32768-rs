#!/bin/bash
# Zen 4 benchmark: the current code's fast paths forced by level in one binary
# (Encode: Scalar 8 / AVX2 32 / AVX-512 32; Decode: Scalar 8 / SSSE3 8 /
# AVX-512 Rank 32), plus the 256-bit AVX-512VL prototype in bench/zen4/vl.
#
# usage: bash bench/zen4/run.sh            (results also go to bench/zen4/result.txt)
# env:   RUNS=3 ROUNDS=31 SAMPLE_MS=10 CPU=2 (core to pin to with taskset)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
Z=$ROOT/bench/zen4
H=$ROOT/target/zen4-bench
export RUSTFLAGS="-C llvm-args=-align-all-functions=6 -C llvm-args=-align-all-nofallthru-blocks=6"
OUT=$Z/result.txt

rm -rf "$H"; mkdir -p "$H"
cp -r "$Z/harness/." "$H/"
for v in head vl; do
  mkdir -p "$H/$v"
  (cd "$ROOT" && git archive HEAD Cargo.toml src base32768-table benches tests test-resources) | tar -xf - -C "$H/$v"
  sed -i.bak "s/^name = \"base32768\"/name = \"base32768-$v\"/; s/^base32768-table = { path = \"base32768-table\" }/base32768-table = { path = \"base32768-table\", package = \"base32768-table-$v\" }/; /^\[workspace\]/,/^members/d" "$H/$v/Cargo.toml"
  sed -i.bak "s/^name = \"base32768-table\"/name = \"base32768-table-$v\"/" "$H/$v/base32768-table/Cargo.toml"
done
cp "$Z/vl/lib.rs" "$H/vl/src/lib.rs"
for v in head vl; do
  cat "$Z/bench_levels.rs" >> "$H/$v/src/lib.rs"
done

{
  echo "# base32768 Zen 4 bench, $(cd "$ROOT" && git rev-parse --short HEAD), $(date -u +%Y-%m-%dT%H:%MZ)"
  echo "rustc: $(rustc --version)"
  uname -srm 2>/dev/null || true
  for v in head vl; do
    echo "tests ($v): $(cd "$H/$v" && RUSTFLAGS= cargo test -q --release --lib 2>&1 | grep -E 'test result|panicked' | tr '\n' ' ')"
  done
} | tee "$OUT"

(cd "$H" && cargo build -q --release)
PIN=()
if command -v taskset >/dev/null; then PIN=(taskset -c "${CPU:-2}"); fi
${PIN[@]+"${PIN[@]}"} "$H/target/release/zen4-bench" | tee -a "$OUT"
echo "saved to $OUT"
