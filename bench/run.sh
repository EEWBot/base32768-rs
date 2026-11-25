#!/bin/bash
# Benchmarks each variant in bench/variants against the commit it is checked
# out with: `old` is the repository's crate, `new` the same crate with the
# variant's lib.rs (and build.rs, if any). Prints the new/old speedup per
# dataset for 3 runs each, plus a HEAD vs HEAD control.
# usage: bench/run.sh [variant...]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
H=$ROOT/target/bench-harness
export RUSTFLAGS="-C llvm-args=-align-all-functions=6 -C llvm-args=-align-all-nofallthru-blocks=6"
RUNS=${RUNS:-3}

rm -rf "$H"; mkdir -p "$H"
cp -r "$ROOT/bench/harness/." "$H/"
for v in old new; do
  mkdir -p "$H/$v"
  (cd "$ROOT" && git ls-files -z -- Cargo.toml src base32768-table | xargs -0 tar -cf -) | tar -xf - -C "$H/$v"
  sed -i.bak "s/^name = \"base32768\"/name = \"base32768-$v\"/; s/^base32768-table = { path = \"base32768-table\" }/base32768-table = { path = \"base32768-table\", package = \"base32768-table-$v\" }/; /^\[workspace\]/,/^members/d" "$H/$v/Cargo.toml"
  sed -i.bak "s/^name = \"base32768-table\"/name = \"base32768-table-$v\"/" "$H/$v/base32768-table/Cargo.toml"
  sed -i.bak '/^\[dev-dependencies\]/,$d' "$H/$v/Cargo.toml"
done
cp "$H/new/src/lib.rs" "$H/head_lib.rs"
cp "$H/new/base32768-table/build.rs" "$H/head_build.rs"

ratios() {
  awk -F'|' -v sec="$1" '
    function us(x) { split(x, a, " "); return a[1] * ((x ~ / ms/) ? 1000 : (x ~ / ns/) ? 0.001 : 1) }
    /^\| (encode|decode) \|/ { cur = ($2 ~ sec) ; next }
    cur && /^\| [0-9]/ { printf "  %-22s %10s -> %10s  %.3f\n", $2, $3, $5, us($3) / us($5) }
  '
}

run_variant() {
  local name=$1 dir=$2
  cp "$H/head_lib.rs" "$H/new/src/lib.rs"
  cp "$H/head_build.rs" "$H/new/base32768-table/build.rs"
  if [ -n "$dir" ]; then
    cp "$dir/lib.rs" "$H/new/src/lib.rs"
    [ -f "$dir/build.rs" ] && cp "$dir/build.rs" "$H/new/base32768-table/build.rs"
  fi
  echo "=================== $name"
  # Tests of the variant itself (the encode/decode level comparisons cover
  # the NEON paths on aarch64).
  if [ -n "$dir" ]; then
    (cd "$H/new" && RUSTFLAGS= cargo test -q --release --lib 2>&1 | grep -E "test result|panicked|FAILED|error" | head -5) || true
  fi
  (cd "$H" && cargo build -q --release --bin abh)
  for i in $(seq "$RUNS"); do
    (cd "$H" && MIN=1 ./target/release/abh 2>/dev/null > out.txt) || { echo "  run $i failed"; continue; }
    echo "-- run $i encode"; ratios encode < "$H/out.txt"
    echo "-- run $i decode"; ratios decode < "$H/out.txt"
  done
}

if [ $# -gt 0 ]; then
  variants=("$@")
else
  variants=()
  for d in "$ROOT"/bench/variants/*/; do variants+=("$(basename "$d")"); done
fi
run_variant "HEAD vs HEAD (control)" ""
for v in "${variants[@]}"; do
  run_variant "$v" "$ROOT/bench/variants/$v"
done
run_variant "HEAD vs HEAD (control, end)" ""
