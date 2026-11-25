# Zen 4 ベンチマーク

現行コード（`HEAD`）の各実行レベルを同一バイナリ内で強制指定して比較し、あわせて 256-bit AVX-512VL 版の試作も測定します。

| 列 | Encode | Decode |
|---|---|---|
| 1 | Scalar 8文字 | Scalar 8文字 |
| 2 | AVX2 32文字 | SSSE3 8文字 |
| 3 | AVX-512 32文字（現行） | AVX-512 Rank 32文字（現行） |
| 4 | AVX-512VL 16文字（試作、YMM） | AVX-512VL Rank 16文字（試作、YMM） |

データセットは 10 KB ramp、1 MB の ramp / random / zeros/64B / zeros/16B / all zero / 1/64 nonzero / 1/4 zero irr / 1/2 zero irr の9種類です。

## 必要なもの

- 最新の stable Rust（`rustup update stable`）と git
- Linux: bash と tar（`taskset` があれば1コアに固定します）
- Windows 10 / 11: 追加のツールは不要です（PowerShell と Windows 標準の `tar.exe` を使います）

## 実行方法

```sh
git clone -b pref/zen4-bench <this repository>
cd base32768-rs
```

- **Linux**: `bash bench/zen4/run.sh`
- **Windows**: エクスプローラーで `bench\zen4\run.cmd` をダブルクリックします。コマンドラインから実行する場合は `powershell -ExecutionPolicy Bypass -File bench\zen4\run.ps1` です。

既定の設定では数分かかります。結果は画面に表示され、`bench/zen4/result.txt` にも保存されます。このファイルをそのまま送ってください。

実行時には次の処理が自動で行われます。

1. `HEAD` のクレートを `target/zen4-bench/head` に、試作版を `target/zen4-bench/vl` にコピーします。
2. それぞれのテストスイートを実行します（結果は `tests (head)` / `tests (vl)` の行に出ます）。
3. ベンチマークの前に、9種類のデータと全15-bit値×全バイト位置について、全列の結果が Scalar と一致することを確認します。

## 測定のコツ

- ブラウザなど負荷の高いアプリケーションは閉じてください。
- Linux では可能であれば `cpupower frequency-set -g performance` を設定してください。
- Windows では電源プランを「高パフォーマンス」（または「最適なパフォーマンス」）にしてください。ベンチマークは論理 CPU 2 に固定し、優先度を「高」にして実行します。
- 固定先のコアは変更できます（既定は 2）。Linux では `CPU=4 bash bench/zen4/run.sh`、Windows の PowerShell では `$env:CPU=4` を設定してから実行します。他の環境変数も同じ方法で指定します。
- 環境変数: `RUNS`（既定 3、全体の繰り返し回数）、`ROUNDS`（既定 31、1回あたりの計測ラウンド数）、`SAMPLE_MS`（既定 10）

## 構成

| ファイル | 内容 |
|---|---|
| `run.sh` | クレートのコピー、テスト、ビルド、実行（Linux / macOS） |
| `run.ps1`, `run.cmd` | 同じ処理の Windows 版（`run.cmd` は `run.ps1` を起動するだけ） |
| `harness/` | ベンチマーク本体（`zen4-bench`） |
| `bench_levels.rs` | 各 `lib.rs` の末尾に追記する、レベル強制用の公開関数 |
| `vl/lib.rs` | AVX-512VL 試作版（`vl/mk_vl.py` で `HEAD` の `src/lib.rs` から生成） |
