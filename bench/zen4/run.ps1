# Windows version of run.sh (Windows PowerShell 5.1 or PowerShell 7).
# usage: double-click run.cmd, or: powershell -ExecutionPolicy Bypass -File bench\zen4\run.ps1
# env:   RUNS=3 ROUNDS=31 SAMPLE_MS=10 CPU=2 (logical CPU to pin the benchmark to)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$utf8 = [System.Text.UTF8Encoding]::new($false)

$Z = $PSScriptRoot
$Root = (Resolve-Path (Join-Path $Z '..\..')).Path
$H = Join-Path $Root 'target\zen4-bench'
$Out = Join-Path $Z 'result.txt'
$env:RUSTFLAGS = '-C llvm-args=-align-all-functions=6 -C llvm-args=-align-all-nofallthru-blocks=6'

function Invoke-Checked([string]$what, [scriptblock]$cmd) {
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}
function Edit-Text([string]$path, [string[]]$pairs) {
    $s = [IO.File]::ReadAllText($path)
    for ($i = 0; $i -lt $pairs.Count; $i += 2) { $s = $s -replace $pairs[$i], $pairs[$i + 1] }
    [IO.File]::WriteAllText($path, $s, $utf8)
}

if (Test-Path $H) { Remove-Item -Recurse -Force $H }
New-Item -ItemType Directory -Force $H | Out-Null
Copy-Item -Recurse -Force (Join-Path $Z 'harness\*') $H

$tarball = Join-Path $H 'src.tar'
# Windows' own bsdtar: a GNU tar earlier in PATH (Git Bash) reads C:\ as a host.
$tar = Join-Path $env:SystemRoot 'System32\tar.exe'
if (-not (Test-Path $tar)) { $tar = 'tar' }
Invoke-Checked 'git archive' { git -C $Root archive -o $tarball HEAD Cargo.toml src base32768-table benches tests test-resources }
foreach ($v in 'head', 'vl') {
    $d = Join-Path $H $v
    New-Item -ItemType Directory -Force $d | Out-Null
    Invoke-Checked 'tar' { & $tar -xf $tarball -C $d }
    Edit-Text (Join-Path $d 'Cargo.toml') @(
        '(?m)^name = "base32768"', "name = `"base32768-$v`"",
        '(?m)^base32768-table = \{ path = "base32768-table" \}', "base32768-table = { path = `"base32768-table`", package = `"base32768-table-$v`" }",
        '(?ms)^\[workspace\]\r?\nmembers = [^\n]*\n', '')
    Edit-Text (Join-Path $d 'base32768-table\Cargo.toml') @('(?m)^name = "base32768-table"', "name = `"base32768-table-$v`"")
}
Remove-Item $tarball
Copy-Item -Force (Join-Path $Z 'vl\lib.rs') (Join-Path $H 'vl\src\lib.rs')
$api = [IO.File]::ReadAllText((Join-Path $Z 'bench_levels.rs'))
foreach ($v in 'head', 'vl') { [IO.File]::AppendAllText((Join-Path $H "$v\src\lib.rs"), $api, $utf8) }

$log = New-Object System.Collections.Generic.List[string]
function Write-Log([string]$line) { Write-Host $line; $log.Add($line) }

$commit = (git -C $Root rev-parse --short HEAD)
Write-Log "# base32768 Zen 4 bench, $commit, $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mmZ'))"
Write-Log "rustc: $(rustc --version)"
Write-Log "os: $([Environment]::OSVersion.VersionString)"
foreach ($v in 'head', 'vl') {
    Push-Location (Join-Path $H $v)
    $saved = $env:RUSTFLAGS; $env:RUSTFLAGS = ''
    $t = (cargo test -q --release --lib 2>&1 | ForEach-Object { "$_" } | Select-String -Pattern 'test result|panicked') -join ' '
    $env:RUSTFLAGS = $saved
    Pop-Location
    Write-Log "tests ($v): $t"
}

Push-Location $H
Invoke-Checked 'cargo build' { cargo build -q --release }
Pop-Location

# Pin to one logical CPU and raise the priority; the benchmark inherits both.
$cpu = if ($env:CPU) { [int]$env:CPU } else { 2 }
$me = [System.Diagnostics.Process]::GetCurrentProcess()
$oldAffinity = $me.ProcessorAffinity
$oldPriority = $me.PriorityClass
try {
    $me.ProcessorAffinity = [IntPtr]([long]1 -shl $cpu)
    $me.PriorityClass = 'High'
    Write-Log "pinned to CPU $cpu"
    & (Join-Path $H 'target\release\zen4-bench.exe') | ForEach-Object { Write-Log "$_" }
    if ($LASTEXITCODE -ne 0) { throw "benchmark failed (exit code $LASTEXITCODE)" }
} finally {
    $me.ProcessorAffinity = $oldAffinity
    $me.PriorityClass = $oldPriority
    [IO.File]::WriteAllLines($Out, $log, $utf8)
}
Write-Host "saved to $Out"
