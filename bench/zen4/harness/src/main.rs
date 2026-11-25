//! Zen 4 level comparison: every fast path of the current code, forced by
//! level in one binary, plus the 256-bit AVX-512VL prototype (crate `vl`).
//!
//! Environment: RUNS (default 3) whole passes, ROUNDS (default 31) timed
//! rounds per pass, SAMPLE_MS (default 10) length of one timed sample.
#[global_allocator]
static G: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::hint::black_box;
use std::time::Instant;

fn env(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn fmt_time(sec: f64) -> String {
    if sec < 1e-3 { format!("{:.2} µs", sec * 1e6) } else { format!("{:.3} ms", sec * 1e3) }
}

#[cfg(target_arch = "x86_64")]
fn cpu_brand() -> String {
    use std::arch::x86_64::__cpuid;
    let max = __cpuid(0x8000_0000).eax;
    if max < 0x8000_0004 {
        return "unknown".into();
    }
    let mut b = Vec::new();
    for leaf in 0x8000_0002..=0x8000_0004u32 {
        let r = __cpuid(leaf);
        for x in [r.eax, r.ebx, r.ecx, r.edx] {
            b.extend_from_slice(&x.to_le_bytes());
        }
    }
    String::from_utf8_lossy(&b).trim_matches(|c: char| c == '\0' || c == ' ').to_string()
}

#[cfg(not(target_arch = "x86_64"))]
fn cpu_brand() -> String {
    "not x86-64".into()
}

/// Times each function in interleaved rounds (rotating the order) and
/// returns the median seconds per call of each.
fn measure(fs: &[&dyn Fn()], rounds: usize, sample: f64) -> Vec<f64> {
    let iters: Vec<usize> = fs
        .iter()
        .map(|f| {
            let mut n = 1;
            loop {
                let t = Instant::now();
                for _ in 0..n {
                    f();
                }
                if t.elapsed().as_secs_f64() > sample {
                    return n;
                }
                n *= 2;
            }
        })
        .collect();
    let mut times = vec![Vec::new(); fs.len()];
    for r in 0..rounds {
        for k in 0..fs.len() {
            let i = (k + r) % fs.len();
            let t = Instant::now();
            for _ in 0..iters[i] {
                fs[i]();
            }
            times[i].push(t.elapsed().as_secs_f64() / iters[i] as f64);
        }
    }
    times.into_iter().map(median).collect()
}

type Enc = fn(&[u8]) -> String;
type Dec = fn(&str) -> Result<Vec<u8>, ()>;

fn main() {
    let runs = env("RUNS", 3);
    let rounds = env("ROUNDS", 31);
    let sample = env("SAMPLE_MS", 10) as f64 / 1e3;

    println!("CPU: {}", cpu_brand());
    #[cfg(target_arch = "x86_64")]
    {
        let f = [
            ("avx2", is_x86_feature_detected!("avx2")),
            ("avx512bw", is_x86_feature_detected!("avx512bw")),
            ("avx512vl", is_x86_feature_detected!("avx512vl")),
            ("avx512vbmi", is_x86_feature_detected!("avx512vbmi")),
            ("avx512vbmi2", is_x86_feature_detected!("avx512vbmi2")),
            ("avx512bitalg", is_x86_feature_detected!("avx512bitalg")),
        ];
        let s: Vec<String> = f.iter().map(|(n, b)| format!("{n}={}", if *b { "yes" } else { "NO" })).collect();
        println!("features: {}", s.join(" "));
        if f.iter().any(|(_, b)| !b) {
            println!("WARNING: a missing feature makes that level fall back to a narrower path; its column then measures the fallback.");
        }
    }
    println!("RUNS={runs} ROUNDS={rounds} SAMPLE_MS={}", (sample * 1e3) as usize);

    let ramp: Vec<u8> = (0..1_000_000).map(|i| (i & 0xff) as u8).collect();
    let mut st = 0x9E37_79B9_7F4A_7C15u64;
    let random: Vec<u8> = (0..1_000_000)
        .map(|_| {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            (st >> 56) as u8
        })
        .collect();
    let ten_kb: Vec<u8> = ramp[..10_000].to_vec();
    // Zero runs make 2-byte characters: about 1 in N 15-bit values.
    let sparse = |every: usize| -> Vec<u8> {
        random.iter().enumerate().map(|(i, &b)| if i % every < 2 { 0 } else { b }).collect()
    };
    let (z64, z16) = (sparse(64), sparse(16));
    let zeros = vec![0u8; 1_000_000];
    let mostly0: Vec<u8> =
        random.iter().enumerate().map(|(i, &b)| if i % 64 == 0 { b } else { 0 }).collect();
    // Zero bytes at irregular positions (each byte zero with probability 1/4 or 1/2).
    let irregular = |shift: u32| -> Vec<u8> {
        let mut st = 0x2545_F491_4F6C_DD1Du64;
        random
            .iter()
            .map(|&b| {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                if (st >> 40) & ((1 << shift) - 1) == 0 { 0 } else { b }
            })
            .collect()
    };
    let (irr4, irr2) = (irregular(2), irregular(1));
    let inputs: [(&str, &Vec<u8>); 9] = [
        ("10 KB ramp", &ten_kb),
        ("1 MB ramp", &ramp),
        ("1 MB random", &random),
        ("1 MB zeros/64B", &z64),
        ("1 MB zeros/16B", &z16),
        ("1 MB all zero", &zeros),
        ("1 MB 1/64 nonzero", &mostly0),
        ("1 MB 1/4 zero irr", &irr4),
        ("1 MB 1/2 zero irr", &irr2),
    ];

    let encs: [(&str, Enc); 4] = [
        ("Scalar 8", head::bench_levels::encode_scalar),
        ("AVX2 32", head::bench_levels::encode_simd),
        ("AVX-512 32", head::bench_levels::encode_avx512),
        ("AVX-512VL 16", vl::bench_levels::encode_avx512),
    ];
    let decs: [(&str, Dec); 4] = [
        ("Scalar 8", |s| head::bench_levels::decode_scalar(s).map_err(drop)),
        ("SSSE3 8", |s| head::bench_levels::decode_simd(s).map_err(drop)),
        ("AVX-512 Rank 32", |s| head::bench_levels::decode_avx512(s).map_err(drop)),
        ("AVX-512VL Rank 16", |s| vl::bench_levels::decode_avx512(s).map_err(drop)),
    ];

    // Correctness: every variant agrees with scalar on every dataset and on
    // every 15-bit value at every byte alignment.
    let mut all = vec![0u8; 32768 * 15 / 8];
    for z in 0..32768usize {
        for k in 0..15 {
            if z >> (14 - k) & 1 == 1 {
                let bit = z * 15 + k;
                all[bit / 8] |= 0x80 >> (bit % 8);
            }
        }
    }
    let mut checks: Vec<Vec<u8>> = inputs.iter().map(|(_, d)| d.to_vec()).collect();
    for shift in 0..15 {
        let mut d = vec![0xA5u8; shift];
        d.extend_from_slice(&all);
        d.push(0xA5);
        checks.push(d);
    }
    for d in &checks {
        let s = encs[0].1(d);
        for (name, f) in &encs {
            assert!(f(d) == s, "encode mismatch: {name}");
        }
        for (name, f) in &decs {
            assert!(f(&s).as_deref() == Ok(&d[..]), "decode mismatch: {name}");
        }
    }
    println!("correctness checks passed");

    let encoded: Vec<String> = inputs.iter().map(|(_, d)| encs[0].1(d)).collect();
    // results[op][dataset][variant] = per-run medians
    let mut results = vec![vec![vec![Vec::new(); 4]; inputs.len()]; 2];
    for run in 1..=runs {
        eprintln!("run {run}/{runs}");
        for (i, (_, data)) in inputs.iter().enumerate() {
            let fs: Vec<Box<dyn Fn()>> = encs
                .iter()
                .map(|&(_, f)| Box::new(move || drop(black_box(f(black_box(data))))) as Box<dyn Fn()>)
                .collect();
            let refs: Vec<&dyn Fn()> = fs.iter().map(|b| b.as_ref()).collect();
            for (v, t) in measure(&refs, rounds, sample).into_iter().enumerate() {
                results[0][i][v].push(t);
            }
            let s = &encoded[i];
            let fs: Vec<Box<dyn Fn()>> = decs
                .iter()
                .map(|&(_, f)| Box::new(move || drop(black_box(f(black_box(s))))) as Box<dyn Fn()>)
                .collect();
            let refs: Vec<&dyn Fn()> = fs.iter().map(|b| b.as_ref()).collect();
            for (v, t) in measure(&refs, rounds, sample).into_iter().enumerate() {
                results[1][i][v].push(t);
            }
        }
    }

    for (op, names) in [("Encode", encs.map(|e| e.0)), ("Decode", decs.map(|d| d.0))] {
        let o = (op == "Decode") as usize;
        println!();
        println!("## {op}: median of {runs} runs, GB/s of input{}", if o == 1 { " (decoded bytes)" } else { "" });
        println!();
        println!(
            "| data | {} | {} | {} | {} | VL16 / 512 |",
            names[0], names[1], names[2], names[3]
        );
        println!("|---|---:|---:|---:|---:|---:|");
        for (i, (name, data)) in inputs.iter().enumerate() {
            let t: Vec<f64> = (0..4).map(|v| median(results[o][i][v].clone())).collect();
            let gbs = |s: f64| data.len() as f64 / s / 1e9;
            let cell = |v: usize| {
                if v == 0 {
                    format!("{} ({:.2})", fmt_time(t[0]), gbs(t[0]))
                } else {
                    format!("{} ({:.2}, {:.2}x)", fmt_time(t[v]), gbs(t[v]), t[0] / t[v])
                }
            };
            println!(
                "| {name} | {} | {} | {} | {} | **{:.2}x** |",
                cell(0),
                cell(1),
                cell(2),
                cell(3),
                t[2] / t[3]
            );
        }
        // Run-to-run spread, as max/min of the per-run medians.
        let spread = (0..inputs.len())
            .flat_map(|i| (0..4).map(move |v| (i, v)))
            .map(|(i, v)| {
                let r = &results[o][i][v];
                r.iter().cloned().fold(0.0, f64::max) / r.iter().cloned().fold(f64::MAX, f64::min)
            })
            .fold(1.0, f64::max);
        println!();
        println!("Largest run-to-run spread (max/min of a cell's runs): {:.3}", spread);
    }
    println!();
    println!("Cells: time (GB/s, speedup over Scalar 8). VL16 / 512: speedup of the 256-bit prototype over the current AVX-512 path.");
}
