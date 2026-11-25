#[global_allocator]
static G: mimalloc::MiMalloc = mimalloc::MiMalloc;
use std::hint::black_box;
use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Iterations so that one sample takes about 20 ms.
fn calibrate<T>(f: &dyn Fn() -> T) -> usize {
    let mut n = 1;
    loop {
        let t = Instant::now();
        for _ in 0..n {
            black_box(f());
        }
        if t.elapsed().as_secs_f64() > 0.02 {
            return n;
        }
        n *= 2;
    }
}

fn fmt_time(sec: f64) -> String {
    if sec < 1e-6 { format!("{:.1} ns", sec * 1e9) }
    else if sec < 1e-3 { format!("{:.2} µs", sec * 1e6) }
    else { format!("{:.2} ms", sec * 1e3) }
}

fn ab<T>(name: &str, bytes: usize, a: impl Fn() -> T, b: impl Fn() -> T) {
    let (na, nb) = (calibrate(&a), calibrate(&b));
    let time = |f: &dyn Fn() -> T, n: usize| {
        let t = Instant::now();
        for _ in 0..n {
            black_box(f());
        }
        t.elapsed().as_secs_f64() / n as f64
    };
    let (mut ta, mut tb) = (vec![], vec![]);
    for round in 0..60 {
        if round % 2 == 0 { ta.push(time(&a, na)); tb.push(time(&b, nb)); } else { tb.push(time(&b, nb)); ta.push(time(&a, na)); }
    }
    let mn = |v: Vec<f64>| v.into_iter().fold(f64::MAX, f64::min); let (ma, mb) = if std::env::var_os("MIN").is_some() { (mn(ta), mn(tb)) } else { (median(ta), median(tb)) };
    let gbs = |s: f64| bytes as f64 / s / 1e9;
    println!("| {name} | {} | {:.3} | {} | {:.2} | **{:.1}x** |", fmt_time(ma), gbs(ma), fmt_time(mb), gbs(mb), ma / mb);
}

fn main() {
    let ramp: Vec<u8> = (0..1_000_000).map(|i| (i & 0xff) as u8).collect();
    let mut st = 0x9E37_79B9_7F4A_7C15u64;
    let random: Vec<u8> = (0..1_000_000)
        .map(|_| { st ^= st << 13; st ^= st >> 7; st ^= st << 17; (st >> 56) as u8 })
        .collect();
    let one = vec![127u8];
    let ten_kb: Vec<u8> = ramp[..10_000].to_vec();
    // Zero runs make 2-byte characters: about 1 in N 15-bit values.
    let sparse = |every: usize| -> Vec<u8> { random.iter().enumerate().map(|(i, &b)| if i % every < 2 { 0 } else { b }).collect() };
    let (z64, z16) = (sparse(64), sparse(16));
    let zeros = vec![0u8; 1_000_000];
    // Mostly zero: one random byte every 64.
    let mostly0: Vec<u8> = random.iter().enumerate().map(|(i, &b)| if i % 64 == 0 { b } else { 0 }).collect();
    // Zero bytes at irregular positions (each byte zero with probability 1/4 or 1/2).
    let irregular = |shift: u32| -> Vec<u8> {
        let mut st = 0x2545_F491_4F6C_DD1Du64;
        random.iter().map(|&b| { st ^= st << 13; st ^= st >> 7; st ^= st << 17; if (st >> 40) & ((1 << shift) - 1) == 0 { 0 } else { b } }).collect()
    };
    let (irr4, irr2) = (irregular(2), irregular(1));
    let inputs: [(&str, &Vec<u8>); 9] =
        [("10 KB", &ten_kb), ("1 MB ramp", &ramp), ("1 MB random", &random), ("1 MB zeros/64B", &z64), ("1 MB zeros/16B", &z16), ("1 MB all zero", &zeros), ("1 MB 1/64 nonzero", &mostly0), ("1 MB 1/4 zero irr", &irr4), ("1 MB 1/2 zero irr", &irr2)];

    // Every 15-bit value, at every byte alignment.
    let mut all = vec![0u8; 32768 * 15 / 8];
    for z in 0..32768usize {
        for k in 0..15 {
            if z >> (14 - k) & 1 == 1 {
                let bit = z * 15 + k;
                all[bit / 8] |= 0x80 >> (bit % 8);
            }
        }
    }
    for shift in 0..15 {
        let mut d = vec![0xA5u8; shift];
        d.extend_from_slice(&all);
        let s = new::encode(&d);
        if std::env::var_os("NOCHECK").is_none() {
        assert_eq!(new::decode(&s).unwrap(), d, "all z, shift {shift}");
        assert_eq!(old::encode(&d), s);
        }
    }
    // An invalid 3-byte character (U+3000) at every character position.
    let s = new::encode(&random[..3000]);
    let starts: Vec<usize> = s.char_indices().map(|(i, _)| i).collect();
    for &i in &starts {
        let c = s[i..].chars().next().unwrap();
        let bad = format!("{}\u{3000}{}", &s[..i], &s[i + c.len_utf8()..]);
        let (o, n) = (old::decode(&bad), new::decode(&bad));
        assert!(n.is_err(), "pos {i}");
        assert_eq!(format!("{o:?}"), format!("{n:?}"), "pos {i}");
    }
    eprintln!("correctness checks passed ({} positions)", starts.len());

    println!("| encode | master time | master GB/s | new time | new GB/s | speedup |");
    println!("|---|---|---|---|---|---|");
    for (name, data) in inputs {
        if std::env::var_os("NOCHECK").is_none() {
        assert_eq!(old::encode(data), new::encode(data));
        }
        ab(name, data.len(), || old::encode(black_box(data)), || new::encode(black_box(data)));
    }
    println!();
    println!("| decode | master time | master GB/s | new time | new GB/s | speedup |");
    println!("|---|---|---|---|---|---|");
    for (name, data) in inputs {
        let s = new::encode(data);
        if std::env::var_os("NOCHECK").is_none() {
        assert_eq!(old::decode(&s).unwrap(), *data);
        }
        if std::env::var_os("NOCHECK").is_none() { assert_eq!(new::decode(&s).unwrap(), *data); }
        ab(name, data.len(), || old::decode(black_box(&s)).unwrap(), || new::decode(black_box(&s)).unwrap());
    }
}
