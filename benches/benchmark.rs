use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn criterion_benchmark(c: &mut Criterion) {
    let one_byte_array: [u8; 1] = [127];
    let ten_kilobytes_array: Vec<u8> = (0..10_000).map(|i| (i & 0xff) as u8).collect();
    let one_megabytes_array: Vec<u8> = (0..1_000_000).map(|i| (i & 0xff) as u8).collect();
    // xorshift64: covers the whole 15-bit value space, unlike the repeating ramp above.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let random_one_megabyte_array: Vec<u8> = (0..1_000_000)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 56) as u8
        })
        .collect();
    let zeros = vec![0u8; 1_000_000];
    // Pack a cycle of all 224 two-byte primary values into 15-bit fields.
    let mut short_values = vec![0u8; 999_990];
    for (group, chunk) in short_values.as_chunks_mut::<15>().0.iter_mut().enumerate() {
        for lane in 0..8 {
            let z = ((8 * group + lane) * 79 % 224) as u8;
            chunk[2 * lane] = z >> (7 - lane);
            if lane < 7 {
                chunk[2 * lane + 1] = z << (1 + lane);
            }
        }
    }
    let datasets: [(&str, &[u8]); 6] = [
        ("OneByte", &one_byte_array),
        ("TenKilobytes", &ten_kilobytes_array),
        ("OneMegabyte", &one_megabytes_array),
        ("ZeroMegabyte", &zeros),
        ("ShortMegabyte", &short_values),
        ("RandomOneMegabyte", &random_one_megabyte_array),
    ];

    let mut group = c.benchmark_group("base32768");
    // Throughput is the binary data size (the encoder's input, the decoder's
    // output), so criterion reports encode and decode in comparable bytes/s.
    for (name, data) in datasets {
        group.throughput(criterion::Throughput::Bytes(data.len() as u64));
        group.bench_function(format!("encoder{name}"), |b| {
            b.iter(|| black_box(base32768::encode(black_box(data))))
        });
    }
    for (name, data) in datasets {
        let encoded = base32768::encode(data);
        group.throughput(criterion::Throughput::Bytes(data.len() as u64));
        group.bench_function(format!("decoder{name}"), |b| {
            b.iter(|| black_box(base32768::decode(black_box(&encoded)).unwrap()))
        });
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
