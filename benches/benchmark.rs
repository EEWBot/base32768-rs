use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn criterion_benchmark(c: &mut Criterion) {
    let one_byte_array: [u8; 1] = [127];
    let ten_kilobytes_array: [u8; 10_000] = (0..10_000)
        .map(|i| (i & 0xff) as u8)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let one_megabytes_array: [u8; 1_000_000] = (0..1_000_000)
        .map(|i| (i & 0xff) as u8)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let one_byte_string = base32768::encode(&one_byte_array);
    let ten_kilobytes_string = base32768::encode(&ten_kilobytes_array);
    let one_megabytes_string = base32768::encode(&one_megabytes_array);

    let mut group = c.benchmark_group("base32758");
    group.throughput(criterion::Throughput::Elements(1));

    group.bench_function("encoderOneByte", |b| {
        b.iter(|| {
            black_box(base32768::encode(black_box(&one_byte_array)));
        })
    });

    group.bench_function("encoderTenKilobytes", |b| {
        b.iter(|| {
            black_box(base32768::encode(black_box(&ten_kilobytes_array)));
        })
    });

    group.bench_function("encoderOneMegabyte", |b| {
        b.iter(|| {
            black_box(base32768::encode(black_box(&one_megabytes_array)));
        })
    });

    group.bench_function("decoderOneByte", |b| {
        b.iter(|| {
            black_box(base32768::decode(black_box(&one_byte_string)).unwrap());
        })
    });

    group.bench_function("decoderTenKilobytes", |b| {
        b.iter(|| {
            black_box(base32768::decode(black_box(&ten_kilobytes_string)).unwrap());
        })
    });

    group.bench_function("decoderOneMegabyte", |b| {
        b.iter(|| {
            black_box(base32768::decode(black_box(&one_megabytes_string)).unwrap());
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
