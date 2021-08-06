use ahash::RandomState;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moka_sketch::MokaSketch;
use std::hash::{BuildHasher, Hash};
use tinylfu::sketch::FrequencySketch;

mod moka_sketch;

fn make_hash<Q: Hash + ?Sized, S: BuildHasher>(hash_builder: &S, val: &Q) -> u64 {
    use core::hash::Hasher;
    let mut state = hash_builder.build_hasher();
    val.hash(&mut state);
    state.finish()
}

fn bench_sketch(c: &mut Criterion) {
    const SIZES: [usize; 6] = [1024, 12_345, 1 << 17, 1 << 25, 1 << 26, 76_543_210];
    let mut group = c.benchmark_group("FrequencySketch::reset");
    for size in SIZES {
        let mut sketch = FrequencySketch::with_capacity(size);
        for i in 0..size {
            sketch.increment(&i);
        }
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter(|| black_box(&mut sketch).reset())
        });
    }
    group.finish();
    let mut group = c.benchmark_group("MokaSketch::frequency");
    for size in SIZES {
        let state = RandomState::new();
        let mut sketch = MokaSketch::with_capacity((size as usize) * 8);
        for i in 0..size {
            sketch.increment(make_hash(&state, &i));
        }
        let sketch = sketch;
        let mut counter: usize = 0;
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter(|| {
                let mut freq = 0;
                let sketch = black_box(&sketch);
                for _ in 0..8 {
                    counter += 1;
                    freq += sketch.frequency(make_hash(&state, &counter)) as u32;
                }
                freq
            })
        });
    }
    group.finish();
    let mut group = c.benchmark_group("FrequencySketch::frequency");
    for size in SIZES {
        let mut sketch = FrequencySketch::with_capacity(size);
        for i in 0..size {
            sketch.increment(&i);
        }
        let sketch = sketch;
        let mut counter: usize = 0;
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter(|| {
                let mut freq = 0;
                let sketch = black_box(&sketch);
                for _ in 0..8 {
                    counter += 1;
                    freq += sketch.frequency(&counter) as u32;
                }
                freq
            })
        });
    }
    group.finish();
    let mut group = c.benchmark_group("MokaSketch::increment");
    for size in SIZES {
        let mut sketch = MokaSketch::with_capacity((size as usize) * 8);
        let state = RandomState::new();
        let mut counter: usize = 0;
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter(|| {
                let mut freq = 0;
                let sketch = black_box(&mut sketch);
                for _ in 0..8 {
                    counter += 1;
                    let hash = make_hash(&state, &counter);
                    sketch.increment(hash);
                    freq += sketch.frequency(hash);
                }
                freq
            })
        });
    }
    group.finish();
    let mut group = c.benchmark_group("FrequencySketch::increment");
    for size in SIZES {
        let mut sketch = FrequencySketch::with_capacity(size);
        let mut counter: usize = 0;
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter(|| {
                let mut freq = 0;
                let sketch = black_box(&mut sketch);
                for _ in 0..8 {
                    counter += 1;
                    freq += sketch.increment(&counter);
                }
                freq
            })
        });
    }
    group.finish();

    let mut group = c.benchmark_group("MokaSketch::reset");
    for size in SIZES {
        let state = RandomState::new();
        let mut sketch = MokaSketch::with_capacity((size as usize) * 8);
        for i in 0..size {
            sketch.increment(make_hash(&state, &i));
        }
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter(|| black_box(&mut sketch).reset())
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sketch);
criterion_main!(benches);
