// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#![allow(unused_mut)]

use criterion::{criterion_group, criterion_main, Criterion};
use grumpkin_msm::utils::{gen_points, gen_scalars};

#[cfg(feature = "cuda")]
use grumpkin_msm::cuda_available;

fn criterion_benchmark(c: &mut Criterion) {
    let bench_npow: usize = std::env::var("BENCH_NPOW")
        .unwrap_or("17".to_string())
        .parse()
        .unwrap();
    let npoints: usize = 1 << bench_npow;

    // println!("generating {} random points, just hang on...", npoints);
    let mut points = gen_points(npoints);
    let mut scalars = gen_scalars(npoints);

    #[cfg(feature = "cuda")]
    {
        unsafe { grumpkin_msm::CUDA_OFF = true };
    }

    let mut group = c.benchmark_group("CPU");
    group.sample_size(10);

    group.bench_function(format!("2**{} points", bench_npow), |b| {
        b.iter(|| {
            let _ = grumpkin_msm::bn256::msm_aux(&points, &scalars, None);
        })
    });

    let context = grumpkin_msm::bn256::init(&points);

    group.bench_function(
        format!("\"preallocate\" 2**{} points", bench_npow),
        |b| {
            b.iter(|| {
                let _ = grumpkin_msm::bn256::with_context_aux(
                    &context, &scalars, None,
                );
            })
        },
    );

    group.finish();

    #[cfg(feature = "cuda")]
    if unsafe { cuda_available() } {
        unsafe { grumpkin_msm::CUDA_OFF = false };

        const EXTRA: usize = 5;
        let bench_npow = bench_npow + EXTRA;
        let npoints: usize = 1 << bench_npow;

        while points.len() < npoints {
            points.append(&mut points.clone());
        }
        scalars.append(&mut gen_scalars(npoints - scalars.len()));

        let mut group = c.benchmark_group("GPU");
        group.sample_size(20);

        group.bench_function(format!("2**{} points", bench_npow), |b| {
            b.iter(|| {
                let _ = grumpkin_msm::bn256::msm_aux(&points, &scalars, None);
            })
        });

        let context = grumpkin_msm::bn256::init(&points);

        let indices = (0..(npoints as u32)).rev().collect::<Vec<_>>();
        group.bench_function(
            format!("preallocate 2**{} points", bench_npow),
            |b| {
                b.iter(|| {
                    let _ = grumpkin_msm::bn256::with_context_aux(
                        &context,
                        &scalars,
                        Some(indices.as_slice()),
                    );
                })
            },
        );

        scalars.reverse();
        group.bench_function(
            format!("preallocate 2**{} points rev", bench_npow),
            |b| {
                b.iter(|| {
                    let _ = grumpkin_msm::bn256::with_context_aux(
                        &context, &scalars, None,
                    );
                })
            },
        );

        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
