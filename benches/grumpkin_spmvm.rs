// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#![allow(unused_mut)]
#![allow(non_snake_case)]

use criterion::{criterion_group, criterion_main, Criterion};
use ff::PrimeField;
use grumpkin_msm::{
    spmvm::{utils::SparseMatrix, CudaSparseMatrix},
    utils::gen_scalars,
};

pub fn gen_sparse_matrix<F: PrimeField>(n: usize, m: usize) -> SparseMatrix<F> {
    let mut rng = rand::thread_rng();

    let mut data = Vec::new();
    let mut col_idx = Vec::new();
    let mut row_ptr = Vec::new();
    row_ptr.push(0);

    for _ in 0..n {
        let num_elements = rng.gen_range(5..=10); // Random number of elements between 5 to 10
        for _ in 0..num_elements {
            data.push(F::random(&mut rng)); // Random data value
            col_idx.push(rng.gen_range(0..m)); // Random column index
        }
        row_ptr.push(data.len()); // Add the index of the next row start
    }

    data.shrink_to_fit();
    col_idx.shrink_to_fit();
    row_ptr.shrink_to_fit();

    SparseMatrix {
        data,
        indices: col_idx,
        indptr: row_ptr,
        cols: m,
    }
}

fn into_witness<F>(scalars: &[F]) -> (&[F], &F, &[F]) {
    let n = scalars.len();
    (&scalars[0..n - 10], &scalars[n - 10], &scalars[n - 9..])
}

#[cfg(feature = "cuda")]
use grumpkin_msm::cuda_available;
use halo2curves::bn256;
use rand::Rng;

fn criterion_benchmark(c: &mut Criterion) {
    let bench_npow: usize = std::env::var("BENCH_NPOW")
        .unwrap_or("24".to_string())
        .parse()
        .unwrap();
    let size: usize = 1 << bench_npow;

    println!("generating sparse matrix with {} rows...", size);
    let csr = gen_sparse_matrix::<bn256::Fr>(size, size);
    let scalars = gen_scalars(size);
    let (W, u, X) = into_witness(&scalars);

    let mut group = c.benchmark_group("GPU");
    group.sample_size(20);

    for block_size in [1 << 21, 1 << 22, 1 << 23, csr.len()] {
        group.bench_function(
            format!("2**{} size, {} block", bench_npow, block_size),
            |b| {
                b.iter(|| {
                    let blocks = csr.blocks(block_size);
                    let blocked_cuda_csr = CudaSparseMatrix::new(
                        &csr.data,
                        &csr.indices,
                        &csr.indptr,
                        size,
                        size,
                        &blocks,
                        block_size,
                    );
                    let _ = grumpkin_msm::spmvm::grumpkin::spmvm(
                        &blocked_cuda_csr,
                        W,
                        u,
                        X,
                        256,
                    );
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
