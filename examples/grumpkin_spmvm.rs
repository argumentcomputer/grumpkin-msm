#![allow(non_snake_case)]

use std::time::Instant;

use ff::PrimeField;
use grumpkin_msm::{
    spmvm::{utils::SparseMatrix, CudaSparseMatrix},
    utils::gen_scalars,
};

#[cfg(feature = "cuda")]
use grumpkin_msm::cuda_available;
use rand::Rng;

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

fn into_witness<F>(
    scalars: &[F],
) -> (&[F], &F, &[F]) {
    let n = scalars.len();
    (&scalars[0..n-10], &scalars[n-10], &scalars[n-9..])
}

/// cargo run --release --example grumpkin_spmvm

fn main() {
    let bench_npow: usize = std::env::var("BENCH_NPOW")
        .unwrap_or("22".to_string())
        .parse()
        .unwrap();
    let size: usize = 1 << bench_npow;

    println!(
        "generating random sparse matrix with {} rows, just hang on...",
        size
    );
    let csr = gen_sparse_matrix(size, size);

    let block_size = 1 << 21;
    let blocks = csr.blocks(block_size);
    let no_blocks = csr.blocks(csr.len());
    let blocked_cuda_csr = CudaSparseMatrix::new(
        &csr.data,
        &csr.indices,
        &csr.indptr,
        size,
        size,
        &blocks,
        block_size,
    );
    let cuda_csr = CudaSparseMatrix::new(
        &csr.data,
        &csr.indices,
        &csr.indptr,
        size,
        size,
        &no_blocks,
        csr.len(),
    );
    let scalars = gen_scalars(size);
    let (W, u, X) = into_witness(&scalars);

    #[cfg(feature = "cuda")]
    if unsafe { cuda_available() } {
        unsafe { grumpkin_msm::CUDA_OFF = false };
    }

    let start = Instant::now();
    let res = grumpkin_msm::spmvm::grumpkin::spmvm(&cuda_csr, W, u, X, 256);
    let gpu = start.elapsed();
    println!("gpu: {:?}", gpu);

    let blocked_res =
        grumpkin_msm::spmvm::grumpkin::spmvm(&blocked_cuda_csr, W, u, X, 256);
    let gpu_blocked = start.elapsed();
    println!("gpu blocked: {:?}", gpu_blocked - gpu);

    let native = csr.multiply_vec_unchecked(&scalars);
    let cpu = start.elapsed();
    println!("cpu: {:?}", cpu - gpu_blocked);
    assert!(res == native);
    assert!(res == blocked_res);
    println!("success!")
}
