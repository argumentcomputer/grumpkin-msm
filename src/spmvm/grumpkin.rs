#![allow(non_snake_case)]

use std::ffi::c_void;

use halo2curves::{bn256, ff::Field};

use crate::spmvm::CudaWitness;

use super::CudaSparseMatrix;

pub fn spmvm(
    csr: &CudaSparseMatrix<'_, bn256::Fr>,
    W: &[bn256::Fr],
    u: &bn256::Fr,
    X: &[bn256::Fr],
    nthreads: usize,
) -> Vec<bn256::Fr> {
    extern "C" {
        fn cuda_sparse_matrix_mul_bn254(
            csr: *const CudaSparseMatrix<'_, bn256::Fr>,
            witness: *const CudaWitness<'_, bn256::Fr>,
            out: *mut bn256::Fr,
            nthreads: usize,
        ) -> sppark::Error;
    }

    let mut out = vec![bn256::Fr::ZERO; csr.num_rows];
    let witness = CudaWitness::new(W, u, X);
    let err = unsafe {
        cuda_sparse_matrix_mul_bn254(
            csr as *const _,
            &witness as *const _,
            out.as_mut_ptr(),
            nthreads,
        )
    };
    assert!(err.code == 0, "{}", String::from(err));

    out
}
