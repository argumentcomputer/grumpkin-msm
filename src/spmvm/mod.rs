#![allow(non_snake_case)]

use std::marker::PhantomData;

use ff::PrimeField;

use self::utils::SparseMatrix;

pub mod grumpkin;
pub mod utils;

#[repr(C)]
pub struct CudaSparseMatrix<'a, F> {
    pub data: *const F,
    pub col_idx: *const usize,
    pub row_ptr: *const usize,
    pub blocks: *const usize,

    pub num_rows: usize,
    pub num_cols: usize,
    pub nnz: usize,

    pub num_blocks: usize,
    pub block_size: usize,

    _p: PhantomData<&'a F>,
}

impl<'a, F> CudaSparseMatrix<'a, F> {
    pub fn new(
        data: &[F],
        col_idx: &[usize],
        row_ptr: &[usize],
        num_rows: usize,
        num_cols: usize,
        blocks: &[usize],
        block_size: usize,
    ) -> Self {
        assert_eq!(
            data.len(),
            col_idx.len(),
            "data and col_idx length mismatch"
        );
        assert_eq!(
            row_ptr.len(),
            num_rows + 1,
            "row_ptr length and num_rows mismatch"
        );

        let nnz = data.len();
        CudaSparseMatrix {
            data: data.as_ptr(),
            col_idx: col_idx.as_ptr(),
            row_ptr: row_ptr.as_ptr(),
            num_rows,
            num_cols,
            nnz,
            blocks: blocks.as_ptr(),
            num_blocks: blocks.len(),
            block_size,
            _p: PhantomData,
        }
    }
}

// impl<'a, F: PrimeField> From<&'a SparseMatrix<F>> for CudaSparseMatrix<'a, F> {
//     fn from(value: &SparseMatrix<F>) -> Self {
//         let mut n = block_size;
//         let mut blocks = vec![0];
//         loop {
//             let i = match row_ptr.binary_search(&block_size) {
//                 Ok(i) => i,
//                 Err(i) => i,
//             };

//             blocks.push(i - 1);
//             n = row_ptr[i - 1] + block_size;
//         }
//         CudaSparseMatrix::new(
//             &value.data,
//             &value.indices,
//             &value.indptr,
//             value.indptr.len() - 1,
//             value.cols,
//         )
//     }
// }
