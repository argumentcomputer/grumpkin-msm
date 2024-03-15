//! # Sparse Matrices
//!
//! This module defines a custom implementation of CSR/CSC sparse matrices.
//! Specifically, we implement sparse matrix / dense vector multiplication
//! to compute the `A z`, `B z`, and `C z` in Nova.

use std::convert::TryInto;

use ff::PrimeField;
use serde::{Deserialize, Serialize};

/// CSR format sparse matrix, We follow the names used by scipy.
/// Detailed explanation here: <https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr>
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseMatrix<F: PrimeField> {
    /// all non-zero values in the matrix
    pub data: Vec<F>,
    /// column indices
    pub indices: Vec<usize>,
    /// row information
    pub indptr: Vec<usize>,
    /// number of columns
    pub cols: usize,
}

impl<F: PrimeField> SparseMatrix<F> {
    /// 0x0 empty matrix
    pub fn empty() -> Self {
        Self {
            data: vec![],
            indices: vec![],
            indptr: vec![0],
            cols: 0,
        }
    }

    /// Construct from the COO representation; Vec<usize(row), usize(col), F>.
    /// We assume that the rows are sorted during construction.
    pub fn new(matrix: &[(usize, usize, F)], rows: usize, cols: usize) -> Self {
        let mut new_matrix = vec![vec![]; rows];
        for (row, col, val) in matrix {
            new_matrix[*row].push((*col, *val));
        }

        for row in new_matrix.iter() {
            assert!(row.windows(2).all(|w| w[0].0 < w[1].0));
        }

        let mut indptr = vec![0; rows + 1];
        for (i, col) in new_matrix.iter().enumerate() {
            indptr[i + 1] = indptr[i] + col.len();
        }

        let mut indices = vec![];
        let mut data = vec![];
        for col in new_matrix {
            let (idx, val): (Vec<_>, Vec<_>) = col.into_iter().unzip();
            indices.extend(idx);
            data.extend(val);
        }

        Self {
            data,
            indices,
            indptr,
            cols,
        }
    }

    /// Retrieves the data for row slice [i..j] from `ptrs`.
    /// We assume that `ptrs` is indexed from `indptrs` and do not check if the
    /// returned slice is actually a valid row.
    pub fn get_row_unchecked(
        &self,
        ptrs: &[usize; 2],
    ) -> impl Iterator<Item = (&F, &usize)> {
        self.data[ptrs[0]..ptrs[1]]
            .iter()
            .zip(&self.indices[ptrs[0]..ptrs[1]])
    }

    /// Multiply by a dense vector; uses rayon to parallelize.
    pub fn multiply_vec_blocked(
        &self,
        vector: &[F],
        block_size: usize,
    ) -> Vec<F> {
        assert_eq!(self.cols, vector.len(), "invalid shape");

        let mut res = vec![F::ZERO; self.indptr.len() - 1];
        self.blocks(block_size).windows(2).for_each(|ptrs| {
            self.indptr[ptrs[0]..ptrs[1] + 1]
                .windows(2)
                .zip(ptrs[0]..ptrs[1])
                .for_each(|(ptrs, i)| {
                    res[i] = self
                        .get_row_unchecked(ptrs.try_into().unwrap())
                        .map(|(val, col_idx)| *val * vector[*col_idx])
                        .sum();
                });
        });
        res
    }

    /// Multiply by a dense vector; uses rayon to parallelize.
    /// This does not check that the shape of the matrix/vector are compatible.
    pub fn multiply_vec_unchecked(&self, vector: &[F]) -> Vec<F> {
        self.indptr
            .windows(2)
            .map(|ptrs| {
                self.get_row_unchecked(ptrs.try_into().unwrap())
                    .map(|(val, col_idx)| *val * vector[*col_idx])
                    .sum()
            })
            .collect()
    }

    pub fn blocks(&self, block_size: usize) -> Vec<usize> {
        let mut n = block_size;
        let mut blocks = vec![0];
        loop {
            let i = match self.indptr.binary_search(&n) {
                Ok(i) | Err(i) => i
            };

            blocks.push(i - 1);
            n = self.indptr[i - 1] + block_size;

            if i == self.indptr.len() {
                break;
            }
        }
        blocks
    }

    /// number of non-zero entries
    pub fn len(&self) -> usize {
        *self.indptr.last().unwrap()
    }

    /// empty matrix
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// returns a custom iterator
    pub fn iter(&self) -> Iter<'_, F> {
        let mut row = 0;
        while self.indptr[row + 1] == 0 {
            row += 1;
        }
        Iter {
            matrix: self,
            row,
            i: 0,
            nnz: *self.indptr.last().unwrap(),
        }
    }
}

/// Iterator for sparse matrix
pub struct Iter<'a, F: PrimeField> {
    matrix: &'a SparseMatrix<F>,
    row: usize,
    i: usize,
    nnz: usize,
}

impl<'a, F: PrimeField> Iterator for Iter<'a, F> {
    type Item = (usize, usize, F);

    fn next(&mut self) -> Option<Self::Item> {
        // are we at the end?
        if self.i == self.nnz {
            return None;
        }

        // compute current item
        let curr_item = (
            self.row,
            self.matrix.indices[self.i],
            self.matrix.data[self.i],
        );

        // advance the iterator
        self.i += 1;
        // edge case at the end
        if self.i == self.nnz {
            return Some(curr_item);
        }
        // if `i` has moved to next row
        while self.i >= self.matrix.indptr[self.row + 1] {
            self.row += 1;
        }

        Some(curr_item)
    }
}
