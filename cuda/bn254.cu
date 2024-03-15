// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

#include <ff/alt_bn128.hpp>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;

#include <msm/pippenger.cuh>
#include <spmvm/spmvm.cuh>

#ifndef __CUDA_ARCH__
extern "C" 
RustError cuda_sparse_matrix_mul_bn254(spmvm_host_t<scalar_t> *csr, const witness_t<scalar_t> *witness, 
                                        scalar_t *out, size_t nthreads)
{
    return sparse_matrix_mul<scalar_t>(csr, witness, out, nthreads);
}

extern "C"
RustError cuda_pippenger_bn254(point_t *out, const affine_t points[], size_t npoints,
                                              const scalar_t scalars[])
{   return mult_pippenger<bucket_t>(out, points, npoints, scalars);   }
#endif
