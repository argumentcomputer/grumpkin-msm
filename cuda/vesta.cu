// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

#include <ff/pasta.hpp>

typedef jacobian_t<vesta_t> point_t;
typedef xyzz_t<vesta_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef pallas_t scalar_t;

#include <msm/pippenger.cuh>

#ifndef __CUDA_ARCH__

extern "C" void drop_msm_context_vesta(msm_context_t<affine_t::mem_t> &ref)
{
    CUDA_OK(cudaFree(ref.d_points));
}

extern "C" RustError
cuda_vesta_init(const affine_t points[], size_t npoints, msm_context_t<affine_t::mem_t> *msm_context)
{
    return mult_pippenger_init<bucket_t, point_t, affine_t, scalar_t>(points, npoints, msm_context);
}

extern "C" RustError cuda_vesta(point_t *out, const affine_t points[], size_t npoints,
                                const scalar_t scalars[], size_t nscalars)
{
    return mult_pippenger<bucket_t>(out, points, npoints, scalars, nscalars);
}

extern "C" RustError cuda_vesta_with(point_t *out, msm_context_t<affine_t::mem_t> *msm_context, size_t npoints,
                                     const scalar_t scalars[], size_t nscalars, uint32_t pidx[])
{
    return mult_pippenger_with<bucket_t, point_t, affine_t, scalar_t>(out, msm_context, npoints, scalars, nscalars, pidx);
}

#endif
