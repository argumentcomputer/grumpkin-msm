// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <msm/pippenger.hpp>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
#include <ff/alt_bn128.hpp>

static thread_pool_t da_pool;

extern "C"
void mult_pippenger_bn254(jacobian_t<fp_t>& ret,
                          const xyzz_t<fp_t>::affine_t points[],
                          size_t npoints, const fr_t scalars[])
{   mult_pippenger<xyzz_t<fp_t>>(ret, points, npoints, scalars, true,
                                    &da_pool);
}

extern "C"
void mult_pippenger_grumpkin(jacobian_t<fr_t>& ret,
                           const xyzz_t<fr_t>::affine_t points[],
                           size_t npoints, const fp_t scalars[])
{   mult_pippenger<xyzz_t<fr_t>>(ret, points, npoints, scalars, true,
                                     &da_pool);
}