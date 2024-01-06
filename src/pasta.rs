// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#![allow(improper_ctypes)]
#![allow(unused)]

extern crate semolina;

use pasta_curves::pallas;

#[cfg(feature = "cuda")]
use crate::{cuda, cuda_available, CUDA_OFF};

extern "C" {
    fn mult_pippenger_pallas(
        out: *mut pallas::Point,
        points: *const pallas::Affine,
        npoints: usize,
        scalars: *const pallas::Scalar,
        is_mont: bool,
    );
}

pub fn pallas(
    points: &[pallas::Affine],
    scalars: &[pallas::Scalar],
) -> pallas::Point {
    let npoints = points.len();
    if npoints != scalars.len() {
        panic!("length mismatch")
    }
    #[cfg(feature = "cuda")]
    if npoints >= 1 << 16 && unsafe { !CUDA_OFF && cuda_available() } {
        extern "C" {
            fn cuda_pippenger_pallas(
                out: *mut pallas::Point,
                points: *const pallas::Affine,
                npoints: usize,
                scalars: *const pallas::Scalar,
                is_mont: bool,
            ) -> cuda::Error;

        }
        let mut ret = pallas::Point::default();
        let err = unsafe {
            cuda_pippenger_pallas(
                &mut ret,
                &points[0],
                npoints,
                &scalars[0],
                true,
            )
        };
        if err.code != 0 {
            panic!("{}", String::from(err));
        }
        return ret;
    }
    let mut ret = pallas::Point::default();
    unsafe {
        mult_pippenger_pallas(&mut ret, &points[0], npoints, &scalars[0], true)
    };
    ret
}

use pasta_curves::vesta;

extern "C" {
    fn mult_pippenger_vesta(
        out: *mut vesta::Point,
        points: *const vesta::Affine,
        npoints: usize,
        scalars: *const vesta::Scalar,
        is_mont: bool,
    );
}

pub fn vesta(
    points: &[vesta::Affine],
    scalars: &[vesta::Scalar],
) -> vesta::Point {
    let npoints = points.len();
    if npoints != scalars.len() {
        panic!("length mismatch")
    }
    #[cfg(feature = "cuda")]
    if npoints >= 1 << 16 && unsafe { !CUDA_OFF && cuda_available() } {
        extern "C" {
            fn cuda_pippenger_vesta(
                out: *mut vesta::Point,
                points: *const vesta::Affine,
                npoints: usize,
                scalars: *const vesta::Scalar,
                is_mont: bool,
            ) -> cuda::Error;

        }
        let mut ret = vesta::Point::default();
        let err = unsafe {
            cuda_pippenger_vesta(
                &mut ret,
                &points[0],
                npoints,
                &scalars[0],
                true,
            )
        };
        if err.code != 0 {
            panic!("{}", String::from(err));
        }
        return ret;
    }
    let mut ret = vesta::Point::default();
    unsafe {
        mult_pippenger_vesta(&mut ret, &points[0], npoints, &scalars[0], true)
    };
    ret
}

#[cfg(test)]
mod tests {
    use halo2curves::group::Curve;

    use crate::{
        pasta::pallas,
        utils::{gen_points, gen_scalars, naive_multiscalar_mul},
    };

    #[test]
    fn it_works() {
        #[cfg(not(debug_assertions))]
        const NPOINTS: usize = 128 * 1024;
        #[cfg(debug_assertions)]
        const NPOINTS: usize = 8 * 1024;

        let points = gen_points(NPOINTS);
        let scalars = gen_scalars(NPOINTS);

        // let naive = naive_multiscalar_mul(&points, &scalars);
        // println!("{:?}", naive);

        let ret = pallas(&points, &scalars).to_affine();
        println!("{:?}", ret);

        assert_eq!(ret, naive);
    }
}
