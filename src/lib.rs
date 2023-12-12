// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#![allow(improper_ctypes)]
#![allow(unused)]

pub mod utils;

extern crate blst;

#[cfg(feature = "cuda")]
sppark::cuda_error!();
#[cfg(feature = "cuda")]
extern "C" {
    pub fn cuda_available() -> bool;
}
#[cfg(feature = "cuda")]
pub static mut CUDA_OFF: bool = false;

use halo2curves::bn256;
use halo2curves::CurveExt;

extern "C" {
    fn mult_pippenger_bn254(
        out: *mut bn256::G1,
        points: *const bn256::G1Affine,
        npoints: usize,
        scalars: *const bn256::Fr,
    );

}

pub fn bn256(points: &[bn256::G1Affine], scalars: &[bn256::Fr]) -> bn256::G1 {
    let npoints = points.len();
    assert!(npoints == scalars.len(), "length mismatch");

    #[cfg(feature = "cuda")]
    if npoints >= 1 << 16 && unsafe { !CUDA_OFF && cuda_available() } {
        extern "C" {
            fn cuda_pippenger_bn254(
                out: *mut bn256::G1,
                points: *const bn256::G1Affine,
                npoints: usize,
                scalars: *const bn256::Fr,
            ) -> cuda::Error;

        }
        let mut ret = bn256::G1::default();
        let err = unsafe {
            cuda_pippenger_bn254(&mut ret, &points[0], npoints, &scalars[0])
        };
        assert!(err.code == 0, "{}", String::from(err));

        return bn256::G1::new_jacobian(ret.x, ret.y, ret.z).unwrap();
    }
    let mut ret = bn256::G1::default();
    unsafe { mult_pippenger_bn254(&mut ret, &points[0], npoints, &scalars[0]) };
    bn256::G1::new_jacobian(ret.x, ret.y, ret.z).unwrap()
}

use halo2curves::grumpkin;

extern "C" {
    fn mult_pippenger_grumpkin(
        out: *mut grumpkin::G1,
        points: *const grumpkin::G1Affine,
        npoints: usize,
        scalars: *const grumpkin::Fr,
    );

}

pub fn grumpkin(
    points: &[grumpkin::G1Affine],
    scalars: &[grumpkin::Fr],
) -> grumpkin::G1 {
    let npoints = points.len();
    assert!(npoints == scalars.len(), "length mismatch");

    #[cfg(feature = "cuda")]
    if npoints >= 1 << 16 && unsafe { !CUDA_OFF && cuda_available() } {
        extern "C" {
            fn cuda_pippenger_grumpkin(
                out: *mut grumpkin::G1,
                points: *const grumpkin::G1Affine,
                npoints: usize,
                scalars: *const grumpkin::Fr,
            ) -> cuda::Error;

        }
        let mut ret = grumpkin::G1::default();
        let err = unsafe {
            cuda_pippenger_grumpkin(&mut ret, &points[0], npoints, &scalars[0])
        };
        assert!(err.code == 0, "{}", String::from(err));

        return grumpkin::G1::new_jacobian(ret.x, ret.y, ret.z).unwrap();
    }
    let mut ret = grumpkin::G1::default();
    unsafe {
        mult_pippenger_grumpkin(&mut ret, &points[0], npoints, &scalars[0])
    };
    grumpkin::G1::new_jacobian(ret.x, ret.y, ret.z).unwrap()
}

#[cfg(test)]
mod tests {
    use halo2curves::group::Curve;

    use crate::utils::{gen_points, gen_scalars, naive_multiscalar_mul};

    #[test]
    fn it_works() {
        #[cfg(not(debug_assertions))]
        const NPOINTS: usize = 128 * 1024;
        #[cfg(debug_assertions)]
        const NPOINTS: usize = 8 * 1024;

        let points = gen_points(NPOINTS);
        let scalars = gen_scalars(NPOINTS);

        let naive = naive_multiscalar_mul(&points, &scalars);
        println!("{:?}", naive);

        let ret = crate::bn256(&points, &scalars).to_affine();
        println!("{:?}", ret);

        assert_eq!(ret, naive);
    }
}
