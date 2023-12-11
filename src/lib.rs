// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#![allow(improper_ctypes)]

extern crate blst;

#[cfg(feature = "cuda")]
sppark::cuda_error!();
#[cfg(feature = "cuda")]
extern "C" {
    fn cuda_available() -> bool;
}
#[cfg(feature = "cuda")]
pub static mut CUDA_OFF: bool = false;

use halo2curves::bn256;

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
    if npoints != scalars.len() {
        panic!("length mismatch")
    }
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
        if err.code != 0 {
            panic!("{}", String::from(err));
        }
        return ret;
    }
    let mut ret = bn256::G1::default();
    unsafe { mult_pippenger_bn254(&mut ret, &points[0], npoints, &scalars[0]) };
    ret
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
    if npoints != scalars.len() {
        panic!("length mismatch")
    }
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
        if err.code != 0 {
            panic!("{}", String::from(err));
        }
        return ret;
    }
    let mut ret = grumpkin::G1::default();
    unsafe {
        mult_pippenger_grumpkin(&mut ret, &points[0], npoints, &scalars[0])
    };
    ret
}

include!("tests.rs");
