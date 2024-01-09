// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#![allow(improper_ctypes)]
#![allow(unused)]

pub mod pasta;
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

#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MSMContextBn256 {
    context: *const std::ffi::c_void,
}

#[cfg(feature = "cuda")]
unsafe impl Send for MSMContextBn256 {}

#[cfg(feature = "cuda")]
unsafe impl Sync for MSMContextBn256 {}

#[cfg(feature = "cuda")]
impl Default for MSMContextBn256 {
    fn default() -> Self {
        Self {
            context: std::ptr::null(),
        }
    }
}

#[cfg(feature = "cuda")]
// TODO: check for device-side memory leaks
impl Drop for MSMContextBn256 {
    fn drop(&mut self) {
        extern "C" {
            fn drop_msm_context_bn254(by_ref: &MSMContextBn256);
        }
        unsafe { drop_msm_context_bn254(std::mem::transmute::<&_, &_>(self)) };
        self.context = core::ptr::null();
    }
}

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
            fn cuda_bn254(
                out: *mut bn256::G1,
                points: *const bn256::G1Affine,
                npoints: usize,
                scalars: *const bn256::Fr,
            ) -> cuda::Error;

        }
        let mut ret = bn256::G1::default();
        let err =
            unsafe { cuda_bn254(&mut ret, &points[0], npoints, &scalars[0]) };
        assert!(err.code == 0, "{}", String::from(err));

        return bn256::G1::new_jacobian(ret.x, ret.y, ret.z).unwrap();
    }
    let mut ret = bn256::G1::default();
    unsafe { mult_pippenger_bn254(&mut ret, &points[0], npoints, &scalars[0]) };
    bn256::G1::new_jacobian(ret.x, ret.y, ret.z).unwrap()
}

#[cfg(feature = "cuda")]
pub fn bn256_init(
    points: &[bn256::G1Affine],
    npoints: usize,
) -> MSMContextBn256 {
    unsafe {
        assert!(
            !CUDA_OFF && cuda_available(),
            "feature = \"cuda\" must be enabled"
        )
    };
    assert!(
        npoints == points.len() && npoints >= 1 << 16,
        "length mismatch or less than 10**16"
    );

    extern "C" {
        fn cuda_bn254_init(
            points: *const bn256::G1Affine,
            npoints: usize,
            msm_context: &mut MSMContextBn256,
        ) -> cuda::Error;

    }

    let mut ret = MSMContextBn256::default();
    let err = unsafe {
        cuda_bn254_init(points.as_ptr() as *const _, npoints, &mut ret)
    };
    assert!(err.code == 0, "{}", String::from(err));

    ret
}

#[cfg(feature = "cuda")]
pub fn bn256_with(
    context: &MSMContextBn256,
    npoints: usize,
    scalars: &[bn256::Fr],
) -> bn256::G1 {
    unsafe {
        assert!(
            !CUDA_OFF && cuda_available(),
            "feature = \"cuda\" must be enabled"
        )
    };
    assert!(
        npoints == scalars.len() && npoints >= 1 << 16,
        "length mismatch or less than 10**16"
    );

    extern "C" {
        fn cuda_bn254_with(
            out: *mut bn256::G1,
            context: &MSMContextBn256,
            npoints: usize,
            scalars: *const bn256::Fr,
            is_mont: bool,
        ) -> cuda::Error;

    }

    let mut ret = bn256::G1::default();
    let err = unsafe {
        cuda_bn254_with(&mut ret, context, npoints, &scalars[0], true)
    };
    assert!(err.code == 0, "{}", String::from(err));

    ret
}

use halo2curves::grumpkin;

#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MSMContextGrumpkin {
    context: *const std::ffi::c_void,
}

#[cfg(feature = "cuda")]
unsafe impl Send for MSMContextGrumpkin {}

#[cfg(feature = "cuda")]
unsafe impl Sync for MSMContextGrumpkin {}

#[cfg(feature = "cuda")]
impl Default for MSMContextGrumpkin {
    fn default() -> Self {
        Self {
            context: std::ptr::null(),
        }
    }
}

#[cfg(feature = "cuda")]
// TODO: check for device-side memory leaks
impl Drop for MSMContextGrumpkin {
    fn drop(&mut self) {
        extern "C" {
            fn drop_msm_context_grumpkin(by_ref: &MSMContextGrumpkin);
        }
        unsafe {
            drop_msm_context_grumpkin(std::mem::transmute::<&_, &_>(self))
        };
        self.context = core::ptr::null();
    }
}

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

#[cfg(feature = "cuda")]
pub fn grumpkin_init(
    points: &[grumpkin::G1Affine],
    npoints: usize,
) -> MSMContextGrumpkin {
    unsafe {
        assert!(
            !CUDA_OFF && cuda_available(),
            "feature = \"cuda\" must be enabled"
        )
    };
    assert!(
        npoints == points.len() && npoints >= 1 << 16,
        "length mismatch or less than 10**16"
    );

    extern "C" {
        fn cuda_grumpkin_init(
            points: *const grumpkin::G1Affine,
            npoints: usize,
            msm_context: &mut MSMContextGrumpkin,
        ) -> cuda::Error;

    }

    let mut ret = MSMContextGrumpkin::default();
    let err = unsafe {
        cuda_grumpkin_init(points.as_ptr() as *const _, npoints, &mut ret)
    };
    assert!(err.code == 0, "{}", String::from(err));

    ret
}

#[cfg(feature = "cuda")]
pub fn grumpkin_with(
    context: &MSMContextGrumpkin,
    npoints: usize,
    scalars: &[grumpkin::Fr],
) -> grumpkin::G1 {
    unsafe {
        assert!(
            !CUDA_OFF && cuda_available(),
            "feature = \"cuda\" must be enabled"
        )
    };
    assert!(
        npoints == scalars.len() && npoints >= 1 << 16,
        "length mismatch or less than 10**16"
    );

    extern "C" {
        fn cuda_grumpkin_with(
            out: *mut grumpkin::G1,
            context: &MSMContextGrumpkin,
            npoints: usize,
            scalars: *const grumpkin::Fr,
            is_mont: bool,
        ) -> cuda::Error;

    }

    let mut ret = grumpkin::G1::default();
    let err = unsafe {
        cuda_grumpkin_with(&mut ret, context, npoints, &scalars[0], true)
    };
    assert!(err.code == 0, "{}", String::from(err));

    ret
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
