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

pub mod bn256 {
    use halo2curves::{
        bn256::{Fr as Scalar, G1Affine as Affine, G1 as Point},
        CurveExt,
    };

    use crate::impl_msm;

    impl_msm!(
        cuda_bn254,
        cuda_bn254_init,
        cuda_bn254_with,
        mult_pippenger_bn254,
        Point,
        Affine,
        Scalar
    );
}

pub mod grumpkin {
    use halo2curves::{
        grumpkin::{Fr as Scalar, G1Affine as Affine, G1 as Point},
        CurveExt,
    };

    use crate::impl_msm;

    impl_msm!(
        cuda_grumpkin,
        cuda_grumpkin_init,
        cuda_grumpkin_with,
        mult_pippenger_grumpkin,
        Point,
        Affine,
        Scalar
    );
}

#[macro_export]
macro_rules! impl_msm {
    (
        $name:ident,
        $name_init:ident,
        $name_with:ident,
        $name_cpu:ident,
        $point:ident,
        $affine:ident,
        $scalar:ident
    ) => {
        #[cfg(feature = "cuda")]
        use $crate::{cuda, cuda_available, CUDA_OFF};

        #[repr(C)]
        #[derive(Debug, Clone)]
        pub struct CudaMSMContext {
            context: *const std::ffi::c_void,
            npoints: usize,
        }

        unsafe impl Send for CudaMSMContext {}

        unsafe impl Sync for CudaMSMContext {}

        impl Default for CudaMSMContext {
            fn default() -> Self {
                Self {
                    context: std::ptr::null(),
                    npoints: 0,
                }
            }
        }

        #[cfg(feature = "cuda")]
        // TODO: check for device-side memory leaks
        impl Drop for CudaMSMContext {
            fn drop(&mut self) {
                extern "C" {
                    fn drop_msm_context_bn254(by_ref: &CudaMSMContext);
                }
                unsafe {
                    drop_msm_context_bn254(std::mem::transmute::<&_, &_>(self))
                };
                self.context = core::ptr::null();
            }
        }

        #[derive(Default, Debug, Clone)]
        pub struct MSMContext<'a> {
            cuda_context: CudaMSMContext,
            on_gpu: bool,
            cpu_context: &'a [$affine],
        }

        unsafe impl<'a> Send for MSMContext<'a> {}

        unsafe impl<'a> Sync for MSMContext<'a> {}

        impl<'a> MSMContext<'a> {
            fn new(points: &'a [$affine]) -> Self {
                Self {
                    cuda_context: CudaMSMContext::default(),
                    on_gpu: false,
                    cpu_context: points,
                }
            }

            fn npoints(&self) -> usize {
                if self.on_gpu {
                    assert_eq!(
                        self.cpu_context.len(),
                        self.cuda_context.npoints
                    );
                }
                self.cpu_context.len()
            }

            fn cuda(&self) -> &CudaMSMContext {
                &self.cuda_context
            }

            fn points(&self) -> &[$affine] {
                &self.cpu_context
            }
        }

        extern "C" {
            fn $name_cpu(
                out: *mut $point,
                points: *const $affine,
                npoints: usize,
                scalars: *const $scalar,
            );

        }

        pub fn msm(points: &[$affine], scalars: &[$scalar]) -> $point {
            let npoints = points.len();
            assert!(npoints == scalars.len(), "length mismatch");

            #[cfg(feature = "cuda")]
            if npoints >= 1 << 16 && unsafe { !CUDA_OFF && cuda_available() } {
                extern "C" {
                    fn $name(
                        out: *mut $point,
                        points: *const $affine,
                        npoints: usize,
                        scalars: *const $scalar,
                    ) -> cuda::Error;

                }
                let mut ret = $point::default();
                let err = unsafe {
                    $name(&mut ret, &points[0], npoints, &scalars[0])
                };
                assert!(err.code == 0, "{}", String::from(err));

                return $point::new_jacobian(ret.x, ret.y, ret.z).unwrap();
            }
            let mut ret = $point::default();
            unsafe { $name_cpu(&mut ret, &points[0], npoints, &scalars[0]) };
            $point::new_jacobian(ret.x, ret.y, ret.z).unwrap()
        }

        pub fn init(points: &[$affine]) -> MSMContext<'_> {
            let npoints = points.len();

            let mut ret = MSMContext::new(points);

            #[cfg(feature = "cuda")]
            if npoints >= 1 << 16 && unsafe { !CUDA_OFF && cuda_available() } {
                extern "C" {
                    fn $name_init(
                        points: *const $affine,
                        npoints: usize,
                        msm_context: &mut CudaMSMContext,
                    ) -> cuda::Error;
                }

                let npoints = points.len();
                let err = unsafe {
                    $name_init(
                        points.as_ptr() as *const _,
                        npoints,
                        &mut ret.cuda_context,
                    )
                };
                assert!(err.code == 0, "{}", String::from(err));
                ret.on_gpu = true;
                return ret;
            }

            ret
        }

        pub fn with(
            context: &MSMContext<'_>,
            scalars: &[$scalar],
            indices: Option<&[u32]>,
        ) -> $point {
            let npoints = context.npoints();
            let nscalars = scalars.len();
            assert!(npoints >= nscalars, "not enough points");

            let mut ret = $point::default();

            #[cfg(feature = "cuda")]
            if nscalars >= 1 << 16
                && context.on_gpu
                && unsafe { !CUDA_OFF && cuda_available() }
            {
                extern "C" {
                    fn $name_with(
                        out: *mut $point,
                        context: &CudaMSMContext,
                        npoints: usize,
                        scalars: *const $scalar,
                        indices: *const u32,
                    ) -> cuda::Error;
                }

                let indices = if let Some(inner) = indices {
                    inner.as_ptr()
                } else {
                    std::ptr::null()
                };
                let err = unsafe {
                    $name_with(
                        &mut ret,
                        &context.cuda_context,
                        nscalars,
                        &scalars[0],
                        indices,
                    )
                };
                assert!(err.code == 0, "{}", String::from(err));
                return $point::new_jacobian(ret.x, ret.y, ret.z).unwrap();
            }

            unsafe {
                $name_cpu(
                    &mut ret,
                    &context.cpu_context[0],
                    nscalars,
                    &scalars[0],
                )
            };
            $point::new_jacobian(ret.x, ret.y, ret.z).unwrap()
        }
    };
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

        let ret = crate::bn256::msm(&points, &scalars).to_affine();
        println!("{:?}", ret);

        let context = crate::bn256::init(&points);
        let ret_other =
            crate::bn256::with(&context, &scalars, None).to_affine();
        println!("{:?}", ret_other);

        assert_eq!(ret, naive);
        assert_eq!(ret, ret_other);
    }
}
