// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#![allow(improper_ctypes)]
#![allow(unused)]

extern crate semolina;

pub mod pallas {
    use pasta_curves::pallas::{Affine, Point, Scalar};

    use crate::impl_pasta;

    impl_pasta!(
        cuda_pallas,
        cuda_pallas_init,
        cuda_pallas_with,
        mult_pippenger_pallas,
        Point,
        Affine,
        Scalar
    );
}

pub mod vesta {
    use pasta_curves::vesta::{Affine, Point, Scalar};

    use crate::impl_pasta;

    impl_pasta!(
        cuda_vesta,
        cuda_vesta_init,
        cuda_vesta_with,
        mult_pippenger_vesta,
        Point,
        Affine,
        Scalar
    );
}

#[macro_export]
macro_rules! impl_pasta {
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
                is_mont: bool,
            );

        }

        pub fn msm_aux(
            points: &[$affine],
            scalars: &[$scalar],
            indices: Option<&[u32]>,
        ) -> $point {
            let npoints = points.len();
            let nscalars = scalars.len();
            if let Some(indices) = indices {
                assert!(nscalars == indices.len(), "length mismatch");
            } else {
                assert!(npoints == nscalars, "length mismatch");
            }

            #[cfg(feature = "cuda")]
            if npoints >= 1 << 16 && unsafe { !CUDA_OFF && cuda_available() } {
                extern "C" {
                    fn $name(
                        out: *mut $point,
                        points: *const $affine,
                        npoints: usize,
                        scalars: *const $scalar,
                        nscalars: usize,
                        indices: *const u32,
                        is_mont: bool,
                    ) -> cuda::Error;

                }

                let indices = if let Some(inner) = indices {
                    inner.as_ptr()
                } else {
                    std::ptr::null()
                };

                let mut ret = $point::default();
                let err = unsafe {
                    $name(
                        &mut ret,
                        &points[0],
                        npoints,
                        &scalars[0],
                        nscalars,
                        indices,
                        true,
                    )
                };
                assert!(err.code == 0, "{}", String::from(err));

                return ret;
            }

            assert!(indices.is_none(), "no cpu support for indexed MSMs");
            let mut ret = $point::default();
            unsafe {
                $name_cpu(&mut ret, &points[0], npoints, &scalars[0], true)
            };
            ret
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

        pub fn msm(points: &[$affine], scalars: &[$scalar]) -> $point {
            msm_aux(points, scalars, None)
        }

        /// An indexed MSM. We do not check if the indices are valid, i.e.
        /// for i in 0..nscalars, 0 <= indices[i] < npoints
        ///
        /// Also, this version carries a performance penalty that all the
        /// points must be moved onto the GPU once instead of in batches.
        /// If the points are to be reused, please use the [`MSMContext`] API
        pub fn indexed_msm(
            points: &[$affine],
            scalars: &[$scalar],
            indices: &[u32],
        ) -> $point {
            msm_aux(points, scalars, Some(indices))
        }

        pub fn with_context_aux(
            context: &MSMContext<'_>,
            scalars: &[$scalar],
            indices: Option<&[u32]>,
        ) -> $point {
            let npoints = context.npoints();
            let nscalars = scalars.len();
            if let Some(indices) = indices {
                assert!(nscalars == indices.len(), "length mismatch");
            } else {
                assert!(npoints >= nscalars, "not enough points");
            }

            let mut ret = $point::default();

            #[cfg(feature = "cuda")]
            if nscalars >= 1 << 16 && unsafe { !CUDA_OFF && cuda_available() } {
                extern "C" {
                    fn $name_with(
                        out: *mut $point,
                        context: &CudaMSMContext,
                        scalars: *const $scalar,
                        nscalars: usize,
                        indices: *const u32,
                        is_mont: bool,
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
                        context.cuda(),
                        &scalars[0],
                        nscalars,
                        indices,
                        true,
                    )
                };
                assert!(err.code == 0, "{}", String::from(err));
                return ret;
            }

            assert!(indices.is_none(), "no cpu support for indexed MSMs");

            unsafe {
                $name_cpu(
                    &mut ret,
                    &context.cpu_context[0],
                    nscalars,
                    &scalars[0],
                    true,
                )
            };

            ret
        }

        pub fn with_context(
            context: &MSMContext<'_>,
            scalars: &[$scalar],
        ) -> $point {
            with_context_aux(context, scalars, None)
        }

        /// An indexed MSM. We do not check if the indices are valid, i.e.
        /// for i in 0..nscalars, 0 <= indices[i] < npoints
        pub fn indexed_with_context(
            context: &MSMContext<'_>,
            scalars: &[$scalar],
            indices: &[u32],
        ) -> $point {
            with_context_aux(context, scalars, Some(indices))
        }
    };
}

pub mod utils {
    use std::{
        mem::transmute,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        },
    };

    use pasta_curves::{
        arithmetic::CurveExt,
        group::{ff::Field, Curve},
        pallas,
    };
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use rayon::iter::{
        IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator,
    };

    pub fn gen_points(npoints: usize) -> Vec<pallas::Affine> {
        let ret = vec![pallas::Affine::default(); npoints];

        let mut rnd = vec![0u8; 32 * npoints];
        ChaCha20Rng::from_entropy().fill_bytes(&mut rnd);

        let n_workers = rayon::current_num_threads();
        let work = AtomicUsize::new(0);
        rayon::scope(|s| {
            for _ in 0..n_workers {
                s.spawn(|_| {
                    let hash = pallas::Point::hash_to_curve("foobar");

                    let mut stride = 1024;
                    let mut tmp = vec![pallas::Point::default(); stride];

                    loop {
                        let work = work.fetch_add(stride, Ordering::Relaxed);
                        if work >= npoints {
                            break;
                        }
                        if work + stride > npoints {
                            stride = npoints - work;
                            unsafe { tmp.set_len(stride) };
                        }
                        for (i, point) in
                            tmp.iter_mut().enumerate().take(stride)
                        {
                            let off = (work + i) * 32;
                            *point = hash(&rnd[off..off + 32]);
                        }
                        #[allow(mutable_transmutes)]
                        pallas::Point::batch_normalize(&tmp, unsafe {
                            transmute::<
                                &[pallas::Affine],
                                &mut [pallas::Affine],
                            >(
                                &ret[work..work + stride]
                            )
                        });
                    }
                })
            }
        });

        ret
    }

    pub fn gen_scalars(npoints: usize) -> Vec<pallas::Scalar> {
        let ret =
            Arc::new(Mutex::new(vec![pallas::Scalar::default(); npoints]));

        let n_workers = rayon::current_num_threads();
        let work = Arc::new(AtomicUsize::new(0));

        rayon::scope(|s| {
            for _ in 0..n_workers {
                let ret_clone = Arc::clone(&ret);
                let work_clone = Arc::clone(&work);

                s.spawn(move |_| {
                    let mut rng = ChaCha20Rng::from_entropy();
                    loop {
                        let work = work_clone.fetch_add(1, Ordering::Relaxed);
                        if work >= npoints {
                            break;
                        }
                        let mut ret = ret_clone.lock().unwrap();
                        ret[work] = pallas::Scalar::random(&mut rng);
                    }
                });
            }
        });

        Arc::try_unwrap(ret).unwrap().into_inner().unwrap()
    }

    pub fn naive_multiscalar_mul(
        points: &[pallas::Affine],
        scalars: &[pallas::Scalar],
    ) -> pallas::Affine {
        let ret: pallas::Point = points
            .par_iter()
            .zip_eq(scalars.par_iter())
            .map(|(p, s)| p * s)
            .sum();

        ret.to_affine()
    }
}

#[cfg(test)]
mod tests {
    use pasta_curves::group::Curve;

    use crate::pasta::{
        pallas,
        utils::{gen_points, gen_scalars, naive_multiscalar_mul},
    };

    #[test]
    fn test_simple() {
        #[cfg(not(debug_assertions))]
        const NPOINTS: usize = 128 * 1024;
        #[cfg(debug_assertions)]
        const NPOINTS: usize = 8 * 1024;

        let points = gen_points(NPOINTS);
        let scalars = gen_scalars(NPOINTS);

        let naive = naive_multiscalar_mul(&points, &scalars);
        println!("{:?}", naive);

        let ret = pallas::msm_aux(&points, &scalars, None).to_affine();
        println!("{:?}", ret);

        let context = pallas::init(&points);
        let ret_other =
            pallas::with_context_aux(&context, &scalars, None).to_affine();
        println!("{:?}", ret_other);

        assert_eq!(ret, naive);
        assert_eq!(ret, ret_other);
    }
}
