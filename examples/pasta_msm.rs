use std::{
    cell::UnsafeCell,
    mem::transmute,
    sync::atomic::{AtomicUsize, Ordering},
};

#[cfg(feature = "cuda")]
use grumpkin_msm::cuda_available;

use pasta_curves::{
    arithmetic::CurveExt,
    group::{ff::Field, Curve},
    pallas,
};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

pub fn gen_points(npoints: usize) -> Vec<pallas::Affine> {
    let mut ret: Vec<pallas::Affine> = Vec::with_capacity(npoints);
    unsafe { ret.set_len(npoints) };

    let mut rnd: Vec<u8> = Vec::with_capacity(32 * npoints);
    unsafe { rnd.set_len(32 * npoints) };
    ChaCha20Rng::from_entropy().fill_bytes(&mut rnd);

    let n_workers = rayon::current_num_threads();
    let work = AtomicUsize::new(0);
    rayon::scope(|s| {
        for _ in 0..n_workers {
            s.spawn(|_| {
                let hash = pallas::Point::hash_to_curve("foobar");

                let mut stride = 1024;
                let mut tmp: Vec<pallas::Point> = Vec::with_capacity(stride);
                unsafe { tmp.set_len(stride) };

                loop {
                    let work = work.fetch_add(stride, Ordering::Relaxed);
                    if work >= npoints {
                        break;
                    }
                    if work + stride > npoints {
                        stride = npoints - work;
                        unsafe { tmp.set_len(stride) };
                    }
                    for i in 0..stride {
                        let off = (work + i) * 32;
                        tmp[i] = hash(&rnd[off..off + 32]);
                    }
                    #[allow(mutable_transmutes)]
                    pallas::Point::batch_normalize(&tmp, unsafe {
                        transmute::<&[pallas::Affine], &mut [pallas::Affine]>(
                            &ret[work..work + stride],
                        )
                    });
                }
            })
        }
    });

    ret
}

fn as_mut<T>(x: &T) -> &mut T {
    unsafe { (*(x as *const _ as *mut UnsafeCell<T>)).get_mut() }
}

pub fn gen_scalars(npoints: usize) -> Vec<pallas::Scalar> {
    let mut ret: Vec<pallas::Scalar> = Vec::with_capacity(npoints);
    unsafe { ret.set_len(npoints) };

    let n_workers = rayon::current_num_threads();
    let work = AtomicUsize::new(0);

    rayon::scope(|s| {
        for _ in 0..n_workers {
            s.spawn(|_| {
                let mut rng = ChaCha20Rng::from_entropy();
                loop {
                    let work = work.fetch_add(1, Ordering::Relaxed);
                    if work >= npoints {
                        break;
                    }
                    *as_mut(&ret[work]) = pallas::Scalar::random(&mut rng);
                }
            })
        }
    });

    ret
}

pub fn naive_multiscalar_mul(
    points: &[pallas::Affine],
    scalars: &[pallas::Scalar],
) -> pallas::Affine {
    let n_workers = rayon::current_num_threads();

    let mut rets: Vec<pallas::Point> = Vec::with_capacity(n_workers);
    unsafe { rets.set_len(n_workers) };

    let npoints = points.len();
    let work = AtomicUsize::new(0);
    let tid = AtomicUsize::new(0);
    rayon::scope(|s| {
        for _ in 0..n_workers {
            s.spawn(|_| {
                let mut ret = pallas::Point::default();

                loop {
                    let work = work.fetch_add(1, Ordering::Relaxed);
                    if work >= npoints {
                        break;
                    }
                    ret += points[work] * scalars[work];
                }

                *as_mut(&rets[tid.fetch_add(1, Ordering::Relaxed)]) = ret;
            })
        }
    });

    let mut ret = pallas::Point::default();
    for i in 0..n_workers {
        ret += rets[i];
    }

    ret.to_affine()
}

fn main() {
    let bench_npow: usize = std::env::var("BENCH_NPOW")
        .unwrap_or("17".to_string())
        .parse()
        .unwrap();
    let npoints: usize = 1 << bench_npow;

    println!("generating {} random points, just hang on...", npoints);
    let points = gen_points(npoints);
    let scalars = gen_scalars(npoints);

    #[cfg(feature = "cuda")]
    if unsafe { cuda_available() } {
        unsafe { grumpkin_msm::CUDA_OFF = false };
    }

    let res = grumpkin_msm::pasta::pallas(&points, &scalars).to_affine();
    let native = naive_multiscalar_mul(&points, &scalars);
    assert_eq!(res, native);
    println!("success!")
}
