use core::mem::transmute;
use core::sync::atomic::*;
use halo2curves::bn256;
use halo2curves::ff::Field;
use halo2curves::group::Curve;
use halo2curves::CurveExt;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::sync::{Arc, Mutex};

pub fn gen_points(npoints: usize) -> Vec<bn256::G1Affine> {
    let ret = vec![bn256::G1Affine::default(); npoints];

    let mut rnd = vec![0u8; 32 * npoints];
    ChaCha20Rng::from_entropy().fill_bytes(&mut rnd);

    let n_workers = rayon::current_num_threads();
    let work = AtomicUsize::new(0);
    rayon::scope(|s| {
        for _ in 0..n_workers {
            s.spawn(|_| {
                let hash = bn256::G1::hash_to_curve("foobar");

                let mut stride = 1024;
                let mut tmp = vec![bn256::G1::default(); stride];

                loop {
                    let work = work.fetch_add(stride, Ordering::Relaxed);
                    if work >= npoints {
                        break;
                    }
                    if work + stride > npoints {
                        stride = npoints - work;
                        unsafe { tmp.set_len(stride) };
                    }
                    for (i, point) in tmp.iter_mut().enumerate().take(stride) {
                        let off = (work + i) * 32;
                        *point = hash(&rnd[off..off + 32]);
                    }
                    #[allow(mutable_transmutes)]
                    bn256::G1::batch_normalize(&tmp, unsafe {
                        transmute::<&[bn256::G1Affine], &mut [bn256::G1Affine]>(
                            &ret[work..work + stride],
                        )
                    });
                }
            })
        }
    });

    ret
}

pub fn gen_scalars(npoints: usize) -> Vec<bn256::Fr> {
    let ret = Arc::new(Mutex::new(vec![bn256::Fr::default(); npoints]));

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
                    ret[work] = bn256::Fr::random(&mut rng);
                }
            });
        }
    });

    Arc::try_unwrap(ret).unwrap().into_inner().unwrap()
}

pub fn naive_multiscalar_mul(
    points: &[bn256::G1Affine],
    scalars: &[bn256::Fr],
) -> bn256::G1Affine {
    let ret: bn256::G1 = points
        .par_iter()
        .zip_eq(scalars.par_iter())
        .map(|(p, s)| p * s)
        .sum();

    ret.to_affine()
}
