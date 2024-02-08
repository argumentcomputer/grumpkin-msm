#[cfg(feature = "cuda")]
use grumpkin_msm::cuda_available;

use grumpkin_msm::pasta::utils::{
    gen_points, gen_scalars, naive_multiscalar_mul,
};
use pasta_curves::group::Curve;

fn main() {
    let bench_npow: usize = std::env::var("BENCH_NPOW")
        .unwrap_or("22".to_string())
        .parse()
        .unwrap();
    let npoints: usize = 1 << bench_npow;

    println!("generating {} random points, just hang on...", npoints);
    let points = gen_points(npoints);
    let mut scalars = gen_scalars(npoints);

    #[cfg(feature = "cuda")]
    if unsafe { cuda_available() } {
        unsafe { grumpkin_msm::CUDA_OFF = false };
    }

    let context = grumpkin_msm::pasta::pallas::init(&points);

    let indices = (0..(npoints as u32)).rev().collect::<Vec<_>>();
    let res = grumpkin_msm::pasta::pallas::with_context_aux(
        &context,
        &scalars,
        Some(indices.as_slice()),
    )
    .to_affine();
    println!("res: {:?}", res);
    let res2 = grumpkin_msm::pasta::pallas::msm_aux(
        &points,
        &scalars,
        Some(indices.as_slice()),
    )
    .to_affine();
    println!("res2: {:?}", res2);
    scalars.reverse();
    let native = naive_multiscalar_mul(&points, &scalars);
    println!("native: {:?}", native);
    println!("success!")
}
