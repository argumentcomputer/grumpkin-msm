use grumpkin_msm::utils::{gen_points, gen_scalars, naive_multiscalar_mul};
use halo2curves::group::Curve;

#[cfg(feature = "cuda")]
use grumpkin_msm::cuda_available;

fn main() {
    let bench_npow: usize = std::env::var("BENCH_NPOW")
        .unwrap_or("23".to_string())
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

    let context = grumpkin_msm::bn256::init(&points);

    let indices = (0..(npoints as u32)).rev().collect::<Vec<_>>();
    let res = grumpkin_msm::bn256::with_context_aux(
        &context,
        &scalars,
        Some(indices.as_slice()),
    )
    .to_affine();
    println!("res: {:?}", res);
    let res2 = grumpkin_msm::bn256::msm_aux(
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
