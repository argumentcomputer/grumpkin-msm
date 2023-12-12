# Multi-Scalar Multiplication for the Grumpkin curve cycle

> [!NOTE] 
> This is basically a copy of the [`pasta_msm`](https://github.com/lurk-lab/pasta-msm) crate.

To compile CUDA support ensure that you have `nvcc`, Nvidia CUDA compiler, on your program search path. Minimal installation suffices. For example on [Ubuntu](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network) it would be sufficient to install `cuda-minimal-build-11-7` instead of complete `cuda` package. If your laptop is equipped with a Turing+ controller, you're likely to have to compile with `--features=cuda-mobile`. Caveat lector. CUDA implementation does not adapt for the actual load **yet**, so that some results would be suboptimal.
