# Optimizing a Metal Matmul Kernel

[siboehm's post](https://siboehm.com/articles/22/CUDA-MMM) explains how to iteratively improve the performance of a CUDA kernel for matrix multiplication.

This repo contains a reimplementation of those kernels (not all yet) on [Metal](https://developer.apple.com/documentation/metal), Apple's GPUs compute API.

## Running a kernel

`./src/run.py`

## Performance

Performance on M1 Pro:

| Kernel                   | GFLOPs/s |
|--------------------------|----------|
| 1: Naive                 | 20       |
| 2: GMEM Coalescing       | 280      |
| 3: SMEM Caching          | -        |
| 4: 1D Blocktiling        | -        |
| 5: 2D Blocktiling        | -        |
| 6: Vectorized Mem Access | -        |
| 9: Autotuning            | -        |
| 10: Warptiling           | -        |
