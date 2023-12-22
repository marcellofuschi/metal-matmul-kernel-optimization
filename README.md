# How to Optimize a Metal Matmul Kernel

[This blog post](https://siboehm.com/articles/22/CUDA-MMM) by siboehm explains how to iteratively improve the performance of a CUDA kernel for matrix multiplication.

In this repo, I reimplemented the kernels on [Metal](https://developer.apple.com/documentation/metal), Apple's GPUs compute API.

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
