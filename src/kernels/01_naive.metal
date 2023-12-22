kernel void fxn(
    const device float* data0,
    const device float* data1,
    device float* data2,
    constant uint& M,
    constant uint& N,
    constant uint& K,
    constant float& ALPHA,
    constant float& BETA,
    uint3 gid [[ threadgroup_position_in_grid ]],
    uint3 lid [[ thread_position_in_threadgroup ]],
    uint3 blockdim [[ threads_per_threadgroup ]]
) {
    const uint x = gid.x * blockdim.x + lid.x;
    const uint y = gid.y * blockdim.y + lid.y;

    if (x < M && y < N) {
        float acc = 0.0;
        for (uint i = 0; i < K; i++) {
            acc += data0[(x * K) + i] * data1[(i * N) + y];
        }
        data2[(x * N) + y] = ALPHA * acc + BETA * data2[(x * N) + y];
    }
}
