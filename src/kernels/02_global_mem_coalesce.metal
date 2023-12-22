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
    uint3 lid [[ thread_position_in_threadgroup ]]
) {
    const uint BLOCKSIZE = 32;
    const uint x = gid.x * BLOCKSIZE + (lid.x / BLOCKSIZE);
    const uint y = gid.y * BLOCKSIZE + (lid.x % BLOCKSIZE);

    if (x < M && y < N) {
        float tmp = 0.0;
        for (uint i = 0; i < K; i++) {
            tmp += data0[(x * K) + i] * data1[(i * N) + y];
        }
        data2[(x * N) + y] = ALPHA * tmp + BETA * data2[(x * N) + y];
    }
}
