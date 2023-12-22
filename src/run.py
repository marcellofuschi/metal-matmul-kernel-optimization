#!/usr/bin/env python3
import Metal
import ctypes
import numpy as np
from math import ceil
from typing import Tuple

M = 1500
K = 3500
N = 2000
ALPHA = 1.62
BETA = 2.4

GRID_DIM = (ceil(M / 32), ceil(N / 32), 1)

USABLE_KERNELS = {
    '01': ('kernels/01_naive.metal', (32, 32, 1)),
    '02': ('kernels/02_global_mem_coalesce.metal', (32*32, 1, 1)),
}
KERNEL_FILE, BLOCK_DIM = USABLE_KERNELS['02']


def main():
    a = np.random.rand(M, K).astype(np.float32)
    b = np.random.rand(K, N).astype(np.float32)
    c = np.random.rand(M, N).astype(np.float32)

    expected_result = ALPHA*(a @ b) + BETA*c  # GEMM
    flop = 2*M*N*K + M*N
    # print(f'{flop/1e9:.1f} gflop')

    result, elapsed_secs = launch_kernel(a, b, c)
    assert np.allclose(result, expected_result), 'Wrong result'
    # print(f'Time on GPU: {elapsed_secs:.2f}s')
    print(f'Performance: {flop/elapsed_secs * 1e-9:.1f} GFLOP/s')


def launch_kernel(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, int]:
    with open(KERNEL_FILE, 'r') as f:
        prg = f.read()

    device = Metal.MTLCreateSystemDefaultDevice()

    options = Metal.MTLCompileOptions.new()
    library, err = device.newLibraryWithSource_options_error_(prg, options, None)
    assert err is None, str(err)

    fxn = library.newFunctionWithName_('fxn')
    pipeline_state, err = device.newComputePipelineStateWithFunction_error_(fxn, None)
    assert err is None, str(err)

    command_queue = device.newCommandQueue()
    command_buffer = command_queue.commandBuffer()

    encoder = command_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline_state)

    # Allocate buffers. fp32 => 4 bytes
    a_buf, = device.newBufferWithLength_options_(4 * a.size, Metal.MTLResourceStorageModeShared),
    b_buf, = device.newBufferWithLength_options_(4 * b.size, Metal.MTLResourceStorageModeShared),
    c_buf, = device.newBufferWithLength_options_(4 * a.shape[0] * b.shape[1], Metal.MTLResourceStorageModeShared),

    # Copy the data into buffers
    get_buf_memory = lambda buf: buf.contents().as_buffer(buf.length())
    get_buf_memory(a_buf)[:] = a.tobytes()
    get_buf_memory(b_buf)[:] = b.tobytes()
    get_buf_memory(c_buf)[:] = c.tobytes()

    encoder.setBuffer_offset_atIndex_(a_buf, 0, 0)
    encoder.setBuffer_offset_atIndex_(b_buf, 0, 1)
    encoder.setBuffer_offset_atIndex_(c_buf, 0, 2)

    encoder.setBytes_length_atIndex_(ctypes.c_int32(a.shape[0]), 4, 3)
    encoder.setBytes_length_atIndex_(ctypes.c_int32(b.shape[1]), 4, 4)
    encoder.setBytes_length_atIndex_(ctypes.c_int32(a.shape[1]), 4, 5)
    encoder.setBytes_length_atIndex_(ctypes.c_float(ALPHA), 4, 6)
    encoder.setBytes_length_atIndex_(ctypes.c_float(BETA), 4, 7)

    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(*GRID_DIM),
        Metal.MTLSizeMake(*BLOCK_DIM)
    )
    encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()

    elapsed_secs = command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
    result = np.frombuffer(get_buf_memory(c_buf), dtype=np.float32).reshape(a.shape[0], b.shape[1])

    return result, elapsed_secs


if __name__ == "__main__":
    main()
