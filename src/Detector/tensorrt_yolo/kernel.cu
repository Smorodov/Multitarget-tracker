
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

inline __device__ float sigmoidGPU(const float& x) { return 1.0f / (1.0f + __expf(-x)); }

__global__ void gpuYoloLayerV3(const float* input, float* output, const uint32_t grid_h_,
								const uint32_t grid_w_, const uint32_t numOutputClasses,
                               const uint32_t numBBoxes)
{
    uint32_t x_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y_id = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t z_id = blockIdx.z * blockDim.z + threadIdx.z;

    if ((x_id >= grid_w_) || (y_id >= grid_h_) || (z_id >= numBBoxes))
    {
        return;
    }

    const int numGridCells = grid_h_ * grid_w_;
    const int bbindex = y_id * grid_w_ + x_id;

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)]);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)]);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)]
        = __expf(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)]);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)]
        = __expf(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)]);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)]);

    for (uint32_t i = 0; i < numOutputClasses; ++i)
    {
        output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))]
            = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))]);
    }
}

cudaError_t cudaYoloLayerV3(const void* input, void* output, const uint32_t& batchSize, 
							const uint32_t& n_grid_h_,const uint32_t& n_grid_w_,
                            const uint32_t& numOutputClasses, const uint32_t& numBBoxes,
                            uint64_t outputSize, cudaStream_t stream)
{
    dim3 threads_per_block(16, 16, 4);
    dim3 number_of_blocks((n_grid_w_ / threads_per_block.x) + 1,
                          (n_grid_h_ / threads_per_block.y) + 1,
                          (numBBoxes / threads_per_block.z) + 1);
    for (int batch = 0; batch < batchSize; ++batch)
    {
        gpuYoloLayerV3<<<number_of_blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<const float*>(input) + (batch * outputSize),
            reinterpret_cast<float*>(output) + (batch * outputSize), n_grid_h_, n_grid_w_, numOutputClasses,
            numBBoxes);
    }
    return cudaGetLastError();
}
