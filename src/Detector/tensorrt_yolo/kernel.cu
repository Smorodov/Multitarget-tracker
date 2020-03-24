/**
MIT License

Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

inline __device__ float sigmoidGPU(const float& x) { return 1.0f / (1.0f + __expf(-x)); }

__global__ void gpuYoloLayerV3(const float* input, float* output, const uint32_t gridSize, const uint32_t numOutputClasses,
                               const uint32_t numBBoxes)
{
    uint32_t x_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y_id = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t z_id = blockIdx.z * blockDim.z + threadIdx.z;

    if ((x_id >= gridSize) || (y_id >= gridSize) || (z_id >= numBBoxes))
    {
        return;
    }

    const int numGridCells = gridSize * gridSize;
    const int bbindex = y_id * gridSize + x_id;

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

cudaError_t cudaYoloLayerV3(const void* input, void* output, const uint32_t& batchSize, const uint32_t& gridSize,
                            const uint32_t& numOutputClasses, const uint32_t& numBBoxes,
                            uint64_t outputSize, cudaStream_t stream)
{
    dim3 threads_per_block(16, 16, 4);
    dim3 number_of_blocks((gridSize / threads_per_block.x) + 1,
                          (gridSize / threads_per_block.y) + 1,
                          (numBBoxes / threads_per_block.z) + 1);
    for (int batch = 0; batch < batchSize; ++batch)
    {
        gpuYoloLayerV3<<<number_of_blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<const float*>(input) + (batch * outputSize),
            reinterpret_cast<float*>(output) + (batch * outputSize), gridSize, numOutputClasses,
            numBBoxes);
    }
    return cudaGetLastError();
}