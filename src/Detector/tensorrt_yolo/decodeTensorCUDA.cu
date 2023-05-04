#include "decodeTensorCUDA.h"
__global__ void decodeTensorKernel(
    float* detections, uint32_t* masks, float* anchors, float* boxes, uint32_t grid_h, uint32_t grid_w, uint32_t numClasses, uint32_t numBBoxes)
{
    // get idx
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= grid_h || x >= grid_w) return;

    const int numGridCells = grid_h * grid_w;

    for (uint32_t b = 0; b < numBBoxes; ++b)
    {
        const float pw = anchors[masks[b] * 2];
        const float ph = anchors[masks[b] * 2 + 1];

        // printf("pw %f, ph %f \n",  pw, ph);
        const uint32_t bbindex = y * grid_w + x;
        boxes[18 * bbindex + 6 * b + 0] = x + detections[bbindex + numGridCells * (b * (5 + numClasses) + 0)];
  
        boxes[18 * bbindex + 6 * b + 1] = y + detections[bbindex + numGridCells * (b * (5 + numClasses) + 1)];
        boxes[18 * bbindex + 6 * b + 2] = pw * detections[bbindex + numGridCells * (b * (5 + numClasses) + 2)];
        boxes[18 * bbindex + 6 * b + 3] = ph * detections[bbindex + numGridCells * (b * (5 + numClasses) + 3)];

        // printf("x %f y %f w %f h %f\n", boxes[18 * bbindex + 6 * b + 0], boxes[18 * bbindex + 6 * b + 1], boxes[18 * bbindex + 6 * b + 2], boxes[18 * bbindex + 6 * b + 3]);

        const float objectness = detections[bbindex + numGridCells * (b * (5 + numClasses) + 4)];
        float maxProb = 0.0f;
        int maxIndex = -1;

        for (uint32_t i = 0; i < numClasses; ++i)
        {
            float prob = detections[bbindex + numGridCells * (b * (5 + numClasses) + (5 + i))];

            if (prob > maxProb)
            {
                maxProb = prob;
                maxIndex = i;
            }
        }
        // printf("objectness * maxProb  %f , objectness %f , maxProb %f \n", objectness * maxProb, objectness, maxProb);
        boxes[18 * bbindex + 6 * b + 4] = objectness * maxProb;
        boxes[18 * bbindex + 6 * b + 5] = (float) maxIndex;
    }
}

float* decodeTensorCUDA(const int imageIdx, const TensorInfo& tensor)
{
    // host
    int boxes_bytes = 6*sizeof(float)*tensor.grid_h*tensor.grid_w*tensor.numBBoxes; // x y w h maxProb maxIndex 6
	const float* detections = &tensor.hostBuffer[imageIdx * tensor.volume];
    float* boxes = (float*) malloc(boxes_bytes);

    uint32_t grid_h = tensor.grid_h;
    uint32_t grid_w = tensor.grid_w;
    uint32_t numClasses = tensor.numClasses;
    uint32_t numBBoxes = tensor.numBBoxes;

    // device
    float* d_detections;
    int detections_bytes = sizeof(float) * grid_h * grid_w * (5 + numClasses) * numBBoxes;
    cudaMalloc((void**) &d_detections, detections_bytes);
    cudaMemcpy((void*) d_detections, (void*) detections, detections_bytes, cudaMemcpyHostToDevice);

    uint32_t* d_masks;
    cudaMalloc((void**) &d_masks, sizeof(uint32_t)*numBBoxes);
    cudaMemcpy((void*) d_masks, (void*) &tensor.masks[0], sizeof(uint32_t)*numBBoxes, cudaMemcpyHostToDevice);

    float* d_anchors;
    cudaMalloc((void**) &d_anchors, sizeof(float)*tensor.anchors.size());
    cudaMemcpy((void*) d_anchors, (void*) &tensor.anchors[0], sizeof(float)*tensor.anchors.size(), cudaMemcpyHostToDevice);    

    float* d_boxes;
    cudaMalloc((void**) &d_boxes, boxes_bytes);

    // CUDA Grid and Block
    dim3 threads_per_block(20, 20);
    dim3 number_of_blocks((tensor.grid_w / threads_per_block.x) + 1, (tensor.grid_h / threads_per_block.y) + 1);

    // start kernel   
    decodeTensorKernel<<<number_of_blocks, threads_per_block>>>(d_detections, d_masks, d_anchors, d_boxes, grid_h, grid_w, numClasses, numBBoxes);

    // asynchronous copy
    cudaMemcpyAsync((void*) boxes, (void*) d_boxes, boxes_bytes, cudaMemcpyDeviceToHost); 
    
    // wait for kernel
    cudaDeviceSynchronize();
    
    // free memory
    cudaFree(d_detections);
    cudaFree(d_masks);
    cudaFree(d_anchors);
    cudaFree(d_boxes);
    
    // decode results
    return boxes;	
}
