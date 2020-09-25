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

#include "yolov2.h"
#include <algorithm>

YoloV2::YoloV2(const NetworkInfo& networkInfo,
               const InferParams& inferParams) :
    Yolo(networkInfo, inferParams){}

std::vector<BBoxInfo> YoloV2::decodeTensor(const int imageIdx, const int imageH, const int imageW,
                                           const TensorInfo& tensor)
{
    float scalingFactor
        = std::min(static_cast<float>(m_InputW) / imageW, static_cast<float>(m_InputH) / imageH);
    float xOffset = (m_InputW - scalingFactor * imageW) / 2;
    float yOffset = (m_InputH - scalingFactor * imageH) / 2;

    float* detections = &tensor.hostBuffer[imageIdx * tensor.volume];

    std::vector<BBoxInfo> binfo;
    for (uint32_t y = 0; y < tensor.gridSize; y++)
    {
        for (uint32_t x = 0; x < tensor.gridSize; x++)
        {
            for (uint32_t b = 0; b < tensor.numBBoxes; b++)
            {
                const float pw = tensor.anchors[2 * b];
                const float ph = tensor.anchors[2 * b + 1];
                const int numGridCells = tensor.gridSize * tensor.gridSize;
                const int bbindex = y * tensor.gridSize + x;
                const float bx
                    = x + detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 0)];
                const float by
                    = y + detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 1)];
                const float bw = pw
                    * exp(detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 2)]);
                const float bh = ph
                    * exp(detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 3)]);

                const float objectness
                    = detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 4)];
                float maxProb = 0.0f;
                int maxIndex = -1;

                for (uint32_t i = 0; i < tensor.numClasses; i++)
                {
                    float prob
                        = detections[bbindex
                                     + numGridCells * (b * (5 + tensor.numClasses) + (5 + i))];

                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxIndex = i;
                    }
                }

                maxProb = objectness * maxProb;

                if (maxProb > m_ProbThresh)
                {
                    addBBoxProposal(bx, by, bw, bh, tensor.stride, scalingFactor, xOffset, yOffset,
                                    maxIndex, maxProb,imageW,imageH, binfo);
                }
            }
        }
    }
    return binfo;
}
