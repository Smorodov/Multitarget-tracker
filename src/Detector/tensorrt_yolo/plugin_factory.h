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

#ifndef __PLUGIN_LAYER_H__
#define __PLUGIN_LAYER_H__

#include <cassert>
#include <cstring>
#include <cudnn.h>
#include <iostream>
#include <memory>

#include "NvInferPlugin.h"

#define NV_CUDA_CHECK(status)                                                                      \
    {                                                                                              \
        if (status != 0)                                                                           \
        {                                                                                          \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) << " in file " << __FILE__ \
                      << " at line " << __LINE__ << std::endl;                                     \
            abort();                                                                               \
        }                                                                                          \
    }

// Forward declaration of cuda kernels
cudaError_t cudaYoloLayerV3(const void* input, void* output, const uint32_t& batchSize,
                            const uint32_t& gridSize, const uint32_t& numOutputClasses,
                            const uint32_t& numBBoxes, uint64_t outputSize, cudaStream_t stream);

class PluginFactory : public nvinfer1::IPluginFactory
{

public:
    PluginFactory();
    nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData,
                                    size_t serialLength) override;
    bool isPlugin(const char* name);
    void destroy();

private:
    static const int m_MaxLeakyLayers = 72;
    static const int m_ReorgStride = 2;
    static constexpr float m_LeakyNegSlope = 0.1;
    static const int m_NumBoxes = 5;
    static const int m_NumCoords = 4;
    static const int m_NumClasses = 80;
    static const int m_MaxYoloLayers = 3;
    int m_LeakyReLUCount = 0;
    int m_YoloLayerCount = 0;
    nvinfer1::plugin::RegionParameters m_RegionParameters{m_NumBoxes, m_NumCoords, m_NumClasses,
                                                          nullptr};

    struct INvPluginDeleter
    {
        void operator()(nvinfer1::plugin::INvPlugin* ptr)
        {
            if (ptr)
            {
                ptr->destroy();
            }
        }
    };
    struct IPluginDeleter
    {
        void operator()(nvinfer1::IPlugin* ptr)
        {
            if (ptr)
            {
                ptr->terminate();
            }
        }
    };
    typedef std::unique_ptr<nvinfer1::plugin::INvPlugin, INvPluginDeleter> unique_ptr_INvPlugin;
    typedef std::unique_ptr<nvinfer1::IPlugin, IPluginDeleter> unique_ptr_IPlugin;

    unique_ptr_INvPlugin m_ReorgLayer;
    unique_ptr_INvPlugin m_RegionLayer;
    unique_ptr_INvPlugin m_LeakyReLULayers[m_MaxLeakyLayers];
    unique_ptr_IPlugin m_YoloLayers[m_MaxYoloLayers];
};

class YoloLayerV3 : public nvinfer1::IPlugin
{
public:
    YoloLayerV3(const void* data, size_t length);
    YoloLayerV3(const uint32_t& numBoxes, const uint32_t& numClasses, const uint32_t& gridSize);
    int getNbOutputs() const override;
    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                       int nbInputDims) override;
    void configure(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims,
                   int nbOutputs, int maxBatchSize) override;
    int initialize() override;
    void terminate() override;
    size_t getWorkspaceSize(int maxBatchSize) const override;
    int enqueue(int batchSize, const void* const* intputs, void** outputs, void* workspace,
                cudaStream_t stream) override;
    size_t getSerializationSize() override;
    void serialize(void* buffer) override;

private:
    template <typename T>
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
    uint32_t m_NumBoxes;
    uint32_t m_NumClasses;
    uint32_t m_GridSize;
    uint64_t m_OutputSize;
};

#endif // __PLUGIN_LAYER_H__