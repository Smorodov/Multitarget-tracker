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

#include "plugin_factory.h"
#include "trt_utils.h"

PluginFactory::PluginFactory() : m_ReorgLayer{nullptr}, m_RegionLayer{nullptr}
{
    for (int i = 0; i < m_MaxLeakyLayers; ++i) m_LeakyReLULayers[i] = nullptr;
}

nvinfer1::IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData,
                                               size_t serialLength)
{
    assert(isPlugin(layerName));
    if (std::string(layerName).find("leaky") != std::string::npos)
    {
        assert(m_LeakyReLUCount >= 0 && m_LeakyReLUCount <= m_MaxLeakyLayers);
        assert(m_LeakyReLULayers[m_LeakyReLUCount] == nullptr);
        m_LeakyReLULayers[m_LeakyReLUCount]
            = unique_ptr_INvPlugin(nvinfer1::plugin::createPReLUPlugin(serialData, serialLength));
        ++m_LeakyReLUCount;
        return m_LeakyReLULayers[m_LeakyReLUCount - 1].get();
    }
    else if (std::string(layerName).find("reorg") != std::string::npos)
    {
        assert(m_ReorgLayer == nullptr);
        m_ReorgLayer = unique_ptr_INvPlugin(
            nvinfer1::plugin::createYOLOReorgPlugin(serialData, serialLength));
        return m_ReorgLayer.get();
    }
    else if (std::string(layerName).find("region") != std::string::npos)
    {
        assert(m_RegionLayer == nullptr);
        m_RegionLayer = unique_ptr_INvPlugin(
            nvinfer1::plugin::createYOLORegionPlugin(serialData, serialLength));
        return m_RegionLayer.get();
    }
    else if (std::string(layerName).find("yolo") != std::string::npos)
    {
        assert(m_YoloLayerCount >= 0 && m_YoloLayerCount < m_MaxYoloLayers);
        assert(m_YoloLayers[m_YoloLayerCount] == nullptr);
        m_YoloLayers[m_YoloLayerCount]
            = unique_ptr_IPlugin(new YoloLayerV3(serialData, serialLength));
        ++m_YoloLayerCount;
        return m_YoloLayers[m_YoloLayerCount - 1].get();
    }
    else
    {
        std::cerr << "ERROR: Unrecognised layer : " << layerName << std::endl;
        assert(0);
        return nullptr;
    }
}

bool PluginFactory::isPlugin(const char* name)
{
    return ((std::string(name).find("leaky") != std::string::npos)
            || (std::string(name).find("reorg") != std::string::npos)
            || (std::string(name).find("region") != std::string::npos)
            || (std::string(name).find("yolo") != std::string::npos));
}

void PluginFactory::destroy()
{
    m_ReorgLayer.reset();
    m_RegionLayer.reset();

    for (int i = 0; i < m_MaxLeakyLayers; ++i)
    {
        m_LeakyReLULayers[i].reset();
    }

    for (int i = 0; i < m_MaxYoloLayers; ++i)
    {
        m_YoloLayers[i].reset();
    }

    m_LeakyReLUCount = 0;
    m_YoloLayerCount = 0;
}

/******* Yolo Layer V3 *******/
/*****************************/
YoloLayerV3::YoloLayerV3(const void* data, size_t length)
{
    const char *d = static_cast<const char*>(data), *a = d;
    read(d, m_NumBoxes);
    read(d, m_NumClasses);
    read(d, m_GridSize);
    read(d, m_OutputSize);
    assert(d = a + length);
}

YoloLayerV3::YoloLayerV3(const uint32_t& numBoxes, const uint32_t& numClasses, const uint32_t& gridSize) :
    m_NumBoxes(numBoxes),
    m_NumClasses(numClasses),
    m_GridSize(gridSize)
{
    assert(m_NumBoxes > 0);
    assert(m_NumClasses > 0);
    assert(m_GridSize > 0);
    m_OutputSize = m_GridSize * m_GridSize * (m_NumBoxes * (4 + 1 + m_NumClasses));
}

int YoloLayerV3::getNbOutputs() const { return 1; }

nvinfer1::Dims YoloLayerV3::getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                                int nbInputDims)
{
    assert(index == 0);
    assert(nbInputDims == 1);
    return inputs[0];
}

void YoloLayerV3::configure(const nvinfer1::Dims* inputDims, int nbInputs,
                            const nvinfer1::Dims* outputDims, int nbOutputs, int maxBatchSize)
{
    assert(nbInputs == 1);
    assert(inputDims != nullptr);
}

int YoloLayerV3::initialize() { return 0; }

void YoloLayerV3::terminate() {}

size_t YoloLayerV3::getWorkspaceSize(int maxBatchSize) const { return 0; }

int YoloLayerV3::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace,
                         cudaStream_t stream)
{
    NV_CUDA_CHECK(cudaYoloLayerV3(inputs[0], outputs[0], batchSize, m_GridSize, m_NumClasses,
                                  m_NumBoxes, m_OutputSize, stream));
    return 0;
}

size_t YoloLayerV3::getSerializationSize()
{
    return sizeof(m_NumBoxes) + sizeof(m_NumClasses) + sizeof(m_GridSize) + sizeof(m_OutputSize);
}

void YoloLayerV3::serialize(void* buffer)
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, m_NumBoxes);
    write(d, m_NumClasses);
    write(d, m_GridSize);
    write(d, m_OutputSize);
    assert(d == a + getSerializationSize());
}