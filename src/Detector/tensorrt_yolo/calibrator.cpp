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

#include "calibrator.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <cuda_runtime.h>
#include <cuda.h>

Int8EntropyCalibrator::Int8EntropyCalibrator(const uint32_t& batchSize, const std::string& calibImages,
                                             const std::string& calibImagesPath,
                                             const std::string& calibTableFilePath,
                                             const uint64_t& inputSize, const uint32_t& inputH,
                                             const uint32_t& inputW, const std::string& inputBlobName, tensor_rt::ModelType netType) :
    m_BatchSize(batchSize),
    m_InputH(inputH),
    m_InputW(inputW),
    m_InputSize(inputSize),
    m_InputCount(batchSize * inputSize),
    m_netType(netType),
    m_InputBlobName(inputBlobName),
    m_CalibTableFilePath(calibTableFilePath)
{
    if (!fileExists(m_CalibTableFilePath, false))
    {
		std::random_device rng;
		std::mt19937 urng(rng());

		m_ImageList = loadImageList(calibImages, calibImagesPath);
		m_ImageList.resize(static_cast<int>(m_ImageList.size() / m_BatchSize) * m_BatchSize);
		std::shuffle(m_ImageList.begin(), m_ImageList.end(), urng);
    }

    NV_CUDA_CHECK(cudaMalloc(&m_DeviceInput, m_InputCount * sizeof(float)));
}

Int8EntropyCalibrator::~Int8EntropyCalibrator()
{
    NV_CUDA_CHECK(cudaFree(m_DeviceInput));
}

bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int /*nbBindings*/) noexcept
{
    if (m_ImageIndex + m_BatchSize >= m_ImageList.size()) return false;

    // Load next batch
    std::vector<DsImage> dsImages(m_BatchSize);
    for (uint32_t j = m_ImageIndex; j < m_ImageIndex + m_BatchSize; ++j)
    {
        dsImages.at(j - m_ImageIndex) = DsImage(m_ImageList.at(j), m_netType, m_InputH, m_InputW);
    }
    m_ImageIndex += m_BatchSize;

    blobFromDsImages(dsImages, m_blob, m_InputH, m_InputW);

    NV_CUDA_CHECK(cudaMemcpy(m_DeviceInput, m_blob.ptr<float>(0), m_InputCount * sizeof(float),
                             cudaMemcpyHostToDevice));
    assert(!strcmp(names[0], m_InputBlobName.c_str()));
    bindings[0] = m_DeviceInput;
    return true;
}

const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) noexcept
{
    void* output;
    m_CalibrationCache.clear();
    assert(!m_CalibTableFilePath.empty());
    std::ifstream input(m_CalibTableFilePath, std::ios::binary | std::ios::in);
    input >> std::noskipws;
    if (m_ReadCache && input.good())
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                  std::back_inserter(m_CalibrationCache));

    length = m_CalibrationCache.size();
    if (length)
    {
        std::cout << "Using cached calibration table to build the engine" << std::endl;
        output = &m_CalibrationCache[0];
    }
    else
    {
        std::cout << "New calibration table will be created to build the engine" << std::endl;
        output = nullptr;
    }

    return output;
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length) noexcept
{
    assert(!m_CalibTableFilePath.empty());
    std::ofstream output(m_CalibTableFilePath, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
    output.close();
}
