#include <chrono>

#define DEFINE_TRT_ENTRYPOINTS 1

#include "YoloONNX.hpp"
#include "../../mtracking/defines.h"

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the YOLO network by parsing the ONNX model and builds
//!          the engine that will be used to run YOLO (m_engine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool YoloONNX::Init(const SampleYoloParams& params)
{
    bool res = false;

    m_params = params;

    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");

    auto GetBindings = [&]()
    {
        auto numBindings = m_engine->getNbIOTensors();

        m_inputDims.clear();
        m_outpuDims.clear();

        std::cout << "** Bindings: " << numBindings << " **" << std::endl;
        for (int32_t i = 0; i < numBindings; ++i)
        {
            std::string bindName = m_engine->getIOTensorName(i);
            nvinfer1::Dims dim = m_engine->getTensorShape(bindName.c_str());
            
            for (const auto& inName : m_params.m_inputTensorNames)
            {
                if (bindName == inName)
                {
                    m_inputDims.emplace_back(dim);
                    break;
                }
            }
            
            for (const auto& outName : m_params.m_outputTensorNames)
            {
                if (bindName == outName)
                {
                    m_outpuDims.emplace_back(dim);
                    break;
                }
            }

            std::cout << i << ": name: " << bindName;
            std::cout << ", size: ";
            for (int j = 0; j < dim.nbDims; ++j)
            {
                std::cout << dim.d[j];
                if (j < dim.nbDims - 1)
                    std::cout << "x";
            }
            std::cout << std::endl;
        }
    };

    if (fs::exists(m_params.m_engineFileName))
    {
        std::vector<char> trtModelStream;
        size_t size{0};
        std::ifstream file(m_params.m_engineFileName, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream.resize(size);
            file.read(trtModelStream.data(), size);
            file.close();
        }

        nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(sample::gLogger);
        if (m_params.m_dlaCore >= 0)
            infer->setDLACore(m_params.m_dlaCore);

        m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(infer->deserializeCudaEngine(trtModelStream.data(), size), samplesCommon::InferDeleter());
#if (NV_TENSORRT_MAJOR < 8)
		infer->destroy();
#else
        //delete infer;
#endif

        if (m_engine)
        {
            GetBindings();
            res = true;
        }
        else
        {
            res = true;
        }
        sample::gLogInfo << "TRT Engine loaded from: " << m_params.m_engineFileName << " with res = " << res << std::endl;
    }
    else
    {
        auto builder = YoloONNXUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
        if (!builder)
            return false;

        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = YoloONNXUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if (!network)
            return false;

        auto parser = YoloONNXUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
        if (!parser)
            return false;

        auto config = YoloONNXUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config)
            return false;

        auto constructed = ConstructNetwork(builder, network, config, parser);
        if (!constructed)
            return false;

        GetBindings();
        assert(m_inputDims[0].nbDims == 4);        

        res = true;
    }

    std::cout << "YoloONNX::Init: loaded = " << res << std::endl;

    if (res)
    {
        m_buffers = std::make_unique<samplesCommon::BufferManager>(m_engine, 0/*m_params.batchSize*/);
        std::cout << "YoloONNX::Init: m_buffers = " << (m_buffers != nullptr) << std::endl;
        m_context = YoloONNXUniquePtr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
        std::cout << "YoloONNX::Init: m_context = " << (m_context != nullptr) << std::endl;
        if (!m_context)
            res = false;
    }

    std::cout << "YoloONNX::Init: res = " << res << std::endl;

    return res;
}

//!
//! \brief Uses an onnx parser to create the YOLO Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the YOLO network
//!
//! \param builder Pointer to the engine builder
//!
bool YoloONNX::ConstructNetwork(YoloONNXUniquePtr<nvinfer1::IBuilder>& builder,
                                YoloONNXUniquePtr<nvinfer1::INetworkDefinition>& network,
                                YoloONNXUniquePtr<nvinfer1::IBuilderConfig>& config,
                                YoloONNXUniquePtr<nvonnxparser::IParser>& parser)
{
    bool res = false;

    // Parse ONNX model file to populate TensorRT INetwork
    //int verbosity = (int) nvinfer1::ILogger::Severity::kERROR;
    int verbosity = (int)nvinfer1::ILogger::Severity::kVERBOSE;

    sample::gLogInfo << "Parsing ONNX file: " << m_params.m_onnxFileName << std::endl;

    if (!parser->parseFromFile(m_params.m_onnxFileName.c_str(), verbosity))
    {
        sample::gLogError << "Unable to parse ONNX model file: " << m_params.m_onnxFileName << std::endl;
        return res;
    }

    {
        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

        //std::cout << "m_params.inputTensorNames.size = " << m_params.inputTensorNames.size() << ", m_inputDims.size = " << m_inputDims.size() << std::endl;
        if (m_params.m_inputTensorNames.size() > 0)
        {
            nvinfer1::Dims dim = network->getInput(0)->getDimensions();
            //std::cout << "dim[0] = " << dim.nbDims << ": [" << dim.d[0] << ", " << dim.d[1] << ", " << dim.d[2] << ", " << dim.d[3] << "]" << std::endl;
            if (dim.d[0] < 1)
                dim.d[0] = 1;   // batch size
            if (dim.d[1] < 1)
                dim.d[1] = 3;   // Channels
            if (dim.d[2] < 1)
                dim.d[2] = 640; // Width
            if (dim.d[3] < 1)
                dim.d[3] = 640; // Height

            profile->setDimensions(m_params.m_inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kMIN, dim);
            profile->setDimensions(m_params.m_inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kOPT, dim);
            profile->setDimensions(m_params.m_inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kMAX, dim);
        }

        // For D-FINE
        if (m_params.m_inputTensorNames.size() > 1)
        {
            nvinfer1::Dims dim = network->getInput(1)->getDimensions();
            //std::cout << "dim[1] = " << dim.nbDims << ": [" << dim.d[0] << ", " << dim.d[1] << "]" << std::endl;
            if (dim.d[0] < 1)
                dim.d[0] = 1;   // batch size
            if (dim.d[1] < 1)
                dim.d[1] = 2;   // Input size

            profile->setDimensions(m_params.m_inputTensorNames[1].c_str(), nvinfer1::OptProfileSelector::kMIN, dim);
            profile->setDimensions(m_params.m_inputTensorNames[1].c_str(), nvinfer1::OptProfileSelector::kOPT, dim);
            profile->setDimensions(m_params.m_inputTensorNames[1].c_str(), nvinfer1::OptProfileSelector::kMAX, dim);
        }

        config->addOptimizationProfile(profile);
    }

#if (NV_TENSORRT_MAJOR < 8)
    builder->setMaxBatchSize(m_params.batchSize);
    config->setMaxWorkspaceSize(m_params.videoMemory ? m_params.videoMemory : 4096_MiB);
#else
    size_t workspaceSize = config->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE);
    size_t dlaManagedSRAMSize = config->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kDLA_MANAGED_SRAM);
    size_t dlaLocalDRAMSize = config->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kDLA_LOCAL_DRAM);
    size_t dlaGlobalDRAMSize = config->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kDLA_GLOBAL_DRAM);
	std::cout << "m_params.videoMemory = " << m_params.m_videoMemory << ", workspaceSize = " << workspaceSize << ", dlaManagedSRAMSize = " << dlaManagedSRAMSize << ", dlaLocalDRAMSize = " << dlaLocalDRAMSize << ", dlaGlobalDRAMSize = " << dlaGlobalDRAMSize << std::endl;

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, m_params.m_videoMemory ? m_params.m_videoMemory : workspaceSize);
#endif

    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);

    switch (m_params.m_precision)
    {
    case tensor_rt::Precision::FP16:
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        break;

    case tensor_rt::Precision::INT8:
    {
        // Calibrator life time needs to last until after the engine is built.
        std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator;

        BatchStream calibrationStream(m_params.m_explicitBatchSize, m_params.m_nbCalBatches, m_params.m_calibrationBatches, m_params.m_dataDirs);
        calibrator.reset(new Int8EntropyCalibrator2<BatchStream>(calibrationStream, 0, "Yolo", m_params.m_inputTensorNames[0].c_str()));
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
    }
        break;

    default:
        break;
    }

    // Enable DLA if mParams.dlaCore is true
    samplesCommon::enableDLA(builder.get(), config.get(), m_params.m_dlaCore);

    sample::gLogInfo << "Building TensorRT engine: " << m_params.m_engineFileName << std::endl;

#if (NV_TENSORRT_MAJOR < 8)
    m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
#else
    nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(sample::gLogger);
    if (m_params.m_dlaCore >= 0)
        infer->setDLACore(m_params.m_dlaCore);
    nvinfer1::IHostMemory* mem = builder->buildSerializedNetwork(*network, *config);
    if (mem)
        m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(infer->deserializeCudaEngine(mem->data(), mem->size()), samplesCommon::InferDeleter());
    else
        sample::gLogError << "Unable to buildSerializedNetwork" << std::endl;
    delete infer;
#endif

    if (!m_engine)
        return res;

    if (m_params.m_engineFileName.size() > 0)
    {
        std::ofstream p(m_params.m_engineFileName, std::ios::binary);
        if (!p)
            return false;

        nvinfer1::IHostMemory* ptr = m_engine->serialize();
        assert(ptr);
        p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
#if (NV_TENSORRT_MAJOR < 8)
		ptr->destroy();
#else
        delete ptr;
#endif
        p.close();
        sample::gLogInfo << "TRT Engine file saved to: " << m_params.m_engineFileName << std::endl;
    }
    res = true;

    return res;
}

///
/// \brief YoloONNX::Detect
/// \param frames
/// \param bboxes
/// \return
///
bool YoloONNX::Detect(const std::vector<cv::Mat>& frames, std::vector<tensor_rt::BatchResult>& bboxes)
{
    // Read the input data into the managed buffers
    if (!ProcessInputAspectRatio(frames))
        return false;

    // Memcpy from host input buffers to device input buffers
    m_buffers->copyInputToDevice();

    bool status = m_context->executeV2(m_buffers->getDeviceBindings().data());
    if (!status)
        return false;

    // Memcpy from device output buffers to host output buffers
    m_buffers->copyOutputToHost();

    // Post-process detections and verify results
    bboxes.resize(frames.size());
    for (size_t i = 0; i < bboxes.size(); ++i)
    {
        VerifyOutputAspectRatio(i, bboxes[i], frames[i].size());
    }

    return true;
}

///
/// \brief YoloONNX::GetInputSize
/// \return Return input size
///
cv::Size YoloONNX::GetInputSize() const
{
    return cv::Size(static_cast<int>(m_inputDims[0].d[3]), static_cast<int>(m_inputDims[0].d[2]));
}

///
/// \brief YoloONNX::GetNumClasses
/// \return
///
size_t YoloONNX::GetNumClasses() const
{
    if (m_outpuDims[0].nbDims == 2) // with nms
    {
        return 0;
    }
    else
    {
        size_t ncInd = 2;
        int nc = static_cast<int>(m_outpuDims[0].d[ncInd] - 5);
        return (size_t)nc;
    }
}

//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
bool YoloONNX::ProcessInputAspectRatio(const std::vector<cv::Mat>& sampleImages)
{
    const int inputB = static_cast<int>(m_inputDims[0].d[0]);
    const int inputC = static_cast<int>(m_inputDims[0].d[1]);
    const int inputH = static_cast<int>(m_inputDims[0].d[2]);
    const int inputW = static_cast<int>(m_inputDims[0].d[3]);

    float* hostInputBuffer = nullptr;
    if (m_params.m_inputTensorNames[0].empty())
        hostInputBuffer = static_cast<float*>(m_buffers->getHostBuffer(0));
    else
        hostInputBuffer = static_cast<float*>(m_buffers->getHostBuffer(m_params.m_inputTensorNames[0]));

    if (static_cast<int>(m_inputChannels.size()) < inputB)
    {
        for (int b = 0; b < inputB; ++b)
        {
            m_inputChannels.push_back(std::vector<cv::Mat> {static_cast<size_t>(inputC)});
        }
    }

    m_resizedROI = cv::Rect(0, 0, inputW, inputH);

#if 1
    // resize the image with scale
    const float imgHeight = static_cast<float>(sampleImages[0].rows);
    const float imgWidth = static_cast<float>(sampleImages[0].cols);
    float dim = std::max(imgHeight, imgWidth);
    int resizeH = cvRound((imgHeight * inputH) / dim);
    int resizeW = cvRound((imgWidth * inputW) / dim);
    //float scalingFactor = static_cast<float>(resizeH) / imgHeight;

    // Additional checks for images with non even dims
    if ((inputW - resizeW) % 2)
        resizeW--;
    if ((inputH - resizeH) % 2)
        resizeH--;
    assert((inputW - resizeW) % 2 == 0);
    assert((inputH - resizeH) % 2 == 0);

    float xOffset = (inputW - resizeW) / 2.f;
    float yOffset = (inputH - resizeH) / 2.f;

    assert(2 * xOffset + resizeW == inputW);
    assert(2 * yOffset + resizeH == inputH);

    cv::Size scaleSize(inputW, inputH);
    m_resizedROI = cv::Rect(cvRound(xOffset), cvRound(yOffset), resizeW, resizeH);

    //std::cout << "m_resizedROI: " << m_resizedROI << ", frameSize: " << sampleImages[0].size() << ", resizeW_H: " << cv::Size2f(resizeW, resizeH) << std::endl;

    if (m_resizedBatch.size() < sampleImages.size())
        m_resizedBatch.resize(sampleImages.size());

    // Each element in batch share the same image matrix
    for (int b = 0; b < inputB; ++b)
    {
        if (m_resizedBatch[b].size() != scaleSize)
            m_resizedBatch[b] = cv::Mat(scaleSize, sampleImages[b].type(), cv::Scalar::all(128));
        cv::resize(sampleImages[b], cv::Mat(m_resizedBatch[b], m_resizedROI), m_resizedROI.size(), 0, 0, cv::INTER_LINEAR);
        cv::split(m_resizedBatch[b], m_inputChannels[b]);
        std::swap(m_inputChannels[b][0], m_inputChannels[b][2]);
    }
#else
    auto scaleSize = cv::Size(inputW, inputH);

    if (m_resizedBatch.size() < sampleImages.size())
        m_resizedBatch.resize(sampleImages.size());

    // Each element in batch share the same image matrix
    for (int b = 0; b < inputB; ++b)
    {
        cv::resize(sampleImages[b], m_resizedBatch[b], scaleSize, 0, 0, cv::INTER_LINEAR);
        cv::split(m_resizedBatch[b], m_inputChannels[b]);
        std::swap(m_inputChannels[b][0], m_inputChannels[b][2]);
    }
#endif

    int volBatch = inputC * inputH * inputW;
    int volChannel = inputH * inputW;

    constexpr float to1 = 1.f / 255.0f;

    int d_batch_pos = 0;
    for (int b = 0; b < inputB; ++b)
    {
        int d_c_pos = d_batch_pos;
        for (int c = 0; c < inputC; ++c)
        {
            m_inputChannels[b][c].convertTo(cv::Mat(inputH, inputW, CV_32FC1, &hostInputBuffer[d_c_pos]), CV_32FC1, to1, 0);
            d_c_pos += volChannel;
        }
        d_batch_pos += volBatch;
    }

    // For D-FINE
    if (m_params.m_inputTensorNames.size() > 1)
    {
        int64_t* hostInput2 = static_cast<int64_t*>(m_buffers->getHostBuffer(m_params.m_inputTensorNames[1]));
        hostInput2[0] = inputW;
        hostInput2[1] = inputH;
    }
    return true;
}

//!
//! \brief Filters output detections and verify result
//!
//! \return whether the detection output matches expectations
//!
bool YoloONNX::VerifyOutputAspectRatio(size_t imgIdx, std::vector<tensor_rt::Result>& nms_bboxes, cv::Size frameSize)
{
    std::vector<float*> outputs;
    for (size_t i = 0; i < m_params.m_outputTensorNames.size();)
    {
        float* output = static_cast<float*>(m_buffers->getHostBuffer(m_params.m_outputTensorNames[i]));
#if 0
        if (output)
            outputs.push_back(output);
#else
        if (!output)
        {
            std::cout << i << " output tensor \"" << m_params.m_outputTensorNames[i] << "\" is null, will be removed" << std::endl;
            m_params.m_outputTensorNames.erase(std::begin(m_params.m_outputTensorNames) + i);
        }
        else
        {
            outputs.push_back(output);
            ++i;
        }
#endif
    }
    if (!outputs.empty())
        nms_bboxes = GetResult(imgIdx, m_params.m_keepTopK, outputs, frameSize);

    return !outputs.empty();
}
