#include <chrono>

#include "YoloONNX.hpp"
#include "trt_utils.h"
#include "../../common/defines.h"

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

        std::cout << "** Bindings: " << numBindings << " **" << std::endl;
        for (int32_t i = 0; i < numBindings; ++i)
        {
            std::string bindName = m_engine->getIOTensorName(i);
            nvinfer1::Dims dim = m_engine->getTensorShape(bindName.c_str());
            for (const auto& outName : m_params.outputTensorNames)
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

    if (fileExists(m_params.engineFileName))
    {
        std::vector<char> trtModelStream;
        size_t size{0};
        std::ifstream file(m_params.engineFileName, std::ios::binary);
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
        if (m_params.dlaCore >= 0)
            infer->setDLACore(m_params.dlaCore);

        m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(infer->deserializeCudaEngine(trtModelStream.data(), size), samplesCommon::InferDeleter());
#if (NV_TENSORRT_MAJOR < 8)
		infer->destroy();
#else
        delete infer;
#endif

        if (m_engine)
        {
            GetBindings();
            m_inputDims = m_engine->getTensorShape(m_engine->getIOTensorName(0));
            res = true;
        }
        else
        {
            res = true;
        }
        sample::gLogInfo << "TRT Engine loaded from: " << m_params.engineFileName << " with res = " << res << std::endl;
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

        auto config = YoloONNXUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config)
            return false;

        auto parser = YoloONNXUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
        if (!parser)
            return false;

        auto constructed = ConstructNetwork(builder, network, config, parser);
        if (!constructed)
            return false;

        assert(network->getNbInputs() == 1);
        m_inputDims = network->getInput(0)->getDimensions();
        std::cout << m_inputDims.nbDims << std::endl;
        assert(m_inputDims.nbDims == 4);

        GetBindings();

        res = true;
    }

    if (res)
    {
        m_buffers = std::make_unique<samplesCommon::BufferManager>(m_engine, 0/*m_params.batchSize*/);
        m_context = YoloONNXUniquePtr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
        if (!m_context)
            res = false;
    }

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
                                YoloONNXUniquePtr<nvinfer1::INetworkDefinition>& network, YoloONNXUniquePtr<nvinfer1::IBuilderConfig>& config,
                                YoloONNXUniquePtr<nvonnxparser::IParser>& parser)
{
    // Parse ONNX model file to populate TensorRT INetwork
    //int verbosity = (int) nvinfer1::ILogger::Severity::kERROR;
    int verbosity = (int)nvinfer1::ILogger::Severity::kVERBOSE;

    sample::gLogInfo << "Parsing ONNX file: " << m_params.onnxFileName << std::endl;

    if (!parser->parseFromFile(m_params.onnxFileName.c_str(), verbosity))
    {
        sample::gLogError << "Unable to parse ONNX model file: " << m_params.onnxFileName << std::endl;
        return false;
    }

#if (NV_TENSORRT_MAJOR < 8)
    builder->setMaxBatchSize(m_params.batchSize);
    config->setMaxWorkspaceSize(m_params.videoMemory ? m_params.videoMemory : 4096_MiB);
#else
    size_t workspaceSize = config->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE);
    size_t dlaManagedSRAMSize = config->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kDLA_MANAGED_SRAM);
    size_t dlaLocalDRAMSize = config->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kDLA_LOCAL_DRAM);
    size_t dlaGlobalDRAMSize = config->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kDLA_GLOBAL_DRAM);
	std::cout << "workspaceSize = " << workspaceSize << ", dlaManagedSRAMSize = " << dlaManagedSRAMSize << ", dlaLocalDRAMSize = " << dlaLocalDRAMSize << ", dlaGlobalDRAMSize = " << dlaGlobalDRAMSize << std::endl;

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, m_params.videoMemory ? m_params.videoMemory : (1 << 20));
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

        BatchStream calibrationStream(m_params.explicitBatchSize, m_params.nbCalBatches, m_params.calibrationBatches, m_params.dataDirs);
        calibrator.reset(new Int8EntropyCalibrator2<BatchStream>(calibrationStream, 0, "Yolo", m_params.inputTensorNames[0].c_str()));
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
    }
        break;

    default:
        break;
    }

    // Enable DLA if mParams.dlaCore is true
    samplesCommon::enableDLA(builder.get(), config.get(), m_params.dlaCore);

    sample::gLogInfo << "Building TensorRT engine: " << m_params.engineFileName << std::endl;

#if (NV_TENSORRT_MAJOR < 8)
    m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
#else
    nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(sample::gLogger);
    if (m_params.dlaCore >= 0)
        infer->setDLACore(m_params.dlaCore);
    nvinfer1::IHostMemory* mem = builder->buildSerializedNetwork(*network, *config);
    m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(infer->deserializeCudaEngine(mem->data(), mem->size()), samplesCommon::InferDeleter());
    delete infer;
#endif

    if (!m_engine)
        return false;

    if (m_params.engineFileName.size() > 0)
    {
        std::ofstream p(m_params.engineFileName, std::ios::binary);
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
        sample::gLogInfo << "TRT Engine file saved to: " << m_params.engineFileName << std::endl;
    }

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool YoloONNX::Detect(cv::Mat frame, std::vector<tensor_rt::Result>& bboxes)
{
    // Read the input data into the managed buffers
    assert(m_params.inputTensorNames.size() == 1);

    if (!ProcessInputAspectRatio(frame))
        return false;

    // Memcpy from host input buffers to device input buffers
    m_buffers->copyInputToDevice();

    bool status = m_context->executeV2(m_buffers->getDeviceBindings().data());
    if (!status)
        return false;

    // Memcpy from device output buffers to host output buffers
    m_buffers->copyOutputToHost();

    // Post-process detections and verify results
    if (!VerifyOutputAspectRatio(0, bboxes, frame.size()))
        return false;

    return true;
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
    assert(m_params.inputTensorNames.size() == 1);

#if 1
    if (!ProcessInputAspectRatio(frames))
        return false;
#else
    std::vector<DsImage> vec_ds_images;
    cv::Size netSize = GetInputSize();
    for (const auto& img : frames)
    {
        vec_ds_images.emplace_back(img, m_params.m_netType, netSize.height, netSize.width);
    }
    int sz[] = { m_params.explicitBatchSize, 3, netSize.height, netSize.width };
    float* hostInputBuffer = nullptr;
    if (m_params.inputTensorNames[0].empty())
        hostInputBuffer = static_cast<float*>(m_buffers->getHostBuffer(0));
    else
        hostInputBuffer = static_cast<float*>(m_buffers->getHostBuffer(m_params.inputTensorNames[0]));
    cv::Mat blob(4, sz, CV_32F, hostInputBuffer);
    blobFromDsImages(vec_ds_images, blob, netSize.height, netSize.width);
#endif
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
    return cv::Size(m_inputDims.d[3], m_inputDims.d[2]);
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
        int nc = m_outpuDims[0].d[ncInd] - 5;
        return (size_t)nc;
    }
}

//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
bool YoloONNX::ProcessInputAspectRatio(const cv::Mat& mSampleImage)
{
    const int inputB = m_inputDims.d[0];
    const int inputC = m_inputDims.d[1];
    const int inputH = m_inputDims.d[2];
    const int inputW = m_inputDims.d[3];

    float* hostInputBuffer = nullptr;
    if (m_params.inputTensorNames[0].empty())
        hostInputBuffer = static_cast<float*>(m_buffers->getHostBuffer(0));
    else
        hostInputBuffer = static_cast<float*>(m_buffers->getHostBuffer(m_params.inputTensorNames[0]));

    if (static_cast<int>(m_inputChannels.size()) < inputB)
    {
        for (int b = 0; b < inputB; ++b)
        {
            m_inputChannels.push_back(std::vector<cv::Mat> {static_cast<size_t>(inputC)});
        }
    }

    auto scaleSize = cv::Size(inputW, inputH);
    cv::resize(mSampleImage, m_resized, scaleSize, 0, 0, cv::INTER_LINEAR);

    // Each element in batch share the same image matrix
    for (int b = 0; b < inputB; ++b)
    {
        cv::split(m_resized, m_inputChannels[b]);
        std::swap(m_inputChannels[b][0], m_inputChannels[b][2]);
    }

    int volBatch = inputC * inputH * inputW;
    int volChannel = inputH * inputW;

    constexpr float to1 =  1.f / 255.0f;

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

    return true;
}

//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
bool YoloONNX::ProcessInputAspectRatio(const std::vector<cv::Mat>& mSampleImage)
{
    const int inputB = m_inputDims.d[0];
    const int inputC = m_inputDims.d[1];
    const int inputH = m_inputDims.d[2];
    const int inputW = m_inputDims.d[3];

    float* hostInputBuffer = nullptr;
    if (m_params.inputTensorNames[0].empty())
        hostInputBuffer = static_cast<float*>(m_buffers->getHostBuffer(0));
    else
        hostInputBuffer = static_cast<float*>(m_buffers->getHostBuffer(m_params.inputTensorNames[0]));

    if (static_cast<int>(m_inputChannels.size()) < inputB)
    {
        for (int b = 0; b < inputB; ++b)
        {
            m_inputChannels.push_back(std::vector<cv::Mat> {static_cast<size_t>(inputC)});
        }
    }

    auto scaleSize = cv::Size(inputW, inputH);

    if (m_resizedBatch.size() < mSampleImage.size())
        m_resizedBatch.resize(mSampleImage.size());

    // Each element in batch share the same image matrix
    for (int b = 0; b < inputB; ++b)
    {
        cv::resize(mSampleImage[b], m_resizedBatch[b], scaleSize, 0, 0, cv::INTER_LINEAR);
        cv::split(m_resizedBatch[b], m_inputChannels[b]);
        std::swap(m_inputChannels[b][0], m_inputChannels[b][2]);
    }

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
    for (size_t i = 0; i < m_params.outputTensorNames.size();)
    {
        float* output = static_cast<float*>(m_buffers->getHostBuffer(m_params.outputTensorNames[i]));
#if 0
        if (output)
            outputs.push_back(output);
#else
        if (!output)
        {
            std::cout << i << " output tensor \"" << m_params.outputTensorNames[i] << "\" is null, will be removed" << std::endl;
            m_params.outputTensorNames.erase(std::begin(m_params.outputTensorNames) + i);
        }
        else
        {
            outputs.push_back(output);
            ++i;
        }
#endif
    }
    if (!outputs.empty())
        nms_bboxes = GetResult(imgIdx, m_params.keepTopK, outputs, frameSize);

    return !outputs.empty();
}
