#include <chrono>

#include "YoloONNX.hpp"
#include "trt_utils.h"

///
/// \brief calculate_num_boxes
/// \param input_h
/// \param input_w
/// \return
///
int calculate_num_boxes(int input_h, int input_w)
{
    int num_anchors = 3;

    int h1 = input_h / 8;
    int h2 = input_h / 16;
    int h3 = input_h / 32;

    int w1 = input_w / 8;
    int w2 = input_w / 16;
    int w3 = input_w / 32;

    return num_anchors * (h1 * w1 + h2 * w2 + h3 * w3);
}

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the YOLO network by parsing the ONNX model and builds
//!          the engine that will be used to run YOLO (this->mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool YoloONNX::Init(const SampleYoloParams& params)
{
    bool res = false;

    mParams = params;

    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");

    if (fileExists(mParams.engingFileName))
    {
        std::vector<char> trtModelStream;
        size_t size{0};
        std::ifstream file(mParams.engingFileName, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream.resize(size);
            file.read(trtModelStream.data(), size);
            file.close();
        }

        IRuntime* infer = nvinfer1::createInferRuntime(sample::gLogger);
        if (mParams.dlaCore >= 0)
            infer->setDLACore(mParams.dlaCore);

        this->mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(infer->deserializeCudaEngine(trtModelStream.data(), size, nullptr), samplesCommon::InferDeleter());
#if (NV_TENSORRT_MAJOR < 8)
		infer->destroy();
#else
        delete infer;
#endif

        sample::gLogInfo << "TRT Engine loaded from: " << mParams.engingFileName << std::endl;

        std::cout << "**Bindings**" << std::endl;
        for (int32_t i = 0; i < mEngine->getNbBindings(); ++i)
        {
            nvinfer1::Dims dim = mEngine->getBindingDimensions(i);

            std::cout << "name: " << mEngine->getBindingName(i) << std::endl;
            std::cout << "size: ";
            for (int j = 0; j < dim.nbDims; j++)
            {
                std::cout << dim.d[j];
                if (j < dim.nbDims - 1)
                    std::cout << "x";
            }
            std::cout << std::endl;
        }

        std::cout << "Num of bindings in engine: " << mEngine->getNbBindings() << std::endl;

        if (!this->mEngine)
        {
            res = false;
        }
        else
        {
            this->mInputDims.nbDims = this->mParams.inputShape.size();
            this->mInputDims.d[0] = this->mParams.inputShape[0];
            this->mInputDims.d[1] = this->mParams.inputShape[1];
            this->mInputDims.d[2] = this->mParams.inputShape[2];
            this->mInputDims.d[3] = this->mParams.inputShape[3];

            res = true;
        }
    }
    else
    {
        auto builder = YoloONNXUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
        if (!builder)
            return false;

        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = YoloONNXUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if (!network)
            return false;

        auto config = YoloONNXUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config)
            return false;

        auto parser = YoloONNXUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
        if (!parser)
            return false;

        auto constructed = constructNetwork(builder, network, config, parser);
        if (!constructed)
            return false;

        assert(network->getNbInputs() == 1);
        this->mInputDims = network->getInput(0)->getDimensions();
        std::cout << this->mInputDims.nbDims << std::endl;
        assert(this->mInputDims.nbDims == 4);

        res = true;
    }

    if (res)
    {
        m_buffers = std::make_unique<samplesCommon::BufferManager>(this->mEngine, mParams.batchSize);
        m_context = YoloONNXUniquePtr<nvinfer1::IExecutionContext>(this->mEngine->createExecutionContext());
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
bool YoloONNX::constructNetwork(YoloONNXUniquePtr<nvinfer1::IBuilder>& builder,
    YoloONNXUniquePtr<nvinfer1::INetworkDefinition>& network, YoloONNXUniquePtr<nvinfer1::IBuilderConfig>& config,
    YoloONNXUniquePtr<nvonnxparser::IParser>& parser)
{
    // Parse ONNX model file to populate TensorRT INetwork
    int verbosity = (int) nvinfer1::ILogger::Severity::kERROR;

    sample::gLogInfo << "Parsing ONNX file: " << mParams.onnxFileName << std::endl;

    if (!parser->parseFromFile(mParams.onnxFileName.c_str(), verbosity))
    {
        sample::gLogError << "Unable to parse ONNX model file: " << mParams.onnxFileName << std::endl;
        return false;
    }

    builder->setMaxBatchSize(mParams.batchSize);

    config->setMaxWorkspaceSize(4096_MiB);

    config->setFlag(BuilderFlag::kGPU_FALLBACK);

    if (mParams.fp16)
        config->setFlag(BuilderFlag::kFP16);

    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;

    // issue for int8 mode
    if (mParams.int8)
    {
        BatchStream calibrationStream(mParams.explicitBatchSize, mParams.nbCalBatches, mParams.calibrationBatches, mParams.dataDirs);
        calibrator.reset(new Int8EntropyCalibrator2<BatchStream>(calibrationStream, 0, "Yolo", mParams.inputTensorNames[0].c_str()));
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
    }

    // Enable DLA if mParams.dlaCore is true
    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    sample::gLogInfo << "Building TensorRT engine: " << mParams.engingFileName << std::endl;

    this->mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());

    if (!this->mEngine)
        return false;

    if (mParams.engingFileName.size() > 0)
    {
        std::ofstream p(mParams.engingFileName, std::ios::binary);
        if (!p)
            return false;

        nvinfer1::IHostMemory* ptr = this->mEngine->serialize();
        assert(ptr);
        p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
#if (NV_TENSORRT_MAJOR < 8)
		ptr->destroy();
#else
        delete ptr;
#endif
        p.close();
        sample::gLogInfo << "TRT Engine file saved to: " << mParams.engingFileName << std::endl;
    }

    return true;
}

bool YoloONNX::infer_iteration(YoloONNXUniquePtr<nvinfer1::IExecutionContext> &context, samplesCommon::BufferManager &buffers, const cv::Mat& image, std::vector<tensor_rt::Result>& bboxes)
{
    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);

    if (!processInput_aspectRatio(buffers, image))
        return false;

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
        return false;

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Post-process detections and verify results
    if (!verifyOutput_aspectRatio(buffers, bboxes, image.size()))
        return false;

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
    if (!this->infer_iteration(m_context, *m_buffers.get(), frame, bboxes))
        return false;
    return true;
}

//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
bool YoloONNX::processInput_aspectRatio(const samplesCommon::BufferManager& buffers, const cv::Mat &mSampleImage)
{
    const int inputB = this->mInputDims.d[0];
    const int inputC = this->mInputDims.d[1];
    const int inputH = this->mInputDims.d[2];
    const int inputW = this->mInputDims.d[3];

    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer(this->mParams.inputTensorNames[0]));

    std::vector<std::vector<cv::Mat>> input_channels;
    for (int b = 0; b < inputB; ++b)
    {
        input_channels.push_back(std::vector<cv::Mat> {static_cast<size_t>(inputC)});
    }

    auto scaleSize = cv::Size(inputW, inputH);
    cv::resize(mSampleImage, m_resized, scaleSize, 0, 0, cv::INTER_LINEAR);

    // Each element in batch share the same image matrix
    for (int b = 0; b < inputB; ++b)
    {
        cv::split(m_resized, input_channels[b]);
        std::swap(input_channels[b][0], input_channels[b][2]);
    }

    int volBatch = inputC * inputH * inputW;
    int volChannel = inputH * inputW;
    int volW = inputW;

    int d_batch_pos = 0;
    for (int b = 0; b < inputB; b++)
    {
        int d_c_pos = d_batch_pos;
        for (int c = 0; c < inputC; c++)
        {
            int s_h_pos = 0;
            int d_h_pos = d_c_pos;
            for (int h = 0; h < inputH; h++)
            {
                int s_pos = s_h_pos;
                int d_pos = d_h_pos;
                for (int w = 0; w < inputW; w++)
                {
                    hostInputBuffer[d_pos] = (float)input_channels[b][c].data[s_pos] / 255.0f;
                    ++s_pos;
                    ++d_pos;
                }
                s_h_pos += volW;
                d_h_pos += volW;
            }
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
bool YoloONNX::verifyOutput_aspectRatio(const samplesCommon::BufferManager& buffers, std::vector<tensor_rt::Result>& nms_bboxes, cv::Size frameSize)
{
    const int keepTopK = mParams.keepTopK;

    float *output = static_cast<float*>(buffers.getHostBuffer(this->mParams.outputTensorNames[0]));

    if (!output)
    {
        std::cout << "NULL value output detected!" << std::endl;
        return false;
    }

    nms_bboxes = this->get_bboxes(this->mParams.outputShapes[0][0], keepTopK, output, frameSize);

    return true;
}

///
/// \brief YoloONNX::get_bboxes
/// \param output
/// \return
///
std::vector<tensor_rt::Result> YoloONNX::get_bboxes(int /*batch_size*/, int /*keep_topk*/, float* output, cv::Size frameSize)
{
    std::vector<tensor_rt::Result> bboxes;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> rectBoxes;

    const float fw = static_cast<float>(frameSize.width) / static_cast<float>(mInputDims.d[3]);
    const float fh = static_cast<float>(frameSize.height) / static_cast<float>(mInputDims.d[2]);

    size_t i = 0;
    int nc = 80;
    while (i < 8400)
    {
        // Box
        size_t k = i * 85;
        float object_conf = output[k + 4];

        if (object_conf < this->mParams.confThreshold)
        {
            i++;
            continue;
        }

        // (center x, center y, width, height) to (x, y, w, h)
        float x = fw * (output[k] - output[k + 2] / 2);
        float y = fh * (output[k + 1] - output[k + 3] / 2);
        float width = fw * output[k + 2];
        float height = fh * output[k + 3];

        // Classes
        float class_conf = output[k + 5];
        int classId = 0;

        for (int j = 1; j < nc; j++)
        {
            if (class_conf < output[k + 5 + j])
            {
                classId = j;
                class_conf = output[k + 5 + j];
            }
        }

        i++;

        class_conf *= object_conf;

        classIds.push_back(classId);
        confidences.push_back(class_conf);
        rectBoxes.emplace_back(cvRound(x), cvRound(y), cvRound(width), cvRound(height));
    }

    // Non-maximum suppression to eliminate redudant overlapping boxes
    std::vector<int> indices;
    cv::dnn::NMSBoxes(rectBoxes, confidences, this->mParams.confThreshold, this->mParams.nmsThreshold, indices);
    bboxes.reserve(indices.size());

    for (size_t bi = 0; bi < indices.size(); ++bi)
    {
        bboxes.emplace_back(classIds[indices[bi]], confidences[indices[bi]], rectBoxes[indices[bi]]);
    }

    return bboxes;
}
