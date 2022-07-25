#pragma once

#include "common/BatchStream.h"
#include "common/EntropyCalibrator.h"
#include "common/buffers.h"
#include "common/common.h"
#include "common/logger.h"

#include "NvOnnxParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>

#include <opencv2/opencv.hpp>
#include "class_detector.h"

//!
//! \brief The SampleYoloParams structure groups the additional parameters required by
//!         the SSD sample.
//!
struct SampleYoloParams
{
    int outputClsSize = 80;              //!< The number of output classes
    int topK = 2000;
    int keepTopK = 1000;                   //!< The maximum number of detection post-NMS
    int nbCalBatches = 100;               //!< The number of batches for calibration
    float confThreshold = 0.3;
    float nmsThreshold = 0.5;

    int explicitBatchSize = 1;
    std::string calibrationBatches; //!< The path to calibration batches
    std::string engingFileName;

    std::string onnxFileName; //!< Filename of ONNX file of a network
    int32_t batchSize{1};              //!< Number of inputs in a batch
    int32_t dlaCore{-1};               //!< Specify the DLA core to run network on.
    tensor_rt::Precision m_precision { tensor_rt::Precision::FP32 }; //!< Allow runnning the network in Int8 mode.
    std::vector<std::string> dataDirs; //!< Directory paths where sample data files are stored
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
};

///
/// \brief The YoloONNX class
///
class YoloONNX
{
    template <typename T>
    using YoloONNXUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    YoloONNX() = default;

    //!
    //! \brief Function builds the network engine
    //!
    bool Init(const SampleYoloParams& params);

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool Detect(cv::Mat frame, std::vector<tensor_rt::Result>& bboxes);

    //!
    //! \brief Cleans up any state created in the sample class
    //!
    bool teardown();

    //!
    //! \brief Return input size
    //!
    cv::Size GetInputSize() const;

private:
    SampleYoloParams m_params; //!< The parameters for the sample.

    nvinfer1::Dims m_inputDims; //!< The dimensions of the input to the network.
    std::vector<nvinfer1::Dims> m_outpuDims; //!< The dimensions of the input to the network.

    std::shared_ptr<nvinfer1::ICudaEngine> m_engine; //!< The TensorRT engine used to run the network

    size_t mImageIdx = 0;
    void* mCudaImg = nullptr;

    cv::Mat m_resized;
    std::vector<std::vector<cv::Mat>> m_inputChannels;

    std::unique_ptr<samplesCommon::BufferManager> m_buffers;
    YoloONNXUniquePtr<nvinfer1::IExecutionContext> m_context;

    //!
    //! \brief Parses an ONNX model for YOLO and creates a TensorRT network
    //!
    bool constructNetwork(YoloONNXUniquePtr<nvinfer1::IBuilder>& builder,
                          YoloONNXUniquePtr<nvinfer1::INetworkDefinition>& network, YoloONNXUniquePtr<nvinfer1::IBuilderConfig>& config,
                          YoloONNXUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
    //!
    bool processInput_aspectRatio(const cv::Mat &mSampleImage);

    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Filters output detections and verify results
    //!
    bool verifyOutput_aspectRatio(std::vector<tensor_rt::Result>& nms_bboxes, cv::Size frameSize);

    bool verifyOutput(const samplesCommon::BufferManager& buffers);

    std::vector<tensor_rt::Result> get_bboxes(int batch_size, int keep_topk, float* output, cv::Size frameSize);
};
