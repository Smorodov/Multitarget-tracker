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
    int m_keepTopK = 1000;                 //!< The maximum number of detection post-NMS
    int m_nbCalBatches = 100;              //!< The number of batches for calibration
    float m_confThreshold = 0.3;
    float m_nmsThreshold = 0.5;

    size_t m_videoMemory = 0     ;         //!< If zero then will use default value

    int m_explicitBatchSize = 1;
    std::string m_calibrationBatches;      //!< The path to calibration batches
    std::string m_engineFileName;

    std::string m_onnxFileName;            //!< Filename of ONNX file of a network
    int32_t m_dlaCore{-1};                 //!< Specify the DLA core to run network on.
    tensor_rt::ModelType m_netType { tensor_rt::ModelType::YOLOV7 };
    tensor_rt::Precision m_precision { tensor_rt::Precision::FP32 }; //!< Allow runnning the network in Int8 mode.
    std::vector<std::string> m_dataDirs;   //!< Directory paths where sample data files are stored
    std::vector<std::string> m_inputTensorNames;
    std::vector<std::string> m_outputTensorNames;
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
    virtual ~YoloONNX() = default;

    //!
    //! \brief Function builds the network engine
    //!
    bool Init(const SampleYoloParams& params);

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool Detect(const std::vector<cv::Mat>& frames, std::vector<tensor_rt::BatchResult>& bboxes);

    //!
    //! \brief Return input size
    //!
    cv::Size GetInputSize() const;

    //!
    //! \brief Return classes count
    //!
    size_t GetNumClasses() const;

protected:
    SampleYoloParams m_params;               //!< The parameters for the sample
    std::vector<nvinfer1::Dims> m_inputDims; //!< The dimensions of the input to the network
    std::vector<nvinfer1::Dims> m_outpuDims; //!< The dimensions of the input to the network
    cv::Rect m_resizedROI;                   //!< Input frame resized into input dimensions with the frame aspect ratio

    virtual std::vector<tensor_rt::Result> GetResult(size_t imgIdx, int keep_topk, const std::vector<float*>& outputs, cv::Size frameSize) = 0;

private:    
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine; //!< The TensorRT engine used to run the network

    cv::Mat m_resized;
    std::vector<cv::Mat> m_resizedBatch;
    std::vector<std::vector<cv::Mat>> m_inputChannels;

    std::unique_ptr<samplesCommon::BufferManager> m_buffers;
    YoloONNXUniquePtr<nvinfer1::IExecutionContext> m_context;

    //!
    //! \brief Parses an ONNX model for YOLO and creates a TensorRT network
    //!
    bool ConstructNetwork(YoloONNXUniquePtr<nvinfer1::IBuilder>& builder,
                          YoloONNXUniquePtr<nvinfer1::INetworkDefinition>& network, YoloONNXUniquePtr<nvinfer1::IBuilderConfig>& config,
                          YoloONNXUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
    //!
    bool ProcessInputAspectRatio(const std::vector<cv::Mat>& sampleImages);

    //!
    //! \brief Filters output detections and verify results
    //!
    bool VerifyOutputAspectRatio(size_t imgIdx, std::vector<tensor_rt::Result>& nms_bboxes, cv::Size frameSize);
};
