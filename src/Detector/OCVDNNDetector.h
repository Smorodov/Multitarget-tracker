#pragma once

#ifdef USE_OCV_DNN

#include "BaseDetector.h"

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>

///
/// \brief The OCVDNNDetector class
///
class OCVDNNDetector final : public BaseDetector
{
public:
    OCVDNNDetector(const cv::UMat& colorFrame);
    OCVDNNDetector(const cv::Mat& colorFrame);
    ~OCVDNNDetector(void) = default;

    bool Init(const config_t& config) override;

    void Detect(const cv::UMat& colorFrame) override;

    bool CanGrayProcessing() const override
    {
        return false;
    }

private:
    enum class ModelType
    {
        Unknown,
        YOLOV3,
        YOLOV3_TINY,
        YOLOV4,
        YOLOV4_TINY,
        YOLOV5,
        YOLOV5_OBB,
        YOLOV5Mask,
        YOLOV6,
        YOLOV7,
        YOLOV7Mask,
        YOLOV8,
        YOLOV8_OBB,
        YOLOV8Mask,
        YOLOV9,
        YOLOV10,
        YOLOV11,
        YOLOV11_OBB,
        YOLOV11Mask,
        YOLOV12,
        RFDETR,
        DFINE
    };

    cv::dnn::Net m_net;

    void DetectInCrop(const cv::UMat& colorFrame, const cv::Rect& crop, regions_t& tmpRegions);

    int m_inWidth = 608;
    int m_inHeight = 608;

    float m_WHRatio = 1.f;
    double m_inScaleFactor = 0.003921; // 1 / 255
    //double m_inScaleFactor = 1.0;
    cv::Scalar m_meanVal = {0, 0, 0};
    float m_confidenceThreshold = 0.24f;
    track_t m_nmsThreshold = static_cast<track_t>(0.4);
    bool m_swapRB = true;
    float m_maxCropRatio = 2.0f;
    ModelType m_netType = ModelType::Unknown;
    std::vector<std::string> m_classNames;
    std::vector<cv::String> m_outNames;
    std::vector<int> m_outLayers;
    std::vector<std::string> m_outLayerTypes;
    cv::UMat m_inputBlob;

    void ParseOldYOLO(const cv::Rect& crop, const std::vector<cv::Mat>& detections, regions_t& tmpRegions);

    void ParseYOLOv5(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions);
    void ParseYOLOv8(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions);
    void ParseYOLOv9(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions);
    void ParseYOLOv10(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions);
    void ParseYOLOv11(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions);
    void ParseYOLOv5_8_11_obb(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions);
    void ParseYOLOv5_8_11_seg(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions);
    void ParseRFDETR(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions);
    void ParseDFINE(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions);
};

#endif
