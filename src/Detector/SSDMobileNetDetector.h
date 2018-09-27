#pragma once

#include "BaseDetector.h"

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>

// MobileNet Single-Shot Detector (https://arxiv.org/abs/1704.04861) to detect objects
// .caffemodel model's file is available here:  https://github.com/chuanqi305/MobileNet-SSD
// Default network is 300x300 and 20-classes VOC

///
/// \brief The SSDMobileNetDetector class
///
class SSDMobileNetDetector : public BaseDetector
{
public:
    SSDMobileNetDetector(bool collectPoints, cv::UMat& colorFrame);
    ~SSDMobileNetDetector(void);

    bool Init(const config_t& config);

    void Detect(cv::UMat& colorFrame);

private:
    cv::dnn::Net m_net;

    void DetectInCrop(cv::Mat colorFrame, const cv::Rect& crop, regions_t& tmpRegions);

    static const int InWidth = 300;
    static const int InHeight = 300;
    float m_WHRatio;
    float m_inScaleFactor;
    float m_meanVal;
    float m_confidenceThreshold;
    float m_maxCropRatio;
    std::vector<std::string> m_classNames;
};
