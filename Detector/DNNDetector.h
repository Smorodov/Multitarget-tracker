#pragma once

#include "BaseDetector.h"

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>

///
/// \brief The DNNDetector class
///
class DNNDetector : public BaseDetector
{
public:
    DNNDetector(bool collectPoints, cv::UMat& colorFrame);
    ~DNNDetector(void);

    bool Init(std::string modelConfiguration = "../data/MobileNetSSD_deploy.prototxt", std::string modelBinary = "../data/MobileNetSSD_deploy.caffemodel");

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
    std::vector<std::string> m_classNames;
};
