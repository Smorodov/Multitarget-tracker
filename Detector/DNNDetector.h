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
};
