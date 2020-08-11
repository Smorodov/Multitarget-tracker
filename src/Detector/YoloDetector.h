#pragma once

#include "BaseDetector.h"

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>

// You only look once (YOLO)-Detector (https://arxiv.org/abs/1612.08242) to detect objects
// Models can be downloaded here: https://pjreddie.com/darknet/yolo/
// Default network is 416x416
// Class names can be downloaded here: https://github.com/pjreddie/darknet/tree/master/data

///
/// \brief The YoloOCVDetector class
///
class YoloOCVDetector : public BaseDetector
{
public:
    YoloOCVDetector(const cv::UMat& colorFrame);
    ~YoloOCVDetector(void);

    bool Init(const config_t& config);

    void Detect(const cv::UMat& colorFrame);

	bool CanGrayProcessing() const
	{
		return false;
	}

private:
    cv::dnn::Net m_net;

    void DetectInCrop(const cv::UMat& colorFrame, const cv::Rect& crop, regions_t& tmpRegions);

	cv::UMat m_inputBlob;

    int m_inWidth = 416;
    int m_inHeight = 416;

    float m_WHRatio = 1.f;
    float m_inScaleFactor = 0.003921f;
    float m_meanVal = 0.f;
    float m_confidenceThreshold = 0.24f;
    float m_maxCropRatio = 2.0f;
    std::vector<std::string> m_classNames;
};
