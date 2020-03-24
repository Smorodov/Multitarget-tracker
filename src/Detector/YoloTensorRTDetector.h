#pragma once

#include "BaseDetector.h"
#include "tensorrt_yolo/class_detector.h"

///
/// \brief The YoloTensorRTDetector class
///
class YoloTensorRTDetector : public BaseDetector
{
public:
	YoloTensorRTDetector(cv::UMat& colorFrame);
	~YoloTensorRTDetector(void);

	bool Init(const config_t& config);

	void Detect(cv::UMat& colorFrame);

	bool CanGrayProcessing() const
	{
		return false;
	}

private:
	std::unique_ptr<tensor_rt::Detector> m_detector;

    float m_confidenceThreshold = 0.5f;
    float m_maxCropRatio = 3.0f;
	std::vector<std::string> m_classNames;
};
