#pragma once

#include "BaseDetector.h"
#include "tensorrt_yolo/class_detector.h"

///
/// \brief The YoloTensorRTDetector class
///
class YoloTensorRTDetector : public BaseDetector
{
public:
	YoloTensorRTDetector(const cv::UMat& colorFrame);
	~YoloTensorRTDetector(void) = default;

	bool Init(const config_t& config);

	void Detect(const cv::UMat& colorFrame);
    void Detect(const std::vector<cv::UMat>& frames, std::vector<regions_t>& regions);

	bool CanGrayProcessing() const
	{
		return false;
	}

private:
	std::unique_ptr<tensor_rt::Detector> m_detector;

    float m_maxCropRatio = 3.0f;
	std::vector<std::string> m_classNames;

	tensor_rt::Config m_localConfig;
    size_t m_batchSize = 1;
};
