#pragma once

#include "BaseDetector.h"

#define OPENCV
#include "darknet/include/yolo_v2_class.hpp"
#undef OPENCV
// You only look once (YOLO)-Detector (https://arxiv.org/abs/1612.08242) to detect objects
// Models can be downloaded here: https://pjreddie.com/darknet/yolo/
// Default network is 416x416
// Class names can be downloaded here: https://github.com/pjreddie/darknet/tree/master/data


///
/// \brief The YoloDarknetDetector class
///
class YoloDarknetDetector : public BaseDetector
{
public:
    YoloDarknetDetector(cv::UMat& colorFrame);
	~YoloDarknetDetector(void);

	bool Init(const config_t& config);

	void Detect(cv::UMat& colorFrame);

	bool CanGrayProcessing() const
	{
		return false;
	}

private:
	std::unique_ptr<Detector> m_detector;

    float m_confidenceThreshold = 0.5f;
    float m_maxCropRatio = 3.0f;
	std::vector<std::string> m_classNames;
	std::set<std::string> m_classesWhiteList;
};
