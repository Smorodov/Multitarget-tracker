#pragma once

#include "BaseDetector.h"
#include "tensorrt_onnx/class_detector.h"

///
/// \brief The ONNXTensorRTDetector class
///
class ONNXTensorRTDetector final : public BaseDetector
{
public:
	ONNXTensorRTDetector(const cv::UMat& colorFrame);
	ONNXTensorRTDetector(const cv::Mat& colorFrame);
	~ONNXTensorRTDetector(void) = default;

	bool Init(const config_t& config) override;

	void Detect(const cv::UMat& colorFrame) override;
    void Detect(const std::vector<cv::UMat>& frames, std::vector<regions_t>& regions) override;

	bool CanGrayProcessing() const override
	{
		return false;
	}

	void CalcMotionMap(cv::Mat& frame);

private:
	std::unique_ptr<tensor_rt::Detector> m_detector;

    float m_maxCropRatio = 3.0f;
	std::vector<std::string> m_classNames;

	tensor_rt::Config m_localConfig;
    size_t m_batchSize = 1;
};
