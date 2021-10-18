#pragma once

#include "BaseDetector.h"

#include "darknet/include/yolo_v2_class.hpp"
// You only look once (YOLO)-Detector (https://arxiv.org/abs/1612.08242) to detect objects
// Models can be downloaded here: https://pjreddie.com/darknet/yolo/
// Default network is 416x416
// Class names can be downloaded here: https://github.com/pjreddie/darknet/tree/master/data


///
/// \brief The YoloDarknetDetector class
///
class YoloDarknetDetector final : public BaseDetector
{
public:
    YoloDarknetDetector(const cv::UMat& colorFrame);
    YoloDarknetDetector(const cv::Mat& colorFrame);
    ~YoloDarknetDetector(void) = default;

    bool Init(const config_t& config) override;

    void Detect(const cv::UMat& colorFrame) override;
    void Detect(const std::vector<cv::UMat>& frames, std::vector<regions_t>& regions) override;

    bool CanGrayProcessing() const override
    {
        return false;
    }

private:
	std::unique_ptr<Detector> m_detector;

    float m_confidenceThreshold = 0.5f;
    float m_maxCropRatio = 3.0f;
	size_t m_batchSize = 1;
	std::vector<std::string> m_classNames;
	cv::Size m_netSize;

	void DetectInCrop(const cv::Mat& colorFrame, const cv::Rect& crop, regions_t& tmpRegions);
	void Detect(const cv::Mat& colorFrame, regions_t& tmpRegions);
	void FillImg(image_t& detImage);
	void FillBatchImg(const std::vector<cv::Mat>& batch, image_t& detImage);

	cv::Mat m_tmpImg;
	std::vector<float> m_tmpBuf;
};
