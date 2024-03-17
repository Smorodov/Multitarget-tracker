#pragma once

#include <opencv2/opencv.hpp>


#if defined(_MSC_VER)
#define LIB_API __declspec(dllexport) 
#else
#define LIB_API __attribute__((visibility("default")))
#endif

///
struct CLIPResult
{
	CLIPResult() = default;
	CLIPResult(const std::string& label, float conf) noexcept
		: m_label(label), m_conf(conf)
	{
	}

	std::string m_label;
	float m_conf = 0.f;
};

///
class LIB_API ClassificationCLIP
{
public:
	ClassificationCLIP() noexcept;
	~ClassificationCLIP() noexcept;

	bool Init(const std::string& pathToClip, const std::string& pathToBPE, int inputImgSize, int indGPU, const std::vector<std::string>& labels);

	bool ProcessFrame(const cv::Mat& frame, const std::vector<cv::Rect>& rois, std::vector<CLIPResult>& result);

	class ClassificationCLIPImpl;

private:
	ClassificationCLIPImpl* m_pImpl = nullptr;
};
