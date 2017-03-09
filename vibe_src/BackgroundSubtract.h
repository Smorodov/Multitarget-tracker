#ifndef _BACKGROUND_SUBTRACT_H_
#define _BACKGROUND_SUBTRACT_H_

#include "defines.h"
#include "vibe.hpp"
#include <opencv2/bgsegm.hpp>

class BackgroundSubtract
{
public:
	enum BGFG_TYPE
	{
		Vibe_t,
		MOG_t,
		GMG_t
	};

	BackgroundSubtract(BGFG_TYPE algType, int channels = 1, int samples = 20, int pixel_neighbor = 1, int distance_threshold = 20, int matching_threshold = 3, int update_factor = 16);
	~BackgroundSubtract();

	void subtract(const cv::Mat& image, cv::Mat& foreground);
	
	int m_channels;
	BGFG_TYPE m_algType;

private:
	std::unique_ptr<vibe::VIBE> m_modelVibe;
	cv::Ptr<cv::BackgroundSubtractor> m_modelOCV;
};

#endif
