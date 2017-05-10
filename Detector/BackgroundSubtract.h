#ifndef _BACKGROUND_SUBTRACT_H_
#define _BACKGROUND_SUBTRACT_H_

#include "defines.h"
#include "vibe_src/vibe.hpp"

#if USE_OCV_BGFG
#include <opencv2/bgsegm.hpp>
#endif

class BackgroundSubtract
{
public:
	enum BGFG_ALGS
	{
        ALG_VIBE,
        ALG_MOG,
        ALG_GMG,
        ALG_CNT
	};

	BackgroundSubtract(BGFG_ALGS algType, int channels = 1, int samples = 20, int pixel_neighbor = 1, int distance_threshold = 20, int matching_threshold = 3, int update_factor = 16);
	~BackgroundSubtract();

	void subtract(const cv::Mat& image, cv::Mat& foreground);
	
	int m_channels;
	BGFG_ALGS m_algType;

private:
	std::unique_ptr<vibe::VIBE> m_modelVibe;
#if USE_OCV_BGFG
	cv::Ptr<cv::BackgroundSubtractor> m_modelOCV;
#endif
};

#endif
