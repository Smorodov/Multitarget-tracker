#ifndef _BACKGROUND_SUBTRACT_H_
#define _BACKGROUND_SUBTRACT_H_

#include "defines.h"
#include "vibe_src/vibe.hpp"
#include "Subsense/BackgroundSubtractorSuBSENSE.h"
#include "Subsense/BackgroundSubtractorLOBSTER.h"

#ifdef USE_OCV_BGFG
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
        ALG_CNT,
        ALG_SuBSENSE,
        ALG_LOBSTER,
        ALG_MOG2
	};

	BackgroundSubtract(BGFG_ALGS algType, int channels = 1, int samples = 20, int pixel_neighbor = 1, int distance_threshold = 20, int matching_threshold = 3, int update_factor = 16);
	~BackgroundSubtract();

    void subtract(const cv::UMat& image, cv::UMat& foreground);
	
	int m_channels;
	BGFG_ALGS m_algType;

private:
	std::unique_ptr<vibe::VIBE> m_modelVibe;
	cv::Ptr<cv::BackgroundSubtractor> m_modelOCV;
    std::unique_ptr<BackgroundSubtractorLBSP> m_modelSuBSENSE;
};

#endif
