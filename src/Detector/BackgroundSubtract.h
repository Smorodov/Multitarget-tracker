#pragma once

#include "defines.h"
#include "vibe_src/vibe.hpp"
#include "Subsense/BackgroundSubtractorSuBSENSE.h"
#include "Subsense/BackgroundSubtractorLOBSTER.h"

#ifdef USE_OCV_BGFG
#include <opencv2/bgsegm.hpp>
#endif

///
/// \brief The BackgroundSubtract class
///
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

    BackgroundSubtract(BGFG_ALGS algType, int channels);
    ~BackgroundSubtract() = default;

    bool Init(const config_t& config);

    void Subtract(const cv::UMat& image, cv::UMat& foreground);

	void ResetModel(const cv::UMat& img, const cv::Rect& roiRect);
	
	int m_channels = 1;
	BGFG_ALGS m_algType = BGFG_ALGS::ALG_MOG2;

private:
	std::unique_ptr<vibe::VIBE> m_modelVibe;
	cv::Ptr<cv::BackgroundSubtractor> m_modelOCV;
    std::unique_ptr<BackgroundSubtractorLBSP> m_modelSuBSENSE;

	cv::UMat m_rawForeground;

	cv::UMat GetImg(const cv::UMat& image);
};
