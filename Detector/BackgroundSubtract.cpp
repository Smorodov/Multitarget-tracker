#include "BackgroundSubtract.h"

//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
BackgroundSubtract::BackgroundSubtract(
	BGFG_ALGS algType,
	int channels,
	int samples,
	int pixel_neighbor,
	int distance_threshold,
	int matching_threshold,
	int update_factor
	)
	:
	m_channels(channels),
    m_algType(algType)
{
    for (bool failed = true; failed;)
    {
        failed = false;

        switch (m_algType)
        {
        case ALG_VIBE:
            m_modelVibe = std::make_unique<vibe::VIBE>(m_channels, samples, pixel_neighbor, distance_threshold, matching_threshold, update_factor);
            break;

#ifdef USE_OCV_BGFG
        case ALG_MOG:
            m_modelOCV = cv::bgsegm::createBackgroundSubtractorMOG(100, 3, 0.7, 0);
            break;

        case ALG_GMG:
            m_modelOCV = cv::bgsegm::createBackgroundSubtractorGMG(50, 0.7);
            break;

        case ALG_CNT:
#if (((CV_VERSION_MAJOR == 3) && (CV_VERSION_MINOR >= 2)) || (CV_VERSION_MAJOR > 3))
            m_modelOCV = cv::bgsegm::createBackgroundSubtractorCNT(15, true, 15 * 60, true);
            break;
#else
            std::cerr << "OpenCV CNT algorithm is not implemented! Used Vibe by default." << std::endl;
            failed = true;
            break;
#endif

#else
        case ALG_MOG:
        case ALG_GMG:
        case ALG_CNT:
            std::cerr << "OpenCV bgfg algorithms are not implemented! Used Vibe by default." << std::endl;
            failed = true;
            break;
#endif

        case ALG_SuBSENSE:
            m_modelSuBSENSE = std::make_unique<BackgroundSubtractorSuBSENSE>(); // default params
            break;

        case ALG_LOBSTER:
            m_modelSuBSENSE = std::make_unique<BackgroundSubtractorLOBSTER>();  // default params
            break;

        case ALG_MOG2:
            m_modelOCV = cv::createBackgroundSubtractorMOG2(500, 16, true).dynamicCast<cv::BackgroundSubtractor>();
            break;

        default:
            m_modelVibe = std::make_unique<vibe::VIBE>(m_channels, samples, pixel_neighbor, distance_threshold, matching_threshold, update_factor);
            break;
        }
        if (failed)
        {
            m_algType = ALG_VIBE;
            failed = false;
        }
    }
}

//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
BackgroundSubtract::~BackgroundSubtract()
{
}

//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
void BackgroundSubtract::subtract(const cv::UMat& image, cv::UMat& foreground)
{
    auto GetImg = [&]() -> cv::UMat
	{
		if (image.channels() != m_channels)
		{
			if (image.channels() == 1)
			{
                cv::UMat newImg;
				cv::cvtColor(image, newImg, CV_GRAY2BGR);
				return newImg;
			}
			else if (image.channels() == 3)
			{
                cv::UMat newImg;
				cv::cvtColor(image, newImg, CV_BGR2GRAY);
				return newImg;
			}
		}
		return image;
	};

	switch (m_algType)
	{
	case ALG_VIBE:
        m_modelVibe->update(GetImg().getMat(cv::ACCESS_READ));
        foreground = m_modelVibe->getMask().getUMat(cv::ACCESS_READ);
		break;

	case ALG_MOG:
	case ALG_GMG:
    case ALG_CNT:
#ifdef USE_OCV_BGFG
		m_modelOCV->apply(GetImg(), foreground);
		break;
#else
        std::cerr << "OpenCV bgfg algorithms are not implemented!" << std::endl;
        break;
#endif

    case ALG_SuBSENSE:
    case ALG_LOBSTER:
        if (foreground.size() != image.size())
        {
            m_modelSuBSENSE->initialize(GetImg().getMat(cv::ACCESS_READ), cv::Mat());
            foreground.create(image.size(), CV_8UC1);
        }
        else
        {
            m_modelSuBSENSE->apply(GetImg(), foreground);
        }
        break;

    case ALG_MOG2:
        m_modelOCV->apply(GetImg(), foreground);
	cv::threshold(foreground, foreground, 200, 255, cv::THRESH_BINARY);
        break;

    default:
        m_modelVibe->update(GetImg().getMat(cv::ACCESS_READ));
        foreground = m_modelVibe->getMask().getUMat(cv::ACCESS_READ);
        break;
	}

    //cv::imshow("before", foreground);

    cv::medianBlur(foreground, foreground, 3);

	cv::Mat dilateElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
	cv::dilate(foreground, foreground, dilateElement, cv::Point(-1, -1), 2);

    //cv::imshow("after", foreground);
}
