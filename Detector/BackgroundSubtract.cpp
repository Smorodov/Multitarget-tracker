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
	switch (m_algType)
	{
	case ALG_VIBE:
		m_modelVibe = std::make_unique<vibe::VIBE>(m_channels, samples, pixel_neighbor, distance_threshold, matching_threshold, update_factor);
		break;

#if USE_OCV_BGFG
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
#endif

#else
    case ALG_MOG:
    case ALG_GMG:
    case ALG_CNT:
        std::cerr << "OpenCV bgfg algorithms are not implemented! Used Vibe by default." << std::endl;
#endif

    default:
        m_modelVibe = std::make_unique<vibe::VIBE>(m_channels, samples, pixel_neighbor, distance_threshold, matching_threshold, update_factor);
        break;
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
void BackgroundSubtract::subtract(const cv::Mat& image, cv::Mat& foreground)
{
	auto GetImg = [&]() -> cv::Mat
	{
		if (image.channels() != m_channels)
		{
			if (image.channels() == 1)
			{
				cv::Mat newImg;
				cv::cvtColor(image, newImg, CV_GRAY2BGR);
				return newImg;
			}
			else if (image.channels() == 3)
			{
				cv::Mat newImg;
				cv::cvtColor(image, newImg, CV_BGR2GRAY);
				return newImg;
			}
		}
		return image;
	};

	switch (m_algType)
	{
	case ALG_VIBE:
		m_modelVibe->update(GetImg());
		foreground = m_modelVibe->getMask();
		break;

	case ALG_MOG:
	case ALG_GMG:
    case ALG_CNT:
#if USE_OCV_BGFG
		m_modelOCV->apply(GetImg(), foreground);
		break;
#else
        std::cerr << "OpenCV bgfg algorithms are not implemented!" << std::endl;
#endif

    default:
        m_modelVibe->update(GetImg());
        foreground = m_modelVibe->getMask();
        break;
	}

    //cv::imshow("before", foreground);

    cv::medianBlur(foreground, foreground, 3);

	cv::Mat dilateElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
	cv::dilate(foreground, foreground, dilateElement, cv::Point(-1, -1), 2);

    //cv::imshow("after", foreground);
}
