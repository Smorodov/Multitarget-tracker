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

	case ALG_MOG:
		m_modelOCV = cv::bgsegm::createBackgroundSubtractorMOG(100, 3, 0.7, 0);
		break;

	case ALG_GMG:
		m_modelOCV = cv::bgsegm::createBackgroundSubtractorGMG(50, 0.7);
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
		m_modelOCV->apply(GetImg(), foreground);
		break;
	}

    //cv::imshow("before", foreground);

	cv::medianBlur(foreground, foreground, 3);

	cv::Mat dilateElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
	cv::dilate(foreground, foreground, dilateElement, cv::Point(-1, -1), 1);

    cv::imshow("after", foreground);
}
