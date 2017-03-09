#include "BackgroundSubtract.h"

BackgroundSubtract::BackgroundSubtract(
	int channels,
	int samples,
	int pixel_neighbor,
	int distance_threshold,
	int matching_threshold,
	int update_factor
	)
{
	m_model = std::make_unique<vibe::VIBE>(channels, samples, pixel_neighbor, distance_threshold, matching_threshold, update_factor);
}

BackgroundSubtract::~BackgroundSubtract()
{
}

void BackgroundSubtract::subtract(const cv::Mat& image, cv::Mat& foreground)
{
	cv::Mat erodeElement = cv::getStructuringElement(0, cv::Size(5, 5), cv::Point(-1, -1));
	cv::Mat dilateElement = cv::getStructuringElement(0, cv::Size(3, 3), cv::Point(-1, -1));

	if (image.channels() != m_model->GetChannels())
	{
		if (image.channels() == 1)
		{
			cv::Mat newImg;
			cv::cvtColor(image, newImg, CV_GRAY2BGR);
			m_model->update(newImg);
		}
		else if (image.channels() == 3)
		{
			cv::Mat newImg;
			cv::cvtColor(image, newImg, CV_BGR2GRAY);
			m_model->update(newImg);
		}
	}
	else
	{
		m_model->update(image);
	}

	foreground = m_model->getMask();

	cv::imshow("before", foreground);

	cv::medianBlur(foreground, foreground, 3);
	cv::dilate(foreground, foreground, dilateElement, cv::Point(-1, -1), 1);

	cv::imshow("after", foreground);
}
