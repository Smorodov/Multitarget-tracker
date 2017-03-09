#ifndef _BACKGROUND_SUBTRACT_H_
#define _BACKGROUND_SUBTRACT_H_

#include "defines.h"
#include "vibe.hpp"

class BackgroundSubtract
{
public:
	BackgroundSubtract(int channels = 1, int samples = 20, int pixel_neighbor = 1, int distance_threshold = 20, int matching_threshold = 3, int update_factor = 16);
	~BackgroundSubtract();
	void subtract(const cv::Mat& image, cv::Mat& foreground);

private:
	std::unique_ptr<vibe::VIBE> m_model;
};

#endif
