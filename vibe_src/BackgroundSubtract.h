#ifndef _BACKGROUND_SUBTRACT_H_
#define _BACKGROUND_SUBTRACT_H_

#include "OpencvInclude.h"
#include "VIBE.h"

class BackgroundSubtract
{
public:
	BackgroundSubtract();
	~BackgroundSubtract();
	void init(cv::Mat &image);
	void subtract(const cv::Mat &image, cv::Mat &foreground);
private:
	vibeModel_t *model;

};

#endif