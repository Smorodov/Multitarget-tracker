#pragma once
#include "BackgroundSubtract.h"
#include <iostream>
#include <vector>
#include "defines.h"

class CDetector
{
private:
	void DetectContour(cv::Mat& img, std::vector<cv::Rect>& Rects, std::vector<Point_t>& centers);
	BackgroundSubtract* bs;
	std::vector<cv::Rect> rects;
	std::vector<Point_t> centers;
	cv::Mat fg;
public:
	CDetector(cv::Mat& gray);
	const std::vector<Point_t>& Detect(cv::Mat& gray);
	~CDetector(void);
};

