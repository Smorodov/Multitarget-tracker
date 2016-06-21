#pragma once
#include "opencv2/opencv.hpp"
#include "BackgroundSubtract.h"
#include <iostream>
#include <vector>

class CDetector
{
private:
	void DetectContour(cv::Mat& img, std::vector<cv::Rect>& Rects,std::vector<cv::Point2d>& centers);
	BackgroundSubtract* bs;
	std::vector<cv::Rect> rects;
	std::vector<cv::Point2d> centers;
	cv::Mat fg;
public:
	CDetector(cv::Mat& gray);
	std::vector<cv::Point2d> Detect(cv::Mat& gray);
	~CDetector(void);
};

