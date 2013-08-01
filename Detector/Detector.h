#pragma once
#include "opencv2/opencv.hpp"
#include "BackgroundSubtract.h"
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;
class CDetector
{
private:
	void DetectContour(Mat& img, vector<Rect>& Rects,vector<Point2d>& centers);
	BackgroundSubtract* bs;
	vector<Rect> rects;
	vector<Point2d> centers;
	Mat fg;
public:
	CDetector(Mat& gray);
	vector<Point2d> Detect(Mat& gray);
	~CDetector(void);
};

