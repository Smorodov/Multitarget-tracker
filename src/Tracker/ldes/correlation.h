#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat gaussianCorrelation(cv::Mat& x1, cv::Mat& x2, int h, int w, int channel, float sigma);

cv::Mat linearCorrelation(cv::Mat& x1, cv::Mat& x2, int h, int w, int channel);

cv::Mat polynomialCorrelation(cv::Mat& x1, cv::Mat& x2, int h, int w, int channel);

cv::Mat phaseCorrelation(cv::Mat& x1, cv::Mat& x2, int h, int w, int channel);

