#pragma once
#include <opencv2/opencv.hpp>

cv::Mat hann1D(int len);
cv::Mat hann2D(const cv::Size& sz);
cv::Mat hann3D(const cv::Size& sz, int chns);