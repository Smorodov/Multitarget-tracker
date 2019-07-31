#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat fftd(const cv::Mat& img, bool reverse = false);

cv::Mat real(const cv::Mat& img);

cv::Mat imag(const cv::Mat& img);

cv::Mat magnitude(const cv::Mat& img);

cv::Mat complexMultiplication(const cv::Mat& a, const cv::Mat& b);

cv::Mat complexDivision(const cv::Mat& a, const cv::Mat& b);

void rearrange(cv::Mat& img);