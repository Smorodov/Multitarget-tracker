#include "hann.h"

cv::Mat hann1D(int len) {
	cv::Mat hann1t = cv::Mat::zeros(1, len, CV_32F);
	float* ptr = (float*)hann1t.data;
	for (int i = 0; i < len; i++)
		ptr[i] = static_cast<float>(0.5 * (1 - std::cos(CV_2PI * i / (len - 1))));

	return hann1t;
}
cv::Mat hann2D(const cv::Size& sz) {
	int w = sz.width, h = sz.height;
	cv::Mat hann_w = hann1D(w);
	cv::Mat hann_h = hann1D(h);

	cv::transpose(hann_h, hann_h);
	cv::Mat hann_hw = hann_h * hann_w;

	return hann_hw.reshape(1, 1);
}

cv::Mat hann3D(const cv::Size& sz, int chns) {
	int col = sz.width*sz.height;
	cv::Mat hanns(chns, col, CV_32F);
	cv::Mat hann_hw = hann2D(sz);

	for (int i = 0; i < chns; ++i) {
		hann_hw.copyTo(hanns(cv::Rect(0, i, col, 1)));
	}
	return hanns;
}