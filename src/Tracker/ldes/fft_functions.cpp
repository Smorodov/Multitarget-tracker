#include "fft_functions.h"

cv::Mat fftd(const cv::Mat& img, bool reverse) {
	cv::Mat tmp;
	if (img.channels() == 1)
	{
		std::vector<cv::Mat> planes = { cv::Mat_<float>(img), cv::Mat_<float>::zeros(img.size()) };
		//cv::Mat planes[] = {cv::Mat_<double> (img), cv::Mat_<double>::zeros(img.size())};
		cv::merge(planes, tmp);
	}
	else
		tmp = img;
	cv::dft(tmp, tmp, reverse ? (cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT) : cv::DFT_COMPLEX_OUTPUT);

	return tmp;
}

cv::Mat real(const cv::Mat& img) {
	std::vector<cv::Mat> planes;
	cv::split(img, planes);
	return planes[0];
}

cv::Mat imag(const cv::Mat& img) {
	std::vector<cv::Mat> planes;
	cv::split(img, planes);
	return planes[1];
}

cv::Mat magnitude(const cv::Mat& img) {
	cv::Mat res;
	std::vector<cv::Mat> planes;
	cv::split(img, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	if (planes.size() == 1) res = cv::abs(img);
	else if (planes.size() == 2) cv::magnitude(planes[0], planes[1], res); // planes[0] = magnitude
	else assert(0);
	return res;
}

cv::Mat complexMultiplication(const cv::Mat& a, const cv::Mat& b) {
	std::vector<cv::Mat> pa;
	std::vector<cv::Mat> pb;
	cv::split(a, pa);
	cv::split(b, pb);

	std::vector<cv::Mat> pres;
	pres.push_back(pa[0].mul(pb[0]) - pa[1].mul(pb[1]));
	pres.push_back(pa[0].mul(pb[1]) + pa[1].mul(pb[0]));

	cv::Mat res;
	cv::merge(pres, res);

	return res;
}

cv::Mat complexDivision(const cv::Mat& a, const cv::Mat& b) {
	std::vector<cv::Mat> pa;
	std::vector<cv::Mat> pb;
	cv::split(a, pa);
	cv::split(b, pb);

	cv::Mat divisor = 1. / (pb[0].mul(pb[0]) + pb[1].mul(pb[1]));

	std::vector<cv::Mat> pres;

	pres.push_back((pa[0].mul(pb[0]) + pa[1].mul(pb[1])).mul(divisor));
	pres.push_back((pa[1].mul(pb[0]) + pa[0].mul(pb[1])).mul(divisor));

	cv::Mat res;
	cv::merge(pres, res);
	return res;
}

void rearrange(cv::Mat& img) {
	int cx = img.cols / 2;
	int cy = img.rows / 2;

	cv::Mat q0(img, cv::Rect(0, 0, cx, cy)); // Top-Left - Create a ROI per quadrant
	cv::Mat q1(img, cv::Rect(cx, 0, cx, cy)); // Top-Right
	cv::Mat q2(img, cv::Rect(0, cy, cx, cy)); // Bottom-Left
	cv::Mat q3(img, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

	cv::Mat tmp; // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
}