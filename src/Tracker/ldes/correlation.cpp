#include "correlation.h"
#include "fft_functions.h"

cv::Mat gaussianCorrelation(cv::Mat& x1, cv::Mat& x2, int h, int w, int channel, float sigma) {
	cv::Mat xy = cv::Mat(cv::Size(w, h), CV_32F, cv::Scalar(0));
	cv::Mat xy_temp;
	cv::Mat x;
	cv::Mat y;
	double xx=0, yy=0;
	for (int i = 0; i < channel; i++) {
		x = x1.row(i).reshape(1, h);   // Procedure do deal with cv::Mat multichannel bug
		y = x2.row(i).reshape(1, h);
		xx +=cv::norm(x)*cv::norm(x) ;
		yy += cv::norm(y)*cv::norm(y) ;
		cv::mulSpectrums(fftd(x), fftd(y), xy_temp, 0, true);
		xy_temp = fftd(xy_temp, true);
		rearrange(xy_temp);	//rearange or not? Doesn't matter
		xy_temp.convertTo(xy_temp, CV_32F);
		xy += xy_temp;
	}
	cv::Mat d;
	cv::max(((xx + yy) - 2. * xy) / (w * h * channel), 0, d);

	cv::Mat k;
	cv::exp((-d / (sigma * sigma)), k);
	return fftd(k);

	//cv::Mat c = cv::Mat(cv::Size(w,h), CV_32F, cv::Scalar(0));
	//cv::Mat caux;
	//cv::Mat x1aux;
	//cv::Mat x2aux;
	//for (int i = 0; i < channel; i++) {
	//	x1aux = x1.row(i);   // Procedure do deal with cv::Mat multichannel bug
	//	x1aux = x1aux.reshape(1, h);
	//	x2aux = x2.row(i).reshape(1, h);
	//	cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
	//	caux = fftd(caux, true);
	//	rearrange(caux);
	//	caux.convertTo(caux, CV_32F);
	//	c = c + real(caux);
	//}
	//
	//cv::Mat d;
	//cv::max(((cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0]) - 2. * c) / (h*w*channel), 0, d);

	//cv::Mat k;
	//cv::exp((-d / (sigma * sigma)), k);
	//return fftd(k);
}

cv::Mat linearCorrelation(cv::Mat& x1, cv::Mat& x2, int h, int w, int channel) {
	cv::Mat xy = cv::Mat(cv::Size(w, h), CV_32FC2, cv::Scalar(0));
	cv::Mat xy_temp;
	cv::Mat x;
	cv::Mat y;
	for (int i = 0; i < channel; i++) {
		x = x1.row(i).reshape(1, h);;
		y = x2.row(i).reshape(1, h);
		cv::mulSpectrums(fftd(x), fftd(y), xy_temp, 0, true);
		xy = xy + xy_temp;
	}
	xy.convertTo(xy, CV_32F);
	return xy / (h*w*channel);
}

cv::Mat polynomialCorrelation(cv::Mat& x1, cv::Mat& x2, int h, int w, int channel) {
	cv::Mat xy = cv::Mat(cv::Size(w, h), CV_32F, cv::Scalar(0));
	cv::Mat xy_temp;
	cv::Mat x;
	cv::Mat y;
	for (int i = 0; i < channel; i++) {
		x = x1.row(i).reshape(1, h);;
		y = x2.row(i).reshape(1, h);
		cv::mulSpectrums(fftd(x), fftd(y), xy_temp, 0, true);
		//rearrange(caux);	//rearange or not?
		//caux.convertTo(caux, CV_32F);
		xy_temp = fftd(xy_temp, true);
		xy_temp.convertTo(xy_temp, CV_32F);
		xy = xy + xy_temp;
	}
	cv::Mat k;
	cv::pow(xy / (h*w*channel) + 1, 9, k);	//polynomal
	return fftd(k);
}

cv::Mat phaseCorrelation(cv::Mat& x1, cv::Mat& x2, int h, int w, int channel) {
	cv::Mat xy = cv::Mat(h, w, CV_32FC2, cv::Scalar(0));
	cv::Mat xy_temp;
	cv::Mat x;
	cv::Mat y;
	cv::Mat d;
	cv::Mat d2;
	for (int i = 0; i < channel; i++) {
		x = x1.row(i).reshape(1, h);;
		y = x2.row(i).reshape(1, h);
		cv::mulSpectrums(fftd(x), fftd(y), xy_temp, 0, true);
		cv::mulSpectrums(xy_temp, xy_temp, d, 0, true);
		cv::sqrt(real(d), d);
		d += 2.2204e-16;
		//d = complexDivision(xy_temp, d);
		std::vector<cv::Mat> planes = { d,d };
		cv::merge(planes, d2);
		cv::divide(xy_temp, d2, xy_temp);
		//xy_temp = complexDivision(xy_temp, d2);
		xy_temp.convertTo(xy_temp, CV_32F);
		xy += xy_temp;
	}
	return xy;
}