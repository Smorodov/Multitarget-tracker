#pragma once
#include <opencv2/opencv.hpp>
#include "fft_functions.h"
#include "correlation.h"
#include "fhog.hpp"
#include "hann.h"

#include "../VOTTracker.hpp"

class LDESTracker final : public VOTTracker
{
public:
	LDESTracker();
	~LDESTracker();

	void Initialize(const cv::Mat &im, cv::Rect region);
	cv::RotatedRect Update(const cv::Mat &im, float& confidence);
	void Train(const cv::Mat &/*im*/, bool /*first*/)
	{
	}

protected:
	float interp_n;
	float interp_factor; // linear interpolation factor for adaptation
	float sigma; // gaussian kernel bandwidth
	float lambda; // regularization
	int cell_size; // HOG cell size
	int cell_sizeQ; // cell size^2, to avoid repeated operations
	int cell_size_scale;
	float padding; // extra area surrounding the target
	float scale_padding;
	float output_sigma_factor; // bandwidth of gaussian target
	int template_size; // template size
	int scale_template_size;

	float scale_step; // scale step for multi-scale estimation
	float scale_weight;  // to downweight detection scores of other scales for added stability

	float train_interp_factor;
	float interp_factor_scale;

	float cscore;
	float sscore;

	cv::Size target_sz;
	cv::Size target_sz0;
	int window_sz;
	int window_sz0;

	int scale_sz;
	int scale_sz0;
	float scale_base;

	cv::Mat hann;

	cv::Mat patch;
	cv::Mat patchL;

	cv::Point2i cur_pos;
	cv::Rect cur_roi;
	std::vector<cv::Point2i> rotated_roi;

	int im_width;
	int im_height;

	const float min_area = 100 * 100;
	const float max_area = 350 * 350;

	cv::Rect cur_position;

	float cur_rot_degree;
	float cur_scale;
	float _scale;
	float _scale2;
	float delta_rot;
	float delta_scale;
	float mag;
	float conf;

	cv::Mat resmap_location;
	float pv_location;
	cv::Point2f peak_loc;


	cv::Rect testKCFTracker(const cv::Mat& image, cv::Rect& rect, bool init = false);
	cv::Mat getFeatures(const cv::Mat & patch, cv::Mat& han, int* sizes, bool inithann = false);
	cv::Mat getPixFeatures(const cv::Mat& patch, int* size);
	float subPixelPeak(float left, float center, float right);
	void weightedPeak(cv::Mat& resmap, cv::Point2f& peak, int pad=2);
	float calcPSR();
	void updateModel(const cv::Mat& image, int polish);	//MATLAB code

	void estimateLocation(cv::Mat& z, cv::Mat x);
	void estimateScale(cv::Mat& z, cv::Mat& x);	

	void trainLocation(cv::Mat& x, float train_interp_factor_);
	void trainScale(cv::Mat& x, float train_interp_factor_);

	void createGaussianPeak(int sizey, int sizex);

	void getTemplates(const cv::Mat& image);

	void getSubWindow(const cv::Mat& image, const char* type="loc");

	cv::Mat padImage(const cv::Mat& image, int& x1, int& y1, int& x2, int& y2);
	cv::Mat cropImage(const cv::Mat& image, const cv::Point2i& pos, int sz);
	cv::Mat cropImageAffine(const cv::Mat& image, const cv::Point2i& pos, int win_sz, float rot);
	

	cv::Mat hogFeatures;
	cv::Mat _alphaf;
	cv::Mat _y;
	cv::Mat _yf;	//alphaf on f domain
	cv::Mat _z;	//template on time domain
	cv::Mat _z_srch;
	cv::Mat modelPatch;
	cv::Mat _num;
	cv::Mat _den;
	cv::Mat _labCentroids;
	cv::Rect _roi;

private:
	int size_patch[3];
	int size_scale[3];
	int size_search[3];
	int _gaussian_size;
	bool _hogfeatures;
	bool _labfeatures;
	bool _rotation;
	bool _scale_hog;
};
