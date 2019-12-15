#include "ldes_tracker.h"

///
LDESTracker::LDESTracker()
{
	lambda = 0.0001;
	padding = 2.5;
	scale_padding = 2.5;
	output_sigma_factor = 0.125;
	_hogfeatures = true;
	_rotation = true;
	_scale_hog = true;
	interp_factor = 0.012;
	sigma = 0.6;
	cell_size = 4;
	template_size = 96;
	scale_template_size = 120;
}

///
LDESTracker::~LDESTracker()
{
}

///
void LDESTracker::Initialize(const cv::Mat &im, cv::Rect region)
{
	cell_size = 4;
	cell_size_scale = _scale_hog ? 4 : 1;
	target_sz = region.size();
	cur_rot_degree = 0.;
	template_size = 96;
	scale_sz0 = 120;

	assert(region.width >= 0 && region.height >= 0);

	cur_pos.x = region.x + region.width / 2;
	cur_pos.y = region.y + region.height / 2;
	cur_roi = region;
	cur_scale = 1.0;

	//for cropping, then resize to window_sz0
	//int padded_sz = static_cast<int>(sqrt(target_sz.area())*padding);	
	float padded_sz = MAX(target_sz.width, target_sz.height)*padding;
	_scale = padded_sz / template_size;
	window_sz0 = cvRound(padded_sz / _scale);
	window_sz0 = window_sz0 / (2 * cell_size)*(2 * cell_size) + 2 * cell_size;
	
	scale_sz0 = scale_template_size / (2 * cell_size_scale)*(2 * cell_size_scale) + 2 * cell_size_scale;
	_scale2 = padded_sz / scale_sz0;

	mag = 30;
	train_interp_factor = 0.012;
	interp_factor_scale = 0.015;

	getTemplates(im);
}

///
void LDESTracker::getSubWindow(const cv::Mat& image, const char* type)
{
	if (strcmp(type, "loc") == 0) {
		if (_rotation) {
			patch = cropImageAffine(image, cur_pos, cvRound(window_sz0*_scale), cur_rot_degree);
		}
		else {
			int win = (int)(window_sz0*_scale);
			patch = cropImage(image, cur_pos, win);
		}
		//cv::imshow("patch", patch);
		cv::resize(patch, patch, cv::Size(window_sz0, window_sz0), cv::INTER_LINEAR);
	}
	else if (strcmp(type, "scale") == 0) {
		if (_rotation) {
			patchL = cropImageAffine(image, cur_pos, cvRound(scale_sz0*_scale2), cur_rot_degree);
		}
		else {
			patchL = cropImage(image, cur_pos, cvRound(scale_sz0*_scale2));
		}
		//cv::imshow("rot_patch", patchL);
		cv::resize(patchL, patchL, cv::Size(scale_sz0, scale_sz0), cv::INTER_LINEAR);
		cv::logPolar(patchL, patchL, cv::Point2f(0.5f*patchL.cols, 0.5f*patchL.rows), mag, cv::INTER_LINEAR);
	}
	else
		assert(0);
}

///
void LDESTracker::getTemplates(const cv::Mat& image)
{
	getSubWindow(image, "loc");
	getSubWindow(image, "scale");

	cv::Mat empty_;
	cv::Mat x = getFeatures(patch, hann, size_patch, true);
	cv::Mat xl;
	if (!_scale_hog)
		xl = getPixFeatures(patchL, size_scale);
	else
		xl = getFeatures(patchL, empty_, size_scale); 

	createGaussianPeak(size_patch[0], size_patch[1]);

	_alphaf = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
	_z = cv::Mat(size_patch[2], size_patch[0] * size_patch[1], CV_32F, float(0));
	modelPatch=cv::Mat(size_scale[2], size_scale[0]*size_scale[1], CV_32F, float(0));
	
	trainLocation(x, 1.0);
	trainScale(xl, 1.0);
}

///
void LDESTracker::trainLocation(cv::Mat& x, float train_interp_factor_)
{
	cv::Mat k = gaussianCorrelation(x, x, size_patch[0], size_patch[1], size_patch[2], sigma);
	cv::Mat alphaf = complexDivision(_yf, (k + lambda));

	_z = (1 - train_interp_factor_) * _z + (train_interp_factor_)* x;
	_alphaf = (1 - train_interp_factor_) * _alphaf + (train_interp_factor_)* alphaf;
}

///
void LDESTracker::trainScale(cv::Mat& x, float interp_factor_)
{
	modelPatch = (1 - interp_factor_)*modelPatch + interp_factor_ * x;
}

///
cv::Mat LDESTracker::padImage(const cv::Mat& image, int& x1, int& y1, int& x2, int& y2)
{
	cv::Mat padded;

	int im_h = image.rows, im_w = image.cols;
	int left, top, right, bottom;

	left = MAX(0, -x1);
	right = MAX(0, x2 - (im_w - 1));
	top = MAX(0, -y1);
	bottom = MAX(0, y2 - (im_h - 1));

	x1 = left;
	x2 = right;
	y1 = top;
	y2 = bottom;

	cv::copyMakeBorder(image, padded, top, bottom, left, right, cv::BORDER_REPLICATE);

	return padded;
}

///
cv::Mat LDESTracker::cropImage(const cv::Mat& image, const cv::Point2i& pos, int sz)
{
	int x1 = pos.x - sz / 2;
	int y1 = pos.y - sz / 2;
	int x2 = x1 + sz - 1;
	int y2 = y1 + sz - 1;
	if (x1 < 0 && x2 < 0) {
		x2 -= x1;
		x1 = 0;
	}
	if (y1 < 0 && y2 < 0) {
		y2 -= y1;
	}

    int tx1 = MAX(0, x1), ty1 = MAX(0, y1), tx2 = MIN(x2, image.cols - 1), ty2 = MIN(y2, image.rows - 1);
	x1 -= tx1;
	x2 -= tx1;
	y1 -= ty1;
	y2 -= ty1;
	cv::Rect rec(tx1, ty1, tx2 - tx1 + 1, ty2 - ty1 + 1);
	cv::Mat patchl;
	image(rec).copyTo(patchl);
	patchl = padImage(patchl, x1, y1, x2, y2);
	return patchl;
}

///
cv::Mat LDESTracker::cropImageAffine(const cv::Mat& image, const cv::Point2i& pos, int win_sz, float rot)
{
	//cv::Mat rot_matrix = cv::getRotationMatrix2D(pos, -rot, scale);
	cv::Mat rot_matrix = cv::getRotationMatrix2D(pos, -rot, 1);
	rot_matrix.convertTo(rot_matrix, CV_32F);
	cv::transpose(rot_matrix, rot_matrix);

	float corners_ptr[12] = {
		pos.x - win_sz * 0.5f, pos.y - win_sz * 0.5f, 1.0f,\
		pos.x - win_sz * 0.5f, pos.y + win_sz * 0.5f, 1.0f,\
		pos.x + win_sz * 0.5f, pos.y + win_sz * 0.5f, 1.0f,\
		pos.x + win_sz * 0.5f, pos.y - win_sz * 0.5f, 1.0f
	};
	cv::Mat corners(4, 3, CV_32F, corners_ptr);

	cv::Mat wcorners = corners * rot_matrix;

	double x1_, y1_, x2_, y2_;
	cv::minMaxLoc(wcorners.col(0).clone(), &x1_, &x2_, NULL, NULL);
	cv::minMaxLoc(wcorners.col(1).clone(), &y1_, &y2_, NULL, NULL);
	int x1 = cvRound(x1_), y1 = cvRound(y1_), x2 = cvRound(x2_), y2 = cvRound(y2_);
	int tx1 = MAX(0, x1), tx2 = MIN(image.cols - 1, x2), ty1 = MAX(0, y1), ty2 = MIN(image.rows-1, y2);
	int ix1 = x1-tx1, ix2 = x2-tx1, iy1 = y1-ty1, iy2 = y2-ty1;

	cv::Mat patchl;
	cv::Rect rec(tx1, ty1, tx2 - tx1 + 1, ty2 - ty1 + 1);
	
	image(rec).copyTo(patchl);
	cv::Mat padded = padImage(patchl, ix1, iy1, ix2, iy2);

	cv::Point2i p(pos.x - tx1 + ix1, pos.y - ty1 + iy1);

	rot_matrix = cv::getRotationMatrix2D(p, -rot, 1);

	rot_matrix.convertTo(rot_matrix, CV_32F);
	rot_matrix.at<float>(0, 2) += win_sz * 0.5f - p.x;
	rot_matrix.at<float>(1, 2) += win_sz * 0.5f - p.y;

	cv::warpAffine(padded, patchl, rot_matrix, cv::Size(win_sz, win_sz));
	return patchl;
}

///
void LDESTracker::estimateLocation(cv::Mat& z, cv::Mat x)
{
	cv::Mat kf = gaussianCorrelation(x, z, size_patch[0], size_patch[1], size_patch[2], sigma);
	cv::Mat res = fftd(complexMultiplication(_alphaf, kf), true);

    res.copyTo(resmap_location);
	
	cv::Point2i pi;
	double pv;
	cv::minMaxLoc(res, NULL, &pv, NULL, &pi);
	float peak_value = (float)pv;
	//cscore=calcPSR();
    peak_loc = cv::Point2f((float)pi.x, (float)pi.y);

    if (pi.x > 0 && pi.x < res.cols - 1)
    {
        peak_loc.x += subPixelPeak(res.at<float>(pi.y, pi.x - 1), peak_value, res.at<float>(pi.y, pi.x + 1));
	}
    if (pi.y > 0 && pi.y < res.rows - 1)
    {
        peak_loc.y += subPixelPeak(res.at<float>(pi.y - 1, pi.x), peak_value, res.at<float>(pi.y + 1, pi.x));
	}

	//weightedPeak(res, p, 2);
    float px = peak_loc.x - res.cols / 2;
    float py = peak_loc.y - res.rows / 2;
	cur_pos.x = MIN(cvRound(cur_pos.x + px * cell_size*_scale), im_width - 1);
	cur_pos.y = MIN(cvRound(cur_pos.y + py * cell_size*_scale), im_height - 1);	
}

///
void LDESTracker::estimateScale(cv::Mat& z, cv::Mat& x)
{
	cv::Mat rf = phaseCorrelation(x, z, size_scale[0], size_scale[1], size_scale[2]);
	cv::Mat res = fftd(rf, true);
	rearrange(res);

	cv::Rect center(5, 5, size_scale[1] - 10, size_scale[0] - 10);
	
	res = res(center).clone();

	cv::Point2i pi;
	double pv_;
	cv::minMaxLoc(res, NULL, &pv_, NULL, &pi);
	float pv = static_cast<float>(pv_);

	cv::Point2f pf(pi.x + 5.f, pi.y + 5.f);
	//weightedPeak(res, pf, 1);
	if (pi.x > 0 && pi.x < res.cols - 1) {
		pf.x += subPixelPeak(res.at<float>(pi.y, pi.x - 1), pv, res.at<float>(pi.y, pi.x + 1));
	}

	if (pi.y > 0 && pi.y < res.rows - 1) {
		pf.y += subPixelPeak(res.at<float>(pi.y - 1, pi.x), pv, res.at<float>(pi.y + 1, pi.x));
	}

	float px = pf.x, py = pf.y;
	
	px -= size_scale[1] * 0.5f;
	py -= size_scale[0] * 0.5f;
	//px *= cell_size_scale;
	//py *= cell_size_scale;

	float rot = -(py) * 180.0f / (size_scale[0] * 0.5f);
	float scale =  exp((px) / mag);

	sscore = static_cast<float>(pv);

	delta_rot = rot;
	delta_scale = scale;
	if (abs(delta_rot) > 5)
		delta_rot = 0;
	delta_scale = MIN(MAX(delta_scale, 0.6f), 1.4f);
}

/*
*Update BGD(Block Gradient Descend, original AAAI Paper MATLAB Code)
*If BGD, more precise but slower
*/
cv::RotatedRect LDESTracker::Update(const cv::Mat &im, float& confidence)
{
	float tmp_scale = 1.0, tmp_scale2 = 1.0;
	float mscore = 0.0;

	updateModel(im, 0);
	tmp_scale = _scale;
	tmp_scale2 = _scale2;
	mscore = calcPSR();
    for (int i = 1; i <= 5; ++i) {	//BGD iterations, <=5, you can have a test
		if (floor(tmp_scale*window_sz0) < 5)
			tmp_scale = 1.0;
		if (floor(tmp_scale2*scale_sz0) < 5)
			tmp_scale2 = 1.0;
		_scale = tmp_scale;
		_scale2 = tmp_scale2;
		updateModel(im, i);
		float psr = calcPSR();

		if (psr > mscore) {
			mscore = psr;
			tmp_scale = _scale;
			tmp_scale2 = _scale2;
		}
		else {
			_scale = tmp_scale;
			_scale2 = tmp_scale2;
			break;
		}
	}
	conf = mscore;
	confidence = conf;
	return cv::RotatedRect(cv::Point2f(cur_roi.x + 0.5f * cur_roi.width, cur_roi.y + 0.5f * cur_roi.height),
		cv::Size2f(cur_roi.width, cur_roi.height), cur_rot_degree);
}

///
void LDESTracker::updateModel(const cv::Mat& image, int /*polish*/)
{
	cv::Mat _han, empty_;
	im_height = image.rows;
	im_width = image.cols;
	//if(polish>=0){
	getSubWindow(image, "loc");

	cv::Mat x = getFeatures(patch, hann, size_patch, false);
	estimateLocation(_z, x);

	getSubWindow(image, "scale");
	cv::Mat xl;
	if(!_scale_hog)
		xl= getPixFeatures(patchL, size_scale);
	else
		xl = getFeatures(patchL, empty_, size_scale);

	estimateScale(modelPatch, xl);
		
	if (_rotation) {
		cur_rot_degree += delta_rot;
	}
	cur_scale *= delta_scale;
	_scale *= delta_scale;
	_scale2 *= delta_scale;
	//cout << "Cur scale: " << cur_scale << " cur rotation:  " << cur_rot_degree << endl;
	cur_roi.width = cvRound(target_sz.width*cur_scale);
	cur_roi.height = cvRound(target_sz.height*cur_scale);
	//cur_roi.width = round(cur_roi.width*delta_scale);
	//cur_roi.height = round(cur_roi.height*delta_scale);
	cur_roi.x = cvRound(cur_pos.x - cur_roi.width / 2);
	cur_roi.y = cvRound(cur_pos.y - cur_roi.height / 2);

	getSubWindow(image, "loc");
	getSubWindow(image, "scale");
		
	x = getFeatures(patch, hann, size_patch, false);
		
	if (!_scale_hog)
		xl = getPixFeatures(patchL, size_scale);
	else
		xl = getFeatures(patchL, empty_, size_scale);
		
	trainLocation(x, train_interp_factor);
	trainScale(xl, interp_factor_scale);
	//}
}

///
void LDESTracker::createGaussianPeak(int sizey, int sizex)
{
	cv::Mat_<float> res(sizey, sizex);

	int syh = (sizey) / 2; 
	int sxh = (sizex) / 2;

	//float output_sigma = std::sqrt((float)sizex * sizey) / cell_size * output_sigma_factor;
	float output_sigma = std::sqrt((float)sizex * sizey) / padding * output_sigma_factor;
	float mult = -0.5f / (output_sigma * output_sigma);

	for (int i = 0; i < sizey; i++)
		for (int j = 0; j < sizex; j++)
		{
			int ih = i - syh;
			int jh = j - sxh;
			res(i, j) = std::exp(mult * (float)(ih * ih + jh * jh));
		}

	res.copyTo(_y);
	_yf = fftd(_y);
}

///
cv::Mat LDESTracker::getFeatures(const cv::Mat & patchl, cv::Mat& han, int* sizes, bool inithann)
{
	cv::Mat FeaturesMap;
	// HOG features
	CvLSVMFeatureMapCaskade *map;
	getFeatureMaps(patchl, cell_size, &map);
	normalizeAndTruncate(map, 0.2f);
	PCAFeatureMaps(map);
	sizes[0] = map->sizeY;
	sizes[1] = map->sizeX;
	sizes[2] = map->numFeatures;

	FeaturesMap = cv::Mat(cv::Size(map->numFeatures, map->sizeX*map->sizeY), CV_32F, map->map);  // Procedure do deal with cv::Mat multichannel bug
	FeaturesMap = FeaturesMap.t();
	freeFeatureMapObject(&map);

	if (inithann) {		
		cv::Size hannSize(sizes[1], sizes[0]);
		cv::Mat hannsMat = hann3D(hannSize, sizes[2]);
		hannsMat.copyTo(han);
		FeaturesMap = han.mul(FeaturesMap);
	}
	else if (!han.empty())
		FeaturesMap = han.mul(FeaturesMap);
	return FeaturesMap;
}

///
cv::Mat LDESTracker::getPixFeatures(const cv::Mat& patchl, int* size)
{
	int h = patchl.rows, w = patchl.cols;
	cv::Mat features(patchl.channels(), w*h, CV_32F);
	std::vector<cv::Mat > planes(3);
	cv::split(patchl, planes);
	planes[0].reshape(1, 1).copyTo(features.row(0));
	planes[1].reshape(1, 1).copyTo(features.row(1));
	planes[2].reshape(1, 1).copyTo(features.row(2));
	size[0] = h;
	size[1] = w;
	size[2] = patchl.channels();
	return features;
}

///
float LDESTracker::subPixelPeak(float left, float center, float right)
{
	float divisor = 2 * center - right - left;
	if (divisor == 0)
		return 0;
	return 0.5f * (right - left) / divisor;
}

///
void LDESTracker::weightedPeak(cv::Mat& resmap, cv::Point2f& peak, int pad)
{
	cv::copyMakeBorder(resmap, resmap, pad, pad, pad, pad, cv::BORDER_REFLECT);
	cv::Rect slobe(cvRound(peak.x), cvRound(peak.y), 2*pad+1, 2*pad+1);
	cv::Mat patchl;
	resmap(slobe).copyTo(patchl);

	int sz = 2 * pad + 1;
	int N = sz * sz;
	
	std::vector<int> xoffset, yoffset;
	for (int i = 0; i < N; ++i) {
		xoffset.push_back((i%sz) - pad);
		yoffset.push_back(i / sz - pad);
	}
	float* data = (float*)patchl.data;
	float xsum = 0, ysum = 0, sum = 0;
	for (int i = 0; i < N; ++i) {
		sum += data[i];
		xsum += data[i] * (peak.x + xoffset[i]);
		ysum += data[i] * (peak.y + yoffset[i]);
	}
	peak.x = xsum / sum;
	peak.y = ysum / sum;
}

///
float LDESTracker::calcPSR()
{
	int px = cvRound(peak_loc.x);
	int py = cvRound(peak_loc.y);

	cv::Mat res = resmap_location.clone();
	float peak = pv_location;

	const float rate = 0.6f / (1 + padding);
	int range = (int)(sqrt(res.cols*res.rows)*rate);

	cv::Rect peak_rect = cv::Rect(px - range / 2, py - range / 2, range, range);

	cv::Mat peakBuff = cv::Mat::zeros(range, range, CV_32FC1);
	peakBuff.copyTo(res(peak_rect));

	float numel = static_cast<float>(res.cols*res.rows);
	float mu = static_cast<float>(cv::sum(res)[0]);// / ();
	mu /= numel;
	cv::Mat subs;
	cv::subtract(res, mu, subs);
	cv::multiply(subs, subs, subs);
	float var = static_cast<float>(cv::sum(subs)[0]);
	var /= (numel - 1);	//sample variance
	float stdev = sqrt(var);

	float psr = (peak - mu) / stdev;

	cscore = psr;
	psr = 0.1f*cscore + 0.9f*sscore;

	return psr;
}

///
cv::Rect LDESTracker::testKCFTracker(const cv::Mat& image, cv::Rect& rect, bool init)
{
	im_width = image.cols;
	im_height = image.rows;
	if (init) {
		_rotation = false;
		this->Initialize(image, rect);
		return rect;
	}
	else {
		getSubWindow(image, "loc");
		cv::Rect rec(cur_pos.x - window_sz / 2, cur_pos.y - window_sz / 2, window_sz, window_sz);
		
		cv::Mat x = getFeatures(patch, hann, size_patch, false);
		estimateLocation(_z, x);
		x = getFeatures(patch, hann, size_patch, false);
		
		cur_roi.width = cvRound(target_sz.width*cur_scale);
		cur_roi.height = cvRound(target_sz.height*cur_scale);
		cur_roi.x = cvRound(cur_pos.x - cur_roi.width / 2);
		cur_roi.y = cvRound(cur_pos.y - cur_roi.height / 2);

		trainLocation(x, train_interp_factor);
		return cur_roi;
	}
}
