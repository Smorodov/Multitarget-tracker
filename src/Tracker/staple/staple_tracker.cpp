/*
 * cv::Size(width, height)
 * cv::Point(y, x)
 * cv::Mat(height, width, channels, ... )
 * cv::Mat save by row after row
 *   2d: address = j * width + i
 *   3d: address = j * width * channels + i * channels + k
 * ------------------------------------------------------------
 * row == heigh == Point.y
 * col == width == Point.x
 * Mat::at(Point(x, y)) == Mat::at(y,x)
 */

#include "fhog.h"
#include "staple_tracker.hpp"
#include <iomanip>

///
/// \brief STAPLE_TRACKER::STAPLE_TRACKER
///
STAPLE_TRACKER::STAPLE_TRACKER()
{
    m_cfg = default_parameters_staple();
    frameno = 0;
}

///
/// \brief STAPLE_TRACKER::~STAPLE_TRACKER
///
STAPLE_TRACKER::~STAPLE_TRACKER()
{

}

///
/// \brief STAPLE_TRACKER::mexResize
///        mexResize got different results using different OpenCV, it's not trustable
///        I found this bug by running vot2015/tunnel, it happened when frameno+1==22 after frameno+1==21
/// \param im
/// \param output
/// \param newsz
/// \param method
///
void STAPLE_TRACKER::mexResize(const cv::Mat &im, cv::Mat &output, cv::Size newsz, const char* /*method*/)
{
    int interpolation = cv::INTER_LINEAR;

#if 0
    if(!strcmp(method, "antialias")){
        interpolation = cv::INTER_AREA;
    } else if (!strcmp(method, "linear")){
        interpolation = cv::INTER_LINEAR;
    } else if (!strcmp(method, "auto")){
        if(newsz.width > im.cols){ // xxx
            interpolation = cv::INTER_LINEAR;
        }else{
            interpolation = cv::INTER_AREA;
        }
    } else {
        assert(0);
        return;
    }
#endif

    cv::resize(im, output, newsz, 0, 0, interpolation);
}

///
/// \brief STAPLE_TRACKER::default_parameters_staple
/// \return
///
staple_cfg STAPLE_TRACKER::default_parameters_staple()
{
	staple_cfg cfg;
    return cfg;
}

///
/// \brief STAPLE_TRACKER::initializeAllAreas
/// \param im
///
void STAPLE_TRACKER::initializeAllAreas(const cv::Mat &im)
{
    // we want a regular frame surrounding the object
    double avg_dim = (m_cfg.target_sz.width + m_cfg.target_sz.height) / 2.0;

    bg_area.width = cvRound(m_cfg.target_sz.width + avg_dim);
    bg_area.height = cvRound(m_cfg.target_sz.height + avg_dim);

    // pick a "safe" region smaller than bbox to avoid mislabeling
    fg_area.width = cvRound(m_cfg.target_sz.width - avg_dim * m_cfg.inner_padding);
    fg_area.height = cvRound(m_cfg.target_sz.height - avg_dim * m_cfg.inner_padding);

    // saturate to image size
    cv::Size imsize = im.size();

    bg_area.width = std::min(bg_area.width, imsize.width - 1);
    bg_area.height = std::min(bg_area.height, imsize.height - 1);

    // make sure the differences are a multiple of 2 (makes things easier later in color histograms)
    bg_area.width = bg_area.width - (bg_area.width - m_cfg.target_sz.width) % 2;
    bg_area.height = bg_area.height - (bg_area.height - m_cfg.target_sz.height) % 2;

    fg_area.width = fg_area.width + (bg_area.width - fg_area.width) % 2;
    fg_area.height = fg_area.height + (bg_area.height - fg_area.width) % 2;

    //std::cout << "bg_area.width " << bg_area.width << " bg_area.height " << bg_area.height << std::endl;
    //std::cout << "fg_area.width " << fg_area.width << " fg_area.height " << fg_area.height << std::endl;

    // Compute the rectangle with (or close to) params.fixedArea
    // and same aspect ratio as the target bbox

    area_resize_factor = sqrt(m_cfg.fixed_area / float(bg_area.width * bg_area.height));
    norm_bg_area.width = cvRound(bg_area.width * area_resize_factor);
    norm_bg_area.height = cvRound(bg_area.height * area_resize_factor);

    //std::cout << "area_resize_factor " << area_resize_factor << " norm_bg_area.width " << norm_bg_area.width << " norm_bg_area.height " << norm_bg_area.height << std::endl;

    // Correlation Filter (HOG) feature space
    // It smaller that the norm bg area if HOG cell size is > 1
    cf_response_size.width = norm_bg_area.width / m_cfg.hog_cell_size;
    cf_response_size.height = norm_bg_area.height / m_cfg.hog_cell_size;

    // given the norm BG area, which is the corresponding target w and h?
    double norm_target_sz_w = 0.75*norm_bg_area.width - 0.25*norm_bg_area.height;
    double norm_target_sz_h = 0.75*norm_bg_area.height - 0.25*norm_bg_area.width;

    // norm_target_sz_w = params.target_sz(2) * params.norm_bg_area(2) / bg_area(2);
    // norm_target_sz_h = params.target_sz(1) * params.norm_bg_area(1) / bg_area(1);
    norm_target_sz.width = cvRound(norm_target_sz_w);
    norm_target_sz.height = cvRound(norm_target_sz_h);

    //std::cout << "norm_target_sz.width " << norm_target_sz.width << " norm_target_sz.height " << norm_target_sz.height << std::endl;

    // distance (on one side) between target and bg area
    cv::Size norm_pad;

    norm_pad.width = (norm_bg_area.width - norm_target_sz.width) / 2;
    norm_pad.height = (norm_bg_area.height - norm_target_sz.height) / 2;

    int radius = std::min(norm_pad.width, norm_pad.height);

    // norm_delta_area is the number of rectangles that are considered.
    // it is the "sampling space" and the dimension of the final merged resposne
    // it is squared to not privilege any particular direction
    norm_delta_area = cv::Size((2*radius+1), (2*radius+1));

    // Rectangle in which the integral images are computed.
    // Grid of rectangles ( each of size norm_target_sz) has size norm_delta_area.
    norm_pwp_search_area.width = norm_target_sz.width + norm_delta_area.width - 1;
    norm_pwp_search_area.height = norm_target_sz.height + norm_delta_area.height - 1;

    //std::cout << "norm_pwp_search_area.width " << norm_pwp_search_area.width << " norm_pwp_search_area.height " << norm_pwp_search_area.height << std::endl;
}

///
/// \brief STAPLE_TRACKER::getSubwindow
///        GET_SUBWINDOW Obtain image sub-window, padding is done by replicating border values.
///        Returns sub-window of image IM centered at POS ([y, x] coordinates),
///        with size MODEL_SZ ([height, width]). If any pixels are outside of the image,
///        they will replicate the values at the borders
/// \param im
/// \param centerCoor
/// \param model_sz
/// \param scaled_sz
/// \param output
///
void STAPLE_TRACKER::getSubwindow(const cv::Mat &im, cv::Point_<float> centerCoor, cv::Size model_sz, cv::Size scaled_sz, cv::Mat &output)
{
    cv::Size sz = scaled_sz; // scale adaptation

    // make sure the size is not to small
    sz.width = std::max(sz.width, 2);
    sz.height = std::max(sz.height, 2);

    cv::Mat subWindow;

    // xs = round(pos(2) + (1:sz(2)) - sz(2)/2);
    // ys = round(pos(1) + (1:sz(1)) - sz(1)/2);

    cv::Point lefttop(
                std::min(im.cols - 1, std::max(-sz.width + 1, int(centerCoor.x + 1 - sz.width/2.0+0.5))),
                std::min(im.rows - 1, std::max(-sz.height + 1, int(centerCoor.y + 1 - sz.height/2.0+0.5)))
                );

    cv::Point rightbottom(
                std::max(0, int(lefttop.x + sz.width - 1)),
                std::max(0, int(lefttop.y + sz.height - 1))
                );

    cv::Point lefttopLimit(
                std::max(lefttop.x, 0),
                std::max(lefttop.y, 0)
                );
    cv::Point rightbottomLimit(
                std::min(rightbottom.x, im.cols - 1),
                std::min(rightbottom.y, im.rows - 1)
                );

    rightbottomLimit.x += 1;
    rightbottomLimit.y += 1;
    cv::Rect roiRect(lefttopLimit, rightbottomLimit);

    im(roiRect).copyTo(subWindow);

    int top = lefttopLimit.y - lefttop.y;
    int bottom = rightbottom.y - rightbottomLimit.y + 1;
    int left = lefttopLimit.x - lefttop.x;
    int right = rightbottom.x - rightbottomLimit.x + 1;

    cv::copyMakeBorder(subWindow, subWindow, top, bottom, left, right, cv::BORDER_REPLICATE);

    // imresize(subWindow, output, model_sz, 'bilinear', 'AntiAliasing', false)
    mexResize(subWindow, output, model_sz, "auto");
}

///
/// \brief STAPLE_TRACKER::updateHistModel
/// UPDATEHISTMODEL create new models for foreground and background or update the current ones
/// \param new_model
/// \param patch
/// \param learning_rate_pwp
///
void STAPLE_TRACKER::updateHistModel(bool new_model, cv::Mat &patch, float learning_rate_pwp)
{
    // Get BG (frame around target_sz) and FG masks (inner portion of target_sz)

    ////////////////////////////////////////////////////////////////////////
    cv::Size pad_offset1;

    // we constrained the difference to be mod2, so we do not have to round here
    pad_offset1.width = (bg_area.width - target_sz.width) / 2;
    pad_offset1.height = (bg_area.height - target_sz.height) / 2;

    // difference between bg_area and target_sz has to be even
    if (
            (
                (pad_offset1.width == round(pad_offset1.width)) &&
                (pad_offset1.height != round(pad_offset1.height))
                ) ||
            (
                (pad_offset1.width != round(pad_offset1.width)) &&
                (pad_offset1.height == round(pad_offset1.height))
                )) {
        assert(0);
    }

    pad_offset1.width = std::max(pad_offset1.width, 1);
    pad_offset1.height = std::max(pad_offset1.height, 1);

    //std::cout << "pad_offset1 " << pad_offset1 << std::endl;

    cv::Mat bg_mask(bg_area, CV_8UC1, cv::Scalar(1)); // init bg_mask

    // xxx: bg_mask(pad_offset1(1)+1:end-pad_offset1(1), pad_offset1(2)+1:end-pad_offset1(2)) = false;

    cv::Rect pad1_rect(
                pad_offset1.width,
                pad_offset1.height,
                bg_area.width - 2 * pad_offset1.width,
                bg_area.height - 2 * pad_offset1.height
                );

    bg_mask(pad1_rect) = false;

    ////////////////////////////////////////////////////////////////////////
    
    // we constrained the difference to be mod2, so we do not have to round here
	cv::Size pad_offset2((bg_area.width - fg_area.width) / 2, (bg_area.height - fg_area.height) / 2);

    // difference between bg_area and fg_area has to be even
    if (
            (
                (pad_offset2.width == round(pad_offset2.width)) &&
                (pad_offset2.height != round(pad_offset2.height))
                ) ||
            (
                (pad_offset2.width != round(pad_offset2.width)) &&
                (pad_offset2.height == round(pad_offset2.height))
                )) {
        assert(0);
    }

    pad_offset2.width = std::max(pad_offset2.width, 1);
    pad_offset2.height = std::max(pad_offset2.height, 1);

    cv::Mat fg_mask(bg_area, CV_8UC1, cv::Scalar(0)); // init fg_mask

    // xxx: fg_mask(pad_offset2(1)+1:end-pad_offset2(1), pad_offset2(2)+1:end-pad_offset2(2)) = true;

	auto Clamp = [](int& v, int& size, int hi) -> int
	{
		int res = 0;

		if (size < 2)
		{
			size = 2;
		}
		if (v < 0)
		{
			res = v;
			v = 0;
			return res;
		}
		else if (v + size > hi - 1)
		{
			v = hi - 1 - size;
			if (v < 0)
			{
				size += v;
				v = 0;
			}
			res = v;
			return res;
		}
		return res;
	};

    cv::Rect pad2_rect(
                pad_offset2.width,
                pad_offset2.height,
                bg_area.width - 2 * pad_offset2.width,
                bg_area.height - 2 * pad_offset2.height
                );

	if (!Clamp(pad2_rect.x, pad2_rect.width, fg_mask.cols) && !Clamp(pad2_rect.y, pad2_rect.height, fg_mask.rows))
	{
		fg_mask(pad2_rect) = true;
	}
    ////////////////////////////////////////////////////////////////////////

    cv::Mat fg_mask_new;
    cv::Mat bg_mask_new;

    mexResize(fg_mask, fg_mask_new, norm_bg_area, "auto");
    mexResize(bg_mask, bg_mask_new, norm_bg_area, "auto");

    int imgCount = 1;
    int dims = 3;
    const int sizes[] = { m_cfg.n_bins, m_cfg.n_bins, m_cfg.n_bins };
    const int channels[] = { 0, 1, 2 };
    float bRange[] = { 0, 256 };
    float gRange[] = { 0, 256 };
    float rRange[] = { 0, 256 };
    const float *ranges[] = { bRange, gRange, rRange };

    if (m_cfg.grayscale_sequence)
        dims = 1;

    // (TRAIN) BUILD THE MODEL
    if (new_model)
	{
        cv::calcHist(&patch, imgCount, channels, bg_mask_new, bg_hist, dims, sizes, ranges);
        cv::calcHist(&patch, imgCount, channels, fg_mask_new, fg_hist, dims, sizes, ranges);

        int bgtotal = std::max(1, cv::countNonZero(bg_mask_new));
        bg_hist = bg_hist / bgtotal;

        int fgtotal = std::max(1, cv::countNonZero(fg_mask_new));
        fg_hist = fg_hist / fgtotal;
    }
	else
	{ // update the model
        cv::MatND bg_hist_tmp;
        cv::MatND fg_hist_tmp;

        cv::calcHist(&patch, imgCount, channels, bg_mask_new, bg_hist_tmp, dims, sizes, ranges);
        cv::calcHist(&patch, imgCount, channels, fg_mask_new, fg_hist_tmp, dims, sizes, ranges);

        int bgtotal = std::max(1, cv::countNonZero(bg_mask_new));
        bg_hist_tmp = bg_hist_tmp / bgtotal;

        int fgtotal = std::max(1, cv::countNonZero(fg_mask_new));
        fg_hist_tmp = fg_hist_tmp / fgtotal;

        // xxx
        bg_hist = (1 - learning_rate_pwp)*bg_hist + learning_rate_pwp*bg_hist_tmp;
        fg_hist = (1 - learning_rate_pwp)*fg_hist + learning_rate_pwp*fg_hist_tmp;
    }
}

///
/// \brief STAPLE_TRACKER::CalculateHann
/// \param sz
/// \param output
///
void STAPLE_TRACKER::CalculateHann(cv::Size sz, cv::Mat &output)
{
    cv::Mat temp1(cv::Size(sz.width, 1), CV_32FC1);
    cv::Mat temp2(cv::Size(sz.height, 1), CV_32FC1);

    float *p1 = temp1.ptr<float>(0);
    float *p2 = temp2.ptr<float>(0);

    for (int i = 0; i < sz.width; ++i)
        p1[i] = static_cast<float>(0.5 * (1 - cos(CV_2PI*i / (sz.width - 1))));

    for (int i = 0; i < sz.height; ++i)
        p2[i] = static_cast<float>(0.5 * (1 - cos(CV_2PI*i / (sz.height - 1))));

    output = temp2.t()*temp1;
}

///
/// \brief meshgrid
/// \param xr
/// \param yr
/// \param outX
/// \param outY
///
void meshgrid(const cv::Range xr, const cv::Range yr, cv::Mat &outX, cv::Mat &outY)
{
    std::vector<int> x;
    x.reserve(xr.end - xr.start + 1);
    std::vector<int> y;
    y.reserve(yr.end - yr.start + 1);

    for (int i = xr.start; i <= xr.end; i++)
        x.push_back(i);
    for (int i = yr.start; i <= yr.end; i++)
        y.push_back(i);

    repeat(cv::Mat(x).t(), static_cast<int>(y.size()), 1, outX);
    repeat(cv::Mat(y), 1, static_cast<int>(x.size()), outY);
}

///
/// \brief STAPLE_TRACKER::gaussianResponse
/// GAUSSIANRESPONSE create the (fixed) target response of the correlation filter response
/// \param rect_size
/// \param sigma
/// \param output
///
void STAPLE_TRACKER::gaussianResponse(cv::Size rect_size, float sigma, cv::Mat &output)
{
    // half = floor((rect_size-1) / 2);
    // i_range = -half(1):half(1);
    // j_range = -half(2):half(2);
    // [i, j] = ndgrid(i_range, j_range);
    cv::Size half;

    half.width = (rect_size.width - 1) / 2;
    half.height = (rect_size.height - 1) / 2;

    cv::Range i_range(-half.width, rect_size.width - (1 + half.width));
    cv::Range j_range(-half.height, rect_size.height - (1 + half.height));
    cv::Mat i, j;

    meshgrid(i_range, j_range, i, j);

    // i_mod_range = mod_one(i_range, rect_size(1));
    // j_mod_range = mod_one(j_range, rect_size(2));

    std::vector<int> i_mod_range;
    i_mod_range.reserve(i_range.end - i_range.start + 1);
    std::vector<int> j_mod_range;
    i_mod_range.reserve(j_range.end - j_range.start + 1);

    for (int k = i_range.start; k <= i_range.end; k++) {
        int val = (int)(k - 1 + rect_size.width) % (int)rect_size.width;
        i_mod_range.push_back(val);
    }

    for (int k = j_range.start; k <= j_range.end; k++) {
        int val = (int)(k - 1 + rect_size.height) % (int)rect_size.height;
        j_mod_range.push_back(val);
    }

    // y = zeros(rect_size);
    // y(i_mod_range, j_mod_range) = exp(-(i.^2 + j.^2) / (2 * sigma^2));

    output = cv::Mat(rect_size.height, rect_size.width, CV_32FC2);

    for (int jj = 0; jj < rect_size.height; jj++)
    {
        int j_idx = j_mod_range[jj];
        assert(j_idx < rect_size.height);

        for (int ii = 0; ii < rect_size.width; ii++)
        {
            int i_idx = i_mod_range[ii];
            assert(i_idx < rect_size.width);

            cv::Vec2f val(exp(-(i.at<int>(jj, ii)*i.at<int>(jj, ii) + j.at<int>(jj, ii)*j.at<int>(jj, ii)) / (2 * sigma*sigma)), 0);
            output.at<cv::Vec2f>(j_idx, i_idx) = val;
        }
    }
}

///
/// \brief STAPLE_TRACKER::tracker_staple_initialize
/// \param im
/// \param region
///
void STAPLE_TRACKER::Initialize(const cv::Mat &im, cv::Rect region)
{
    int n = im.channels();
    if (n == 1)
        m_cfg.grayscale_sequence = true;

    // xxx: only support 3 channels, TODO: fix updateHistModel
    //assert(!cfg.grayscale_sequence);

    m_cfg.init_pos.x = region.x + region.width / 2.0f;
    m_cfg.init_pos.y = region.y + region.height / 2.0f;

    m_cfg.target_sz.width = region.width;
    m_cfg.target_sz.height = region.height;

    initializeAllAreas(im);

    pos = m_cfg.init_pos;
    target_sz = m_cfg.target_sz;

    // patch of the target + padding
    cv::Mat patch_padded;
    getSubwindow(im, pos, norm_bg_area, bg_area, patch_padded);

    // initialize hist model
    updateHistModel(true, patch_padded);

    CalculateHann(cf_response_size, hann_window);

    // gaussian-shaped desired response, centred in (1,1)
    // bandwidth proportional to target size
    float output_sigma = sqrt(static_cast<float>(norm_target_sz.width * norm_target_sz.height)) * m_cfg.output_sigma_factor / m_cfg.hog_cell_size;

    cv::Mat y;
    gaussianResponse(cf_response_size, output_sigma, y);
    cv::dft(y, yf);

    // SCALE ADAPTATION INITIALIZATION
    if (m_cfg.scale_adaptation)
	{
        // Code from DSST
        scale_factor = 1;
        base_target_sz = target_sz; // xxx
        float scale_sigma = sqrt(static_cast<float>(m_cfg.num_scales)) * m_cfg.scale_sigma_factor;

        cv::Mat ys = cv::Mat(1, m_cfg.num_scales, CV_32FC2);
        for (int i = 0; i < m_cfg.num_scales; i++)
        {
            cv::Vec2f val((i + 1) - ceil(m_cfg.num_scales/2.0f), 0.f);
            val[0] = exp(-0.5f * (val[0] * val[0]) / (scale_sigma * scale_sigma));
            ys.at<cv::Vec2f>(i) = val;

            // SS = (1:p.num_scales) - ceil(p.num_scales/2);
            // ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
        }

        cv::dft(ys, ysf, cv::DFT_ROWS);
        //std::cout << ysf << std::endl;

        scale_window = cv::Mat(1, m_cfg.num_scales, CV_32FC1);
        if (m_cfg.num_scales % 2 == 0)
        {
            for (int i = 0; i < m_cfg.num_scales + 1; ++i)
            {
                if (i > 0)
                    scale_window.at<float>(i - 1) = 0.5f * (1 - cos(static_cast<float>(CV_2PI) * i / (m_cfg.num_scales + 1 - 1)));
            }
        }
        else
        {
            for (int i = 0; i < m_cfg.num_scales; ++i)
            {
                scale_window.at<float>(i) = 0.5f * (1 - cos(static_cast<float>(CV_2PI) * i / (m_cfg.num_scales - 1)));
            }
        }


        scale_factors = cv::Mat(1, m_cfg.num_scales, CV_32FC1);
        for (int i = 0; i < m_cfg.num_scales; i++)
        {
            scale_factors.at<float>(i) = pow(m_cfg.scale_step, (ceil(m_cfg.num_scales/2.0f)  - (i+1)));
        }

        //std::cout << scale_factors << std::endl;

        //ss = 1:p.num_scales;
        //scale_factors = p.scale_step.^(ceil(p.num_scales/2) - ss);

        if ((m_cfg.scale_model_factor * m_cfg.scale_model_factor) * (norm_target_sz.width * norm_target_sz.height) > m_cfg.scale_model_max_area)
            m_cfg.scale_model_factor = sqrt(m_cfg.scale_model_max_area / (norm_target_sz.width * norm_target_sz.height));

        //std::cout << cfg.scale_model_factor << std::endl;

        scale_model_sz.width = static_cast<int>(norm_target_sz.width * m_cfg.scale_model_factor);
        scale_model_sz.height = static_cast<int>(norm_target_sz.height * m_cfg.scale_model_factor);
        //scale_model_sz = floor(p.norm_target_sz * p.scale_model_factor);

        //std::cout << scale_model_sz << std::endl;

        // find maximum and minimum scales
        min_scale_factor = pow(m_cfg.scale_step, ceil(log(std::max(5.0f / bg_area.width, 5.0f / bg_area.height)) / log(m_cfg.scale_step)));
        max_scale_factor = pow(m_cfg.scale_step, floor(log(std::min(im.cols / (float)target_sz.width, im.rows / (float)target_sz.height)) / log(m_cfg.scale_step)));

        //min_scale_factor = p.scale_step ^ ceil(log(max(5 ./ bg_area)) / log(p.scale_step));
        //max_scale_factor = p.scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ target_sz)) / log(p.scale_step));

        //std::cout << min_scale_factor << " " << max_scale_factor << std::endl;
    }
}

///
/// \brief STAPLE_TRACKER::getFeatureMap
/// code from DSST
/// \param im_patch
/// \param feature_type
/// \param output
///
void STAPLE_TRACKER::getFeatureMap(cv::Mat &im_patch, const char* feature_type, cv::MatND &output)
{
    assert(!strcmp(feature_type, "fhog"));

    // allocate space
#if 0
    cv::Mat tmp_image;
    im_patch.convertTo(tmp_image, CV_32FC1);
    fhog28(output, tmp_image, cfg.hog_cell_size, 9);
#else
    fhog28(output, im_patch, m_cfg.hog_cell_size, 9);
#endif
    int w = cf_response_size.width;
    int h = cf_response_size.height;

    // hog28 already generate this matrix of (w,h,28)
    // out = zeros(h, w, 28, 'single');
    // out(:,:,2:28) = temp(:,:,1:27);

    cv::Mat new_im_patch;

    if (m_cfg.hog_cell_size > 1)
        mexResize(im_patch, new_im_patch, cv::Size(w, h), "auto");
    else
        new_im_patch = im_patch;

    cv::Mat grayimg;

    if (new_im_patch.channels() > 1)
        cv::cvtColor(new_im_patch, grayimg, cv::COLOR_BGR2GRAY);
    else
        grayimg = new_im_patch;

    // out(:,:,1) = single(im_patch)/255 - 0.5;

    float alpha = 1.f / 255.0f;
    float betta = 0.5f;

    typedef cv::Vec<float, 28> Vecf28;

    for (int j = 0; j < h; ++j)
    {
        Vecf28* pDst = output.ptr<Vecf28>(j);
        const float* pHann = hann_window.ptr<float>(j);
        const uchar* pGray = grayimg.ptr<uchar>(j);

        for (int i = 0; i < w; ++i)
        {
            // apply Hann window
            Vecf28& val = pDst[0];

            val = val * pHann[0];
            val[0] = (alpha * pGray[0] - betta) * pHann[0];

            ++pDst;
            ++pHann;
            ++pGray;
        }
    }
}

///
/// \brief matsplit
/// \param xt
/// \param xtsplit
///
void matsplit(const cv::MatND &xt, std::vector<cv::Mat> &xtsplit)
{
    int w = xt.cols;
    int h = xt.rows;
    int cn = xt.channels();

    assert(cn == 28);

    for (int k = 0; k < cn; k++)
    {
        cv::Mat dim = cv::Mat(h, w, CV_32FC2);

        for (int j = 0; j < h; ++j)
        {
            float* pDst = dim.ptr<float>(j);
            const float* pSrc = xt.ptr<float>(j);

            for (int i = 0; i < w; ++i)
            {
                pDst[0] = pSrc[k];
                pDst[1] = 0.0f;

                pSrc += cn;
                pDst += 2;
            }
        }

        xtsplit.push_back(dim);
    }
}

///
/// \brief STAPLE_TRACKER::getSubwindowFloor
///        GET_SUBWINDOW Obtain image sub-window, padding is done by replicating border values.
///        Returns sub-window of image IM centered at POS ([y, x] coordinates),
///        with size MODEL_SZ ([height, width]). If any pixels are outside of the image,
///        they will replicate the values at the borders
/// \param im
/// \param centerCoor
/// \param model_sz
/// \param scaled_sz
/// \param output
///
void STAPLE_TRACKER::getSubwindowFloor(const cv::Mat &im, cv::Point_<float> centerCoor, cv::Size model_sz, cv::Size scaled_sz, cv::Mat &output)
{
    cv::Size sz = scaled_sz; // scale adaptation

    // make sure the size is not to small
    sz.width = std::max(sz.width, 2);
    sz.height = std::max(sz.height, 2);

    cv::Mat subWindow;

    // xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
    // ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);

    cv::Point lefttop(
                std::min(im.cols - 1, std::max(-sz.width + 1, int(centerCoor.x + 1) - int(sz.width/2.0))),
                std::min(im.rows - 1, std::max(-sz.height + 1, int(centerCoor.y + 1) - int(sz.height/2.0)))
                );

    cv::Point rightbottom(
                std::max(0, int(lefttop.x + sz.width - 1)),
                std::max(0, int(lefttop.y + sz.height - 1))
                );

    cv::Point lefttopLimit(
                std::max(lefttop.x, 0),
                std::max(lefttop.y, 0)
                );
    cv::Point rightbottomLimit(
                std::min(rightbottom.x, im.cols - 1),
                std::min(rightbottom.y, im.rows - 1)
                );

    rightbottomLimit.x += 1;
    rightbottomLimit.y += 1;
    cv::Rect roiRect(lefttopLimit, rightbottomLimit);

    im(roiRect).copyTo(subWindow);

    // imresize(subWindow, output, model_sz, 'bilinear', 'AntiAliasing', false)
    mexResize(subWindow, output, model_sz, "auto");
}

///
/// \brief STAPLE_TRACKER::getScaleSubwindow
/// code from DSST
/// \param im
/// \param centerCoor
/// \param output
///
void STAPLE_TRACKER::getScaleSubwindow(const cv::Mat &im, cv::Point_<float> centerCoor, cv::Mat &output)
{
    int ch = 0;
    int total = 0;

    for (int s = 0; s < m_cfg.num_scales; s++)
    {
        cv::Size_<float> patch_sz;

        patch_sz.width = floor(base_target_sz.width * scale_factor * scale_factors.at<float>(s));
        patch_sz.height = floor(base_target_sz.height * scale_factor * scale_factors.at<float>(s));

        cv::Mat im_patch_resized;
        getSubwindowFloor(im, centerCoor, scale_model_sz, patch_sz, im_patch_resized);

        // extract scale features
        cv::MatND temp;
        fhog31(temp, im_patch_resized, m_cfg.hog_cell_size, 9);

        if (s == 0)
        {
            ch = temp.channels();
            total = temp.cols * temp.rows * ch;

            output = cv::Mat(total, m_cfg.num_scales, CV_32FC2);
        }

        int tempw = temp.cols;
        int temph = temp.rows;
        int tempch = temp.channels();

        int count = 0;

        float scaleWnd = scale_window.at<float>(s);

        float* outData = (float*)output.data;

        // window
        for (int j = 0; j < temph; ++j)
        {
            const float* tmpData = temp.ptr<float>(j);

            for (int i = 0; i < tempw; ++i)
            {
                for (int k = 0; k < tempch; ++k)
                {
                    outData[(count * m_cfg.num_scales + s) * 2 + 0] = tmpData[k] * scaleWnd;
                    outData[(count * m_cfg.num_scales + s) * 2 + 1] = 0.0;

                    ++count;
                }
                tmpData += ch;
            }
        }
    }
}

///
/// \brief STAPLE_TRACKER::tracker_staple_train
/// TRAINING
/// \param im
/// \param first
///
void STAPLE_TRACKER::Train(const cv::Mat &im, bool first)
{
    // extract patch of size bg_area and resize to norm_bg_area
    cv::Mat im_patch_bg;
    getSubwindow(im, pos, norm_bg_area, bg_area, im_patch_bg);

    // compute feature map, of cf_response_size
    cv::MatND xt;
    getFeatureMap(im_patch_bg, m_cfg.feature_type, xt);

    // apply Hann window in getFeatureMap
    // xt = bsxfun(@times, hann_window, xt);

    // compute FFT
    // cv::MatND xtf;
    std::vector<cv::Mat> xtsplit;
    std::vector<cv::Mat> xtf; // xtf is splits of xtf

    matsplit(xt, xtsplit);

    for (int i =  0; i < xt.channels(); i++) {
        cv::Mat dimf;
        cv::dft(xtsplit[i], dimf);
        xtf.push_back(dimf);
    }

    // FILTER UPDATE
    // Compute expectations over circular shifts,
    // therefore divide by number of pixels.
    // new_hf_num = bsxfun(@times, conj(yf), xtf) / prod(p.cf_response_size);
    // new_hf_den = (conj(xtf) .* xtf) / prod(p.cf_response_size);

    {
        std::vector<cv::Mat> new_hf_num;
        std::vector<cv::Mat> new_hf_den;

        int w = xt.cols;
        int h = xt.rows;
        float invArea = 1.f / (cf_response_size.width * cf_response_size.height);

        for (int ch = 0; ch < xt.channels(); ch++)
        {
            cv::Mat dim = cv::Mat(h, w, CV_32FC2);

            for (int j = 0; j < h; ++j)
            {
                const float* pXTF = xtf[ch].ptr<float>(j);
                const float* pYF = yf.ptr<float>(j);
                cv::Vec2f* pDst = dim.ptr<cv::Vec2f>(j);

                for (int i = 0; i < w; ++i)
                {
                    cv::Vec2f val(pYF[1] * pXTF[1] + pYF[0] * pXTF[0], pYF[0] * pXTF[1] - pYF[1] * pXTF[0]);
                    *pDst = invArea * val;

                    pXTF += 2;
                    pYF += 2;
                    ++pDst;
                }
            }
            new_hf_num.push_back(dim);
        }

        for (int ch = 0; ch < xt.channels(); ch++)
        {
            cv::Mat dim = cv::Mat(h, w, CV_32FC1);

            for (int j = 0; j < h; ++j)
            {
                const float* pXTF = xtf[ch].ptr<float>(j);
                float* pDst = dim.ptr<float>(j);

                for (int i = 0; i < w; ++i)
                {
                    *pDst = invArea * (pXTF[0]*pXTF[0] + pXTF[1]*pXTF[1]);

                    pXTF += 2;
                    ++pDst;
                }
            }
            new_hf_den.push_back(dim);
        }

        if (first) {
            // first frame, train with a single image
            hf_den.assign(new_hf_den.begin(), new_hf_den.end());
            hf_num.assign(new_hf_num.begin(), new_hf_num.end());
        } else {
            // subsequent frames, update the model by linear interpolation
            for (int ch =  0; ch < xt.channels(); ch++) {
                hf_den[ch] = (1 - m_cfg.learning_rate_cf) * hf_den[ch] + m_cfg.learning_rate_cf * new_hf_den[ch];
                hf_num[ch] = (1 - m_cfg.learning_rate_cf) * hf_num[ch] + m_cfg.learning_rate_cf * new_hf_num[ch];
            }

            updateHistModel(false, im_patch_bg, m_cfg.learning_rate_pwp);

            // BG/FG MODEL UPDATE
            // patch of the target + padding
            // [bg_hist, fg_hist] = updateHistModel(new_pwp_model, im_patch_bg, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence, bg_hist, fg_hist, p.learning_rate_pwp);
        }
    }

    // SCALE UPDATE
    if (m_cfg.scale_adaptation) {
        cv::Mat im_patch_scale;

        getScaleSubwindow(im, pos, im_patch_scale);

        cv::Mat xsf;
        cv::dft(im_patch_scale, xsf, cv::DFT_ROWS);

        // new_sf_num = bsxfun(@times, ysf, conj(xsf));
        // new_sf_den = sum(xsf .* conj(xsf), 1);

        cv::Mat new_sf_num;
        cv::Mat new_sf_den;

        int w = xsf.cols;
        int h = xsf.rows;

        new_sf_num = cv::Mat(h, w, CV_32FC2);

        for (int j = 0; j < h; ++j) // xxx
        {
            float* pDst = new_sf_num.ptr<float>(j);

            const float* pXSF = xsf.ptr<float>(j);
            const float* pYSF = ysf.ptr<float>(0);

            for (int i = 0; i < w; ++i)
            {
                pDst[0] = (pYSF[1] * pXSF[1] + pYSF[0] * pXSF[0]);
                pDst[1] = (pYSF[1] * pXSF[0] - pYSF[0] * pXSF[1]);

                pXSF += 2;
                pYSF += 2;
                pDst += 2;
            }
        }

        new_sf_den = cv::Mat(1, w, CV_32FC1, cv::Scalar(0, 0, 0));
        float* pDst = new_sf_den.ptr<float>(0);

        for (int j = 0; j < h; ++j)
        {
            const float* pSrc = xsf.ptr<float>(j);

            for (int i = 0; i < w; ++i)
            {
                pDst[i] += (pSrc[0] * pSrc[0] + pSrc[1] * pSrc[1]);
                pSrc += 2;
            }
        }

        if (first) {
            // first frame, train with a single image
            new_sf_den.copyTo(sf_den);
            new_sf_num.copyTo(sf_num);
        } else {
            sf_den = (1 - m_cfg.learning_rate_scale) * sf_den + m_cfg.learning_rate_scale * new_sf_den;
            sf_num = (1 - m_cfg.learning_rate_scale) * sf_num + m_cfg.learning_rate_scale * new_sf_num;
        }
    }

    // update bbox position
    if (first) {
        rect_position.x = cvRound(pos.x - target_sz.width / 2);
        rect_position.y = cvRound(pos.y - target_sz.height / 2);
        rect_position.width = target_sz.width;
        rect_position.height = target_sz.height;
    }

    frameno += 1;
}

///
/// \brief ensure_real
/// xxx: improve later
/// \param complex
/// \return
///
cv::Mat ensure_real(const cv::Mat &complex)
{
    int w = complex.cols;
    int h = complex.rows;

    cv::Mat real = cv::Mat(h, w, CV_32FC1);

    for (int j = 0; j < h; ++j)
    {
        float* pDst = real.ptr<float>(j);
        const float* pSrc = complex.ptr<float>(j);

        for (int i = 0; i < w; ++i)
        {
            *pDst = *pSrc;
            ++pDst;
            pSrc += 2;
        }
    }
    return real;
}

///
/// \brief STAPLE_TRACKER::cropFilterResponse
/// \param response_cf
/// \param response_size
/// \param output
///
void STAPLE_TRACKER::cropFilterResponse(const cv::Mat &response_cf, cv::Size response_size, cv::Mat& output)
{
    int w = response_cf.cols;
    int h = response_cf.rows;

    // newh and neww must be odd, as we want an exact center
    assert(((response_size.width % 2) == 1) && ((response_size.height % 2) == 1));

    int half_width = response_size.width / 2;
    int half_height = response_size.height / 2;

    cv::Range i_range(-half_width, response_size.width - (1 + half_width));
    cv::Range j_range(-half_height, response_size.height - (1 + half_height));

    std::vector<int> i_mod_range;
    i_mod_range.reserve(i_range.end - i_range.start + 1);
    std::vector<int> j_mod_range;
    i_mod_range.reserve(j_range.end - j_range.start + 1);

    for (int k = i_range.start; k <= i_range.end; k++) {
        int val = (k - 1 + w) % w;
        i_mod_range.push_back(val);
    }

    for (int k = j_range.start; k <= j_range.end; k++) {
        int val = (k - 1 + h) % h;
        j_mod_range.push_back(val);
    }

    cv::Mat tmp = cv::Mat(response_size.height, response_size.width, CV_32FC1, cv::Scalar(0, 0, 0));

    for (int j = 0; j < response_size.height; j++)
    {
        int j_idx = j_mod_range[j];
        assert(j_idx < h);

        float* pDst = tmp.ptr<float>(j);
        const float* pSrc = response_cf.ptr<float>(j_idx);

        for (int i = 0; i < response_size.width; i++)
        {
            int i_idx = i_mod_range[i];
            assert(i_idx < w);

            *pDst = pSrc[i_idx];
            ++pDst;
        }
    }
    output = tmp;
}

///
/// \brief STAPLE_TRACKER::getColourMap
/// GETCOLOURMAP computes pixel-wise probabilities (PwP) given PATCH and models BG_HIST and FG_HIST
/// \param patch
/// \param output
///
void STAPLE_TRACKER::getColourMap(const cv::Mat &patch, cv::Mat& output)
{
    // check whether the patch has 3 channels
    int h = patch.rows;
    int w = patch.cols;
    int d = patch.channels();

    // figure out which bin each pixel falls into
    int bin_width = 256 / m_cfg.n_bins;

    // convert image to d channels array
    //patch_array = reshape(double(patch), w*h, d);

    output = cv::Mat(h, w, CV_32FC1);

    if (!m_cfg.grayscale_sequence)
    {
        for (int j = 0; j < h; ++j)
        {
            const uchar* pSrc = patch.ptr<uchar>(j);
            float* pDst = output.ptr<float>(j);

            for (int i = 0; i < w; ++i)
            {
                int b1 = pSrc[0] / bin_width;
                int b2 = pSrc[1] / bin_width;
                int b3 = pSrc[2] / bin_width;

                float* histd = (float*)bg_hist.data;
                float probg = histd[b1 * m_cfg.n_bins * m_cfg.n_bins + b2 * m_cfg.n_bins + b3];

                histd = (float*)fg_hist.data;
                float profg = histd[b1 * m_cfg.n_bins * m_cfg.n_bins + b2 * m_cfg.n_bins + b3];

                // xxx
                *pDst = profg / (profg + probg);
                if (std::isnan(*pDst))
                    *pDst = 0.0;

                pSrc += d;
                ++pDst;

                // (TODO) in theory it should be at 0.5 (unseen colors shoud have max entropy)
                //likelihood_map(isnan(likelihood_map)) = 0;
            }
        }
    }
    else
    {
        for (int j = 0; j < h; j++)
        {
            const uchar* pSrc = patch.ptr<uchar>(j);
            float* pDst = output.ptr<float>(j);

            for (int i = 0; i < w; i++)
            {
                int b = *pSrc;

                float* histd = (float*)bg_hist.data;
                float probg = histd[b];

                histd = (float*)fg_hist.data;
                float profg = histd[b];

                // xxx
                *pDst = profg / (profg + probg);
                if (std::isnan(*pDst))
                    *pDst = 0.0;

                pSrc += d;
                ++pDst;

                // (TODO) in theory it should be at 0.5 (unseen colors shoud have max entropy)
                //likelihood_map(isnan(likelihood_map)) = 0;
            }
        }
    }

    // to which bin each pixel (for all d channels) belongs to
    //bin_indices = floor(patch_array/bin_width) + 1;

    // Get pixel-wise posteriors (PwP)
    // P_bg = getP(bg_hist, h, w, bin_indices, grayscale_sequence);
    // P_fg = getP(fg_hist, h, w, bin_indices, grayscale_sequence);

    // Object-likelihood map
    //P_O = P_fg ./ (P_fg + P_bg);
}

///
/// \brief STAPLE_TRACKER::getCenterLikelihood
/// GETCENTERLIKELIHOOD computes the sum over rectangles of size M.
/// \param object_likelihood
/// \param m
/// \param center_likelihood
///
void STAPLE_TRACKER::getCenterLikelihood(const cv::Mat &object_likelihood, cv::Size m, cv::Mat& center_likelihood)
{
    // CENTER_LIKELIHOOD is the 'colour response'
    int h = object_likelihood.rows;
    int w = object_likelihood.cols;
    int n1 = w - m.width + 1;
    int n2 = h - m.height + 1;
    float invArea = 1.f / (m.width * m.height);

    cv::Mat temp;

    // integral images
    cv::integral(object_likelihood, temp);

    center_likelihood = cv::Mat(n2, n1, CV_32FC1);

    for (int j = 0; j < n2; ++j)
    {
        float* pLike = reinterpret_cast<float*>(center_likelihood.ptr(j));

        for (int i = 0; i < n1; ++i)
        {
            *pLike = invArea * static_cast<float>(temp.at<double>(j, i) + temp.at<double>(j+m.height, i+m.width) - temp.at<double>(j, i+m.width) - temp.at<double>(j+m.height, i));
            ++pLike;
        }
    }

    // SAT = integralImage(object_likelihood);
    // i = 1:n1;
    // j = 1:n2;
    // center_likelihood = (SAT(i,j) + SAT(i+m(1), j+m(2)) - SAT(i+m(1), j) - SAT(i, j+m(2))) / prod(m);
}

///
/// \brief STAPLE_TRACKER::mergeResponses
/// \param response_cf
/// \param response_pwp
/// \param response
///
void STAPLE_TRACKER::mergeResponses(const cv::Mat &response_cf, const cv::Mat &response_pwp, cv::Mat &response)
{
    auto alpha = m_cfg.merge_factor;
    //const char *merge_method = cfg.merge_method;

    // MERGERESPONSES interpolates the two responses with the hyperparameter ALPHA
    response = (1 - alpha) * response_cf + alpha * response_pwp;

    // response = (1 - alpha) * response_cf + alpha * response_pwp;
}

///
/// \brief STAPLE_TRACKER::tracker_staple_update
/// TESTING step
/// \param im
/// \param confidence
/// \return
///
cv::RotatedRect STAPLE_TRACKER::Update(const cv::Mat &im, float& confidence)
{
    confidence = 0;

    // extract patch of size bg_area and resize to norm_bg_area
    cv::Mat im_patch_cf;
    getSubwindow(im, pos, norm_bg_area, bg_area, im_patch_cf);

    cv::Size pwp_search_area;

    pwp_search_area.width = static_cast<int>(norm_pwp_search_area.width / area_resize_factor);
    pwp_search_area.height = static_cast<int>(norm_pwp_search_area.height / area_resize_factor);

    // extract patch of size pwp_search_area and resize to norm_pwp_search_area
    getSubwindow(im, pos, norm_pwp_search_area, pwp_search_area, im_patch_pwp);

    // compute feature map
    cv::MatND xt_windowed;
    getFeatureMap(im_patch_cf, m_cfg.feature_type, xt_windowed);

    // apply Hann window in getFeatureMap

    // compute FFT
    // cv::MatND xtf;
    std::vector<cv::Mat> xtsplit;
    std::vector<cv::Mat> xtf; // xtf is splits of xtf

    matsplit(xt_windowed, xtsplit);

    for (int i =  0; i < xt_windowed.channels(); i++) {
        cv::Mat dimf;
        cv::dft(xtsplit[i], dimf);
        xtf.push_back(dimf);
    }

	const int w = xt_windowed.cols;
	const int h = xt_windowed.rows;
	std::vector<cv::Mat> hf(xt_windowed.channels(), cv::Mat(h, w, CV_32FC2));

    // Correlation between filter and test patch gives the response
    // Solve diagonal system per pixel.
    if (m_cfg.den_per_channel)
    {
        for (int ch = 0; ch < xt_windowed.channels(); ++ch)
        {
            for (int j = 0; j < h; ++j)
            {
                const cv::Vec2f* pSrc = hf_num[ch].ptr<cv::Vec2f>(j);
                const float* pDen = hf_den[ch].ptr<float>(j);
                cv::Vec2f* pDst = hf[ch].ptr<cv::Vec2f>(j);

                for (int i = 0; i < w; ++i)
                {
                    pDst[i] = pSrc[i] / (pDen[i] + m_cfg.lambda);
                }
            }
        }
    }
    else
    {
        //hf = bsxfun(@rdivide, hf_num, sum(hf_den, 3)+p.lambda);

        std::vector<float> DIM1(static_cast<size_t>(w) * static_cast<size_t>(h), m_cfg.lambda);

        for (int ch = 0; ch < xt_windowed.channels(); ++ch)
        {
            float* pDim1 = &DIM1[0];
            for (int j = 0; j < h; ++j)
            {
                const float* pDen = hf_den[ch].ptr<float>(j);
                for (int i = 0; i < w; ++i)
                {
                    *pDim1 += pDen[i];
                    ++pDim1;
                }
            }
        }

        for (int ch = 0; ch < xt_windowed.channels(); ++ch)
        {
            const float* pDim1 = &DIM1[0];
            for (int j = 0; j < h; ++j)
            {
                const cv::Vec2f* pSrc = hf_num[ch].ptr<cv::Vec2f>(j);
                cv::Vec2f* pDst = hf[ch].ptr<cv::Vec2f>(j);

                for (int i = 0; i < w; ++i)
                {
                    *pDst = *pSrc / *pDim1;
                    ++pDim1;
                    ++pDst;
                    ++pSrc;
                }
            }
        }
    }

    cv::Mat response_cff = cv::Mat(h, w, CV_32FC2);

    for (int j = 0; j < h; j++)
    {
        cv::Vec2f* pDst = response_cff.ptr<cv::Vec2f>(j);

        for (int i = 0; i < w; i++)
        {
            float sum = 0.0;
            float sumi = 0.0;

            for (size_t ch = 0; ch < hf.size(); ch++)
            {
                cv::Vec2f pHF = hf[ch].at<cv::Vec2f>(j,i);
                cv::Vec2f pXTF = xtf[ch].at<cv::Vec2f>(j,i);

                sum += (pHF[0] * pXTF[0] + pHF[1] * pXTF[1]);
                sumi += (pHF[0] * pXTF[1] - pHF[1] * pXTF[0]);
                // assert(norm(imag(x(:))) <= 1e-5 * norm(real(x(:))));
            }

            *pDst = cv::Vec2f(sum, sumi);
            ++pDst;
        }
    }

    cv::Mat response_cfi;
    cv::dft(response_cff, response_cfi, cv::DFT_SCALE|cv::DFT_INVERSE);
    cv::Mat response_cf = ensure_real(response_cfi);

    // response_cf = ensure_real(ifft2(sum(conj(hf) .* xtf, 3)));

    // Crop square search region (in feature pixels).
    cv::Size newsz = norm_delta_area;
    newsz.width = (newsz.width / m_cfg.hog_cell_size);
    newsz.height = (newsz.height / m_cfg.hog_cell_size);

    if (newsz.width % 2 == 0)
		newsz.width -= 1;
    if (newsz.height % 2 == 0)
		newsz.height -= 1;

    cropFilterResponse(response_cf, newsz, response_cf);

    if (m_cfg.hog_cell_size > 1)
    {
        cv::Mat temp;
        mexResize(response_cf, temp, norm_delta_area, "auto");
        response_cf = temp; // xxx: low performance
    }

    cv::Mat likelihood_map;
    getColourMap(im_patch_pwp, likelihood_map);
    //[likelihood_map] = getColourMap(im_patch_pwp, bg_hist, fg_hist, p.n_bins, p.grayscale_sequence);

    // each pixel of response_pwp loosely represents the likelihood that
    // the target (of size norm_target_sz) is centred on it
    cv::Mat response_pwp;
    getCenterLikelihood(likelihood_map, norm_target_sz, response_pwp);

    // ESTIMATION
    cv::Mat response;
    mergeResponses(response_cf, response_pwp, response);

    double maxVal = 0;
    cv::Point maxLoc;

    cv::minMaxLoc(response, nullptr, &maxVal, nullptr, &maxLoc);
    //[row, col] = find(response == max(response(:)), 1);

    //std::cout << "maxLoc = " << maxLoc << ", maxVal = " << maxVal << std::endl;
    confidence = static_cast<float>(maxVal);

    float centerx = static_cast<float>((1 + norm_delta_area.width) / 2 - 1);
    float centery = static_cast<float>((1 + norm_delta_area.height) / 2 - 1);

    pos.x += (maxLoc.x - centerx) / area_resize_factor;
    pos.y += (maxLoc.y - centery) / area_resize_factor;

    // Report current location
    cv::Rect_<float> location;

    location.x = pos.x - target_sz.width / 2.0f;
    location.y = pos.y - target_sz.height / 2.0f;
    location.width = static_cast<float>(target_sz.width);
    location.height = static_cast<float>(target_sz.height);

    //std::cout << location << std::endl;

    // center = (1+p.norm_delta_area) / 2;
    // pos = pos + ([row, col] - center) / area_resize_factor;
    // rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];

    // SCALE SPACE SEARCH
    if (m_cfg.scale_adaptation)
    {
        cv::Mat im_patch_scale;

        getScaleSubwindow(im, pos, im_patch_scale);

        cv::Mat xsf;
        cv::dft(im_patch_scale, xsf, cv::DFT_ROWS);

        // im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor * scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size);
        // xsf = fft(im_patch_scale,[],2);

        const int cols = xsf.cols;
        const int rows = xsf.rows;

        cv::Mat scale_responsef = cv::Mat(1, cols, CV_32FC2, cv::Scalar(0, 0, 0));

        for (int j = 0; j < rows; ++j)
        {
            const float* pXSF = xsf.ptr<float>(j);
            const float* pXSFNUM = sf_num.ptr<float>(j);
            const float* pDen = sf_den.ptr<float>(0);
            float* pscale = scale_responsef.ptr<float>(0);

            for (int i = 0; i < cols; ++i)
            {
                float invDen = 1.f / (*pDen + m_cfg.lambda);

                pscale[0] += invDen * (pXSFNUM[0]*pXSF[0] - pXSFNUM[1]*pXSF[1]);
                pscale[1] += invDen * (pXSFNUM[0]*pXSF[1] + pXSFNUM[1]*pXSF[0]);

                pscale += 2;
                pXSF += 2;
                pXSFNUM += 2;
                ++pDen;
            }
        }

        cv::Mat scale_response;
        cv::dft(scale_responsef, scale_response, cv::DFT_SCALE|cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);

        //scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + p.lambda) ));
        cv::minMaxLoc(scale_response, nullptr, &maxVal, nullptr, &maxLoc);

        //recovered_scale = ind2sub(size(scale_response),find(scale_response == max(scale_response(:)), 1));

        int recovered_scale = maxLoc.x;

        // set the scale
        scale_factor = scale_factor * scale_factors.at<float>(recovered_scale);

        if (scale_factor < min_scale_factor) {
            scale_factor = min_scale_factor;
        } else if (scale_factor > max_scale_factor) {
            scale_factor = max_scale_factor;
        }

        // use new scale to update bboxes for target, filter, bg and fg models
        target_sz.width = cvRound(base_target_sz.width * scale_factor);
        target_sz.height = cvRound(base_target_sz.height * scale_factor);

        float avg_dim = (target_sz.width + target_sz.height) / 2.0f;

        bg_area.width = std::min(im.cols - 1, cvRound(target_sz.width + avg_dim));
        bg_area.height = std::min(im.rows - 1, cvRound(target_sz.height + avg_dim));

        bg_area.width = bg_area.width - (bg_area.width - target_sz.width) % 2;
        bg_area.height = bg_area.height - (bg_area.height - target_sz.height) % 2;

        fg_area.width = cvRound(target_sz.width - avg_dim * m_cfg.inner_padding);
        fg_area.height = cvRound(target_sz.height - avg_dim * m_cfg.inner_padding);

        fg_area.width = fg_area.width + int(bg_area.width - fg_area.width) % 2;
        fg_area.height = fg_area.height + int(bg_area.height - fg_area.height) % 2;

        // Compute the rectangle with (or close to) params.fixed_area and
        // same aspect ratio as the target bboxgetScaleSubwindow
        area_resize_factor = sqrt(m_cfg.fixed_area / (float)(bg_area.width * bg_area.height));
    }

    return cv::RotatedRect(cv::Point2f(location.x + 0.5f * location.width, location.y + 0.5f * location.height),
		cv::Size2f(location.width, location.height), 0.f);
}
