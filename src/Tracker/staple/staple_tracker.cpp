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
    cfg = default_parameters_staple(cfg);
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
void STAPLE_TRACKER::mexResize(const cv::Mat &im, cv::Mat &output, cv::Size newsz, const char *method) {
    int interpolation = cv::INTER_LINEAR;

    cv::Size sz = im.size();

    if(!strcmp(method, "antialias")){
        interpolation = cv::INTER_AREA;
    } else if (!strcmp(method, "linear")){
        interpolation = cv::INTER_LINEAR;
    } else if (!strcmp(method, "auto")){
        if(newsz.width > sz.width){ // xxx
            interpolation = cv::INTER_LINEAR;
        }else{
            interpolation = cv::INTER_AREA;
        }
    } else {
        assert(0);
        return;
    }

    resize(im, output, newsz, 0, 0, interpolation);
}

///
/// \brief STAPLE_TRACKER::default_parameters_staple
/// \param cfg
/// \return
///
staple_cfg STAPLE_TRACKER::default_parameters_staple(staple_cfg cfg)
{
    return cfg;
}

///
/// \brief STAPLE_TRACKER::initializeAllAreas
/// \param im
///
void STAPLE_TRACKER::initializeAllAreas(const cv::Mat &im)
{
    // we want a regular frame surrounding the object
    double avg_dim = (cfg.target_sz.width + cfg.target_sz.height) / 2.0;

    bg_area.width = round(cfg.target_sz.width + avg_dim);
    bg_area.height = round(cfg.target_sz.height + avg_dim);

    // pick a "safe" region smaller than bbox to avoid mislabeling
    fg_area.width = round(cfg.target_sz.width - avg_dim * cfg.inner_padding);
    fg_area.height = round(cfg.target_sz.height - avg_dim * cfg.inner_padding);

    // saturate to image size
    cv::Size imsize = im.size();

    bg_area.width = std::min(bg_area.width, imsize.width - 1);
    bg_area.height = std::min(bg_area.height, imsize.height - 1);

    // make sure the differences are a multiple of 2 (makes things easier later in color histograms)
    bg_area.width = bg_area.width - (bg_area.width - cfg.target_sz.width) % 2;
    bg_area.height = bg_area.height - (bg_area.height - cfg.target_sz.height) % 2;

    fg_area.width = fg_area.width + (bg_area.width - fg_area.width) % 2;
    fg_area.height = fg_area.height + (bg_area.height - fg_area.width) % 2;

    std::cout << "bg_area.width " << bg_area.width << " bg_area.height " << bg_area.height << std::endl;
    std::cout << "fg_area.width " << fg_area.width << " fg_area.height " << fg_area.height << std::endl;

    // Compute the rectangle with (or close to) params.fixedArea
    // and same aspect ratio as the target bbox

    area_resize_factor = sqrt(cfg.fixed_area / double(bg_area.width * bg_area.height));
    norm_bg_area.width = round(bg_area.width * area_resize_factor);
    norm_bg_area.height = round(bg_area.height * area_resize_factor);

    std::cout << "area_resize_factor " << area_resize_factor << " norm_bg_area.width " << norm_bg_area.width << " norm_bg_area.height " << norm_bg_area.height << std::endl;

    // Correlation Filter (HOG) feature space
    // It smaller that the norm bg area if HOG cell size is > 1
    cf_response_size.width = floor(norm_bg_area.width / cfg.hog_cell_size);
    cf_response_size.height = floor(norm_bg_area.height / cfg.hog_cell_size);

    // given the norm BG area, which is the corresponding target w and h?
    double norm_target_sz_w = 0.75*norm_bg_area.width - 0.25*norm_bg_area.height;
    double norm_target_sz_h = 0.75*norm_bg_area.height - 0.25*norm_bg_area.width;

    // norm_target_sz_w = params.target_sz(2) * params.norm_bg_area(2) / bg_area(2);
    // norm_target_sz_h = params.target_sz(1) * params.norm_bg_area(1) / bg_area(1);
    norm_target_sz.width = round(norm_target_sz_w);
    norm_target_sz.height = round(norm_target_sz_h);

    std::cout << "norm_target_sz.width " << norm_target_sz.width << " norm_target_sz.height " << norm_target_sz.height << std::endl;

    // distance (on one side) between target and bg area
    cv::Size norm_pad;

    norm_pad.width = floor((norm_bg_area.width - norm_target_sz.width) / 2.0);
    norm_pad.height = floor((norm_bg_area.height - norm_target_sz.height) / 2.0);

    int radius = floor(fmin(norm_pad.width, norm_pad.height));

    // norm_delta_area is the number of rectangles that are considered.
    // it is the "sampling space" and the dimension of the final merged resposne
    // it is squared to not privilege any particular direction
    norm_delta_area = cv::Size((2*radius+1), (2*radius+1));

    // Rectangle in which the integral images are computed.
    // Grid of rectangles ( each of size norm_target_sz) has size norm_delta_area.
    norm_pwp_search_area.width = norm_target_sz.width + norm_delta_area.width - 1;
    norm_pwp_search_area.height = norm_target_sz.height + norm_delta_area.height - 1;

    std::cout << "norm_pwp_search_area.width " << norm_pwp_search_area.width << " norm_pwp_search_area.height " << norm_pwp_search_area.height << std::endl;
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
    sz.width = fmax(sz.width, 2);
    sz.height = fmax(sz.height, 2);

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
void STAPLE_TRACKER::updateHistModel(bool new_model, cv::Mat &patch, double learning_rate_pwp)
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

    pad_offset1.width = fmax(pad_offset1.width, 1);
    pad_offset1.height = fmax(pad_offset1.height, 1);

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
    cv::Size pad_offset2;

    // we constrained the difference to be mod2, so we do not have to round here
    pad_offset2.width = (bg_area.width - fg_area.width) / 2;
    pad_offset2.height = (bg_area.height - fg_area.height) / 2;

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

    pad_offset2.width = fmax(pad_offset2.width, 1);
    pad_offset2.height = fmax(pad_offset2.height, 1);

    //std::cout << "pad_offset2 " << pad_offset2 << std::endl;

    cv::Mat fg_mask(bg_area, CV_8UC1, cv::Scalar(0)); // init fg_mask

    // xxx: fg_mask(pad_offset2(1)+1:end-pad_offset2(1), pad_offset2(2)+1:end-pad_offset2(2)) = true;

    cv::Rect pad2_rect(
                pad_offset2.width,
                pad_offset2.height,
                bg_area.width - 2 * pad_offset2.width,
                bg_area.height - 2 * pad_offset2.height
                );

    fg_mask(pad2_rect) = true;
    ////////////////////////////////////////////////////////////////////////

    cv::Mat fg_mask_new;
    cv::Mat bg_mask_new;

    mexResize(fg_mask, fg_mask_new, norm_bg_area, "auto");
    mexResize(bg_mask, bg_mask_new, norm_bg_area, "auto");

    int imgCount = 1;
    int dims = 3;
    const int sizes[] = { cfg.n_bins, cfg.n_bins, cfg.n_bins };
    const int channels[] = { 0, 1, 2 };
    float bRange[] = { 0, 256 };
    float gRange[] = { 0, 256 };
    float rRange[] = { 0, 256 };
    const float *ranges[] = { bRange, gRange, rRange };

    if (cfg.grayscale_sequence) {
        dims = 1;
    }

    // (TRAIN) BUILD THE MODEL
    if (new_model) {
        cv::calcHist(&patch, imgCount, channels, bg_mask_new, bg_hist, dims, sizes, ranges);
        cv::calcHist(&patch, imgCount, channels, fg_mask_new, fg_hist, dims, sizes, ranges);

        int bgtotal = cv::countNonZero(bg_mask_new);
        (bgtotal == 0) && (bgtotal = 1);
        bg_hist = bg_hist / bgtotal;

        int fgtotal = cv::countNonZero(fg_mask_new);
        (fgtotal == 0) && (fgtotal = 1);
        fg_hist = fg_hist / fgtotal;
    } else { // update the model
        cv::MatND bg_hist_tmp;
        cv::MatND fg_hist_tmp;

        cv::calcHist(&patch, imgCount, channels, bg_mask_new, bg_hist_tmp, dims, sizes, ranges);
        cv::calcHist(&patch, imgCount, channels, fg_mask_new, fg_hist_tmp, dims, sizes, ranges);

        int bgtotal = cv::countNonZero(bg_mask_new);
        (bgtotal == 0) && (bgtotal = 1);
        bg_hist_tmp = bg_hist_tmp / bgtotal;

        int fgtotal = cv::countNonZero(fg_mask_new);
        (fgtotal == 0) && (fgtotal = 1);
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
        p1[i] = 0.5*(1 - cos(CV_2PI*i / (sz.width - 1)));

    for (int i = 0; i < sz.height; ++i)
        p2[i] = 0.5*(1 - cos(CV_2PI*i / (sz.height - 1)));

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
    std::vector<int> x, y;

    for (int i = xr.start; i <= xr.end; i++)
        x.push_back(i);
    for (int i = yr.start; i <= yr.end; i++)
        y.push_back(i);

    repeat(cv::Mat(x).t(), y.size(), 1, outX);
    repeat(cv::Mat(y), 1, x.size(), outY);
}

///
/// \brief STAPLE_TRACKER::gaussianResponse
/// GAUSSIANRESPONSE create the (fixed) target response of the correlation filter response
/// \param rect_size
/// \param sigma
/// \param output
///
void STAPLE_TRACKER::gaussianResponse(cv::Size rect_size, double sigma, cv::Mat &output)
{
    // half = floor((rect_size-1) / 2);
    // i_range = -half(1):half(1);
    // j_range = -half(2):half(2);
    // [i, j] = ndgrid(i_range, j_range);
    cv::Size half;

    half.width = floor((rect_size.width - 1) / 2);
    half.height = floor((rect_size.height - 1) / 2);

    cv::Range i_range(-half.width, rect_size.width - (1 + half.width));
    cv::Range j_range(-half.height, rect_size.height - (1 + half.height));
    cv::Mat i, j;

    meshgrid(i_range, j_range, i, j);

    // i_mod_range = mod_one(i_range, rect_size(1));
    // j_mod_range = mod_one(j_range, rect_size(2));

    std::vector<int> i_mod_range, j_mod_range;

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

    float *OUTPUT = new float[rect_size.width*rect_size.height*2];

    for (int ii = 0; ii < rect_size.width; ii++)
        for (int jj = 0; jj < rect_size.height; jj++) {
            int i_idx = i_mod_range[ii];
            int j_idx = j_mod_range[jj];

            assert((i_idx < rect_size.width) && (j_idx < rect_size.height));

            OUTPUT[j_idx*rect_size.width*2+i_idx*2] = exp(-(i.at<int>(jj, ii)*i.at<int>(jj, ii) + j.at<int>(jj, ii)*j.at<int>(jj, ii)) / (2 * sigma*sigma));
            OUTPUT[j_idx*rect_size.width*2+i_idx*2+1] = 0;
        }

    output = cv::Mat(rect_size.height, rect_size.width, CV_32FC2, OUTPUT).clone();

    delete[] OUTPUT;
}

///
/// \brief STAPLE_TRACKER::tracker_staple_initialize
/// \param im
/// \param region
///
void STAPLE_TRACKER::Initialize(const cv::Mat &im, cv::Rect region)
{
    int n = im.channels();

    if (n == 1) {
        cfg.grayscale_sequence = true;
    }

    // xxx: only support 3 channels, TODO: fix updateHistModel
    //assert(!cfg.grayscale_sequence);

    cfg.init_pos.x = region.x + region.width / 2.0;
    cfg.init_pos.y = region.y + region.height / 2.0;

    cfg.target_sz.width = region.width;
    cfg.target_sz.height = region.height;

    initializeAllAreas(im);

    pos = cfg.init_pos;
    target_sz = cfg.target_sz;

    // patch of the target + padding
    cv::Mat patch_padded;

    getSubwindow(im, pos, norm_bg_area, bg_area, patch_padded);

    // initialize hist model
    updateHistModel(true, patch_padded);

    CalculateHann(cf_response_size, hann_window);

    // gaussian-shaped desired response, centred in (1,1)
    // bandwidth proportional to target size
    double output_sigma
            = sqrt(norm_target_sz.width * norm_target_sz.height) * cfg.output_sigma_factor / cfg.hog_cell_size;

    cv::Mat y;
    gaussianResponse(cf_response_size, output_sigma, y);
    cv::dft(y, yf);

    // SCALE ADAPTATION INITIALIZATION
    if (cfg.scale_adaptation) {
        // Code from DSST
        scale_factor = 1;
        base_target_sz = target_sz; // xxx
        float scale_sigma = sqrt(cfg.num_scales) * cfg.scale_sigma_factor;
        float *SS = new float[cfg.num_scales*2];
        float *YSBUF = SS;

        for (int i = 0; i < cfg.num_scales; i++) {
            SS[i*2] = (i+1) - ceil(cfg.num_scales/2.0);
            YSBUF[i*2] = exp(-0.5 * (SS[i*2]*SS[i*2]) / (scale_sigma*scale_sigma));
            YSBUF[i*2+1] = 0.0;
            // SS = (1:p.num_scales) - ceil(p.num_scales/2);
            // ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
        }

        cv::Mat ys = cv::Mat(1, cfg.num_scales, CV_32FC2, YSBUF).clone();
        delete[] SS;

        cv::dft(ys, ysf, cv::DFT_ROWS);
        //std::cout << ysf << std::endl;

        float *SWBUFF = new float[cfg.num_scales];

        if (cfg.num_scales % 2 == 0) {
            for (int i = 0; i < cfg.num_scales + 1; ++i) {
                if (i > 0) {
                    SWBUFF[i - 1] = 0.5*(1 - cos(CV_2PI*i / (cfg.num_scales + 1 - 1)));
                }
            }
        } else {
            for (int i = 0; i < cfg.num_scales; ++i)
                SWBUFF[i] = 0.5*(1 - cos(CV_2PI*i / (cfg.num_scales - 1)));
        }

        scale_window = cv::Mat(1, cfg.num_scales, CV_32FC1, SWBUFF).clone();
        delete[] SWBUFF;

        float *SFBUF = new float[cfg.num_scales];

        for (int i = 0; i < cfg.num_scales; i++) {
            SFBUF[i] = pow(cfg.scale_step, (ceil(cfg.num_scales/2.0)  - (i+1)));
        }

        scale_factors = cv::Mat(1, cfg.num_scales, CV_32FC1, SFBUF).clone();
        delete[] SFBUF;

        //std::cout << scale_factors << std::endl;

        //ss = 1:p.num_scales;
        //scale_factors = p.scale_step.^(ceil(p.num_scales/2) - ss);

        if ((cfg.scale_model_factor*cfg.scale_model_factor) * (norm_target_sz.width*norm_target_sz.height) > cfg.scale_model_max_area) {
            cfg.scale_model_factor = sqrt(cfg.scale_model_max_area/(norm_target_sz.width*norm_target_sz.height));
        }

        //std::cout << cfg.scale_model_factor << std::endl;

        scale_model_sz.width = floor(norm_target_sz.width * cfg.scale_model_factor);
        scale_model_sz.height = floor(norm_target_sz.height * cfg.scale_model_factor);
        //scale_model_sz = floor(p.norm_target_sz * p.scale_model_factor);

        //std::cout << scale_model_sz << std::endl;

        cv::Size sz = im.size();
        // find maximum and minimum scales
        min_scale_factor = pow(cfg.scale_step, ceil(log(std::max(5.0/bg_area.width, 5.0/bg_area.height))/log(cfg.scale_step)));
        max_scale_factor = pow(cfg.scale_step, floor(log(std::min(sz.width/(float)target_sz.width, sz.height/(float)target_sz.height))/log(cfg.scale_step)));
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
void STAPLE_TRACKER::getFeatureMap(cv::Mat &im_patch, const char *feature_type, cv::MatND &output)
{
    assert(!strcmp(feature_type, "fhog"));

    // allocate space
#if 0
    cv::Mat tmp_image;
    im_patch.convertTo(tmp_image, CV_32FC1);
    fhog28(output, tmp_image, cfg.hog_cell_size, 9);
#else
    fhog28(output, im_patch, cfg.hog_cell_size, 9);
#endif
    int w = cf_response_size.width;
    int h = cf_response_size.height;

    // hog28 already generate this matrix of (w,h,28)
    // out = zeros(h, w, 28, 'single');
    // out(:,:,2:28) = temp(:,:,1:27);

    cv::Mat new_im_patch;

    if (cfg.hog_cell_size > 1) {
        cv::Size newsz(w, h);

        mexResize(im_patch, new_im_patch, newsz, "auto");
    } else {
        new_im_patch = im_patch;
    }

    cv::Mat grayimg;

    if (new_im_patch.channels() > 1) {
        cv::cvtColor(new_im_patch, grayimg, cv::COLOR_BGR2GRAY);
    } else {
        grayimg = new_im_patch;
    }

    // out(:,:,1) = single(im_patch)/255 - 0.5;

    cv::Mat grayimgf;

    grayimg.convertTo(grayimgf, CV_32FC1);
    grayimgf /= 255.0;
    grayimgf -= 0.5;

    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++) {
            typedef cv::Vec<float, 28> Vecf28;

            // apply Hann window
            output.at<Vecf28>(j, i) = output.at<Vecf28>(j, i) * hann_window.at<float>(j, i);
            output.at<Vecf28>(j, i)[0] = grayimgf.at<float>(j, i) * hann_window.at<float>(j, i);
        }
}

///
/// \brief matsplit
/// \param xt
/// \param xtsplit
///
void matsplit(const cv::MatND &xt, std::vector<cv::Mat> &xtsplit)
{
    cv::Size sz = xt.size();
    int w = sz.width;
    int h = sz.height;
    int cn = xt.channels();

    assert(cn == 28);

    float *XT = new float[w*h*2];

    for (int k = 0; k < cn; k++) {
        int count = 0;

        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++) {
                typedef cv::Vec<float, 28> Vecf28;

                Vecf28 p = xt.at<Vecf28>(i, j); // by rows

                XT[count] = p[k];
                count++;
                XT[count] = 0.0;
                count++;
            }

        cv::Mat dim = cv::Mat(h, w, CV_32FC2, XT).clone();
        xtsplit.push_back(dim);
    }

    delete[] XT;
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
    sz.width = fmax(sz.width, 2);
    sz.height = fmax(sz.height, 2);

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
    float *OUTPUT = NULL;
    int w = 0;
    int h = 0;
    int ch = 0;
    int total = 0;

    for (int s = 0; s < cfg.num_scales; s++) {
        cv::Size_<float> patch_sz;

        patch_sz.width = floor(base_target_sz.width * scale_factor * scale_factors.at<float>(s));
        patch_sz.height = floor(base_target_sz.height * scale_factor * scale_factors.at<float>(s));

        cv::Mat im_patch_resized;
        getSubwindowFloor(im, centerCoor, scale_model_sz, patch_sz, im_patch_resized);

        // extract scale features
        cv::MatND temp;
        fhog31(temp, im_patch_resized, cfg.hog_cell_size, 9);

        if (s == 0) {
            cv::Size sz = temp.size();

            w = sz.width;
            h = sz.height;
            ch = temp.channels();
            total = w*h*ch;

            OUTPUT = new float[cfg.num_scales*total*2](); // xxx
        }

        cv::Size tempsz = temp.size();
        int tempw = tempsz.width;
        int temph = tempsz.height;
        int tempch = temp.channels();

        int count = 0;

        // window
        for (int i = 0; i < tempw; i++)
            for (int j = 0; j < temph; j++)
                for (int k = 0; k < tempch; k++) {
                    int off = j*tempw*ch+i*tempch+k;

                    OUTPUT[(count*cfg.num_scales + s)*2 + 0] = ((float *)temp.data)[off] * scale_window.at<float>(s);
                    OUTPUT[(count*cfg.num_scales + s)*2 + 1] = 0.0;
                    count++;
                }
    }

    output = cv::Mat(total, cfg.num_scales, CV_32FC2, OUTPUT).clone();

    delete[] OUTPUT;
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
    getFeatureMap(im_patch_bg, cfg.feature_type, xt);

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

        cv::Size sz = xt.size();
        int w = sz.width;
        int h = sz.height;
        float area = cf_response_size.width*cf_response_size.height;

        float *DIM = new float[w*h*2];

        for (int ch = 0; ch < xt.channels(); ch++) {
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++) {
                    cv::Vec2f pXTF = xtf[ch].at<cv::Vec2f>(i,j);
                    cv::Vec2f pYF = yf.at<cv::Vec2f>(i,j);

                    DIM[i*w*2+j*2+0] = (pYF[1]*pXTF[1] + pYF[0]*pXTF[0]) / area;
                    DIM[i*w*2+j*2+1] = (pYF[0]*pXTF[1] - pYF[1]*pXTF[0]) / area;
                }

            cv::Mat dim = cv::Mat(h, w, CV_32FC2, DIM).clone();

            new_hf_num.push_back(dim);
        }

        delete[] DIM;

        float *DIM1 = new float[w*h];

        for (int ch = 0; ch < xt.channels(); ch++) {
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++) {
                    cv::Vec2f pXTF = xtf[ch].at<cv::Vec2f>(i,j);

                    DIM1[i*w+j] = (pXTF[0]*pXTF[0] + pXTF[1]*pXTF[1]) / area;
                }

            cv::Mat dim = cv::Mat(h, w, CV_32FC1, DIM1).clone();

            new_hf_den.push_back(dim);
        }

        delete[] DIM1;

        if (first) {
            // first frame, train with a single image
            hf_den.assign(new_hf_den.begin(), new_hf_den.end());
            hf_num.assign(new_hf_num.begin(), new_hf_num.end());
        } else {
            // subsequent frames, update the model by linear interpolation
            for (int ch =  0; ch < xt.channels(); ch++) {
                hf_den[ch] = (1 - cfg.learning_rate_cf) * hf_den[ch] + cfg.learning_rate_cf * new_hf_den[ch];
                hf_num[ch] = (1 - cfg.learning_rate_cf) * hf_num[ch] + cfg.learning_rate_cf * new_hf_num[ch];
            }

            updateHistModel(false, im_patch_bg, cfg.learning_rate_pwp);

            // BG/FG MODEL UPDATE
            // patch of the target + padding
            // [bg_hist, fg_hist] = updateHistModel(new_pwp_model, im_patch_bg, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence, bg_hist, fg_hist, p.learning_rate_pwp);
        }
    }

    // SCALE UPDATE
    if (cfg.scale_adaptation) {
        cv::Mat im_patch_scale;

        getScaleSubwindow(im, pos, im_patch_scale);

        cv::Mat xsf;
        cv::dft(im_patch_scale, xsf, cv::DFT_ROWS);

        // new_sf_num = bsxfun(@times, ysf, conj(xsf));
        // new_sf_den = sum(xsf .* conj(xsf), 1);

        cv::Mat new_sf_num;
        cv::Mat new_sf_den;

        cv::Size sz = xsf.size();
        int w = sz.width;
        int h = sz.height;

        float *NEW_SF_NUM = new float[w*h*2];

        for (int i = 0; i < h; i++) // xxx
            for (int j = 0; j < w; j++) {
                cv::Vec2f pXSF = xsf.at<cv::Vec2f>(i,j);
                cv::Vec2f pYSF = ysf.at<cv::Vec2f>(j);

                NEW_SF_NUM[i*w*2+j*2+0] = (pYSF[1]*pXSF[1] + pYSF[0]*pXSF[0]);
                NEW_SF_NUM[i*w*2+j*2+1] = (pYSF[1]*pXSF[0] - pYSF[0]*pXSF[1]);
            }

        new_sf_num = cv::Mat(h, w, CV_32FC2, NEW_SF_NUM).clone();
        delete[] NEW_SF_NUM;

        float *NEW_SF_DEN = new float[w]();

        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++) {
                cv::Vec2f pXSF = xsf.at<cv::Vec2f>(i,j);

                NEW_SF_DEN[j] += (pXSF[0]*pXSF[0] + pXSF[1]*pXSF[1]);
            }

        new_sf_den = cv::Mat(1, w, CV_32FC1, NEW_SF_DEN).clone();
        delete[] NEW_SF_DEN;

        if (first) {
            // first frame, train with a single image
            new_sf_den.copyTo(sf_den);
            new_sf_num.copyTo(sf_num);
        } else {
            sf_den = (1 - cfg.learning_rate_scale) * sf_den + cfg.learning_rate_scale * new_sf_den;
            sf_num = (1 - cfg.learning_rate_scale) * sf_num + cfg.learning_rate_scale * new_sf_num;
        }
    }

    // update bbox position
    if (first) {
        rect_position.x = pos.x - target_sz.width/2;
        rect_position.y = pos.y - target_sz.height/2;
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
    cv::Size sz = complex.size();
    int w = sz.width;
    int h = sz.height;
    int cn = complex.channels();
    float *REAL = new float[w*h];

    for (int k = 0; k < cn; k++) {
        int count = 0;

        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++) {
                cv::Vec2f p = complex.at<cv::Vec2f>(i, j); // by rows

                REAL[count] = p[k];
                count++;
            }

        break;
    }

    cv::Mat real = cv::Mat(h, w, CV_32FC1, REAL).clone();
    delete[] REAL;

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
    cv::Size sz = response_cf.size();
    int w = sz.width;
    int h = sz.height;

    // newh and neww must be odd, as we want an exact center
    assert(((response_size.width % 2) == 1) && ((response_size.height % 2) == 1));

    int half_width = floor(response_size.width / 2);
    int half_height = floor(response_size.height / 2);

    cv::Range i_range(-half_width, response_size.width - (1 + half_width));
    cv::Range j_range(-half_height, response_size.height - (1 + half_height));

    std::vector<int> i_mod_range, j_mod_range;

    for (int k = i_range.start; k <= i_range.end; k++) {
        int val = (k - 1 + w) % w;
        i_mod_range.push_back(val);
    }

    for (int k = j_range.start; k <= j_range.end; k++) {
        int val = (k - 1 + h) % h;
        j_mod_range.push_back(val);
    }

    float *OUTPUT = new float[response_size.width*response_size.height];

    for (int i = 0; i < response_size.width; i++)
        for (int j = 0; j < response_size.height; j++) {
            int i_idx = i_mod_range[i];
            int j_idx = j_mod_range[j];

            assert((i_idx < w) && (j_idx < h));

            OUTPUT[j*response_size.width+i] = response_cf.at<float>(j_idx,i_idx);
        }

    output = cv::Mat(response_size.height, response_size.width, CV_32FC1, OUTPUT).clone();
    delete[] OUTPUT;
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
    cv::Size sz = patch.size();
    int h = sz.height;
    int w = sz.width;
    //int d = patch.channels();

    // figure out which bin each pixel falls into
    int bin_width = 256 / cfg.n_bins;

    // convert image to d channels array
    //patch_array = reshape(double(patch), w*h, d);

    float probg;
    float profg;
    float *P_O = new float[w*h];

    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++) {
            if (!cfg.grayscale_sequence) {
                cv::Vec3b p = patch.at<cv::Vec3b>(j,i);

                int b1 = floor(p[0] / bin_width);
                int b2 = floor(p[1] / bin_width);
                int b3 = floor(p[2] / bin_width);

                float* histd;

                histd = (float*)bg_hist.data;
                probg = histd[b1*cfg.n_bins*cfg.n_bins + b2*cfg.n_bins + b3];

                histd = (float*)fg_hist.data;
                profg = histd[b1*cfg.n_bins*cfg.n_bins + b2*cfg.n_bins + b3];

                // xxx
                P_O[j*w+i] = profg / (profg + probg);

                isnan(P_O[j*w+i]) && (P_O[j*w+i] = 0.0);

                // (TODO) in theory it should be at 0.5 (unseen colors shoud have max entropy)
                //likelihood_map(isnan(likelihood_map)) = 0;
            } else {
                int b = patch.at<uchar>(j,i);

                float* histd;

                histd = (float*)bg_hist.data;
                probg = histd[b];

                histd = (float*)fg_hist.data;
                profg = histd[b];

                // xxx
                P_O[j*w+i] = profg / (profg + probg);

                isnan(P_O[j*w+i]) && (P_O[j*w+i] = 0.0);

                // (TODO) in theory it should be at 0.5 (unseen colors shoud have max entropy)
                //likelihood_map(isnan(likelihood_map)) = 0;
            }
        }

    // to which bin each pixel (for all d channels) belongs to
    //bin_indices = floor(patch_array/bin_width) + 1;

    // Get pixel-wise posteriors (PwP)
    // P_bg = getP(bg_hist, h, w, bin_indices, grayscale_sequence);
    // P_fg = getP(fg_hist, h, w, bin_indices, grayscale_sequence);

    // Object-likelihood map
    //P_O = P_fg ./ (P_fg + P_bg);

    output = cv::Mat(h, w, CV_32FC1, P_O).clone();
    delete[] P_O;
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
    cv::Size sz = object_likelihood.size();
    int h = sz.height;
    int w = sz.width;
    int n1 = w - m.width + 1;
    int n2 = h - m.height + 1;
    int area = m.width * m.height;

    cv::Mat temp;

    // integral images
    cv::integral(object_likelihood, temp);

    float *CENTER_LIKELIHOOD = new float[n1*n2];

    for (int i = 0; i < n1; i++)
        for (int j = 0; j < n2; j++) {
            CENTER_LIKELIHOOD[j*n1 + i]
                    = (temp.at<double>(j, i) + temp.at<double>(j+m.height, i+m.width) - temp.at<double>(j, i+m.width) - temp.at<double>(j+m.height, i)) / area;
        }

    // SAT = integralImage(object_likelihood);
    // i = 1:n1;
    // j = 1:n2;
    // center_likelihood = (SAT(i,j) + SAT(i+m(1), j+m(2)) - SAT(i+m(1), j) - SAT(i, j+m(2))) / prod(m);

    center_likelihood = cv::Mat(n2, n1, CV_32FC1, CENTER_LIKELIHOOD).clone();
    delete[] CENTER_LIKELIHOOD;
}

///
/// \brief STAPLE_TRACKER::mergeResponses
/// \param response_cf
/// \param response_pwp
/// \param response
///
void STAPLE_TRACKER::mergeResponses(const cv::Mat &response_cf, const cv::Mat &response_pwp, cv::Mat &response)
{
    double alpha = cfg.merge_factor;
    //const char *merge_method = cfg.merge_method;

    // MERGERESPONSES interpolates the two responses with the hyperparameter ALPHA
    response = (1 - alpha) * response_cf + alpha * response_pwp;

    // response = (1 - alpha) * response_cf + alpha * response_pwp;
}

///
/// \brief STAPLE_TRACKER::tracker_staple_update
/// TESTING step
/// \param im
/// \return
///
cv::Rect STAPLE_TRACKER::Update(const cv::Mat &im)
{
    // extract patch of size bg_area and resize to norm_bg_area
    cv::Mat im_patch_cf;
    getSubwindow(im, pos, norm_bg_area, bg_area, im_patch_cf);

    cv::Size pwp_search_area;

    pwp_search_area.width = round(norm_pwp_search_area.width / area_resize_factor);
    pwp_search_area.height = round(norm_pwp_search_area.height / area_resize_factor);

    // extract patch of size pwp_search_area and resize to norm_pwp_search_area
    getSubwindow(im, pos, norm_pwp_search_area, pwp_search_area, im_patch_pwp);

    // compute feature map
    cv::MatND xt_windowed;
    getFeatureMap(im_patch_cf, cfg.feature_type, xt_windowed);

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

    std::vector<cv::Mat> hf;
    cv::Size sz = xt_windowed.size();
    int w = sz.width;
    int h = sz.height;

    // Correlation between filter and test patch gives the response
    // Solve diagonal system per pixel.
    if (cfg.den_per_channel) {
        float *DIM = new float[w*h*2];

        for (int ch = 0; ch < xt_windowed.channels(); ch++) {
            for (int i = 0; i < w; i++)
                for (int j = 0; j < h; j++) {
                    cv::Vec2f p = hf_num[ch].at<cv::Vec2f>(j,i);

                    DIM[j*w*2+i*2+0] = p[0] / (hf_den[ch].at<float>(j,i) + cfg.lambda);
                    DIM[j*w*2+i*2+1] = p[1] / (hf_den[ch].at<float>(j,i) + cfg.lambda);
                }

            cv::Mat dim = cv::Mat(h, w, CV_32FC2, DIM).clone();

            hf.push_back(dim);
        }

        delete[] DIM;
    } else {
        //hf = bsxfun(@rdivide, hf_num, sum(hf_den, 3)+p.lambda);

        float *DIM1 = new float[w*h];

        for (int i = 0; i < w; i++)
            for (int j = 0; j < h; j++) {
                float sum=0.0;

                for (int ch = 0; ch < xt_windowed.channels(); ch++) {
                    sum += hf_den[ch].at<float>(j,i);
                }

                DIM1[j*w+i] = sum + cfg.lambda;
            }

        float *DIM = new float[w*h*2];

        for (int ch = 0; ch < xt_windowed.channels(); ch++) {
            for (int i = 0; i < w; i++)
                for (int j = 0; j < h; j++) {
                    cv::Vec2f p = hf_num[ch].at<cv::Vec2f>(j,i);

                    DIM[j*w*2+i*2+0] = p[0] / DIM1[j*w+i];
                    DIM[j*w*2+i*2+1] = p[1] / DIM1[j*w+i];
                }

            cv::Mat dim = cv::Mat(h, w, CV_32FC2, DIM).clone();

            hf.push_back(dim);
        }

        delete[] DIM;

        delete[] DIM1;
    }

    float *RESPONSE_CF = new float[w*h*2];

    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++) {
            float sum=0.0;
            float sumi=0.0;

            for (size_t ch = 0; ch < hf.size(); ch++) {
                cv::Vec2f pHF = hf[ch].at<cv::Vec2f>(j,i);
                cv::Vec2f pXTF = xtf[ch].at<cv::Vec2f>(j,i);

                sum += (pHF[0]*pXTF[0] + pHF[1]*pXTF[1]);
                sumi += (pHF[0]*pXTF[1] - pHF[1]*pXTF[0]);

                // assert(norm(imag(x(:))) <= 1e-5 * norm(real(x(:))));
            }

            RESPONSE_CF[j*w*2+i*2+0] = sum;
            RESPONSE_CF[j*w*2+i*2+1] = sumi;
        }

    cv::Mat response_cff = cv::Mat(h, w, CV_32FC2, RESPONSE_CF).clone();
    delete[] RESPONSE_CF;

    cv::Mat response_cfi;
    cv::dft(response_cff, response_cfi, cv::DFT_SCALE|cv::DFT_INVERSE);
    cv::Mat response_cf = ensure_real(response_cfi);

    // response_cf = ensure_real(ifft2(sum(conj(hf) .* xtf, 3)));

    // Crop square search region (in feature pixels).
    cv::Size newsz = norm_delta_area;
    newsz.width = floor(newsz.width / cfg.hog_cell_size);
    newsz.height = floor(newsz.height / cfg.hog_cell_size);

    (newsz.width % 2 == 0) && (newsz.width -= 1);
    (newsz.height % 2 == 0) && (newsz.height -= 1);

    cropFilterResponse(response_cf, newsz, response_cf);

    if (cfg.hog_cell_size > 1) {
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

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;

    cv::minMaxLoc(response, &minVal, &maxVal, &minLoc, &maxLoc);
    //[row, col] = find(response == max(response(:)), 1);

    //std::cout << maxLoc.x << " " << maxLoc.y << std::endl;

    float centerx = (1 + norm_delta_area.width) / 2 - 1;
    float centery = (1 + norm_delta_area.height) / 2 - 1;

    pos.x += (maxLoc.x - centerx) / area_resize_factor;
    pos.y += (maxLoc.y - centery) / area_resize_factor;

    // Report current location
    cv::Rect_<float> location;

    location.x = pos.x - target_sz.width/2.0;
    location.y = pos.y - target_sz.height/2.0;
    location.width = target_sz.width;
    location.height = target_sz.height;

    //std::cout << location << std::endl;

    // center = (1+p.norm_delta_area) / 2;
    // pos = pos + ([row, col] - center) / area_resize_factor;
    // rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];

    // SCALE SPACE SEARCH
    if (cfg.scale_adaptation) {
        cv::Mat im_patch_scale;

        getScaleSubwindow(im, pos, im_patch_scale);

        cv::Mat xsf;
        cv::dft(im_patch_scale, xsf, cv::DFT_ROWS);

        // im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor * scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size);
        // xsf = fft(im_patch_scale,[],2);

        cv::Size sz = xsf.size();
        int w = sz.width;
        int h = sz.height;
        float *SCALE_RESPONSEF = new float[w*2]();

        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                cv::Vec2f pXSF = xsf.at<cv::Vec2f>(j,i);
                cv::Vec2f pXSFNUM = sf_num.at<cv::Vec2f>(j,i);

                SCALE_RESPONSEF[i*2] += (pXSFNUM[0]*pXSF[0] - pXSFNUM[1]*pXSF[1]) / (sf_den.at<float>(i) + cfg.lambda);
                SCALE_RESPONSEF[i*2 + 1] += (pXSFNUM[0]*pXSF[1] + pXSFNUM[1]*pXSF[0]) / (sf_den.at<float>(i) + cfg.lambda);
            }
        }

        cv::Mat scale_responsef = cv::Mat(1, w, CV_32FC2, SCALE_RESPONSEF).clone();
        delete[] SCALE_RESPONSEF;

        cv::Mat scale_response;

        cv::dft(scale_responsef, scale_response, cv::DFT_SCALE|cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);

        //scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + p.lambda) ));

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;

        cv::minMaxLoc(scale_response, &minVal, &maxVal, &minLoc, &maxLoc);

        //recovered_scale = ind2sub(size(scale_response),find(scale_response == max(scale_response(:)), 1));

        int recovered_scale =  maxLoc.x;

        // set the scale
        scale_factor = scale_factor * scale_factors.at<float>(recovered_scale);

        if (scale_factor < min_scale_factor) {
            scale_factor = min_scale_factor;
        } else if (scale_factor > max_scale_factor) {
            scale_factor = max_scale_factor;
        }

        // use new scale to update bboxes for target, filter, bg and fg models
        target_sz.width = round(base_target_sz.width * scale_factor);
        target_sz.height = round(base_target_sz.height * scale_factor);

        float avg_dim = (target_sz.width + target_sz.height)/2.0;

        bg_area.width= round(target_sz.width + avg_dim);
        bg_area.height = round(target_sz.height + avg_dim);

        (bg_area.width > im.cols) && (bg_area.width = im.cols - 1);
        (bg_area.height > im.rows) && (bg_area.height = im.rows - 1);

        bg_area.width = bg_area.width - (bg_area.width - target_sz.width) % 2;
        bg_area.height = bg_area.height - (bg_area.height - target_sz.height) % 2;

        fg_area.width = round(target_sz.width - avg_dim * cfg.inner_padding);
        fg_area.height = round(target_sz.height - avg_dim * cfg.inner_padding);

        fg_area.width = fg_area.width + int(bg_area.width - fg_area.width) % 2;
        fg_area.height = fg_area.height + int(bg_area.height - fg_area.height) % 2;

        // Compute the rectangle with (or close to) params.fixed_area and
        // same aspect ratio as the target bboxgetScaleSubwindow
        area_resize_factor = sqrt(cfg.fixed_area / (float)(bg_area.width * bg_area.height));
    }

    return location;
}
