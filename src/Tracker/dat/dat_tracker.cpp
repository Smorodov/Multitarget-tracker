#include <opencv2/imgproc/imgproc_c.h>
#include "dat_tracker.hpp"

///
/// \brief DAT_TRACKER::DAT_TRACKER
///
DAT_TRACKER::DAT_TRACKER()
{
    cfg = default_parameters_dat(cfg);
}

///
/// \brief DAT_TRACKER::~DAT_TRACKER
///
DAT_TRACKER::~DAT_TRACKER()
{

}

///
/// \brief DAT_TRACKER::tracker_dat_initialize
/// \param I
/// \param region
///
void DAT_TRACKER::Initialize(const cv::Mat &im, cv::Rect region)
{
    double cx = region.x + double(region.width - 1) / 2.0;
    double cy = region.y + double(region.height - 1) / 2.0;
    double w = region.width;
    double h = region.height;

    cv::Point target_pos(round(cx),round(cy));
    cv::Size target_sz(round(w),round(h));

    scale_factor_ = std::min(1.0, round(10.0 * double(cfg.img_scale_target_diagonal) / cv::norm(cv::Point(target_sz.width,target_sz.height))) / 10.0);
    target_pos.x = target_pos.x * scale_factor_; target_pos.y = target_pos.y * scale_factor_;
    target_sz.width = target_sz.width * scale_factor_; target_sz.height = target_sz.height * scale_factor_;

    cv::Mat img;
    cv::resize(im, img, cv::Size(), scale_factor_, scale_factor_);
    switch (cfg.color_space) {
    case 1: //1rgb
		if (img.channels() == 1)
		{
			cv::cvtColor(img, img, CV_GRAY2BGR);
		}
        break;
    case 2: //2lab
        cv::cvtColor(img, img, CV_BGR2Lab);
        break;
    case 3: //3hsv
        cv::cvtColor(img, img, CV_BGR2HSV);
        break;
    case 4: //4gray
		if (img.channels() == 3)
		{
			cv::cvtColor(img, img, CV_BGR2GRAY);
		}
        break;
    default:
        std::cout << "int_variable does not equal any of the above cases" << std::endl;
    }
    cv::Size surr_sz(floor(cfg.surr_win_factor * target_sz.width),
                     floor(cfg.surr_win_factor * target_sz.height));
    cv::Rect surr_rect = pos2rect(target_pos, surr_sz, img.size());
    cv::Rect obj_rect_surr = pos2rect(target_pos, target_sz, img.size());
    obj_rect_surr.x -= surr_rect.x;
    obj_rect_surr.y -= surr_rect.y;
    cv::Mat surr_win = getSubwindow(img, target_pos, surr_sz);
    cv::Mat prob_map;
    getForegroundBackgroundProbs(surr_win, obj_rect_surr, cfg.num_bins, cfg.bin_mapping, prob_lut_, prob_map);

    prob_lut_distractor_ = prob_lut_.clone();
    prob_lut_masked_ = prob_lut_.clone();
    adaptive_threshold_ = getAdaptiveThreshold(prob_map, obj_rect_surr);

    target_pos_history_.push_back(cv::Point(target_pos.x / scale_factor_, target_pos.y / scale_factor_));
    target_sz_history_.push_back(cv::Size(target_sz.width / scale_factor_, target_sz.height / scale_factor_));
}

///
/// \brief DAT_TRACKER::tracker_dat_update
/// \param I
/// \param confidence
/// \return
///
cv::RotatedRect DAT_TRACKER::Update(const cv::Mat &im, float& confidence)
{
    confidence = 0;

    cv::Mat img_preprocessed;
    cv::resize(im, img_preprocessed, cv::Size(), scale_factor_, scale_factor_);
    cv::Mat img;
    switch (cfg.color_space) {
    case 1://1rgb
		if (img_preprocessed.channels() == 1)
		{
			cv::cvtColor(img_preprocessed, img, CV_GRAY2BGR);
		}
		else
		{
			img_preprocessed.copyTo(img);
		}
        break;
    case 2://2lab
        cv::cvtColor(img_preprocessed, img, CV_BGR2Lab);
        break;
    case 3://3hsv
        cv::cvtColor(img_preprocessed, img, CV_BGR2HSV);
        break;
    case 4://4gray
		if (img_preprocessed.channels() == 3)
		{
			cv::cvtColor(img_preprocessed, img, CV_BGR2GRAY);
		}
        break;
    default:
        std::cout << "int_variable does not equal any of the above cases" << std::endl;
    }
    cv::Point prev_pos = target_pos_history_.back();
    cv::Size prev_sz = target_sz_history_.back();

    if (cfg.motion_estimation_history_size > 0)
        prev_pos = prev_pos + getMotionPrediction(target_pos_history_, cfg.motion_estimation_history_size);

    cv::Point2f target_pos(prev_pos.x*scale_factor_, prev_pos.y*scale_factor_);
    cv::Size target_sz(prev_sz.width*scale_factor_, prev_sz.height*scale_factor_);

    cv::Size search_sz;
    search_sz.width = floor(target_sz.width + cfg.search_win_padding*std::max(target_sz.width, target_sz.height));
    search_sz.height = floor(target_sz.height + cfg.search_win_padding*std::max(target_sz.width, target_sz.height));
    cv::Rect search_rect = pos2rect(target_pos, search_sz);
    cv::Mat search_win, padded_search_win;
    getSubwindowMasked(img, target_pos, search_sz, search_win, padded_search_win);

    // Apply probability LUT
    cv::Mat pm_search = getForegroundProb(search_win, prob_lut_, cfg.bin_mapping);
    cv::Mat pm_search_dist;
    if (cfg.distractor_aware) {
        pm_search_dist = getForegroundProb(search_win, prob_lut_distractor_, cfg.bin_mapping);
        pm_search = (pm_search + pm_search_dist)/2.;
    }
    pm_search.setTo(0, padded_search_win);

    // Cosine / Hanning window
    cv::Mat cos_win = CalculateHann(search_sz);

    std::vector<cv::Rect> hypotheses;
    std::vector<double> vote_scores;
    std::vector<double> dist_scores;
    getNMSRects(pm_search, target_sz, cfg.nms_scale, cfg.nms_overlap,
                cfg.nms_score_factor, cos_win, cfg.nms_include_center_vote,
                hypotheses, vote_scores, dist_scores);

    std::vector<cv::Point2f> candidate_centers;
    std::vector<double> candidate_scores;
    for (size_t i = 0; i < hypotheses.size(); ++i) {
        candidate_centers.push_back(cv::Point2f(float(hypotheses[i].x) + float(hypotheses[i].width) / 2.,
                                                float(hypotheses[i].y) + float(hypotheses[i].height) / 2.));
        candidate_scores.push_back(vote_scores[i] * dist_scores[i]);
    }
    auto maxEl = std::max_element(candidate_scores.begin(), candidate_scores.end());
    size_t best_candidate = maxEl - candidate_scores.begin();
    confidence = *maxEl;

    target_pos = candidate_centers[best_candidate];

    std::vector<cv::Rect> distractors;
    std::vector<double> distractor_overlap;
    if (hypotheses.size() > 1) {
        distractors.clear();
        distractor_overlap.clear();
        cv::Rect target_rect = pos2rect(target_pos, target_sz, pm_search.size());
        for (size_t i = 0; i < hypotheses.size(); ++i){
            if (i != best_candidate) {
                distractors.push_back(hypotheses[i]);
                distractor_overlap.push_back(intersectionOverUnion(target_rect, distractors.back()));
            }
        }
    } else {
        distractors.clear();
        distractor_overlap.clear();
    }

    // Localization visualization
    if (cfg.show_figures)
	{
        cv::Mat pm_search_color;
        pm_search.convertTo(pm_search_color,CV_8UC1,255);
        applyColorMap(pm_search_color, pm_search_color, cv::COLORMAP_JET);
        for (size_t i = 0; i < hypotheses.size(); ++i){
            cv::rectangle(pm_search_color, hypotheses[i], cv::Scalar(0, 255, 255 * (i != best_candidate)), 2);
        }
#ifndef SILENT_WORK
        //cv::imshow("Search Window", pm_search_color);
        //cv::waitKey(1);
#endif
    }

    // Appearance update
    // Get current target position within full(possibly downscaled) image coorinates
    cv::Point2f target_pos_img;
    target_pos_img.x = target_pos.x + search_rect.x;
    target_pos_img.y = target_pos.y + search_rect.y;
    if (cfg.prob_lut_update_rate > 0) {
        // Extract surrounding region
        cv::Size surr_sz;
        surr_sz.width = floor(cfg.surr_win_factor * target_sz.width);
        surr_sz.height = floor(cfg.surr_win_factor * target_sz.height);
        cv::Rect surr_rect = pos2rect(target_pos_img, surr_sz, img.size());
        cv::Rect obj_rect_surr = pos2rect(target_pos_img, target_sz, img.size());
        obj_rect_surr.x -= surr_rect.x;
        obj_rect_surr.y -= surr_rect.y;

        cv::Mat surr_win = getSubwindow(img, target_pos_img, surr_sz);

        cv::Mat prob_lut_bg;
        getForegroundBackgroundProbs(surr_win, obj_rect_surr, cfg.num_bins, prob_lut_bg);

        cv::Mat prob_map;
        if (cfg.distractor_aware) {
            // Handle distractors
            if (distractors.size() > 1) {
                cv::Rect obj_rect = pos2rect(target_pos, target_sz, search_win.size());
                cv::Mat prob_lut_dist = getForegroundDistractorProbs(search_win, obj_rect, distractors, cfg.num_bins);

                prob_lut_distractor_ = (1 - cfg.prob_lut_update_rate) * prob_lut_distractor_ + cfg.prob_lut_update_rate * prob_lut_dist;
            }
            else {
                // If there are no distractors, trigger decay of distractor LUT
                prob_lut_distractor_ = (1 - cfg.prob_lut_update_rate) * prob_lut_distractor_ + cfg.prob_lut_update_rate * prob_lut_bg;
            }

            // Only update if distractors are not overlapping too much
            if (distractors.empty() || (*max_element(distractor_overlap.begin(), distractor_overlap.end()) < 0.1)) {
                prob_lut_ = (1 - cfg.prob_lut_update_rate) * prob_lut_ + cfg.prob_lut_update_rate * prob_lut_bg;
            }

            prob_map = getForegroundProb(surr_win, prob_lut_, cfg.bin_mapping);
            cv::Mat dist_map = getForegroundProb(surr_win, prob_lut_distractor_, cfg.bin_mapping);
            prob_map = .5 * prob_map + .5 * dist_map;
        }
        else { // No distractor - awareness
            prob_lut_ = (1 - cfg.prob_lut_update_rate) * prob_lut_ + cfg.prob_lut_update_rate * prob_lut_bg;
            prob_map = getForegroundProb(surr_win, prob_lut_, cfg.bin_mapping);
        }
        // Update adaptive threshold
        adaptive_threshold_ = getAdaptiveThreshold(prob_map, obj_rect_surr);
    }

    // Store current location
    target_pos.x = target_pos.x + search_rect.x ;
    target_pos.y = target_pos.y + search_rect.y;
    cv::Point target_pos_original;
    cv::Size target_sz_original;
    target_pos_original.x = target_pos.x / scale_factor_;
    target_pos_original.y = target_pos.y / scale_factor_;
    target_sz_original.width = target_sz.width / scale_factor_;
    target_sz_original.height = target_sz.height / scale_factor_;

    target_pos_history_.push_back(target_pos_original);
    target_sz_history_.push_back(target_sz_original);

    // Report current location
    cv::Rect location = pos2rect(target_pos_history_.back(), target_sz_history_.back(), im.size());

    // Adapt image scale factor
    scale_factor_ = std::min(1.0, round(10.0 * double(cfg.img_scale_target_diagonal) / cv::norm(cv::Point(target_sz_original.width, target_sz_original.height))) / 10.0);
    
	return cv::RotatedRect(cv::Point2f(location.x + 0.5f * location.width, location.y + 0.5f * location.height),
		cv::Size2f(location.width, location.height), 0.f);
}

///
/// \brief DAT_TRACKER::Train
/// \param im
/// \param first
///
void DAT_TRACKER::Train(const cv::Mat &/*im*/, bool /*first*/)
{

}

///
/// \brief DAT_TRACKER::getNMSRects
/// \param prob_map
/// \param obj_sz
/// \param scale
/// \param overlap
/// \param score_frac
/// \param dist_map
/// \param include_inner
/// \param top_rects
/// \param top_vote_scores
/// \param top_dist_scores
///
void DAT_TRACKER::getNMSRects(cv::Mat prob_map, cv::Size obj_sz, double scale,
                              double overlap, double score_frac, cv::Mat dist_map, bool include_inner,
                              std::vector<cv::Rect> &top_rects, std::vector<double> &top_vote_scores, std::vector<double> &top_dist_scores){
    int height = prob_map.rows;
    int width = prob_map.cols;
    cv::Size rect_sz(floor(obj_sz.width * scale), floor(obj_sz.height * scale));

    int stepx = std::max(1, int(round(rect_sz.width * (1.0 - overlap))));
    int stepy = std::max(1, int(round(rect_sz.height * (1.0 - overlap))));

    std::vector<int> posx, posy;
    for (int i = 0; i <= (width -1 - rect_sz.width); i += stepx)
    {
        posx.push_back(i);
    }
    for (int i = 0; i <= (height -1 - rect_sz.height); i += stepy)
    {
        posy.push_back(i);
    }
    cv::Mat xgv(posx); cv::Mat ygv(posy); cv::Mat x; cv::Mat y;
    cv::repeat(xgv.reshape(1, 1), ygv.total(), 1, x);
    cv::repeat(ygv.reshape(1, 1).t(), 1, xgv.total(), y);

    cv::Mat r = x + rect_sz.width;;
    cv::Mat b = y + rect_sz.height;
    r.setTo(width-1, r > (width-1));
    b.setTo(height-1, b > (height-1));

    std::vector<cv::Rect> boxes;
    int n = x.rows*x.cols;
    int *p_x = x.ptr<int>(0);
    int *p_y = y.ptr<int>(0);
    int *p_r = r.ptr<int>(0);
    int *p_b = b.ptr<int>(0);
    for (int i = 0; i < n; ++i)
        boxes.push_back(cv::Rect(p_x[i], p_y[i], p_r[i] - p_x[i], p_b[i] - p_y[i]));

    std::vector<cv::Rect> boxes_inner;
	int o_x = round(std::max(1.0, rect_sz.width*0.2));
	int o_y = round(std::max(1.0, rect_sz.height*0.2));
    if (include_inner) {
        for (int i = 0; i < n; ++i)
            boxes_inner.push_back(cv::Rect(p_x[i] + o_x, p_y[i] + o_y, p_r[i] - p_x[i] - 2 * o_x, p_b[i] - p_y[i] - 2 * o_y));
    }

    // Linear indices
    cv::Mat l = x;
    cv::Mat t = y;
    std::vector<cv::Point>bl, br, tl, tr;

    int *p_l = l.ptr<int>(0);
    int *p_t = t.ptr<int>(0);
    for (int i = 0; i < n; ++i){
        bl.push_back(cv::Point(p_l[i], p_b[i]));
        br.push_back(cv::Point(p_r[i], p_b[i]));
        tl.push_back(cv::Point(p_l[i], p_t[i]));
        tr.push_back(cv::Point(p_r[i], p_t[i]));
    }
    cv::Size rect_sz_inner;
    std::vector<cv::Point>bl_inner, br_inner, tl_inner, tr_inner;
    if (include_inner){
        rect_sz_inner.width = rect_sz.width - 2 * o_x;
        rect_sz_inner.height = rect_sz.height - 2 *o_y;

        for (int i = 0; i < n; ++i){
            bl_inner.push_back(cv::Point(p_l[i]+o_x, p_b[i]-o_y));
            br_inner.push_back(cv::Point(p_r[i]-o_x, p_b[i]-o_y));
            tl_inner.push_back(cv::Point(p_l[i]+o_x, p_t[i]+o_y));
            tr_inner.push_back(cv::Point(p_r[i]-o_x, p_t[i]+o_y));
        }
    }

    cv::Mat intProbMap;
    cv::integral(prob_map, intProbMap);
    cv::Mat intDistMap;
    cv::integral(dist_map, intDistMap);

    std::vector<float> v_scores(n, 0);
    std::vector<float> d_scores(n, 0);
    for (size_t i = 0; i < bl.size(); ++i){
        v_scores[i] = intProbMap.at<double>(br[i]) - intProbMap.at<double>(bl[i]) - intProbMap.at<double>(tr[i]) + intProbMap.at<double>(tl[i]);
        d_scores[i] = intDistMap.at<double>(br[i]) - intDistMap.at<double>(bl[i]) - intDistMap.at<double>(tr[i]) + intDistMap.at<double>(tl[i]);
    }
    std::vector<float> scores_inner(n, 0);
    if (include_inner){
        for (size_t i = 0; i < bl.size(); ++i){
            scores_inner[i] = intProbMap.at<double>(br_inner[i]) - intProbMap.at<double>(bl_inner[i]) - intProbMap.at<double>(tr_inner[i]) + intProbMap.at<double>(tl_inner[i]);
            v_scores[i] = v_scores[i] / double(rect_sz.area()) + scores_inner[i] / double(rect_sz_inner.area());
        }
    }

    top_rects.clear();;
    top_vote_scores.clear();
    top_dist_scores.clear();
    int midx = max_element(v_scores.begin(), v_scores.end()) - v_scores.begin();
    double ms = v_scores[midx];

    double best_score = ms;

    while (ms > score_frac * best_score){
        prob_map(boxes[midx]) = cv::Scalar(0.0);
        top_rects.push_back(boxes[midx]);
        top_vote_scores.push_back(v_scores[midx]);
        top_dist_scores.push_back(d_scores[midx]);
        boxes.erase(boxes.begin() + midx);
        if (include_inner)
            boxes_inner.erase(boxes_inner.begin() + midx);

        bl.erase(bl.begin() + midx);
        br.erase(br.begin() + midx);
        tl.erase(tl.begin() + midx);
        tr.erase(tr.begin() + midx);
        if (include_inner){
            bl_inner.erase(bl_inner.begin() + midx);
            br_inner.erase(br_inner.begin() + midx);
            tl_inner.erase(tl_inner.begin() + midx);
            tr_inner.erase(tr_inner.begin() + midx);
        }

        cv::integral(prob_map, intProbMap);
        cv::integral(dist_map, intDistMap);

        v_scores.resize(bl.size(), 0);
        d_scores.resize(bl.size(), 0);
        for (size_t i = 0; i < bl.size(); ++i){
            v_scores[i] = intProbMap.at<double>(br[i]) - intProbMap.at<double>(bl[i]) - intProbMap.at<double>(tr[i]) + intProbMap.at<double>(tl[i]);
            d_scores[i] = intDistMap.at<double>(br[i]) - intDistMap.at<double>(bl[i]) - intDistMap.at<double>(tr[i]) + intDistMap.at<double>(tl[i]);
        }
        scores_inner.resize(bl.size(), 0);
        if (include_inner){
            for (size_t i = 0; i < bl.size(); ++i){
                scores_inner[i] = intProbMap.at<double>(br_inner[i]) - intProbMap.at<double>(bl_inner[i]) - intProbMap.at<double>(tr_inner[i]) + intProbMap.at<double>(tl_inner[i]);
                v_scores[i] = v_scores[i] / (rect_sz.area()) + scores_inner[i] / (rect_sz_inner.area());
            }
        }
        midx = max_element(v_scores.begin(), v_scores.end()) - v_scores.begin();
        ms = v_scores[midx];
    }
}

///
/// \brief DAT_TRACKER::intersectionOverUnion
/// \param target_rect
/// \param candidates
/// \return
///
double DAT_TRACKER::intersectionOverUnion(cv::Rect target_rect, cv::Rect candidates) {
    return double((target_rect & candidates).area()) / double(target_rect.area() + candidates.area() - (target_rect & candidates).area());
}

///
/// \brief DAT_TRACKER::getForegroundDistractorProbs
/// \param frame
/// \param obj_rect
/// \param distractors
/// \param num_bins
/// \return
///
cv::Mat DAT_TRACKER::getForegroundDistractorProbs(cv::Mat frame, cv::Rect obj_rect, std::vector<cv::Rect> distractors, int num_bins) {
    int imgCount = 1;
    int dims = 3;
    const int sizes[] = { num_bins, num_bins, num_bins };
    const int channels[] = { 0, 1, 2 };
    float rRange[] = { 0, 256 };
    float gRange[] = { 0, 256 };
    float bRange[] = { 0, 256 };
    const float *ranges[] = { rRange, gRange, bRange };

    cv::Mat Md(frame.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat Mo(frame.size(), CV_8UC1, cv::Scalar(0));
    for (size_t i = 0; i < distractors.size(); ++i) {
        Mo(distractors[i]) = true;
    }
    Mo(obj_rect) = true;

    cv::Mat obj_hist, distr_hist;
    cv::calcHist(&frame, imgCount, channels, Md, distr_hist, dims, sizes, ranges);
    cv::calcHist(&frame, imgCount, channels, Mo, obj_hist, dims, sizes, ranges);
    cv::Mat prob_lut = (obj_hist*distractors.size() + 1) / (distr_hist + obj_hist*distractors.size() + 2);
    return prob_lut;
}

///
/// \brief DAT_TRACKER::CalculateHann
/// \param sz
/// \return
///
cv::Mat DAT_TRACKER::CalculateHann(cv::Size sz) {
    cv::Mat temp1(cv::Size(sz.width, 1), CV_32FC1);
    cv::Mat temp2(cv::Size(sz.height, 1), CV_32FC1);
    float *p1 = temp1.ptr<float>(0);
    float *p2 = temp2.ptr<float>(0);
    for (int i = 0; i < sz.width; ++i)
        p1[i] = 0.5*(1 - cos(CV_2PI*i / (sz.width - 1)));
    for (int i = 0; i < sz.height; ++i)
        p2[i] = 0.5*(1 - cos(CV_2PI*i / (sz.height - 1)));
    return temp2.t()*temp1;
}

///
/// \brief DAT_TRACKER::getForegroundProb
/// \param frame
/// \param prob_lut
/// \param bin_mapping
/// \return
///
cv::Mat DAT_TRACKER::getForegroundProb(cv::Mat frame, cv::Mat prob_lut, cv::Mat bin_mapping){
    cv::Mat frame_bin;
    cv::Mat prob_map(frame.size(), CV_32FC1);
    cv::LUT(frame, bin_mapping, frame_bin);
    float *p_prob_map = prob_map.ptr<float>(0);
    cv::MatIterator_<cv::Vec3b> it, end;
    for (it = frame_bin.begin<cv::Vec3b>(), end = frame_bin.end<cv::Vec3b>(); it != end; ++it)
    {
        *p_prob_map++ = prob_lut.at<float>((*it)[0], (*it)[1], (*it)[2]);
    }
    return prob_map;
}

///
/// \brief DAT_TRACKER::getSubwindowMasked
/// \param im
/// \param pos
/// \param sz
/// \param out
/// \param mask
///
void DAT_TRACKER::getSubwindowMasked(cv::Mat im, cv::Point pos, cv::Size sz, cv::Mat &out, cv::Mat &mask){

    int xs_1 = floor(pos.x) + 1 - floor(double(sz.width) / 2.);
    //int xs_2 = floor(pos.x) + sz.width - floor(double(sz.width) / 2.);
    int ys_1 = floor(pos.y) + 1 - floor(double(sz.height) / 2.);
    //int ys_2 = floor(pos.y) + sz.height - floor(double(sz.height) / 2.);

    out = getSubwindow(im, pos, sz);

    cv::Rect bbox(xs_1, ys_1, sz.width, sz.height);
    bbox = bbox&cv::Rect(0, 0, im.cols - 1, im.rows - 1);
    bbox.x = bbox.x - xs_1;
    bbox.y = bbox.y - ys_1;
    mask = cv::Mat(sz, CV_8UC1,cv::Scalar(1));
    mask(bbox) = cv::Scalar(0.0);
}

///
/// \brief DAT_TRACKER::getMotionPrediction
/// \param values
/// \param maxNumFrames
/// \return
///
cv::Point DAT_TRACKER::getMotionPrediction(std::vector<cv::Point>values, int maxNumFrames){
    cv::Point2f pred(0, 0);
    if (values.size() < 3){
        pred.x = 0; pred.y = 0;
    }
    else {
        maxNumFrames = maxNumFrames + 2;
        double A1 = 0.8;
        double A2 = -1;

        std::vector<cv::Point> V;
        for (size_t i = std::max(0, int(int(values.size()) - maxNumFrames)); i < values.size(); ++i)
            V.push_back(values[i]);

        std::vector<cv::Point2f> P;
        for (size_t i = 2; i < V.size(); ++i){
            P.push_back(cv::Point2f(A1*(V[i].x - V[i - 2].x) + A2*(V[i - 1].x - V[i - 2].x),
                    A1*(V[i].y - V[i - 2].y) + A2*(V[i - 1].y - V[i - 2].y)));
        }
        for (size_t i = 0; i < P.size(); ++i){
            pred.x += P[i].x;
            pred.y += P[i].y;
        }
        pred.x = pred.x / P.size();
        pred.y = pred.y / P.size();
    }
    return pred;
}

///
/// \brief DAT_TRACKER::getForegroundBackgroundProbs
/// \param frame
/// \param obj_rect
/// \param num_bins
/// \param bin_mapping
/// \param prob_lut
/// \param prob_map
///
void DAT_TRACKER::getForegroundBackgroundProbs(cv::Mat frame, cv::Rect obj_rect, int num_bins, cv::Mat bin_mapping, cv::Mat &prob_lut, cv::Mat &prob_map) {
    int imgCount = 1;
    const int channels[] = { 0, 1, 2 };
    cv::Mat mask = cv::Mat();
    int dims = 3;
    const int sizes[] = { num_bins, num_bins, num_bins };
    float bRange[] = { 0, 256 };
    float gRange[] = { 0, 256 };
    float rRange[] = { 0, 256 };
    const float *ranges[] = { bRange, gRange, rRange };

    cv::Mat surr_hist, obj_hist;
    cv::calcHist(&frame, imgCount, channels, mask, surr_hist, dims, sizes, ranges);

    int obj_col = round(obj_rect.x);
    int obj_row = round(obj_rect.y);
    int obj_width = round(obj_rect.width);
    int obj_height = round(obj_rect.height);

    if ((obj_col + obj_width) > (frame.cols - 1))
        obj_width = (frame.cols - 1) - obj_col;
    if ((obj_row + obj_height) > (frame.rows-1))
        obj_height = (frame.rows-1) - obj_row;

    cv::Mat obj_win;
    cv::Rect obj_region(std::max(0, obj_col), std::max(0, obj_row),
                        obj_col + obj_width + 1 - std::max(0, obj_col), obj_row + obj_height + 1 - std::max(0, obj_row));
    obj_win = frame(obj_region);
    cv::calcHist(&obj_win, imgCount, channels, mask, obj_hist, dims, sizes, ranges);
    prob_lut = (obj_hist + 1.) / (surr_hist + 2.);

    prob_map = cv::Mat(frame.size(), CV_32FC1);
    cv::Mat frame_bin;
    cv::LUT(frame, bin_mapping, frame_bin);

    float *p_prob_map = prob_map.ptr<float>(0);
    cv::MatIterator_<cv::Vec3b> it, end;
    for (it = frame_bin.begin<cv::Vec3b>(), end = frame_bin.end<cv::Vec3b>(); it != end; ++it)
    {
        *p_prob_map++ = prob_lut.at<float>((*it)[0], (*it)[1], (*it)[2]);
    }
}

///
/// \brief DAT_TRACKER::getForegroundBackgroundProbs
/// \param frame
/// \param obj_rect
/// \param num_bins
/// \param prob_lut
///
void DAT_TRACKER::getForegroundBackgroundProbs(cv::Mat frame, cv::Rect obj_rect, int num_bins, cv::Mat &prob_lut) {
    int imgCount = 1;
    const int channels[] = { 0, 1, 2 };
    cv::Mat mask = cv::Mat();
    int dims = 3;
    const int sizes[] = { num_bins, num_bins, num_bins };
    float bRange[] = { 0, 256 };
    float gRange[] = { 0, 256 };
    float rRange[] = { 0, 256 };
    const float *ranges[] = { bRange, gRange, rRange };

    cv::Mat surr_hist, obj_hist;
    cv::calcHist(&frame, imgCount, channels, mask, surr_hist, dims, sizes, ranges);

    int obj_col = round(obj_rect.x);
    int obj_row = round(obj_rect.y);
    int obj_width = round(obj_rect.width);
    int obj_height = round(obj_rect.height);

    if ((obj_col + obj_width) > (frame.cols - 1))
        obj_width = (frame.cols - 1) - obj_col;
    if ((obj_row + obj_height) > (frame.rows - 1))
        obj_height = (frame.rows - 1) - obj_row;

    cv::Mat obj_win;
    frame(cv::Rect(std::max(0, obj_col), std::max(0, obj_row), obj_width + 1, obj_height + 1)).copyTo(obj_win);
    cv::calcHist(&obj_win, imgCount, channels, mask, obj_hist, dims, sizes, ranges);
    prob_lut = (obj_hist + 1) / (surr_hist + 2);
}

///
/// \brief DAT_TRACKER::getAdaptiveThreshold
/// \param prob_map
/// \param obj_coords
/// \return
///
double DAT_TRACKER::getAdaptiveThreshold(cv::Mat prob_map, cv::Rect obj_coords){
    obj_coords.width++; obj_coords.width = std::min(prob_map.cols - obj_coords.x, obj_coords.width);
    obj_coords.height++; obj_coords.height = std::min(prob_map.rows - obj_coords.y, obj_coords.height);
    cv::Mat obj_prob_map = prob_map(obj_coords);
    int bins = 21;
    float range[] = { -0.025, 1.025 };
    const float* histRange = { range };
    bool uniform = true; bool accumulate = false;

    cv::Mat H_obj, H_dist;
    /// Compute the histograms:
    cv::calcHist(&obj_prob_map, 1, 0, cv::Mat(), H_obj, 1, &bins, &histRange, uniform, accumulate);

    H_obj = H_obj / cv::sum(H_obj)[0];
    cv::Mat cum_H_obj = H_obj.clone();
    for (int i = 1; i < cum_H_obj.rows; ++i)
        cum_H_obj.at<float>(i, 0) += cum_H_obj.at<float>(i-1, 0);

    cv::calcHist(&prob_map, 1, 0, cv::Mat(), H_dist, 1, &bins, &histRange, uniform, accumulate);
    H_dist = H_dist - H_obj;
    H_dist = H_dist / cv::sum(H_dist)[0];
    cv::Mat cum_H_dist = H_dist.clone();
    for (int i = 1; i < cum_H_dist.rows; ++i)
        cum_H_dist.at<float>(i, 0) += cum_H_dist.at<float>(i - 1, 0);

    cv::Mat k(cum_H_obj.size(), cum_H_obj.type(), cv::Scalar(0.0));
    for (int i = 0; i < (k.rows-1); ++i)
        k.at<float>(i, 0) = cum_H_obj.at<float>(i + 1, 0) - cum_H_obj.at<float>(i, 0);
    cv::Mat cum_H_obj_lt = (cum_H_obj < (1 - cum_H_dist));
    cum_H_obj_lt.convertTo(cum_H_obj_lt, CV_32FC1, 1.0/255);
    cv::Mat x = abs(cum_H_obj - (1 - cum_H_dist)) + cum_H_obj_lt + (1 - k);
    float xmin = 100;
    int min_index = 0;
    for (int i = 0; i < x.rows; ++i) {
        if (xmin > x.at<float>(i, 0))
        {
            xmin = x.at<float>(i, 0);
            min_index = i;
        }
    }
    //Final threshold result should lie between 0.4 and 0.7 to be not too restrictive
    double threshold = std::max(.4, std::min(.7, cfg.adapt_thresh_prob_bins[min_index]));
    return threshold;
}

///
/// \brief DAT_TRACKER::pos2rect
/// \param obj_center
/// \param obj_size
/// \param win_size
/// \return
///
cv::Rect DAT_TRACKER::pos2rect(cv::Point obj_center, cv::Size obj_size, cv::Size win_size){
    cv::Rect rect(round(obj_center.x - obj_size.width / 2), round(obj_center.y - obj_size.height / 2), obj_size.width, obj_size.height);
    cv::Rect border(0, 0, win_size.width - 1, win_size.height - 1);
    return rect&border;
}

///
/// \brief DAT_TRACKER::pos2rect
/// \param obj_center
/// \param obj_size
/// \return
///
cv::Rect DAT_TRACKER::pos2rect(cv::Point obj_center, cv::Size obj_size){
    cv::Rect rect(round(obj_center.x - obj_size.width / 2), round(obj_center.y - obj_size.height / 2), obj_size.width, obj_size.height);
    return rect;
}

///
/// \brief DAT_TRACKER::default_parameters_dat
/// \param cfg
/// \return
///
dat_cfg DAT_TRACKER::default_parameters_dat(dat_cfg cfg){
    for (double i = 0; i <= 20; i++)
        cfg.adapt_thresh_prob_bins.push_back(i*0.05);

    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.data;
    for (int i = 0; i < 256; ++i)
        p[i] = uchar(i / (256 / cfg.num_bins));
    cfg.bin_mapping = lookUpTable;
    return cfg;
}

///
/// \brief DAT_TRACKER::getSubwindow
/// \param frame
/// \param centerCoor
/// \param sz
/// \return
///
cv::Mat DAT_TRACKER::getSubwindow(const cv::Mat &frame, cv::Point centerCoor, cv::Size sz) {
    cv::Mat subWindow;
    cv::Point lefttop(std::min(frame.cols - 1, std::max(-sz.width + 1, centerCoor.x - cvFloor(float(sz.width) / 2.0) + 1)),
                      std::min(frame.rows - 1, std::max(-sz.height + 1, centerCoor.y - cvFloor(float(sz.height) / 2.0) + 1)));
    cv::Point rightbottom(lefttop.x + sz.width - 1, lefttop.y + sz.height - 1);

    cv::Rect border(-std::min(lefttop.x, 0), -std::min(lefttop.y, 0),
                    std::max(rightbottom.x - frame.cols + 1, 0), std::max(rightbottom.y - frame.rows + 1, 0));
    cv::Point lefttopLimit(std::max(lefttop.x, 0), std::max(lefttop.y, 0));
    cv::Point rightbottomLimit(std::min(rightbottom.x, frame.cols - 1), std::min(rightbottom.y, frame.rows - 1));

    rightbottomLimit.x += 1;
    rightbottomLimit.y += 1;
    cv::Rect roiRect(lefttopLimit, rightbottomLimit);

    frame(roiRect).copyTo(subWindow);

    if (border != cv::Rect(0, 0, 0, 0))
        cv::copyMakeBorder(subWindow, subWindow, border.y, border.height, border.x, border.width, cv::BORDER_REPLICATE);
    return subWindow;
}
