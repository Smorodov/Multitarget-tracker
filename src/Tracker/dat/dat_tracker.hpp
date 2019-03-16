#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "../VOTTracker.hpp"

///
/// \brief The dat_cfg struct
///
struct dat_cfg
{
    bool show_figures = false;
    int  img_scale_target_diagonal = 75;
    double search_win_padding = 2;
    double surr_win_factor = 1.9;
    int color_space = 1; //1rgb 2lab 3hsv 4gray
    int num_bins = 16;
    cv::Mat bin_mapping; //getBinMapping(cfg.num_bins);
    double prob_lut_update_rate = 0.05;
    bool distractor_aware = true;
    std::vector<double> adapt_thresh_prob_bins; // 0:0.05 : 1;
    int motion_estimation_history_size = 5;

    int nms_scale = 1;
    double nms_overlap = 0.9;
    double nms_score_factor = 0.5;
    bool nms_include_center_vote = true;
};

///
/// \brief The DAT_TRACKER class
///
class DAT_TRACKER : public VOTTracker
{
public:
    DAT_TRACKER();
    ~DAT_TRACKER();

    void Initialize(const cv::Mat &im, cv::Rect region);
    cv::Rect Update(const cv::Mat &im, float& confidence);
    void Train(const cv::Mat &im, bool first);

protected:
    void getNMSRects(cv::Mat prob_map, cv::Size obj_sz, double scale,
                     double overlap, double score_frac, cv::Mat dist_map, bool include_inner,
                     std::vector<cv::Rect> &top_rects, std::vector<double> &top_vote_scores, std::vector<double> &top_dist_scores);

    void getForegroundBackgroundProbs(cv::Mat frame, cv::Rect obj_rect, int num_bins, cv::Mat bin_mapping, cv::Mat &prob_lut, cv::Mat &prob_map);

    void getForegroundBackgroundProbs(cv::Mat frame, cv::Rect obj_rect, int num_bins, cv::Mat &prob_lut);

    cv::Mat getForegroundDistractorProbs(cv::Mat frame, cv::Rect obj_rect, std::vector<cv::Rect> distractors, int num_bins);

    double getAdaptiveThreshold(cv::Mat prob_map, cv::Rect obj_rect_surr);

    cv::Mat getForegroundProb(cv::Mat frame, cv::Mat prob_lut, cv::Mat bin_mapping);

    cv::Mat CalculateHann(cv::Size sz);

    double intersectionOverUnion(cv::Rect target_rect, cv::Rect candidates);

    void getSubwindowMasked(cv::Mat im, cv::Point pos, cv::Size sz, cv::Mat &out, cv::Mat &mask);

    cv::Point getMotionPrediction(std::vector<cv::Point>values, int maxNumFrames);

    cv::Rect pos2rect(cv::Point obj_center, cv::Size obj_size, cv::Size win_size);

    cv::Rect pos2rect(cv::Point obj_center, cv::Size obj_size);

    cv::Mat getSubwindow(const cv::Mat &frame, cv::Point centerCoor, cv::Size sz);

    dat_cfg default_parameters_dat(dat_cfg cfg);

private:
    dat_cfg cfg;
    double scale_factor_;
    cv::Mat prob_lut_;
    cv::Mat prob_lut_distractor_;
    cv::Mat prob_lut_masked_;
    double adaptive_threshold_;
    std::vector<cv::Point>target_pos_history_;
    std::vector<cv::Size>target_sz_history_;
};
