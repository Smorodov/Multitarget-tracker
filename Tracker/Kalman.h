#pragma once
#include "defines.h"
#include <memory>

#include <opencv/cv.h>

#ifdef USE_OCV_UKF
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/kalman_filters.hpp>
#endif

// http://www.morethantechnical.com/2011/06/17/simple-kalman-filter-for-tracking-using-opencv-2-2-w-code/
class TKalmanFilter
{
public:
    TKalmanFilter(tracking::KalmanType type, Point_t pt, track_t deltaTime = 0.2, track_t accelNoiseMag = 0.5);
    TKalmanFilter(tracking::KalmanType type, cv::Rect rect, track_t deltaTime = 0.2, track_t accelNoiseMag = 0.5);
	~TKalmanFilter();

    Point_t GetPointPrediction();
    Point_t Update(Point_t pt, bool dataCorrect);

    cv::Rect GetRectPrediction();
    cv::Rect Update(cv::Rect rect, bool dataCorrect);

private:
    tracking::KalmanType m_type;
    std::unique_ptr<cv::KalmanFilter> m_linearKalman;
#ifdef USE_OCV_UKF
    cv::Ptr<cv::tracking::UnscentedKalmanFilter> m_uncsentedKalman;
#endif

    std::deque<Point_t> m_initialPoints;
    std::deque<cv::Rect> m_initialRects;
    static const size_t MIN_INIT_VALS = 4;

    Point_t m_lastPointResult;
    cv::Rect_<track_t> m_lastRectResult;
    cv::Rect_<track_t> m_lastRect;

    bool m_initialized;
    track_t m_deltaTime;
    track_t m_accelNoiseMag;

    void CreateLinear(Point_t xy0, Point_t xyv0);
    void CreateLinear(cv::Rect_<track_t> rect0, Point_t rectv0);
#ifdef USE_OCV_UKF
    void CreateUnscented(Point_t xy0, Point_t xyv0);
    void CreateUnscented(cv::Rect_<track_t> rect0, Point_t rectv0);
    void CreateAugmentedUnscented(Point_t xy0, Point_t xyv0);
    void CreateAugmentedUnscented(cv::Rect_<track_t> rect0, Point_t rectv0);
#endif
};
