#pragma once
#include "defines.h"
#include <memory>
#include <deque>

#include <opencv2/opencv.hpp>

#ifdef USE_OCV_UKF
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/kalman_filters.hpp>
#endif

///
/// \brief The TKalmanFilter class
/// http://www.morethantechnical.com/2011/06/17/simple-kalman-filter-for-tracking-using-opencv-2-2-w-code/
///
class TKalmanFilter
{
public:
    TKalmanFilter(tracking::KalmanType type, bool useAcceleration, track_t deltaTime, track_t accelNoiseMag);
    ~TKalmanFilter() = default;

    Point_t GetPointPrediction();
    Point_t Update(Point_t pt, bool dataCorrect);

    cv::Rect GetRectPrediction();
    cv::Rect Update(cv::Rect rect, bool dataCorrect);

	cv::RotatedRect GetRRectPrediction();
	cv::RotatedRect Update(cv::RotatedRect rrect, bool dataCorrect);

    cv::Vec<track_t, 2> GetVelocity() const;

    void GetPtStateAndResCov(cv::Mat& covar, cv::Mat& state) const;

private:
    cv::KalmanFilter m_linearKalman;
#ifdef USE_OCV_UKF
#if (((CV_VERSION_MAJOR == 4) && (CV_VERSION_MINOR < 5)) || ((CV_VERSION_MAJOR == 4) && (CV_VERSION_MINOR == 5) && (CV_VERSION_REVISION < 1)) || (CV_VERSION_MAJOR == 3))
    cv::Ptr<cv::tracking::UnscentedKalmanFilter> m_uncsentedKalman;
#else
    cv::Ptr<cv::detail::tracking::UnscentedKalmanFilter> m_uncsentedKalman;
#endif
#endif

    static constexpr size_t MIN_INIT_VALS = 2;
    std::vector<Point_t> m_initialPoints;
    std::vector<cv::Rect> m_initialRects;
	std::vector<cv::RotatedRect> m_initialRRects;

	cv::RotatedRect m_lastRRectResult;
    cv::Rect_<track_t> m_lastRectResult;
    cv::Rect_<track_t> m_lastRect;
    Point_t m_lastPointResult;
    track_t m_accelNoiseMag = 0.5f;
    track_t m_deltaTime = 0.2f;
    track_t m_deltaTimeMin = 0.2f;
    track_t m_deltaTimeMax = 2 * 0.2f;
    track_t m_lastDist = 0;
    track_t m_deltaStep = 0;
    static constexpr int m_deltaStepsCount = 20;
    tracking::KalmanType m_type = tracking::KalmanLinear;
    bool m_useAcceleration = false; // If set true then will be used motion model x(t) = x0 + v0 * t + a * t^2 / 2
    bool m_initialized = false;

	// Constant velocity model
    void CreateLinear(Point_t xy0, Point_t xyv0);
    void CreateLinear(cv::Rect_<track_t> rect0, Point_t rectv0);
	void CreateLinear(cv::RotatedRect rrect0, Point_t rrectv0);
	
	// Constant acceleration model
	// https://www.mathworks.com/help/driving/ug/linear-kalman-filters.html
	void CreateLinearAcceleration(Point_t xy0, Point_t xyv0);
	void CreateLinearAcceleration(cv::Rect_<track_t> rect0, Point_t rectv0);
	void CreateLinearAcceleration(cv::RotatedRect rrect0, Point_t rrectv0);

#ifdef USE_OCV_UKF
    void CreateUnscented(Point_t xy0, Point_t xyv0);
    void CreateUnscented(cv::Rect_<track_t> rect0, Point_t rectv0);
    void CreateAugmentedUnscented(Point_t xy0, Point_t xyv0);
    void CreateAugmentedUnscented(cv::Rect_<track_t> rect0, Point_t rectv0);
#endif
};

