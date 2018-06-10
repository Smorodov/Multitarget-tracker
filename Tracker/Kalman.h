#pragma once
#include "defines.h"
#include <memory>

#include <opencv/cv.h>

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

//---------------------------------------------------------------------------
///
/// \brief sqr
/// \param val
/// \return
///
template<class T> inline
T sqr(T val)
{
    return val * val;
}

///
/// \brief get_lin_regress_params
/// \param in_data
/// \param start_pos
/// \param in_data_size
/// \param kx
/// \param bx
/// \param ky
/// \param by
///
template<typename T, typename CONT>
void get_lin_regress_params(
        const CONT& in_data,
        size_t start_pos,
        size_t in_data_size,
        T& kx, T& bx, T& ky, T& by)
{
    T m1(0.), m2(0.);
    T m3_x(0.), m4_x(0.);
    T m3_y(0.), m4_y(0.);

    const T el_count = static_cast<T>(in_data_size - start_pos);
    for (size_t i = start_pos; i < in_data_size; ++i)
    {
        m1 += i;
        m2 += sqr(i);

        m3_x += in_data[i].x;
        m4_x += i * in_data[i].x;

        m3_y += in_data[i].y;
        m4_y += i * in_data[i].y;
    }
    T det_1 = 1. / (el_count * m2 - sqr(m1));

    m1 *= -1.;

    kx = det_1 * (m1 * m3_x + el_count * m4_x);
    bx = det_1 * (m2 * m3_x + m1 * m4_x);

    ky = det_1 * (m1 * m3_y + el_count * m4_y);
    by = det_1 * (m2 * m3_y + m1 * m4_y);
}
