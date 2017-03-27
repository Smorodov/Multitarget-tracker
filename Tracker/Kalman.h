#pragma once
#include "defines.h"
#include <opencv/cv.h>

// http://www.morethantechnical.com/2011/06/17/simple-kalman-filter-for-tracking-using-opencv-2-2-w-code/
class TKalmanFilter
{
public:
    TKalmanFilter(Point_t pt, track_t deltaTime = 0.2, track_t accelNoiseMag = 0.5);
    TKalmanFilter(cv::Rect rect, track_t deltaTime = 0.2, track_t accelNoiseMag = 0.5);
	~TKalmanFilter();

    Point_t GetPointPrediction();
    Point_t Update(Point_t p, bool dataCorrect);

    cv::Rect GetRectPrediction();
    cv::Rect Update(cv::Rect rect, bool dataCorrect);

private:
    cv::KalmanFilter* kalman;
    Point_t lastPointResult;
    cv::Rect_<track_t> lastRectResult;
    cv::Rect_<track_t> lastRect;
};
