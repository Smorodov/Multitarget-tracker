#include "Kalman.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
TKalmanFilter::TKalmanFilter(
        Point_t pt,
        track_t deltaTime, // time increment (lower values makes target more "massive")
        track_t accelNoiseMag
        )
{
	// We don't know acceleration, so, assume it to process noise.
	// But we can guess, the range of acceleration values thich can be achieved by tracked object. 
    // Process noise. (standard deviation of acceleration: m/s^2)
	// shows, woh much target can accelerate.
    //track_t accelNoiseMag = 0.5;

	//4 state variables, 2 measurements
    kalman = new cv::KalmanFilter(4, 2, 0);
	// Transition cv::Matrix
    kalman->transitionMatrix = (cv::Mat_<track_t>(4, 4) << 1, 0, deltaTime, 0, 0, 1, 0, deltaTime, 0, 0, 1, 0, 0, 0, 0, 1);

	// init... 
    lastPointResult = pt;
	kalman->statePre.at<track_t>(0) = pt.x; // x
	kalman->statePre.at<track_t>(1) = pt.y; // y

	kalman->statePre.at<track_t>(2) = 0;
	kalman->statePre.at<track_t>(3) = 0;

	kalman->statePost.at<track_t>(0) = pt.x;
	kalman->statePost.at<track_t>(1) = pt.y;

	cv::setIdentity(kalman->measurementMatrix);

	kalman->processNoiseCov = (cv::Mat_<track_t>(4, 4) <<
        pow(deltaTime,4.0)/4.0	,0						,pow(deltaTime,3.0)/2.0		,0,
        0						,pow(deltaTime,4.0)/4.0	,0							,pow(deltaTime,3.0)/2.0,
        pow(deltaTime,3.0)/2.0	,0						,pow(deltaTime,2.0)			,0,
        0						,pow(deltaTime,3.0)/2.0	,0							,pow(deltaTime,2.0));


    kalman->processNoiseCov *= accelNoiseMag;

	setIdentity(kalman->measurementNoiseCov, cv::Scalar::all(0.1));

	setIdentity(kalman->errorCovPost, cv::Scalar::all(.1));
}

//---------------------------------------------------------------------------
TKalmanFilter::TKalmanFilter(
        cv::Rect rect,
        track_t deltaTime, // time increment (lower values makes target more "massive")
        track_t accelNoiseMag
        )
{
    // We don't know acceleration, so, assume it to process noise.
    // But we can guess, the range of acceleration values thich can be achieved by tracked object.
    // Process noise. (standard deviation of acceleration: m/s^2)
    // shows, woh much target can accelerate.
    //track_t accelNoiseMag = 0.5;

    //4 state variables (x, y, dx, dy, width, height), 4 measurements (x, y, width, height)
    kalman = new cv::KalmanFilter(6, 4, 0);
    // Transition cv::Matrix
    kalman->transitionMatrix = (cv::Mat_<track_t>(6, 6) <<
                                1, 0, 0, 0, deltaTime, 0,
                                0, 1, 0, 0, 0,         deltaTime,
                                0, 0, 1, 0, 0,         0,
                                0, 0, 0, 1, 0,         0,
                                0, 0, 0, 0, 1,         0,
                                0, 0, 0, 0, 0,         1);

    // init...
    lastRectResult = rect;
	kalman->statePre.at<track_t>(0) = static_cast<track_t>(rect.x);      // x
	kalman->statePre.at<track_t>(1) = static_cast<track_t>(rect.y);      // y
	kalman->statePre.at<track_t>(2) = static_cast<track_t>(rect.width);  // width
	kalman->statePre.at<track_t>(3) = static_cast<track_t>(rect.height); // height
    kalman->statePre.at<track_t>(4) = 0;                                 // dx
    kalman->statePre.at<track_t>(5) = 0;                                 // dy

	kalman->statePost.at<track_t>(0) = static_cast<track_t>(rect.x);
	kalman->statePost.at<track_t>(1) = static_cast<track_t>(rect.y);
	kalman->statePost.at<track_t>(2) = static_cast<track_t>(rect.width);
	kalman->statePost.at<track_t>(3) = static_cast<track_t>(rect.height);

    cv::setIdentity(kalman->measurementMatrix);

    kalman->processNoiseCov = (cv::Mat_<track_t>(6, 6) <<
        pow(deltaTime,4.)/4., 0,                    0,                    0,                    pow(deltaTime,3.)/2., 0,
        0,                    pow(deltaTime,4.)/4., 0,                    0,                    pow(deltaTime,3.)/2., 0,
        0,                    0,                    pow(deltaTime,4.)/4., 0,                    0,                    0,
        0,                    0,                    0,                    pow(deltaTime,4.)/4., 0,                    0,
        pow(deltaTime,3.)/2., 0,                    0,                    0,                    pow(deltaTime,2.),    0,
        0,                    pow(deltaTime,3.)/2., 0,                    0,                    0,                    pow(deltaTime,2.));


    kalman->processNoiseCov *= accelNoiseMag;

    setIdentity(kalman->measurementNoiseCov, cv::Scalar::all(0.1));

    setIdentity(kalman->errorCovPost, cv::Scalar::all(.1));
}

//---------------------------------------------------------------------------
TKalmanFilter::~TKalmanFilter()
{
	delete kalman;
}

//---------------------------------------------------------------------------
Point_t TKalmanFilter::GetPointPrediction()
{
	cv::Mat prediction = kalman->predict();
    lastPointResult = Point_t(prediction.at<track_t>(0), prediction.at<track_t>(1));
    return lastPointResult;
}

//---------------------------------------------------------------------------
Point_t TKalmanFilter::Update(Point_t p, bool dataCorrect)
{
	cv::Mat measurement(2, 1, Mat_t(1));
    if (!dataCorrect)
	{
        measurement.at<track_t>(0) = lastPointResult.x;  //update using prediction
        measurement.at<track_t>(1) = lastPointResult.y;
	}
	else
	{
		measurement.at<track_t>(0) = p.x;  //update using measurements
		measurement.at<track_t>(1) = p.y;
	}
	// Correction
	cv::Mat estiMated = kalman->correct(measurement);
    lastPointResult.x = estiMated.at<track_t>(0);   //update using measurements
    lastPointResult.y = estiMated.at<track_t>(1);

    return lastPointResult;
}
//---------------------------------------------------------------------------

cv::Rect TKalmanFilter::GetRectPrediction()
{
    cv::Mat prediction = kalman->predict();
    lastRectResult = cv::Rect_<track_t>(prediction.at<track_t>(0), prediction.at<track_t>(1), prediction.at<track_t>(2), prediction.at<track_t>(3));
	return cv::Rect(static_cast<int>(lastRectResult.x), static_cast<int>(lastRectResult.y), static_cast<int>(lastRectResult.width), static_cast<int>(lastRectResult.height));
}

//---------------------------------------------------------------------------
cv::Rect TKalmanFilter::Update(cv::Rect rect, bool dataCorrect)
{
    cv::Mat measurement(4, 1, Mat_t(1));
    if (!dataCorrect)
    {
        measurement.at<track_t>(0) = lastRectResult.x;  // update using prediction
        measurement.at<track_t>(1) = lastRectResult.y;
        measurement.at<track_t>(2) = lastRectResult.width;
        measurement.at<track_t>(3) = lastRectResult.height;
    }
    else
    {
		measurement.at<track_t>(0) = static_cast<track_t>(rect.x);  // update using measurements
		measurement.at<track_t>(1) = static_cast<track_t>(rect.y);
		measurement.at<track_t>(2) = static_cast<track_t>(rect.width);
		measurement.at<track_t>(3) = static_cast<track_t>(rect.height);
    }
    // Correction
    cv::Mat estiMated = kalman->correct(measurement);
    lastRectResult.x = estiMated.at<track_t>(0);   //update using measurements
    lastRectResult.y = estiMated.at<track_t>(1);
    lastRectResult.width = estiMated.at<track_t>(2);
    lastRectResult.height = estiMated.at<track_t>(3);

	return cv::Rect(static_cast<int>(lastRectResult.x), static_cast<int>(lastRectResult.y), static_cast<int>(lastRectResult.width), static_cast<int>(lastRectResult.height));
}
//---------------------------------------------------------------------------
