#include "Kalman.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
TKalmanFilter::TKalmanFilter(Point_t pt, track_t dt, track_t Accel_noise_mag)
{
	//time increment (lower values makes target more "massive")
	deltatime = dt; //0.2

	// We don't know acceleration, so, assume it to process noise.
	// But we can guess, the range of acceleration values thich can be achieved by tracked object. 
    // Process noise. (standard deviation of acceleration: m/s^2)
	// shows, woh much target can accelerate.
	//track_t Accel_noise_mag = 0.5; 

	//4 state variables, 2 measurements
	kalman = new cv::KalmanFilter( 4, 2, 0 );  
	// Transition cv::Matrix
	kalman->transitionMatrix = (cv::Mat_<track_t>(4, 4) << 1, 0, deltatime, 0, 0, 1, 0, deltatime, 0, 0, 1, 0, 0, 0, 0, 1);

	// init... 
	LastResult = pt;
	kalman->statePre.at<track_t>(0) = pt.x; // x
	kalman->statePre.at<track_t>(1) = pt.y; // y

	kalman->statePre.at<track_t>(2) = 0;
	kalman->statePre.at<track_t>(3) = 0;

	kalman->statePost.at<track_t>(0) = pt.x;
	kalman->statePost.at<track_t>(1) = pt.y;

	cv::setIdentity(kalman->measurementMatrix);

	kalman->processNoiseCov = (cv::Mat_<track_t>(4, 4) <<
		pow(deltatime,4.0)/4.0	,0						,pow(deltatime,3.0)/2.0		,0,
		0						,pow(deltatime,4.0)/4.0	,0							,pow(deltatime,3.0)/2.0,
		pow(deltatime,3.0)/2.0	,0						,pow(deltatime,2.0)			,0,
		0						,pow(deltatime,3.0)/2.0	,0							,pow(deltatime,2.0));


	kalman->processNoiseCov*=Accel_noise_mag;

	setIdentity(kalman->measurementNoiseCov, cv::Scalar::all(0.1));

	setIdentity(kalman->errorCovPost, cv::Scalar::all(.1));

}
//---------------------------------------------------------------------------
TKalmanFilter::~TKalmanFilter()
{
	delete kalman;
}

//---------------------------------------------------------------------------
Point_t TKalmanFilter::GetPrediction()
{
	cv::Mat prediction = kalman->predict();
	LastResult = Point_t(prediction.at<track_t>(0), prediction.at<track_t>(1));
	return LastResult;
}
//---------------------------------------------------------------------------
Point_t TKalmanFilter::Update(Point_t p, bool DataCorrect)
{
	cv::Mat measurement(2, 1, Mat_t(1));
	if(!DataCorrect)
	{
		measurement.at<track_t>(0) = LastResult.x;  //update using prediction
		measurement.at<track_t>(1) = LastResult.y;
	}
	else
	{
		measurement.at<track_t>(0) = p.x;  //update using measurements
		measurement.at<track_t>(1) = p.y;
	}
	// Correction
	cv::Mat estiMated = kalman->correct(measurement);
	LastResult.x = estiMated.at<track_t>(0);   //update using measurements
	LastResult.y = estiMated.at<track_t>(1);
	return LastResult;
}
//---------------------------------------------------------------------------
