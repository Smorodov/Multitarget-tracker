#pragma once
#include "Kalman.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
TKalmanFilter::TKalmanFilter(Point2f pt,float dt,float Accel_noise_mag)
{
	//приращение времени (чем меньше эта величина, тем "инертнее" цель)
	deltatime = dt; //0.2
	
	// Ускорение мы не знаем, поэтому относим его к шуму процесса.
	// Зато мы можем предполагать, какие величины ускорения может выдать отслеживаемый объект. 
	// Шум процесса. (стандартное отклонение величины ускорения: м/с^2)
	// показывает, насколько сильно объект может ускориться.
	//float Accel_noise_mag = 0.5; 

	//4 переменных состояния, 2 переменных измерения
	kalman = new KalmanFilter( 4, 2, 0 );  
	// Матрица перехода
	kalman->transitionMatrix = (Mat_<float>(4, 4) << 1,0,deltatime,0,   0,1,0,deltatime,  0,0,1,0,  0,0,0,1);

	// init... 
	LastResult = pt;
	kalman->statePre.at<float>(0) = pt.x; // x
	kalman->statePre.at<float>(1) = pt.y; // y

	kalman->statePre.at<float>(2) = 0;
	kalman->statePre.at<float>(3) = 0;

	kalman->statePost.at<float>(0)=pt.x;
	kalman->statePost.at<float>(1)=pt.y;

	setIdentity(kalman->measurementMatrix);

	kalman->processNoiseCov=(Mat_<float>(4, 4) << 
		pow(deltatime,4.0)/4.0	,0						,pow(deltatime,3.0)/2.0		,0,
		0						,pow(deltatime,4.0)/4.0	,0							,pow(deltatime,3.0)/2.0,
		pow(deltatime,3.0)/2.0	,0						,pow(deltatime,2.0)			,0,
		0						,pow(deltatime,3.0)/2.0	,0							,pow(deltatime,2.0));


	kalman->processNoiseCov*=Accel_noise_mag;

	setIdentity(kalman->measurementNoiseCov, Scalar::all(0.1));

	setIdentity(kalman->errorCovPost, Scalar::all(.1));

}
//---------------------------------------------------------------------------
TKalmanFilter::~TKalmanFilter()
{
	delete kalman;
}

//---------------------------------------------------------------------------
Point2f TKalmanFilter::GetPrediction()
{
	Mat prediction = kalman->predict();
	LastResult=Point2f(prediction.at<float>(0),prediction.at<float>(1)); 
	return LastResult;
}
//---------------------------------------------------------------------------
Point2f TKalmanFilter::Update(Point2f p, bool DataCorrect)
{
	Mat measurement(2,1,CV_32FC1);
	if(!DataCorrect)
	{
		measurement.at<float>(0) = LastResult.x;  //уточняем используя предсказание
		measurement.at<float>(1) = LastResult.y;
	}
	else
	{
		measurement.at<float>(0) = p.x;  //уточняем, используя данные измерений
		measurement.at<float>(1) = p.y;
	}
	// Коррекция
		Mat estimated = kalman->correct(measurement);
		LastResult.x=estimated.at<float>(0);  //уточняем, используя данные измерений
		LastResult.y=estimated.at<float>(1);
	return LastResult;
}
//---------------------------------------------------------------------------