#pragma once
#include "Kalman.h"
#include "HungarianAlg.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>

class CTrack
{
public:
	std::vector<cv::Point2d> trace;
	static size_t NextTrackID;
	size_t track_id;
	size_t skipped_frames; 
	cv::Point2d prediction;
	TKalmanFilter* KF;
	CTrack(cv::Point2f p, float dt, float Accel_noise_mag);
	~CTrack();
};


class CTracker
{
public:
	
	// Ўаг времени опроса фильтра
	float dt; 

	float Accel_noise_mag;

	// ѕорог рассто€ни€. ≈сли точки наход€тс€ дуг от друга на рассто€нии,
	// превышающем этот порог, то эта пара не рассматриваетс€ в задаче о назначени€х.
	double dist_thres;
	// ћаксимальное количество кадров которое трек сохран€етс€ не получа€ данных о измерений.
	int maximum_allowed_skipped_frames;
	// ћаксимальна€ длина следа
	int max_trace_length;

	std::vector<CTrack*> tracks;
	void Update(std::vector<cv::Point2d>& detections);
	CTracker(float _dt, float _Accel_noise_mag, double _dist_thres=60, int _maximum_allowed_skipped_frames=10,int _max_trace_length=10);
	~CTracker(void);
};

