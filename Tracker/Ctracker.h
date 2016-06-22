#pragma once
#include "Kalman.h"
#include "HungarianAlg.h"
#include "defines.h"
#include <iostream>
#include <vector>
#include <memory>

class CTrack
{
public:
	std::vector<Point_t> trace;
	static size_t NextTrackID;
	size_t track_id;
	size_t skipped_frames; 
	Point_t prediction;
	TKalmanFilter* KF;
	CTrack(Point_t p, track_t dt, track_t Accel_noise_mag);
	~CTrack();
};


class CTracker
{
public:
	
	// Ўаг времени опроса фильтра
	track_t dt;

	track_t Accel_noise_mag;

	// ѕорог рассто€ни€. ≈сли точки наход€тс€ дуг от друга на рассто€нии,
	// превышающем этот порог, то эта пара не рассматриваетс€ в задаче о назначени€х.
	track_t dist_thres;
	// ћаксимальное количество кадров которое трек сохран€етс€ не получа€ данных о измерений.
    size_t maximum_allowed_skipped_frames;
	// ћаксимальна€ длина следа
    size_t max_trace_length;

	std::vector<std::unique_ptr<CTrack>> tracks;
	void Update(std::vector<Point_t>& detections);
    CTracker(track_t dt_, track_t Accel_noise_mag_, track_t dist_thres_ = 60, size_t maximum_allowed_skipped_frames_ = 10, size_t max_trace_length_ = 10);
	~CTracker(void);
};

