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
	
	// Шаг времени опроса фильтра
	track_t dt;

	track_t Accel_noise_mag;

	// Порог расстояния. Если точки находятся дуг от друга на расстоянии,
	// превышающем этот порог, то эта пара не рассматривается в задаче о назначениях.
	track_t dist_thres;
	// Максимальное количество кадров которое трек сохраняется не получая данных о измерений.
    size_t maximum_allowed_skipped_frames;
	// Максимальная длина следа
    size_t max_trace_length;

	std::vector<std::unique_ptr<CTrack>> tracks;
	void Update(const std::vector<Point_t>& detections);
    CTracker(track_t dt_, track_t Accel_noise_mag_, track_t dist_thres_ = 60, size_t maximum_allowed_skipped_frames_ = 10, size_t max_trace_length_ = 10);
	~CTracker(void);
};

