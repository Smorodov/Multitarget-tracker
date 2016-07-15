#pragma once
#include "Kalman.h"
#include "HungarianAlg.h"
#include "defines.h"
#include <iostream>
#include <vector>
#include <memory>

// --------------------------------------------------------------------------
class CTrack
{
public:
	CTrack(const Point_t& p, const cv::Rect& rect, track_t dt, track_t Accel_noise_mag, size_t trackID)
		:
		track_id(trackID),
		skipped_frames(0),
		prediction(p),
		lastRect(rect),
		KF(p, dt, Accel_noise_mag)
	{
	}

	void Update(const Point_t& p, const cv::Rect& rect, bool dataCorrect, size_t max_trace_length)
	{
		KF.GetPrediction();
		prediction = KF.Update(p, dataCorrect);

		if (dataCorrect)
		{
			lastRect = rect;
		}

		if (trace.size() > max_trace_length)
		{
			trace.erase(trace.begin(), trace.end() - max_trace_length);
		}

		trace.push_back(prediction);
	}

	std::vector<Point_t> trace;
	size_t track_id;
	size_t skipped_frames; 
	
	Point_t prediction;

	cv::Rect GetLastRect()
	{
		return cv::Rect(
			static_cast<int>(prediction.x - lastRect.width / 2),
			static_cast<int>(prediction.y - lastRect.height / 2),
			lastRect.width,
			lastRect.height);
	}

private:
	cv::Rect lastRect;
	TKalmanFilter KF;
};

// --------------------------------------------------------------------------
class CTracker
{
public:
	CTracker(track_t dt_, track_t Accel_noise_mag_, track_t dist_thres_ = 60, size_t maximum_allowed_skipped_frames_ = 10, size_t max_trace_length_ = 10);
	~CTracker(void);

	std::vector<std::unique_ptr<CTrack>> tracks;
	void Update(const std::vector<Point_t>& detections, const std::vector<cv::Rect>& rects);

private:
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

	size_t NextTrackID;
};

