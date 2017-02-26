#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <array>

#include "defines.h"
#include "Kalman.h"

// --------------------------------------------------------------------------
class CTrack
{
public:
    CTrack(
            const Point_t& p,
            const CRegion& region,
            track_t dt,
            track_t Accel_noise_mag,
            size_t trackID
            )
		:
		track_id(trackID),
		skipped_frames(0),
        lastRegion(region),
        pointsCount(0),
		prediction(p),
		KF(p, dt, Accel_noise_mag)
	{
	}

	track_t CalcDist(const Point_t& p)
	{
		Point_t diff = prediction - p;
		return sqrtf(diff.x * diff.x + diff.y * diff.y);
	}

	track_t CalcDist(const cv::Rect& r)
	{
		std::array<track_t, 4> diff;
        diff[0] = prediction.x - lastRegion.m_rect.width / 2 - r.x;
        diff[1] = prediction.y - lastRegion.m_rect.height / 2 - r.y;
        diff[2] = static_cast<track_t>(lastRegion.m_rect.width - r.width);
        diff[3] = static_cast<track_t>(lastRegion.m_rect.height - r.height);

		track_t dist = 0;
		for (size_t i = 0; i < diff.size(); ++i)
		{
			dist += diff[i] * diff[i];
		}
		return sqrtf(dist);
	}

    void Update(const Point_t& p, const CRegion& region, bool dataCorrect, size_t max_trace_length)
	{
		KF.GetPrediction();

        if (pointsCount)
        {
            if (dataCorrect)
            {
                prediction = KF.Update((p + averagePoint) / 2, dataCorrect);
                //prediction = (prediction + averagePoint) / 2;
            }
            else
            {
                prediction = KF.Update(averagePoint, dataCorrect);
            }
        }
        else
        {
            prediction = KF.Update(p, dataCorrect);
        }

		if (dataCorrect)
		{
            lastRegion = region;
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
    CRegion lastRegion;
    int pointsCount;
    Point_t averagePoint;

	cv::Rect GetLastRect()
	{
		return cv::Rect(
            static_cast<int>(prediction.x - lastRegion.m_rect.width / 2),
            static_cast<int>(prediction.y - lastRegion.m_rect.height / 2),
            lastRegion.m_rect.width,
            lastRegion.m_rect.height);
	}

private:
	Point_t prediction;
	TKalmanFilter KF;
};

typedef std::vector<std::unique_ptr<CTrack>> tracks_t;
