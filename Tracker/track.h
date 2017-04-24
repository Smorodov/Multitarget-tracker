#pragma once
#include <iostream>
#include <vector>
#include <deque>
#include <memory>
#include <array>

#include "defines.h"
#include "Kalman.h"

// --------------------------------------------------------------------------
struct TrajectoryPoint
{
    TrajectoryPoint()
        : m_hasRaw(false)
    {
    }

    TrajectoryPoint(const Point_t& prediction)
        : m_hasRaw(false), m_prediction(prediction)
    {
    }

    TrajectoryPoint(const Point_t& prediction, const Point_t& raw)
        : m_hasRaw(true), m_prediction(prediction), m_raw(raw)
    {
    }

    bool m_hasRaw;
    Point_t m_prediction;
    Point_t m_raw;
};

// --------------------------------------------------------------------------
class Trace
{
public:
    const Point_t& operator[](size_t i) const
    {
        return m_trace[i].m_prediction;
    }

    Point_t& operator[](size_t i)
    {
        return m_trace[i].m_prediction;
    }

    size_t size() const
    {
        return m_trace.size();
    }

    void push_back(const Point_t& prediction)
    {
        m_trace.push_back(TrajectoryPoint(prediction));
    }
    void push_back(const Point_t& prediction, const Point_t& raw)
    {
        m_trace.push_back(TrajectoryPoint(prediction, raw));
    }

    void pop_front(size_t count)
    {
        if (count < size())
        {
            m_trace.erase(m_trace.begin(), m_trace.begin() + count);
        }
        else
        {
            m_trace.clear();
        }
    }

    size_t GetRawCount(size_t lastPeriod) const
    {
        size_t res = 0;

        size_t i = 0;
        if (lastPeriod < m_trace.size())
        {
            i = m_trace.size() - lastPeriod;
        }
        for (; i < m_trace.size(); ++i)
        {
            if (m_trace[i].m_hasRaw)
            {
                ++res;
            }
        }

        return res;
    }

private:
    std::deque<TrajectoryPoint> m_trace;
};

// --------------------------------------------------------------------------
class CTrack
{
public:
    CTrack(
            const Point_t& pt,
            const CRegion& region,
            track_t deltaTime,
            track_t accelNoiseMag,
            size_t trackID,
            bool filterObjectSize
            )
		:
		track_id(trackID),
		skipped_frames(0),
        lastRegion(region),
        pointsCount(0),
        m_predictionPoint(pt),
        m_filterObjectSize(filterObjectSize)
	{
        if (filterObjectSize)
        {
            m_kalman = new TKalmanFilter(region.m_rect, deltaTime, accelNoiseMag);
        }
        else
        {
            m_kalman = new TKalmanFilter(pt, deltaTime, accelNoiseMag);
        }
        trace.push_back(pt, pt);
	}

    track_t CalcDist(const Point_t& pt)
	{
        Point_t diff = m_predictionPoint - pt;
		return sqrtf(diff.x * diff.x + diff.y * diff.y);
	}

	track_t CalcDist(const cv::Rect& r)
	{
		std::array<track_t, 4> diff;
        diff[0] = m_predictionPoint.x - lastRegion.m_rect.width / 2 - r.x;
        diff[1] = m_predictionPoint.y - lastRegion.m_rect.height / 2 - r.y;
        diff[2] = static_cast<track_t>(lastRegion.m_rect.width - r.width);
        diff[3] = static_cast<track_t>(lastRegion.m_rect.height - r.height);

		track_t dist = 0;
		for (size_t i = 0; i < diff.size(); ++i)
		{
			dist += diff[i] * diff[i];
		}
		return sqrtf(dist);
	}

    void Update(const Point_t& pt, const CRegion& region, bool dataCorrect, size_t max_trace_length)
	{
        if (m_filterObjectSize)
        {
            m_kalman->GetRectPrediction();

			if (boundidgRect.area() > 0)
			{
				if (dataCorrect)
				{
					cv::Rect prect(
						(boundidgRect.x + region.m_rect.x) / 2,
						(boundidgRect.y + region.m_rect.y) / 2,
						(boundidgRect.width + region.m_rect.width) / 2,
						(boundidgRect.height + region.m_rect.height) / 2);

					m_predictionRect = m_kalman->Update(prect, dataCorrect);
				}
				else
				{
					m_predictionRect = m_kalman->Update(boundidgRect, dataCorrect);
				}
			}
			else
            {
                m_predictionRect = m_kalman->Update(region.m_rect, dataCorrect);
            }
            m_predictionPoint = (m_predictionRect.tl() + m_predictionRect.br()) / 2;
        }
        else // Kalman filter only for object center
        {
            m_kalman->GetPointPrediction();

            if (pointsCount)
            {
                if (dataCorrect)
                {
                    m_predictionPoint = m_kalman->Update((pt + averagePoint) / 2, dataCorrect);
                }
                else
                {
                    m_predictionPoint = m_kalman->Update(averagePoint, dataCorrect);
                }
            }
            else
            {
                m_predictionPoint = m_kalman->Update(pt, dataCorrect);
            }
        }

        if (dataCorrect)
        {
            lastRegion = region;
            trace.push_back(m_predictionPoint, pt);
        }
        else
        {
            trace.push_back(m_predictionPoint);
        }

        if (trace.size() > max_trace_length)
        {
            trace.pop_front(trace.size() - max_trace_length);
        }
    }

    bool IsRobust(int minTraceSize, float minRawRatio, cv::Size2f sizeRatio) const
    {
        bool res = trace.size() > static_cast<size_t>(minTraceSize);
        res &= trace.GetRawCount(minTraceSize) / static_cast<float>(minTraceSize) > minRawRatio;
        if (sizeRatio.width + sizeRatio.height > 0)
        {
            float sr = lastRegion.m_rect.width / static_cast<float>(lastRegion.m_rect.height);
            if (sizeRatio.width > 0)
            {
                res &= (sr > sizeRatio.width);
            }
            if (sizeRatio.height > 0)
            {
                res &= (sr < sizeRatio.height);
            }
        }
        return res;
    }

    Trace trace;
	size_t track_id;
	size_t skipped_frames; 
    CRegion lastRegion;
    int pointsCount;
    Point_t averagePoint;   ///< Average point after LocalTracking
	cv::Rect boundidgRect;  ///< Bounding rect after LocalTracking

    cv::Rect GetLastRect() const
	{
        if (m_filterObjectSize)
        {
            return m_predictionRect;
        }
        else
        {
            return cv::Rect(
                        static_cast<int>(m_predictionPoint.x - lastRegion.m_rect.width / 2),
                        static_cast<int>(m_predictionPoint.y - lastRegion.m_rect.height / 2),
                        lastRegion.m_rect.width,
                        lastRegion.m_rect.height);
        }
    }

private:
    Point_t m_predictionPoint;
    cv::Rect m_predictionRect;
    TKalmanFilter* m_kalman;
    bool m_filterObjectSize;
};

typedef std::vector<std::unique_ptr<CTrack>> tracks_t;
