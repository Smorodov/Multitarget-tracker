#pragma once
#include <iostream>
#include <vector>
#include <deque>
#include <memory>
#include <array>

#if USE_OCV_KCF
#include <opencv2/tracking.hpp>
#endif

#include "defines.h"
#include "Kalman.h"

// --------------------------------------------------------------------------
///
/// \brief The TrajectoryPoint struct
///
struct TrajectoryPoint
{
    ///
    /// \brief TrajectoryPoint
    ///
    TrajectoryPoint()
        : m_hasRaw(false)
    {
    }

    ///
    /// \brief TrajectoryPoint
    /// \param prediction
    ///
    TrajectoryPoint(const Point_t& prediction)
        : m_hasRaw(false), m_prediction(prediction)
    {
    }

    ///
    /// \brief TrajectoryPoint
    /// \param prediction
    /// \param raw
    ///
    TrajectoryPoint(const Point_t& prediction, const Point_t& raw)
        : m_hasRaw(true), m_prediction(prediction), m_raw(raw)
    {
    }

    bool m_hasRaw;
    Point_t m_prediction;
    Point_t m_raw;
};

// --------------------------------------------------------------------------
///
/// \brief The Trace class
///
class Trace
{
public:
    ///
    /// \brief operator []
    /// \param i
    /// \return
    ///
    const Point_t& operator[](size_t i) const
    {
        return m_trace[i].m_prediction;
    }

    ///
    /// \brief operator []
    /// \param i
    /// \return
    ///
    Point_t& operator[](size_t i)
    {
        return m_trace[i].m_prediction;
    }

    ///
    /// \brief HasRaw
    /// \param i
    /// \return
    ///
    bool HasRaw(size_t i) const
    {
        return m_trace[i].m_hasRaw;
    }

    ///
    /// \brief size
    /// \return
    ///
    size_t size() const
    {
        return m_trace.size();
    }

    ///
    /// \brief push_back
    /// \param prediction
    ///
    void push_back(const Point_t& prediction)
    {
        m_trace.push_back(TrajectoryPoint(prediction));
    }
    void push_back(const Point_t& prediction, const Point_t& raw)
    {
        m_trace.push_back(TrajectoryPoint(prediction, raw));
    }

    ///
    /// \brief pop_front
    /// \param count
    ///
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

    ///
    /// \brief GetRawCount
    /// \param lastPeriod
    /// \return
    ///
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
///
/// \brief The CTrack class
///
class CTrack
{
public:
    ///
    /// \brief CTrack
    /// \param pt
    /// \param region
    /// \param deltaTime
    /// \param accelNoiseMag
    /// \param trackID
    /// \param filterObjectSize
    /// \param externalTrackerForLost
    ///
    CTrack(
            const Point_t& pt,
            const CRegion& region,
            TKalmanFilter::KalmanType kalmanType,
            track_t deltaTime,
            track_t accelNoiseMag,
            size_t trackID,
            bool filterObjectSize,
            bool externalTrackerForLost
            )
		:
        m_trackID(trackID),
        m_skippedFrames(0),
        m_lastRegion(region),
        m_predictionPoint(pt),
        m_filterObjectSize(filterObjectSize),
        m_externalTrackerForLost(externalTrackerForLost)
	{
        if (filterObjectSize)
        {
            m_kalman = new TKalmanFilter(kalmanType, region.m_rect, deltaTime, accelNoiseMag);
        }
        else
        {
            m_kalman = new TKalmanFilter(kalmanType, pt, deltaTime, accelNoiseMag);
        }
        m_trace.push_back(pt, pt);
	}

    ///
    /// \brief CalcDist
    /// \param pt
    /// \return
    ///
    track_t CalcDist(const Point_t& pt)
	{
        Point_t diff = m_predictionPoint - pt;
		return sqrtf(diff.x * diff.x + diff.y * diff.y);
	}

    ///
    /// \brief CalcDist
    /// \param r
    /// \return
    ///
    track_t CalcDist(const cv::Rect& r)
	{
		std::array<track_t, 4> diff;
        diff[0] = m_predictionPoint.x - m_lastRegion.m_rect.width / 2 - r.x;
        diff[1] = m_predictionPoint.y - m_lastRegion.m_rect.height / 2 - r.y;
        diff[2] = static_cast<track_t>(m_lastRegion.m_rect.width - r.width);
        diff[3] = static_cast<track_t>(m_lastRegion.m_rect.height - r.height);

		track_t dist = 0;
		for (size_t i = 0; i < diff.size(); ++i)
		{
			dist += diff[i] * diff[i];
		}
		return sqrtf(dist);
	}

    ///
    /// \brief Update
    /// \param pt
    /// \param region
    /// \param dataCorrect
    /// \param max_trace_length
    /// \param prevFrame
    /// \param currFrame
    ///
    void Update(
            const Point_t& pt,
            const CRegion& region,
            bool dataCorrect,
            size_t max_trace_length,
            cv::Mat prevFrame,
            cv::Mat currFrame
            )
	{
        if (m_filterObjectSize) // Kalman filter for object coordinates and size
        {
            RectUpdate(region, dataCorrect, prevFrame, currFrame);
        }
        else // Kalman filter only for object center
        {
            PointUpdate(pt, dataCorrect, currFrame.size());
        }

        if (dataCorrect)
        {
            m_lastRegion = region;
            m_trace.push_back(m_predictionPoint, pt);
        }
        else
        {
            m_trace.push_back(m_predictionPoint);
        }

        if (m_trace.size() > max_trace_length)
        {
            m_trace.pop_front(m_trace.size() - max_trace_length);
        }
    }

    ///
    /// \brief IsRobust
    /// \param minTraceSize
    /// \param minRawRatio
    /// \param sizeRatio
    /// \return
    ///
    bool IsRobust(int minTraceSize, float minRawRatio, cv::Size2f sizeRatio) const
    {
        bool res = m_trace.size() > static_cast<size_t>(minTraceSize);
        res &= m_trace.GetRawCount(minTraceSize) / static_cast<float>(minTraceSize) > minRawRatio;
        if (sizeRatio.width + sizeRatio.height > 0)
        {
            float sr = m_lastRegion.m_rect.width / static_cast<float>(m_lastRegion.m_rect.height);
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

    Trace m_trace;
    size_t m_trackID;
    size_t m_skippedFrames;
    CRegion m_lastRegion;
    Point_t m_averagePoint;   ///< Average point after LocalTracking
    cv::Rect m_boundidgRect;  ///< Bounding rect after LocalTracking

    ///
    /// \brief GetLastRect
    /// \return
    ///
    cv::Rect GetLastRect() const
	{
        if (m_filterObjectSize)
        {
            return m_predictionRect;
        }
        else
        {
            return cv::Rect(
                        static_cast<int>(m_predictionPoint.x - m_lastRegion.m_rect.width / 2),
                        static_cast<int>(m_predictionPoint.y - m_lastRegion.m_rect.height / 2),
                        m_lastRegion.m_rect.width,
                        m_lastRegion.m_rect.height);
        }
    }

private:
    Point_t m_predictionPoint;
    cv::Rect m_predictionRect;
    TKalmanFilter* m_kalman;
    bool m_filterObjectSize;

    bool m_externalTrackerForLost;
#if USE_OCV_KCF
    cv::Ptr<cv::TrackerKCF> m_tracker;
#endif

    ///
    /// \brief RectUpdate
    /// \param region
    /// \param dataCorrect
    /// \param prevFrame
    /// \param currFrame
    ///
    void RectUpdate(
            const CRegion& region,
            bool dataCorrect,
            cv::Mat prevFrame,
            cv::Mat currFrame
            )
    {
        m_kalman->GetRectPrediction();

        bool recalcPrediction = true;
        if (m_externalTrackerForLost)
        {
#if USE_OCV_KCF
            if (!dataCorrect)
            {
                if (!m_tracker || m_tracker.empty())
                {
                    cv::TrackerKCF::Params params;
                    params.compressed_size = 1;
                    params.desc_pca = cv::TrackerKCF::GRAY;
                    params.desc_npca = cv::TrackerKCF::GRAY;
                    params.resize = false;

                    m_tracker = cv::TrackerKCF::createTracker(params);
                    cv::Rect2d lastRect(m_predictionRect.x, m_predictionRect.y, m_predictionRect.width, m_predictionRect.height);
                    m_tracker->init(prevFrame, lastRect);
                }
                cv::Rect2d newRect;
                if (m_tracker->update(currFrame, newRect))
                {
                    cv::Rect prect(cvRound(newRect.x), cvRound(newRect.y), cvRound(newRect.width), cvRound(newRect.height));

                    m_predictionRect = m_kalman->Update(prect, true);

                    recalcPrediction = false;

                    m_boundidgRect = cv::Rect();
                    m_lastRegion.m_points.clear();
                }
            }
            else
            {
                if (m_tracker && !m_tracker.empty())
                {
                    m_tracker.release();
                }
            }
#else
            std::cerr << "KCF tracker was disabled in CMAKE! Set useExternalTrackerForLostObjects = TrackNone in constructor." << std::endl;
#endif
        }

        if (recalcPrediction)
        {
            if (m_boundidgRect.area() > 0)
            {
                if (dataCorrect)
                {
                    cv::Rect prect(
                                (m_boundidgRect.x + region.m_rect.x) / 2,
                                (m_boundidgRect.y + region.m_rect.y) / 2,
                                (m_boundidgRect.width + region.m_rect.width) / 2,
                                (m_boundidgRect.height + region.m_rect.height) / 2);

                    m_predictionRect = m_kalman->Update(prect, dataCorrect);
                }
                else
                {
                    m_predictionRect = m_kalman->Update(m_boundidgRect, true);
                }
            }
            else
            {
                m_predictionRect = m_kalman->Update(region.m_rect, dataCorrect);
            }
        }
        if (m_predictionRect.width < 2)
        {
            m_predictionRect.width = 2;
        }
        if (m_predictionRect.x < 0)
        {
            m_predictionRect.x = 0;
        }
        else if (m_predictionRect.x + m_predictionRect.width > currFrame.cols - 1)
        {
            m_predictionRect.x = currFrame.cols - 1 - m_predictionRect.width;
        }
        if (m_predictionRect.height < 2)
        {
            m_predictionRect.height = 2;
        }
        if (m_predictionRect.y < 0)
        {
            m_predictionRect.y = 0;
        }
        else if (m_predictionRect.y + m_predictionRect.height > currFrame.rows - 1)
        {
            m_predictionRect.y = currFrame.rows - 1 - m_predictionRect.height;
        }

        m_predictionPoint = (m_predictionRect.tl() + m_predictionRect.br()) / 2;
    }

    ///
    /// \brief PointUpdate
    /// \param pt
    /// \param dataCorrect
    ///
    void PointUpdate(
            const Point_t& pt,
            bool dataCorrect,
            const cv::Size& frameSize
            )
    {
        m_kalman->GetPointPrediction();

        if (m_averagePoint.x + m_averagePoint.y > 0)
        {
            if (dataCorrect)
            {
                m_predictionPoint = m_kalman->Update((pt + m_averagePoint) / 2, dataCorrect);
            }
            else
            {
                m_predictionPoint = m_kalman->Update(m_averagePoint, true);
            }
        }
        else
        {
            m_predictionPoint = m_kalman->Update(pt, dataCorrect);
        }

        if (m_predictionPoint.x < 0)
        {
            m_predictionPoint.x = 0;
        }
        else if (m_predictionPoint.x > frameSize.width - 1)
        {
            m_predictionPoint.x = frameSize.width - 1;
        }
        if (m_predictionPoint.y < 0)
        {
            m_predictionPoint.y = 0;
        }
        else if (m_predictionPoint.y > frameSize.height - 1)
        {
            m_predictionPoint.y = frameSize.height - 1;
        }
    }
};

typedef std::vector<std::unique_ptr<CTrack>> tracks_t;
