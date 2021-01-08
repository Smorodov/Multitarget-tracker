#pragma once
#include <iostream>
#include <vector>
#include <deque>
#include <memory>
#include <array>

#ifdef USE_OCV_KCF
#include <opencv2/tracking.hpp>
#endif

#include "defines.h"
#include "object_types.h"
#include "Kalman.h"
#include "VOTTracker.hpp"

///
/// \brief The TrajectoryPoint struct
///
struct TrajectoryPoint
{
    ///
    /// \brief TrajectoryPoint
    ///
    TrajectoryPoint() = default;

    ///
    /// \brief TrajectoryPoint
    /// \param prediction
    ///
    TrajectoryPoint(const Point_t& prediction)
        : m_prediction(prediction)
    {
    }

    ///
    /// \brief TrajectoryPoint
    /// \param prediction
    /// \param raw
    ///
    TrajectoryPoint(const Point_t& prediction, const Point_t& raw)
        :
          m_prediction(prediction),
          m_raw(raw),
          m_hasRaw(true)
    {
    }

	///
	TrajectoryPoint(const TrajectoryPoint& tp) noexcept
		: m_prediction(tp.m_prediction), m_raw(tp.m_raw), m_hasRaw(tp.m_hasRaw)
	{
	}

	///
	TrajectoryPoint& operator=(const TrajectoryPoint& tp) noexcept
	{
		m_prediction = tp.m_prediction;
		m_raw = tp.m_raw;
		m_hasRaw = tp.m_hasRaw;
		return *this;
	}

	///
	TrajectoryPoint(TrajectoryPoint&&) = default;

    Point_t m_prediction;
    Point_t m_raw;
	bool m_hasRaw = false;
};

///
/// \brief The Trace class
///
class Trace
{
public:
	///
	Trace() = default;
	///
    Trace(const Trace&) = default;
    ///
    Trace(Trace&&) = default;

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
    /// \brief at
    /// \param i
    /// \return
    ///
    const TrajectoryPoint& at(size_t i) const
    {
        return m_trace[i];
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
        m_trace.emplace_back(prediction);
    }
    void push_back(const Point_t& prediction, const Point_t& raw)
    {
        m_trace.emplace_back(prediction, raw);
    }

    ///
    /// \brief pop_front
    /// \param count
    ///
    void pop_front(size_t count)
    {
        if (count < size())
            m_trace.erase(m_trace.begin(), m_trace.begin() + count);
        else
            m_trace.clear();
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
            i = m_trace.size() - lastPeriod;

        for (; i < m_trace.size(); ++i)
        {
            if (m_trace[i].m_hasRaw)
                ++res;
        }

        return res;
    }

	///
	/// \brief Reserve
	/// \param capacity
	/// \return
	///
	void Reserve(size_t capacity)
	{
		m_trace.reserve(capacity);
	}

private:
    std::vector<TrajectoryPoint> m_trace;
};

///
/// \brief The TrackingObject class
///
struct TrackingObject
{
	Trace m_trace;                     // Trajectory
	size_t m_ID = 0;                   // Objects ID
	cv::RotatedRect m_rrect;           // Coordinates
	cv::Vec<track_t, 2> m_velocity;    // pixels/sec
	objtype_t m_type = bad_type;       // Objects type name or empty
	float m_confidence = -1;           // From Detector with score (YOLO or SSD)
	bool m_isStatic = false;           // Object is abandoned
	bool m_outOfTheFrame = false;      // Is object out of freme
	mutable bool m_lastRobust = false; // saved latest robust value

	///
    TrackingObject(const cv::RotatedRect& rrect, size_t ID, const Trace& trace,
		bool isStatic, bool outOfTheFrame, objtype_t type, float confidence, cv::Vec<track_t, 2> velocity)
		:
        m_trace(trace), m_ID(ID), m_rrect(rrect), m_velocity(velocity), m_type(type), m_confidence(confidence), m_isStatic(isStatic), m_outOfTheFrame(outOfTheFrame)
	{
	}

	///
	TrackingObject(TrackingObject&&) = default;

    ///
    /// \brief IsRobust
    /// \param minTraceSize
    /// \param minRawRatio
    /// \param sizeRatio
    /// \return
    ///
	bool IsRobust(int minTraceSize, float minRawRatio, cv::Size2f sizeRatio) const
	{
		m_lastRobust = m_trace.size() > static_cast<size_t>(minTraceSize);
		m_lastRobust &= m_trace.GetRawCount(m_trace.size() - 1) / static_cast<float>(m_trace.size()) > minRawRatio;
		if (sizeRatio.width + sizeRatio.height > 0)
		{
            float sr = m_rrect.size.width / m_rrect.size.height;
			if (sizeRatio.width > 0)
				m_lastRobust &= (sr > sizeRatio.width);

			if (sizeRatio.height > 0)
				m_lastRobust &= (sr < sizeRatio.height);
		}
		if (m_outOfTheFrame)
			m_lastRobust = false;
		return m_lastRobust;
	}
};

///
/// \brief The RegionEmbedding struct
///
struct RegionEmbedding
{
    cv::Mat m_hist;
	cv::Mat m_embedding;
	double m_embDot = 0.;
};


///
/// \brief The CTrack class
///
class CTrack
{
public:
    CTrack(const CRegion& region,
           tracking::KalmanType kalmanType,
           track_t deltaTime,
           track_t accelNoiseMag,
           bool useAcceleration,
           size_t trackID,
           bool filterObjectSize,
           tracking::LostTrackType externalTrackerForLost);

    CTrack(const CRegion& region,
           const RegionEmbedding& regionEmbedding,
           tracking::KalmanType kalmanType,
           track_t deltaTime,
           track_t accelNoiseMag,
           bool useAcceleration,
           size_t trackID,
           bool filterObjectSize,
           tracking::LostTrackType externalTrackerForLost);

    ///
    /// \brief CalcDistCenter
    /// Euclidean distance from 0 to 1  between objects centres on two N and N+1 frames
    /// \param reg
    /// \return
    ///
    track_t CalcDistCenter(const CRegion& reg) const;
    ///
    /// \brief CalcDistRect
    /// Euclidean distance from 0 to 1 between object contours on two N and N+1 frames
    /// \param reg
    /// \return
    ///
    track_t CalcDistRect(const CRegion& reg) const;
    ///
    /// \brief CalcDistJaccard
    /// Jaccard distance from 0 to 1 between object bounding rectangles on two N and N+1 frames
    /// \param reg
    /// \return
    ///
    track_t CalcDistJaccard(const CRegion& reg) const;
	///
	/// \brief CalcDistHist
	/// Distance from 0 to 1 between objects histogramms on two N and N+1 frames
	/// \param reg
	/// \param currFrame
	/// \return
	///
    track_t CalcDistHist(const CRegion& reg, RegionEmbedding& embedding, cv::UMat currFrame) const;
	///
	/// \brief CalcCosine
	/// Distance from 0 to 1 between objects embeddings on two N and N+1 frames
	/// \param reg
	/// \param currFrame
	/// \return
	///
	track_t CalcCosine(const CRegion& reg, RegionEmbedding& embedding, cv::UMat currFrame) const;

	cv::RotatedRect CalcPredictionEllipse(cv::Size_<track_t> minRadius) const;
	///
	/// \brief IsInsideArea
	/// Test point inside in prediction area: prediction area + object velocity
	/// \param pt
	/// \param minVal
	/// \return
	///
	track_t IsInsideArea(const Point_t& pt, const cv::RotatedRect& rrect) const;
    track_t WidthDist(const CRegion& reg) const;
    track_t HeightDist(const CRegion& reg) const;

    void Update(const CRegion& region, bool dataCorrect, size_t max_trace_length, cv::UMat prevFrame, cv::UMat currFrame, int trajLen, int maxSpeedForStatic);
    void Update(const CRegion& region, const RegionEmbedding& regionEmbedding, bool dataCorrect, size_t max_trace_length, cv::UMat prevFrame, cv::UMat currFrame, int trajLen, int maxSpeedForStatic);

    bool IsStatic() const;
    bool IsStaticTimeout(int framesTime) const;
    bool IsOutOfTheFrame() const;

    cv::RotatedRect GetLastRect() const;

    const Point_t& AveragePoint() const;
    Point_t& AveragePoint();
    const CRegion& LastRegion() const;
    size_t SkippedFrames() const;
    size_t& SkippedFrames();

    TrackingObject ConstructObject() const;

private:
    TKalmanFilter m_kalman;
    CRegion m_lastRegion;
    Trace m_trace;
    cv::RotatedRect m_predictionRect;
    Point_t m_predictionPoint;

    size_t m_trackID = 0;
    size_t m_skippedFrames = 0;

    tracking::LostTrackType m_externalTrackerForLost;
#ifdef USE_OCV_KCF
    cv::Ptr<cv::Tracker> m_tracker;
#endif
    std::unique_ptr<VOTTracker> m_VOTTracker;

    ///
    void RectUpdate(const CRegion& region, bool dataCorrect, cv::UMat prevFrame, cv::UMat currFrame);

    ///
    void CreateExternalTracker(int channels);

    ///
    void PointUpdate(const Point_t& pt, const cv::Size& newObjSize, bool dataCorrect, const cv::Size& frameSize);

    RegionEmbedding m_regionEmbedding;

    ///
    bool CheckStatic(int trajLen, cv::UMat currFrame, const CRegion& region, int maxSpeedForStatic);
    cv::UMat m_staticFrame;
    cv::Rect m_staticRect;
    int m_staticFrames = 0;
    bool m_isStatic = false;

    bool m_filterObjectSize = false;
    bool m_outOfTheFrame = false;
};

typedef std::vector<std::unique_ptr<CTrack>> tracks_t;
