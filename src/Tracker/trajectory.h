#pragma once
#include <vector>
#include "defines.h"

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
	track_id_t m_ID;                   // Objects ID
	cv::RotatedRect m_rrect;           // Coordinates
	cv::Vec<track_t, 2> m_velocity;    // pixels/sec
	objtype_t m_type = bad_type;       // Objects type name or empty value
	float m_confidence = -1;           // From Detector with score (YOLO or SSD)
	bool m_isStatic = false;           // Object is abandoned
	int m_isStaticTime = 0;            // Object is abandoned, frames
	bool m_outOfTheFrame = false;      // Is object out of the frame
	mutable bool m_lastRobust = false; // saved latest robust value

	///
    TrackingObject(const cv::RotatedRect& rrect, track_id_t ID, const Trace& trace,
		bool isStatic, int isStaticTime, bool outOfTheFrame, objtype_t type, float confidence, cv::Vec<track_t, 2> velocity)
		:
        m_trace(trace), m_ID(ID), m_rrect(rrect), m_velocity(velocity), m_type(type), m_confidence(confidence),
        m_isStatic(isStatic), m_isStaticTime(isStaticTime),
        m_outOfTheFrame(outOfTheFrame)
	{
	}

    ///
	TrackingObject() = default;
    ///
	TrackingObject(const TrackingObject&) = default;
	///
	TrackingObject(TrackingObject&&) = default;

    ///
	~TrackingObject() = default;

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
