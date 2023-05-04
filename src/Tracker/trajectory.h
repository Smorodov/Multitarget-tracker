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
	Trace& operator=(const Trace& trace)
	{
		m_trace = trace.m_trace;
		return *this;
	}

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
	int m_staticTime = 0;              // Object is abandoned, frames
	bool m_outOfTheFrame = false;      // Is object out of the frame
	mutable bool m_lastRobust = false; // saved latest robust value

	///
    TrackingObject(const cv::RotatedRect& rrect, track_id_t ID, const Trace& trace,
		bool isStatic, int staticTime, bool outOfTheFrame, objtype_t type, float confidence, cv::Vec<track_t, 2> velocity)
		:
        m_trace(trace), m_ID(ID), m_rrect(rrect), m_velocity(velocity), m_type(type), m_confidence(confidence),
        m_isStatic(isStatic), m_staticTime(staticTime),
        m_outOfTheFrame(outOfTheFrame)
	{
		//std::cout << "TrackingObject.m_rrect: " << m_rrect.center << ", " << m_rrect.angle << ", " << m_rrect.size << std::endl;
	}

    ///
	TrackingObject() = default;
    ///
	TrackingObject(const TrackingObject&) = default;
	///
	TrackingObject(TrackingObject&&) = default;
	///
	TrackingObject & operator=(const TrackingObject& track)
	{
		m_trace = track.m_trace;
		m_ID = track.m_ID;
		m_rrect = track.m_rrect;
		m_velocity = track.m_velocity;
		m_type = track.m_type;
		m_confidence = track.m_confidence;
		m_isStatic = track.m_isStatic;
		m_staticTime = track.m_staticTime;
		m_outOfTheFrame = track.m_outOfTheFrame;
		m_lastRobust = track.m_lastRobust;

		return *this;
	}
    ///
	~TrackingObject() = default;

    ///
    /// \brief IsRobust
    /// \param minTraceSize
    /// \param minRawRatio
    /// \param sizeRatio
    /// \return
    ///
	bool IsRobust(int minTraceSize, float minRawRatio, cv::Size2f sizeRatio, size_t lastDetectsCount = 0) const
	{
		m_lastRobust = m_trace.size() > static_cast<size_t>(minTraceSize);
		if (lastDetectsCount)
		{
			size_t raws = m_trace.GetRawCount(lastDetectsCount);
			m_lastRobust = (raws > 0);
		}
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

    ///
    /// \brief GetTrajectory
    /// \return
    ///
    std::vector<cv::Point> GetTrajectory() const
    {
        std::vector<cv::Point> trajectory(m_trace.size());
        for (size_t i = 0; i < m_trace.size(); ++i)
        {
            trajectory[i] = m_trace.at(i).m_prediction;
        }
        return trajectory;
    }

    ///
    /// \brief LeastSquarespoly2
    /// \return
    ///
	void LeastSquarespoly2(size_t posFrom, size_t count, track_t& ax, track_t& v0x, track_t& x0, track_t& ay, track_t& v0y, track_t& y0) const
	{
		double b1_x(0), b2_x(0), b3_x(0);
		double b1_y(0), b2_y(0), b3_y(0);
		double t_0(0.), t_1(0.), t_2(0.), t_3(0.), t_4(0.);
        double j = static_cast<double>(posFrom);
		for (size_t i = posFrom; i < count; ++i, j += 1.)
		{
			double sqr_j = sqr(j);

			t_0 += 1.;
			t_1 += j;
			t_2 += sqr_j;
			t_3 += j * sqr_j;
			t_4 += sqr(sqr_j);

            const auto& pt = m_trace.at(i).m_prediction;

			b1_x += pt.x;
			b2_x += j * pt.x;
			b3_x += sqr_j * pt.x;

			b1_y += pt.y;
			b2_y += j * pt.y;
			b3_y += sqr_j * pt.y;
		}

		// Cramers rule for system of linear equations 3x3
        double a11(t_0), a12(t_1), a13(t_2), a21(t_1), a22(t_2), a23(t_3), a31(t_2), a32(t_3), a33(t_4);

        double det_1 = 1. / (a11 * a22 * a33 + a21 * a32 * a13 + a12 * a23 * a31 - a31 * a22 * a13 - a11 * a23 * a32 - a12 * a21 * a33);
		x0 =  static_cast<track_t>(det_1 * (b1_x * a22 * a33 + b2_x * a32 * a13 + a12 * a23 * b3_x - b3_x * a22 * a13 - b1_x * a23 * a32 - a12 * b2_x * a33));
		v0x = static_cast<track_t>(det_1 * (a11 * b2_x * a33 + a21 * b3_x * a13 + b1_x * a23 * a31 - a31 * b2_x * a13 - a11 * a23 * b3_x - b1_x * a21 * a33));
		ax =  static_cast<track_t>(det_1 * (a11 * a22 * b3_x + a21 * a32 * b1_x + a12 * b2_x * a31 - a31 * a22 * b1_x - a11 * b2_x * a32 - a12 * a21 * b3_x));
		y0 =  static_cast<track_t>(det_1 * (b1_y * a22 * a33 + b2_y * a32 * a13 + a12 * a23 * b3_y - b3_y * a22 * a13 - b1_y * a23 * a32 - a12 * b2_y * a33));
		v0y = static_cast<track_t>(det_1 * (a11 * b2_y * a33 + a21 * b3_y * a13 + b1_y * a23 * a31 - a31 * b2_y * a13 - a11 * a23 * b3_y - b1_y * a21 * a33));
		ay =  static_cast<track_t>(det_1 * (a11 * a22 * b3_y + a21 * a32 * b1_y + a12 * b2_y * a31 - a31 * a22 * b1_y - a11 * b2_y * a32 - a12 * a21 * b3_y));
	}

	///
	struct LSParams
	{
		track_t m_ax = 0;
		track_t m_v0x = 0;
		track_t m_x0 = 0;
		track_t m_ay = 0;
		track_t m_v0y = 0;
		track_t m_y0 = 0;

		friend std::ostream& operator<<(std::ostream& os, const LSParams& lsParaml)
		{
			os << "(" << lsParaml.m_ax << ", " << lsParaml.m_v0x << ", " << lsParaml.m_x0 << "), (" << lsParaml.m_ay << ", " << lsParaml.m_v0y << ", " << lsParaml.m_y0 << ")";
			return os;
		}
	};

    ///
    /// \brief LeastSquares2
    /// \return
    ///
    bool LeastSquares2(size_t framesCount, track_t& mean, track_t& stddev, LSParams& lsParams) const
    {
        bool res = m_trace.size() > 3;

        if (res)
        {
            size_t startPos = 0;
#if 0
            if (framesCount < m_trace.size())
                startPos = m_trace.size() - framesCount;
            else
                framesCount = m_trace.size();
#else
			framesCount = m_trace.size();
#endif

            LeastSquarespoly2(startPos, framesCount, lsParams.m_ax, lsParams.m_v0x, lsParams.m_x0, lsParams.m_ay, lsParams.m_v0y, lsParams.m_y0);

            track_t sum = 0;
            track_t sum2 = 0;
            for (size_t i = startPos; i < framesCount; ++i)
            {
                track_t t = static_cast<track_t>(i);
                track_t dist = distance<track_t>(m_trace[i], Point_t(lsParams.m_ax * sqr(t) + lsParams.m_v0x * t + lsParams.m_x0,
                                                                     lsParams.m_ay * sqr(t) + lsParams.m_v0y * t + lsParams.m_y0));
                sum += dist;
                sum2 += sqr(dist);
            }
            mean = sum / static_cast<track_t>(framesCount);
            stddev = sqrt(sum2 / static_cast<track_t>(framesCount) - sqr(mean));
        }
        return res;
    }

    ///
    /// \brief GetTrajectory
    /// \return
    ///
    cv::Rect GetBoundingRect() const
    {
        return m_rrect.boundingRect();
    }
};
