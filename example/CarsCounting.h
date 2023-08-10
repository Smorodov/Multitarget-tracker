#pragma once

#include <unordered_set>
#include "VideoExample.h"

///
constexpr double DEG_TO_RAD = 0.017453292519943295769236907684886;
constexpr double EARTH_RADIUS_IN_METERS = 6372797.560856;

template<typename T>
T Haversine(const cv::Point_<T>& from, const cv::Point_<T>& to)
{
	constexpr T Deg2Rad = static_cast<T>(DEG_TO_RAD);

	T lat_arc = (from.x - to.x) * Deg2Rad;
	T lon_arc = (from.y - to.y) * Deg2Rad;
	T lat_h = sin(lat_arc * static_cast<T>(0.5));
	lat_h *= lat_h;
	T lon_h = sin(lon_arc * static_cast<T>(0.5));
	lon_h *= lon_h;
	T tmp = cos(from.x * Deg2Rad) * cos(to.y * Deg2Rad);
	return static_cast<T>(2.0) * asin(sqrt(lat_h + tmp * lon_h));
}

///
template<typename T>
T DistanceInMeters(const cv::Point_<T>& from, const cv::Point_<T>& to)
{
	constexpr T EarthRadius = static_cast<T>(EARTH_RADIUS_IN_METERS);
	return EarthRadius * Haversine(from, to);
}

// Fix cv::getPerspectiveTransform for double
template<class T>
cv::Mat GetPerspectiveTransform(const std::vector<cv::Point_<T>>& src, const std::vector<cv::Point_<T>>& dst, int solveMethod)
{
    cv::Mat M(3, 3, CV_64F), X(8, 1, CV_64F, M.ptr());
    double a[8][8], b[8];
    cv::Mat A(8, 8, CV_64F, a), B(8, 1, CV_64F, b);

    for (int i = 0; i < 4; ++i)
    {
        a[i][0] = a[i + 4][3] = src[i].x;
        a[i][1] = a[i + 4][4] = src[i].y;
        a[i][2] = a[i + 4][5] = 1;
        a[i][3] = a[i][4] = a[i][5] = a[i + 4][0] = a[i + 4][1] = a[i + 4][2] = 0;
        a[i][6] = -src[i].x * dst[i].x;
        a[i][7] = -src[i].y * dst[i].x;
        a[i + 4][6] = -src[i].x * dst[i].y;
        a[i + 4][7] = -src[i].y * dst[i].y;
        b[i] = dst[i].x;
        b[i + 4] = dst[i].y;
    }

    cv::solve(A, B, X, solveMethod);
    M.ptr<double>()[8] = 1.;

    return M;
}

///
/// \brief The GeoParams class
///
template<typename T>
class GeoParams
{
public:
	///
	GeoParams() = default;

	///
	GeoParams(const std::vector<cv::Point>& framePoints, const std::vector<cv::Point_<T>>& geoPoints)
	{
		SetKeyPoints(framePoints, geoPoints);
	}

	///
	bool SetKeyPoints(const std::vector<cv::Point>& framePoints, const std::vector<cv::Point_<T>>& geoPoints)
	{
		m_framePoints = framePoints;
		m_geoPoints = geoPoints;

		assert(m_framePoints.size() == m_geoPoints.size());
		assert(m_framePoints.size() >= 4);

		bool res = true;

		std::vector<cv::Point_<T>> tmpPix;
		tmpPix.reserve(m_framePoints.size());
		for (const auto& pix : m_framePoints)
		{
			tmpPix.emplace_back(static_cast<T>(pix.x), static_cast<T>(pix.y));
		}
#if 0
		std::cout << "Coords pairs: ";
		for (size_t i = 0; i < tmpPix.size(); ++i)
		{
			std::cout << tmpPix[i] << " - " << m_geoPoints[i] << "; ";
		}
		std::cout << std::endl;
#endif
		cv::Mat toGeo = GetPerspectiveTransform(tmpPix, m_geoPoints, (int)cv::DECOMP_LU);
		cv::Mat toPix = GetPerspectiveTransform(m_geoPoints, tmpPix, (int)cv::DECOMP_LU);
		m_toGeo = toGeo;
		m_toPix = toPix;
		//std::cout << "To Geo: " << m_toGeo << std::endl;
		//std::cout << "To Pix: " << m_toPix << std::endl;

		return res;
	}

    ///
    bool SetMapParams(const std::string& mapfile, const std::vector<cv::Point_<T>>& mapGeoCorners)
    {
        m_map = cv::imread(mapfile);
        m_mapGeoCorners.assign(std::begin(mapGeoCorners), std::end(mapGeoCorners));

        if (!m_map.empty())
            m_p2mParams.Init(m_mapGeoCorners, m_map.cols, m_map.rows);

        return !m_map.empty();
    }

	///
	cv::Point Geo2Pix(const cv::Point_<T>& geo) const
	{
		cv::Vec<T, 3> g(geo.x, geo.y, 1);
		auto p = m_toPix * g;
		return cv::Point(cvRound(p[0] / p[2]), cvRound(p[1] / p[2]));
	}

	///
	cv::Point_<T> Pix2Geo(const cv::Point& pix) const
	{
		cv::Vec<T, 3> p(static_cast<T>(pix.x), static_cast<T>(pix.y), 1);
		auto g = m_toGeo * p;
		return cv::Point_<T>(g[0] / g[2], g[1] / g[2]);
	}

	///
	std::vector<cv::Point> GetFramePoints() const
	{
		return m_framePoints;
	}

	///
	bool Empty() const
	{
		return m_framePoints.size() != m_geoPoints.size() || m_framePoints.size() < 4;
	}

    ///
    cv::Mat DrawTracksOnMap(const std::vector<TrackingObject>& tracks)
    {
        if (m_map.empty())
            return cv::Mat();

        cv::Mat map = m_map.clone();

#if 1 // Debug output
        // Draw bindings points
        for (size_t i = 0; i < m_geoPoints.size(); ++i)
        {
            cv::Point_<T> gp1(m_geoPoints[i % m_geoPoints.size()]);
            cv::Point p1(m_p2mParams.Geo2Pix(gp1));

            cv::Point_<T> gp2(m_geoPoints[(i + 1) % m_geoPoints.size()]);
            cv::Point p2(m_p2mParams.Geo2Pix(gp2));

            cv::line(map, p1, p2, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

            //std::cout << p1 << " - " << p2 << std::endl;

            std::stringstream label;
            label << std::fixed << std::setw(5) << std::setprecision(5) << "[" << m_geoPoints[i].x << ", " << m_geoPoints[i].y << "]";
            int baseLine = 0;
            double fontScale = (map.cols < 1000 && map.rows < 1000) ? 0.5 : 1.;
            cv::Size labelSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_TRIPLEX, fontScale, 1, &baseLine);
            cv::putText(map, label.str(), p1, cv::FONT_HERSHEY_TRIPLEX, fontScale, cv::Scalar(255, 255, 255));
        }
#endif

        // Draw tracks
        for (const auto& track : tracks)
        {
            if (track.m_lastRobust)
            {
                auto geoCenter = Pix2Geo(track.m_rrect.center);
                auto center = m_p2mParams.Geo2Pix(geoCenter);
                //std::cout << "Convert: " << track.m_rrect.center << " -> " << geoCenter << " -> " << center << std::endl;
                if (center.x > 0 && center.x < map.cols && center.y > 0 && center.y < map.rows)
                    cv::ellipse(map, center, cv::Size(5, 5), 0, 0, 360, cv::Scalar(255, 0, 255), cv::FILLED, 8);
            }
        }
 
        return map;
    }

private:
	std::vector<cv::Point> m_framePoints;
	std::vector<cv::Point_<T>> m_geoPoints;

	cv::Matx<T, 3, 3> m_toGeo;
	cv::Matx<T, 3, 3> m_toPix;

    cv::Mat m_map;
    std::vector<cv::Point_<T>> m_mapGeoCorners;

    ///
    struct Pix2MapParams
    {
        cv::Matx<T, 3, 3> m_toPix;

        ///
        cv::Point Geo2Pix(const cv::Point_<T>& geo) const
        {
            cv::Vec<T, 3> g(geo.x, geo.y, 1);
            auto p = m_toPix * g;
            return cv::Point(cvRound(p[0] / p[2]), cvRound(p[1] / p[2]));
        }

        ///
        void Init(const std::vector<cv::Point_<T>>& geoCorners, int mapWidth, int mapHeight)
        {
            std::vector<cv::Point_<T>> frameCorners { cv::Point_<T>(0, 0), cv::Point_<T>(mapWidth, 0), cv::Point_<T>(mapWidth, mapHeight), cv::Point_<T>(0, mapHeight) };
            cv::Mat toPix = GetPerspectiveTransform(geoCorners, frameCorners, (int)cv::DECOMP_LU);
            m_toPix = toPix;
        }
    };
    Pix2MapParams m_p2mParams;
};

///
/// \brief The RoadLine struct
///
class RoadLine
{
public:
    ///
    /// \brief RoadLine
    ///
    RoadLine() = default;
    RoadLine(const cv::Point2f& pt1, const cv::Point2f& pt2, unsigned int uid)
        :
          m_pt1(pt1), m_pt2(pt2), m_uid(uid)
    {
    }

    cv::Point2f m_pt1;
    cv::Point2f m_pt2;

    unsigned int m_uid = 0;

    int m_intersect1 = 0;
    int m_intersect2 = 0;

    ///
    /// \brief operator ==
    /// \param line
    /// \return
    ///
    bool operator==(const RoadLine &line) const
    {
        return line.m_uid == m_uid;
    }

    ///
    /// \brief Draw
    /// \param frame
    ///
    void Draw(cv::Mat frame) const
    {
        auto Ptf2i = [&](cv::Point2f pt)
        {
            return cv::Point(cvRound(frame.cols * pt.x), cvRound(frame.rows * pt.y));
        };

        cv::line(frame, Ptf2i(m_pt1), Ptf2i(m_pt2), cv::Scalar(0, 255, 255), 1, cv::LINE_8, 0);

        std::string label = "Line " + std::to_string(m_uid) + ": " + std::to_string(m_intersect1) + "/" + std::to_string(m_intersect2);
		int baseLine = 0;
		double fontScale = 0.7;
		int thickness = 1;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontScale, 1, &baseLine);
		cv::Point pt(Ptf2i(0.5f * (m_pt1 + m_pt2)));
		pt.y += labelSize.height;
		//pt.x += labelSize.width;
        cv::putText(frame, label, pt, cv::FONT_HERSHEY_TRIPLEX, fontScale, cv::Scalar(0, 0, 0), thickness);
    }

    ///
    /// \brief IsIntersect
    /// \param pt1
    /// \param pt2
    /// \return
    ///
    int IsIntersect(track_id_t objID, cv::Point2f pt1, cv::Point2f pt2)
    {
        int direction = 0;

        if (m_lastIntersections.find(objID) != m_lastIntersections.end())
            return direction;

        bool isIntersect = CheckIntersection(pt1, pt2);

        if (isIntersect)
        {
            m_lastIntersections.emplace(objID);

            cv::Point2f pt;
            if ((m_pt1.x <= m_pt2.x) && (m_pt1.y > m_pt2.y))
            {
                pt.x = (m_pt1.x + m_pt2.x) / 2.f - 0.01f;
                pt.y = (m_pt1.y + m_pt1.y) / 2.f - 0.01f;
            }
            else
            {
                if ((m_pt1.x <= m_pt2.x) && (m_pt1.y <= m_pt2.y))
                {
                    pt.x = (m_pt1.x + m_pt2.x) / 2.f + 0.01f;
                    pt.y = (m_pt1.y + m_pt1.y) / 2.f - 0.01f;
                }
                else
                {
                    if ((m_pt1.x > m_pt2.x) && (m_pt1.y > m_pt2.y))
                    {
                        pt.x = (m_pt1.x + m_pt2.x) / 2.f - 0.01f;
                        pt.y = (m_pt1.y + m_pt1.y) / 2.f + 0.01f;
                    }
                    else
                    {
                        if ((m_pt1.x > m_pt2.x) && (m_pt1.y <= m_pt2.y))
                        {
                            pt.x = (m_pt1.x + m_pt2.x) / 2.f + 0.01f;
                            pt.y = (m_pt1.y + m_pt1.y) / 2.f + 0.01f;
                        }
                    }
                }
            }
            if (CheckIntersection(pt1, pt))
            {
                direction = 1;
                ++m_intersect1;
            }
            else
            {
                direction = 2;
                ++m_intersect2;
            }
        }

        return direction;
    }

private:

    std::unordered_set<track_id_t> m_lastIntersections;

    ///
    /// \brief CheckIntersection
    /// \param pt1
    /// \param pt2
    /// \return
    ///
    bool CheckIntersection(cv::Point2f pt1, cv::Point2f pt2) const
    {
        const float eps = 0.00001f; // Epsilon for equal comparing

        // First line equation
        float a1 = 0;
        float b1 = 0;
        bool trivial1 = false; // Is first line is perpendicular with OX

        if (fabs(m_pt1.x - m_pt2.x) < eps)
        {
            trivial1 = true;
        }
        else
        {
            a1 = (m_pt2.y - m_pt1.y) / (m_pt2.x - m_pt1.x);
            b1 = (m_pt2.x * m_pt1.y - m_pt1.x * m_pt2.y) / (m_pt2.x - m_pt1.x);
        }

        // Second line equation
        float a2 = 0;
        float b2 = 0;
        bool trivial2 = false; // Is second line is perpendicular with OX

        if (fabs(pt1.x - pt2.x) < eps)
        {
            trivial2 = true;
        }
        else
        {
            a2 = (pt2.y - pt1.y) / (pt2.x - pt1.x);
            b2 = (pt2.x * pt1.y - pt1.x * pt2.y) / (pt2.x - pt1.x);
        }

        // Intersection coords
        cv::Point2f intersectPt;

        bool isIntersect = true;
        if (trivial1)
        {
            if (trivial2)
                isIntersect = (fabs(m_pt1.x - pt1.x) < eps);
            else
                intersectPt.x = m_pt1.x;

            intersectPt.y = a2 * intersectPt.x + b2;
        }
        else
        {
            if (trivial2)
            {
                intersectPt.x = pt1.x;
            }
            else
            {
                if (fabs(a2 - a1) > eps)
                    intersectPt.x = (b1 - b2) / (a2 - a1);
                else
                    isIntersect = false;
            }
            intersectPt.y = a1 * intersectPt.x + b1;
        }

        if (isIntersect)
        {
            auto InRange = [](float val, float minVal, float  maxVal) -> bool
            {
                return (val >= minVal) && (val <= maxVal);
            };

            isIntersect = InRange(intersectPt.x, std::min(m_pt1.x, m_pt2.x) - eps, std::max(m_pt1.x, m_pt2.x) + eps) &&
                    InRange(intersectPt.x, std::min(pt1.x, pt2.x) - eps, std::max(pt1.x, pt2.x) + eps) &&
                    InRange(intersectPt.y, std::min(m_pt1.y, m_pt2.y) - eps, std::max(m_pt1.y, m_pt2.y) + eps) &&
                    InRange(intersectPt.y, std::min(pt1.y, pt2.y) - eps, std::max(pt1.y, pt2.y) + eps);
        }

        return isIntersect;
    }
};

///
/// \brief The CarsCounting class
///
class CarsCounting final : public VideoExample
{
public:
    CarsCounting(const cv::CommandLineParser& parser);

    // Lines API
    void AddLine(const RoadLine& newLine);
    bool GetLine(unsigned int lineUid, RoadLine& line);
    bool RemoveLine(unsigned int lineUid);

private:

	bool m_drawHeatMap = false;

    bool InitDetector(cv::UMat frame) override;
    bool InitTracker(cv::UMat frame) override;

    void DrawData(cv::Mat frame, const std::vector<TrackingObject>& tracks, int framesCounter, int currTime) override;
    void DrawTrack(cv::Mat frame, const TrackingObject& track, bool drawTrajectory, int framesCounter) override;

    // Road lines
    std::deque<RoadLine> m_lines;
    void CheckLinesIntersection(const TrackingObject& track, float xMax, float yMax);

	// Binding frame coordinates to geographical coordinates
	GeoParams<double> m_geoParams;
	std::string m_geoBindFile;
	bool ReadGeobindings(cv::Size frameSize);

	// Heat map for visualization long term detections
	cv::Mat m_keyFrame;
	cv::Mat m_heatMap;
	cv::Mat m_normHeatMap;
	cv::Mat m_colorMap;

	void AddToHeatMap(const cv::Rect& rect);
	cv::Mat DrawHeatMap();
};
