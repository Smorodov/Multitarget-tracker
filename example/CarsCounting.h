#pragma once

#include <cmath>
#include <mutex>
#include <unordered_set>
#include "VideoExample.h"

#ifdef USE_GEOGRAPHICLIB
#include <GeographicLib/Geocentric.hpp>
#include <GeographicLib/LocalCartesian.hpp>
#endif

/// Degrees to radians
constexpr double DEG_TO_RAD = 0.017453292519943295769236907684886;

/// WGS84 ellipsoid: equatorial radius (meters) and flattening
constexpr double WGS84_A = 6378137.0;
constexpr double WGS84_F = 1.0 / 298.257223563;

///
/// \brief The distance between the two points of the local plane in meters.
///
inline double DistanceInMeters(const cv::Point2d& p1, const cv::Point2d& p2)
{
	return cv::norm(p2 - p1);
}

///
/// \brief The LocalPlane class
///
/// The local ENU (East-North-Up) tangent plane in meters around the scene origin.
/// All the geographic coordinates (latitude/longitude) are projected into this plane, so:
///  - the perspective transform is estimated in the metric coordinates (meters) and not in
///    degrees (no float precision problems, no anisotropic longitude scale);
///  - the distance between the two points is a simple Euclidean distance in meters.
///
/// GeographicLib::LocalCartesian is used if USE_GEOGRAPHICLIB is defined, otherwise the
/// same ENU transformation is calculated directly (WGS84 ellipsoid).
///
class LocalPlane
{
public:
	///
	LocalPlane() = default;

	///
	LocalPlane(double lat0, double lon0)
	{
		SetOrigin(lat0, lon0);
	}

	///
	void SetOrigin(double lat0, double lon0)
	{
		m_lat0 = lat0;
		m_lon0 = lon0;
#ifdef USE_GEOGRAPHICLIB
		m_lc.Reset(lat0, lon0, 0.0);
#endif
	}

	///
	double OriginLat() const { return m_lat0; }

	///
	double OriginLon() const { return m_lon0; }

	///
	/// \brief Convert the geographic coordinates to the local plane.
	/// \param lat - latitude (degrees)
	/// \param lon - longitude (degrees)
	/// \return (East, North) in meters
	cv::Point2d ToLocal(double lat, double lon) const
	{
#ifdef USE_GEOGRAPHICLIB
		double east = 0.0, north = 0.0, up = 0.0;
		m_lc.Forward(lat, lon, 0.0, east, north, up);
		return cv::Point2d(east, north);
#else
		const double e2 = WGS84_F * (2.0 - WGS84_F);
		const double sinLat = std::sin(lat * DEG_TO_RAD), cosLat = std::cos(lat * DEG_TO_RAD);
		const double sinLon = std::sin(lon * DEG_TO_RAD), cosLon = std::cos(lon * DEG_TO_RAD);
		const double n = WGS84_A / std::sqrt(1.0 - e2 * sinLat * sinLat);
		const double x = n * cosLat * cosLon;
		const double y = n * cosLat * sinLon;
		const double z = n * (1.0 - e2) * sinLat;

		const double sinLat0 = std::sin(m_lat0 * DEG_TO_RAD), cosLat0 = std::cos(m_lat0 * DEG_TO_RAD);
		const double sinLon0 = std::sin(m_lon0 * DEG_TO_RAD), cosLon0 = std::cos(m_lon0 * DEG_TO_RAD);
		const double n0 = WGS84_A / std::sqrt(1.0 - e2 * sinLat0 * sinLat0);
		const double x0 = n0 * cosLat0 * cosLon0;
		const double y0 = n0 * cosLat0 * sinLon0;
		const double z0 = n0 * (1.0 - e2) * sinLat0;

		const double dx = x - x0, dy = y - y0, dz = z - z0;
		const double east = -sinLon0 * dx + cosLon0 * dy;
		const double north = -sinLat0 * cosLat0 * dx - sinLat0 * sinLon0 * dy + cosLat0 * dz;
		return cv::Point2d(east, north);
#endif
	}

	///
	/// \brief Convert the local plane coordinates back to the geographic ones.
	/// \param east - East (meters)
	/// \param north - North (meters)
	/// \param[out] lat - latitude (degrees)
	/// \param[out] lon - longitude (degrees)
	void ToGeo(double east, double north, double& lat, double& lon) const
	{
#ifdef USE_GEOGRAPHICLIB
		double up = 0.0;
		m_lc.Reverse(east, north, up, lat, lon, up);
#else
		const double e2 = WGS84_F * (2.0 - WGS84_F);
		const double sinLat0 = std::sin(m_lat0 * DEG_TO_RAD), cosLat0 = std::cos(m_lat0 * DEG_TO_RAD);
		const double sinLon0 = std::sin(m_lon0 * DEG_TO_RAD), cosLon0 = std::cos(m_lon0 * DEG_TO_RAD);
		const double n0 = WGS84_A / std::sqrt(1.0 - e2 * sinLat0 * sinLat0);
		const double x0 = n0 * cosLat0 * cosLon0;
		const double y0 = n0 * cosLat0 * sinLon0;
		const double z0 = n0 * (1.0 - e2) * sinLat0;

		const double dx = -sinLon0 * east - sinLat0 * cosLat0 * north;
		const double dy =  cosLon0 * east - sinLat0 * sinLon0 * north;
		const double dz =  cosLat0 * north;

		const double x = x0 + dx, y = y0 + dy, z = z0 + dz;
		const double p = std::sqrt(x * x + y * y);
		const double theta = std::atan2(z * WGS84_A, p * WGS84_A * std::sqrt(1.0 - e2));
		const double sinTheta = std::sin(theta), cosTheta = std::cos(theta);
		lat = std::atan2(z + WGS84_A * e2 * sinTheta * sinTheta * sinTheta,
						 p - WGS84_A * e2 * cosTheta * cosTheta * cosTheta) / DEG_TO_RAD;
		lon = std::atan2(y, x) / DEG_TO_RAD;
#endif
	}

	///
	/// \brief The distance between the two geographic points in meters.
	double DistanceInMeters(double lat1, double lon1, double lat2, double lon2) const
	{
		return ::DistanceInMeters(ToLocal(lat1, lon1), ToLocal(lat2, lon2));
	}

private:
	double m_lat0 = 0.0;
	double m_lon0 = 0.0;
#ifdef USE_GEOGRAPHICLIB
	GeographicLib::LocalCartesian m_lc;
#endif
};

///
/// \brief Normalize the points (Hartley normalization): shift them to the centroid and
///        scale them so that the mean distance to the centroid is sqrt(2).
///
inline void NormalizePoints(const std::vector<cv::Point2d>& pts, std::vector<cv::Point2d>& normPts, cv::Matx<double, 3, 3>& T)
{
	cv::Point2d mean(0.0, 0.0);
	for (const auto& p : pts)
		mean += p;
	mean *= 1.0 / static_cast<double>(pts.size());

	double meanDist = 0.0;
	for (const auto& p : pts)
		meanDist += cv::norm(p - mean);
	meanDist /= static_cast<double>(pts.size());

	const double scale = (meanDist > 1e-12) ? std::sqrt(2.0) / meanDist : 1.0;

	T = cv::Matx<double, 3, 3>(scale, 0.0, -scale * mean.x,
							   0.0, scale, -scale * mean.y,
							   0.0, 0.0, 1.0);

	normPts.resize(pts.size());
	for (size_t i = 0; i < pts.size(); ++i)
	{
		normPts[i] = cv::Point2d((pts[i].x - mean.x) * scale, (pts[i].y - mean.y) * scale);
	}
}

#if 0
// Prev version - copy from OpenCV
// Fix cv::getPerspectiveTransform for double
template<class T>
cv::Mat GetPerspectiveTransform(const std::vector<cv::Point_<T>>& src, const std::vector<cv::Point_<T>>& dst, int solveMethod = (int)cv::DECOMP_LU)
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
#else
///
/// \brief Estimate the perspective transform (homography) from the point correspondences.
///
/// The estimation is performed in double precision (float is not enough to represent the
/// geographic coordinates and to keep the matrix well conditioned) and uses all the given
/// points (not only the first 4 ones).
///
/// \param src - source points
/// \param dst - destination points
/// \return 3x3 homography matrix (CV_64F) or an empty matrix if the estimation failed
///
inline cv::Mat GetPerspectiveTransform(const std::vector<cv::Point2d>& src, const std::vector<cv::Point2d>& dst)
{
    const size_t count = std::min(src.size(), dst.size());
    if (count < 4)
    {
        std::cerr << "GetPerspectiveTransform: points less than 4: " << count << std::endl;
        return cv::Mat();
    }

    // Hartley normalization
    std::vector<cv::Point2d> nsrc, ndst;
    cv::Matx<double, 3, 3> Tsrc, Tdst;
    NormalizePoints(src, nsrc, Tsrc);
    NormalizePoints(dst, ndst, Tdst);

    // The direct linear transform system
    cv::Mat A(static_cast<int>(2 * count), 9, CV_64F);
    for (size_t i = 0; i < count; ++i)
    {
        const double x = nsrc[i].x, y = nsrc[i].y;
        const double u = ndst[i].x, v = ndst[i].y;
        double* r0 = A.ptr<double>(static_cast<int>(2 * i));
        double* r1 = A.ptr<double>(static_cast<int>(2 * i + 1));
        // u * (h6 * x + h7 * y + h8) = h0 * x + h1 * y + h2
        r0[0] = x; r0[1] = y; r0[2] = 1.0;
        r0[3] = 0.0; r0[4] = 0.0; r0[5] = 0.0;
        r0[6] = -u * x; r0[7] = -u * y; r0[8] = -u;
        // v * (h6 * x + h7 * y + h8) = h3 * x + h4 * y + h5
        r1[0] = 0.0; r1[1] = 0.0; r1[2] = 0.0;
        r1[3] = x; r1[4] = y; r1[5] = 1.0;
        r1[6] = -v * x; r1[7] = -v * y; r1[8] = -v;
    }

    // The null vector of the system is the right singular vector with the smallest singular value
    cv::Mat w, vh;
    cv::SVD::compute(A, w, cv::noArray(), vh, cv::SVD::FULL_UV);
    if (vh.rows < 9 || vh.cols < 9)
    {
        std::cerr << "GetPerspectiveTransform: cv::SVD::compute vh size less than 9: " << vh.size() << std::endl;
        return cv::Mat();
    }

	cv::Mat Hn(3, 3, CV_64F);
	const cv::Mat h = vh.row(vh.rows - 1).reshape(1, 9);
	for (int i = 0; i < 9; ++i)
		Hn.at<double>(i / 3, i % 3) = h.at<double>(i);

	// Undo the normalization: H = Tdst^-1 * Hn * Tsrc
	const cv::Matx<double, 3, 3> TdstInv( 1.0 / Tdst(0, 0), 0.0, -Tdst(0, 2) / Tdst(0, 0),
										   0.0, 1.0 / Tdst(1, 1), -Tdst(1, 2) / Tdst(1, 1),
										   0.0, 0.0, 1.0);
	cv::Mat H = cv::Mat(TdstInv) * Hn * cv::Mat(Tsrc);

	const double h33 = H.at<double>(2, 2);
	if (std::abs(h33) < 1e-12)
    {
        std::cerr << "GetPerspectiveTransform: std::abs(h33) < 1e-12: " << std::abs(h33) << std::endl;
        return cv::Mat();
    }
    H /= h33;

	return H;
}
#endif

///
/// \brief Convert the 3x3 matrix to the fixed size one.
///
inline cv::Matx<double, 3, 3> ToMatx(const cv::Mat& m)
{
	cv::Matx<double, 3, 3> res;
	for (int r = 0; r < 3; ++r)
	{
		for (int c = 0; c < 3; ++c)
			res(r, c) = m.at<double>(r, c);
	}
	return res;
}

///
/// \brief The minimal positive homogeneous coordinate w for a valid projection.
///
constexpr double kMinHomogeneousW = 1e-9;

///
/// \brief Check that the homogeneous coordinate of a projected point can be used
/// as the divisor: zero gives inf/nan, a negative one is a point behind the
/// plane (mirrored garbage coordinates).
///
inline bool ValidHomogeneousW(double w)
{
	return w > kMinHomogeneousW;
}

///
/// \brief Normalize the homography so that the homogeneous coordinate w is 1
/// at the given point of its source plane.
///
/// After the DLT solve (H /= h33) the sign and the scale of the homogeneous
/// part are arbitrary, while the projection x' = (H*x)/w is invariant to such
/// a scaling. Without the normalization the w > 0 checks in Pix2Ground /
/// Ground2Pix could reject perfectly valid points only because of a negative
/// overall scale. The scaling is element-wise (cv::Matx does not overload
/// operator* by a scalar in all OpenCV versions).
/// \return false if w at the given point is nearly zero.
///
inline bool NormalizeHomographyW(cv::Matx<double, 3, 3>& H, const cv::Point2d& at)
{
	const cv::Vec<double, 3> p = H * cv::Vec<double, 3>(at.x, at.y, 1.0);
	if (std::abs(p[2]) < kMinHomogeneousW)
		return false;
	const double s = 1.0 / p[2];
	for (int r = 0; r < 3; ++r)
	{
		for (int c = 0; c < 3; ++c)
			H(r, c) *= s;
	}
	return true;
}

///
/// \brief Check that the point cloud is not degenerate: no duplicated points
/// and no 3 points on one line.
///
/// An exactly collinear cloud is mapped by the DLT with a zero reprojection
/// error, but the resulting homography is meaningless outside the line, so an
/// explicit check is needed in addition to the reprojection error test.
/// \param pts - points (pixels or meters)
/// \param what - cloud name for the error messages
///
inline bool CheckPointsNotDegenerate(const std::vector<cv::Point2d>& pts, const char* what)
{
	// Diagonal of the cloud for the relative thresholds.
	double diag = 0.0;
	for (size_t i = 0; i < pts.size(); ++i)
	{
		for (size_t j = i + 1; j < pts.size(); ++j)
			diag = std::max(diag, cv::norm(pts[j] - pts[i]));
	}

	const double minDist = 1e-6 * diag;
	for (size_t i = 0; i < pts.size(); ++i)
	{
		for (size_t j = i + 1; j < pts.size(); ++j)
		{
			if (cv::norm(pts[j] - pts[i]) < minDist)
			{
				std::cerr << "CheckPointsNotDegenerate: " << what << " duplicated points: "
					<< pts[i] << " and " << pts[j] << std::endl;
				return false;
			}
		}
	}

	// Note the threshold: the points exactly on one line give a zero triangle
	// area, but the points derived from real geo coordinates through the
	// ellipsoid ToLocal are never exactly collinear (the curvature adds a
	// relative deviation of the order of the angular step, about 2e-6 of the
	// cloud diagonal per 0.001 deg step). 1e-5 * diag^2 still passes real
	// bindings (a lateral GPS jitter of 2 m over a 100 m span gives an area
	// of about 1e-2 * diag^2) and rejects the degenerate ones.
	const double minArea = 1e-5 * diag * diag;
	for (size_t i = 0; i + 2 < pts.size(); ++i)
	{
		for (size_t j = i + 1; j + 1 < pts.size(); ++j)
		{
			for (size_t k = j + 1; k < pts.size(); ++k)
			{
				const cv::Point2d d1 = pts[j] - pts[i];
				const cv::Point2d d2 = pts[k] - pts[i];
				if (std::abs(d1.x * d2.y - d1.y * d2.x) < minArea)
				{
					std::cerr << "CheckPointsNotDegenerate: " << what << " collinear points: "
						<< pts[i] << ", " << pts[j] << ", " << pts[k] << std::endl;
					return false;
				}
			}
		}
	}

	return true;
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
		m_groundPoints.clear();

		if (m_framePoints.size() != m_geoPoints.size() || m_framePoints.size() < 4)
		{
			m_valid = false;
			return false;
		}

		// The origin of the local plane is the center of the bound points
		double lat0 = 0.0, lon0 = 0.0;
		for (const auto& p : m_geoPoints)
		{
			lat0 += static_cast<double>(p.x);
			lon0 += static_cast<double>(p.y);
		}
		lat0 /= static_cast<double>(m_geoPoints.size());
		lon0 /= static_cast<double>(m_geoPoints.size());
		m_plane.SetOrigin(lat0, lon0);

		// Convert the geographic coordinates to the metric local plane (meters)
		std::vector<cv::Point2d> tmpPix, tmpGround;
		tmpPix.reserve(m_framePoints.size());
		tmpGround.reserve(m_geoPoints.size());
		m_groundPoints.reserve(m_geoPoints.size());
		for (size_t i = 0; i < m_framePoints.size(); ++i)
		{
			tmpPix.emplace_back(static_cast<double>(m_framePoints[i].x), static_cast<double>(m_framePoints[i].y));
			const cv::Point2d ground = m_plane.ToLocal(static_cast<double>(m_geoPoints[i].x), static_cast<double>(m_geoPoints[i].y));
			tmpGround.push_back(ground);
			m_groundPoints.emplace_back(static_cast<T>(ground.x), static_cast<T>(ground.y));
		}
#if 1
		std::cout << "SetKeyPoints: Coords pairs: ";
		for (size_t i = 0; i < tmpPix.size(); ++i)
		{
			std::cout << tmpPix[i] << " - " << m_groundPoints[i] << "; ";
		}
		std::cout << std::endl;
#endif

		// Reject the degenerate bindings: duplicated or collinear points give a
		// formally valid but meaningless homography.
		if (!CheckPointsNotDegenerate(tmpPix, "frame points") ||
			!CheckPointsNotDegenerate(tmpGround, "ground points"))
		{
			m_valid = false;
			return false;
		}

		// The perspective transforms are estimated in the metric coordinates (meters)
		const cv::Mat toGround = GetPerspectiveTransform(tmpPix, tmpGround);
		const cv::Mat toPix = GetPerspectiveTransform(tmpGround, tmpPix);
		if (toGround.empty() || toPix.empty())
		{
			std::cerr << "SetKeyPoints: toGround = " << toGround << ", toPix = " << toPix << std::endl;
			m_valid = false;
			return false;
		}
		m_toGround = ToMatx(toGround);
		m_toPix = ToMatx(toPix);

		// Make the homogeneous coordinate w equal to 1 at the scene centroids:
		// a projectively invariant normalization, without it the w > 0 checks
		// below would reject valid points because of an arbitrary sign/scale
		// of the solved homography. A centroid on the horizon line means a
		// degenerate binding anyway.
		cv::Point2d meanPix(0.0, 0.0), meanGround(0.0, 0.0);
		for (size_t i = 0; i < tmpPix.size(); ++i)
		{
			meanPix += tmpPix[i];
			meanGround += tmpGround[i];
		}
		meanPix *= 1.0 / static_cast<double>(tmpPix.size());
		meanGround *= 1.0 / static_cast<double>(tmpGround.size());
		if (!NormalizeHomographyW(m_toGround, meanPix) || !NormalizeHomographyW(m_toPix, meanGround))
		{
			std::cerr << "SetKeyPoints: the scene centroid is on the horizon line" << std::endl;
			m_valid = false;
			return false;
		}

		// Reprojection error check: a homography over a bad point configuration
		// must not pass silently. Thresholds are relative to the point clouds
		// (an exact 4-point solution gives an error of about 1e-10).
		double diagPix = 0.0, diagGround = 0.0;
		for (size_t i = 0; i < tmpPix.size(); ++i)
		{
			for (size_t j = i + 1; j < tmpPix.size(); ++j)
			{
				diagPix = std::max(diagPix, cv::norm(tmpPix[j] - tmpPix[i]));
				diagGround = std::max(diagGround, cv::norm(tmpGround[j] - tmpGround[i]));
			}
		}
		const double maxPixError = std::max(1.0, 0.01 * diagPix);
		const double maxGroundError = std::max(0.1, 0.01 * diagGround);
		for (size_t i = 0; i < tmpPix.size(); ++i)
		{
			cv::Point2d ground;
			cv::Point pix;
			const bool groundOk = Pix2Ground(cv::Point(tmpPix[i].x, tmpPix[i].y), ground) &&
				cv::norm(ground - tmpGround[i]) <= maxGroundError;
			const bool pixOk = Ground2Pix(tmpGround[i], pix) &&
				cv::norm(cv::Point2d(pix.x, pix.y) - tmpPix[i]) <= maxPixError;
			if (!groundOk || !pixOk)
			{
				std::cerr << "SetKeyPoints: reprojection error is too big for the point "
					<< i << ": pixel " << tmpPix[i] << ", ground " << tmpGround[i] << std::endl;
				m_valid = false;
				return false;
			}
		}

		std::cout << "To Ground: " << m_toGround << std::endl;
		std::cout << "To Pix: " << m_toPix << std::endl;

		m_valid = true;

		return true;
	}

    ///
    bool SetMapParams(const std::string& mapfile, const std::vector<cv::Point_<T>>& mapGeoCorners)
    {
        if (!m_valid)
            return false;

        m_map = cv::imread(mapfile);
        m_mapGeoCorners.assign(std::begin(mapGeoCorners), std::end(mapGeoCorners));

        if (!m_map.empty())
            m_p2mParams.Init(m_mapGeoCorners, m_map.cols, m_map.rows, m_plane);

        // The map is usable only together with a valid ground -> map transform.
        return !m_map.empty() && m_p2mParams.Valid();
    }

	///
	/// \brief Convert the local plane coordinates (East, North) to the frame pixels.
	/// \return false for a point projected behind the frame plane.
	///
	bool Ground2Pix(const cv::Point2d& ground, cv::Point& pix) const
	{
		cv::Vec<double, 3> g(ground.x, ground.y, 1.0);
		const cv::Vec<double, 3> p = m_toPix * g;
		if (!ValidHomogeneousW(p[2]))
			return false;
		pix = cv::Point(cvRound(p[0] / p[2]), cvRound(p[1] / p[2]));
		return true;
	}

	///
	/// \brief Convert the frame pixels to the local plane coordinates (East, North) in meters.
	/// \return false for the pixels on the horizon line or above it: the point is
	/// behind the ground plane, the division gives inf/nan or mirrored garbage.
	///
	bool Pix2Ground(const cv::Point& pix, cv::Point2d& ground) const
	{
		cv::Vec<double, 3> p(static_cast<double>(pix.x), static_cast<double>(pix.y), 1.0);
		const cv::Vec<double, 3> g = m_toGround * p;
		if (!ValidHomogeneousW(g[2]))
			return false;
		ground = cv::Point2d(g[0] / g[2], g[1] / g[2]);
		return true;
	}

	///
	/// \brief The local plane (metric coordinates) of the bound points.
	const LocalPlane& Plane() const
    {
        return m_plane;
    }

	///
	std::vector<cv::Point> GetFramePoints() const
	{
		return m_framePoints;
	}

	///
	bool Empty() const
	{
		return !m_valid;
	}

    ///
    cv::Mat DrawTracksOnMap(const std::vector<TrackingObject>& tracks)
    {
        if (m_map.empty() || !m_p2mParams.Valid())
            return cv::Mat();

        cv::Mat map = m_map.clone();

#if 1 // Debug output
        // Draw bindings points
        for (size_t i = 0; i < m_geoPoints.size(); ++i)
        {
            const cv::Point_<T>& g1 = m_groundPoints[i % m_groundPoints.size()];
            const cv::Point_<T>& g2 = m_groundPoints[(i + 1) % m_groundPoints.size()];
            cv::Point p1, p2;
            if (!m_p2mParams.Ground2Pix(cv::Point2d(g1.x, g1.y), p1) ||
                !m_p2mParams.Ground2Pix(cv::Point2d(g2.x, g2.y), p2))
            {
                continue;
            }

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
                cv::Point2d ground;
                cv::Point center;
                if (!Pix2Ground(track.m_rrect.center, ground) || !m_p2mParams.Ground2Pix(ground, center))
                    continue;
                //std::cout << "Convert: " << track.m_rrect.center << " -> " << ground << " -> " << center << std::endl;
                if (center.x > 0 && center.x < map.cols && center.y > 0 && center.y < map.rows)
                    cv::ellipse(map, center, cv::Size(5, 5), 0, 0, 360, cv::Scalar(255, 0, 255), cv::FILLED, 8);
            }
        }
 
        return map;
    }

private:
	std::vector<cv::Point> m_framePoints;
	std::vector<cv::Point_<T>> m_geoPoints;       // Geographic coordinates: x = lat, y = lon
	std::vector<cv::Point_<T>> m_groundPoints;    // Local plane coordinates (meters): x = East, y = North

	LocalPlane m_plane;

	cv::Matx<double, 3, 3> m_toGround;   // Frame pixels -> local plane (meters)
	cv::Matx<double, 3, 3> m_toPix;      // Local plane (meters) -> frame pixels

	bool m_valid = false;

    cv::Mat m_map;
    std::vector<cv::Point_<T>> m_mapGeoCorners;

    ///
    /// \brief Frame ground plane (meters) -> map image pixels transform.
    ///
    struct Pix2MapParams
    {
        cv::Matx<double, 3, 3> m_toPix;
        bool m_valid = false;

        ///
        bool Valid() const
        {
            return m_valid;
        }

        ///
        bool Ground2Pix(const cv::Point2d& ground, cv::Point& pix) const
        {
            if (!m_valid)
                return false;

            cv::Vec<double, 3> g(ground.x, ground.y, 1.0);
            const cv::Vec<double, 3> p = m_toPix * g;
            if (!ValidHomogeneousW(p[2]))
                return false;

            pix = cv::Point(cvRound(p[0] / p[2]), cvRound(p[1] / p[2]));
            return true;
        }

        ///
        void Init(const std::vector<cv::Point_<T>>& geoCorners, int mapWidth, int mapHeight, const LocalPlane& plane)
        {
            m_valid = false;

            std::vector<cv::Point2d> groundCorners;
            groundCorners.reserve(geoCorners.size());
            for (const auto& c : geoCorners)
            {
                const cv::Point2d g = plane.ToLocal(static_cast<double>(c.x), static_cast<double>(c.y));
                groundCorners.push_back(g);
            }

            // Duplicated or collinear map corners give a degenerate transform.
            if (!CheckPointsNotDegenerate(groundCorners, "map ground corners"))
                return;

            const std::vector<cv::Point2d> frameCorners { cv::Point2d(0.0, 0.0), cv::Point2d(static_cast<double>(mapWidth), 0.0),
                                                          cv::Point2d(static_cast<double>(mapWidth), static_cast<double>(mapHeight)), cv::Point2d(0.0, static_cast<double>(mapHeight)) };
            const cv::Mat toPix = GetPerspectiveTransform(groundCorners, frameCorners);
            if (toPix.empty())
                return;

            m_toPix = ToMatx(toPix);

            // w = 1 at the centroid of the ground corners: see NormalizeHomographyW.
            cv::Point2d meanGround(0.0, 0.0);
            for (const auto& g : groundCorners)
                meanGround += g;
            meanGround *= 1.0 / static_cast<double>(groundCorners.size());
            if (!NormalizeHomographyW(m_toPix, meanGround))
                return;

            m_valid = true;
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
                pt.y = (m_pt1.y + m_pt2.y) / 2.f - 0.01f;
            }
            else
            {
                if ((m_pt1.x <= m_pt2.x) && (m_pt1.y <= m_pt2.y))
                {
                    pt.x = (m_pt1.x + m_pt2.x) / 2.f + 0.01f;
                    pt.y = (m_pt1.y + m_pt2.y) / 2.f - 0.01f;
                }
                else
                {
                    if ((m_pt1.x > m_pt2.x) && (m_pt1.y > m_pt2.y))
                    {
                        pt.x = (m_pt1.x + m_pt2.x) / 2.f - 0.01f;
                        pt.y = (m_pt1.y + m_pt2.y) / 2.f + 0.01f;
                    }
                    else
                    {
                        if ((m_pt1.x > m_pt2.x) && (m_pt1.y <= m_pt2.y))
                        {
                            pt.x = (m_pt1.x + m_pt2.x) / 2.f + 0.01f;
                            pt.y = (m_pt1.y + m_pt2.y) / 2.f + 0.01f;
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

#ifdef H12_TEST_ACCESS
    // Test-only hook: build/h12_test checks the degenerate geometry of
    // CheckIntersection directly (roadmap H12).
    friend class H12Test;
#endif

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

        auto InRange = [](float val, float minVal, float  maxVal) -> bool
        {
            return (val >= minVal) && (val <= maxVal);
        };

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

        if (trivial1 && trivial2)
        {
            // Both lines are vertical: there is no unique intersection point,
            // so the InRange check of the point does not apply. The segments
            // meet only if they lie on the same vertical line and their
            // y-ranges overlap.
            return (fabs(m_pt1.x - pt1.x) < eps) &&
                (std::min(m_pt1.y, m_pt2.y) - eps <= std::max(pt1.y, pt2.y) + eps) &&
                (std::min(pt1.y, pt2.y) - eps <= std::max(m_pt1.y, m_pt2.y) + eps);
        }

        // Intersection coords (initialized: the reads of the uninitialized
        // value below were undefined behavior)
        cv::Point2f intersectPt(0.f, 0.f);

        if (trivial1)
        {
            intersectPt.x = m_pt1.x;
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
                {
                    // Parallel or coincident non-vertical lines: no unique
                    // intersection point, conservatively no crossing (a
                    // collinear overlap is not resolved either).
                    return false;
                }
            }
        }

        if (trivial1)
            intersectPt.y = a2 * intersectPt.x + b2;
        else
            intersectPt.y = a1 * intersectPt.x + b1;

        const bool isIntersect =
                InRange(intersectPt.x, std::min(m_pt1.x, m_pt2.x) - eps, std::max(m_pt1.x, m_pt2.x) + eps) &&
                InRange(intersectPt.x, std::min(pt1.x, pt2.x) - eps, std::max(pt1.x, pt2.x) + eps) &&
                InRange(intersectPt.y, std::min(m_pt1.y, m_pt2.y) - eps, std::max(m_pt1.y, m_pt2.y) + eps) &&
                InRange(intersectPt.y, std::min(pt1.y, pt2.y) - eps, std::max(pt1.y, pt2.y) + eps);

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

#ifdef H10_TEST_ACCESS
	// Test-only hook: build/h10_test checks that InitTracker really uses the
	// [tracking] settings parsed from the ini file (roadmap H10).
	friend class H10Test;
#endif

	bool m_drawHeatMap = false;

    bool InitDetector(cv::UMat frame) override;
    bool InitTracker(cv::UMat frame) override;

    void Detection(FrameInfo& frame) override;
    void DrawData(cv::Mat frame, const std::vector<TrackingObject>& tracks, int framesCounter, int currTime) override;
    void DrawTrack(cv::Mat frame, const TrackingObject& track, bool drawTrajectory, int framesCounter, const std::string& userLabel = "") override;

    // Road lines
    std::deque<RoadLine> m_lines;
    void CheckLinesIntersection(const TrackingObject& track, float xMax, float yMax);

	// Binding frame coordinates to geographical coordinates.
	// std::call_once: ReadGeobindings may be called by Detection() from the
	// capture thread (async mode) or by InitTracker() from the main thread,
	// whichever comes first; the other one blocks until it is done, so
	// m_geoParams is never initialized twice or read while being written.
	GeoParams<double> m_geoParams;
	std::string m_geoBindFile;
	std::once_flag m_geoBindingsOnce;
	bool ReadGeobindings(cv::Size frameSize);

	// Heat map for visualization long term detections
	cv::Mat m_keyFrame;
	cv::Mat m_heatMap;
	cv::Mat m_normHeatMap;
	cv::Mat m_colorMap;

	void AddToHeatMap(const cv::Rect& rect);
	cv::Mat DrawHeatMap();
};
