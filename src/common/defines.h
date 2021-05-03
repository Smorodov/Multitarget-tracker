#pragma once

#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
#include "object_types.h"

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
typedef float track_t;
typedef cv::Point_<track_t> Point_t;
#define El_t CV_32F
#define Mat_t CV_32FC

typedef std::vector<int> assignments_t;
typedef std::vector<track_t> distMatrix_t;

///
template<typename T>
class TrackID
{
public:
    typedef T value_type;

    TrackID() = default;
    TrackID(const TrackID& id) = default;
    TrackID(TrackID&& id) = default;
    TrackID& operator=(TrackID&& id) noexcept
    {
        m_val = std::move(id.m_val);
        return *this;
    }
    TrackID& operator=(const TrackID& id) noexcept
    {
        m_val = id.m_val;
        return *this;
    }
    TrackID(T val)
        : m_val(val)
    {
    }
    const TrackID& operator=(T val)
    {
        m_val = val;
    }

    bool operator==(const TrackID& id) const
    {
        return m_val == id.m_val;
    }

    std::string ID2Str() const
    {
        return std::to_string(m_val);
    }
    static TrackID Str2ID(const std::string& id)
    {
        return TrackID(std::stoi(id));
    }
    TrackID NextID() const
    {
        return TrackID(m_val + 1);
    }
    size_t ID2Module(size_t module) const
    {
        return m_val % module;
    }

    T m_val{ 0 };
};

typedef TrackID<size_t> track_id_t;
namespace std
{
  template <>
  struct hash<track_id_t>
  {
    std::size_t operator()(const track_id_t& k) const
    {
      return std::hash<track_id_t::value_type>()(k.m_val);
    }
  };

}

///
/// \brief config_t
///
typedef std::multimap<std::string, std::string> config_t;

///
/// \brief The CRegion class
///
class CRegion
{
public:
	CRegion() = default;

    CRegion(const cv::Rect& rect) noexcept
        : m_brect(rect)
    {
        B2RRect();
    }

    CRegion(const cv::RotatedRect& rrect) noexcept
        : m_rrect(rrect)
    {
        R2BRect();
    }

    CRegion(const cv::Rect& rect, objtype_t type, float confidence) noexcept
        : m_type(type), m_brect(rect), m_confidence(confidence)
    {
        B2RRect();
    }

	objtype_t m_type = bad_type;
    cv::RotatedRect m_rrect;
    cv::Rect m_brect;
	float m_confidence = -1;

private:
    ///
    /// \brief R2BRect
    /// \return
    ///
    cv::Rect R2BRect() noexcept
    {
        m_brect = m_rrect.boundingRect();
        return m_brect;
    }
    ///
    /// \brief B2RRect
    /// \return
    ///
    cv::RotatedRect B2RRect() noexcept
    {
        m_rrect = cv::RotatedRect(m_brect.tl(), cv::Point2f(static_cast<float>(m_brect.x + m_brect.width), static_cast<float>(m_brect.y)), m_brect.br());
        return m_rrect;
    }
};

typedef std::vector<CRegion> regions_t;

///
///
///
namespace tracking
{
///
/// \brief The Detectors enum
///
enum Detectors
{
    Motion_VIBE = 0,
    Motion_MOG = 1,
    Motion_GMG = 2,
    Motion_CNT = 3,
    Motion_SuBSENSE = 4,
    Motion_LOBSTER = 5,
    Motion_MOG2 = 6,
    Face_HAAR = 7,
    Pedestrian_HOG = 8,
    Pedestrian_C4 = 9,
    Yolo_Darknet = 10,
    Yolo_TensorRT = 11,
    DNN_OCV = 12,
    DetectorsCount
};

///
/// \brief The DistType enum
///
enum DistType
{
    DistCenters,    // Euclidean distance between centers, [0, 1]
    DistRects,      // Euclidean distance between bounding rectangles, [0, 1]
    DistJaccard,    // Intersection over Union, IoU, [0, 1]
    DistHist,       // Bhatacharia distance between histograms, [0, 1]
    DistFeatureCos, // Cosine distance between embeddings, [0, 1]
    DistsCount
};

///
/// \brief The FilterGoal enum
///
enum FilterGoal
{
    FilterCenter,
    FilterRect,
    FiltersCount
};

///
/// \brief The KalmanType enum
///
enum KalmanType
{
    KalmanLinear,
    KalmanUnscented,
    KalmanAugmentedUnscented,
    KalmanCount
};

///
/// \brief The MatchType enum
///
enum MatchType
{
    MatchHungrian,
    MatchBipart,
    MatchCount
};

///
/// \brief The LostTrackType enum
///
enum LostTrackType
{
    TrackNone,
    TrackKCF,
    TrackMIL,
    TrackMedianFlow,
    TrackGOTURN,
    TrackMOSSE,
    TrackCSRT,
    TrackDAT,
    TrackSTAPLE,
    TrackLDES,
    SingleTracksCount
};
}
