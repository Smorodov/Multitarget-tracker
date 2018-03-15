#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>


// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
typedef float track_t;
typedef cv::Point_<track_t> Point_t;
#define Mat_t CV_32FC

///
/// \brief The CRegion class
///
class CRegion
{
public:
    CRegion()
        : m_type(""), m_confidence(-1)
    {
    }

    CRegion(const cv::Rect& rect)
        : m_rect(rect)
    {

    }

    CRegion(const cv::Rect& rect, const std::string& type, float confidence)
        : m_rect(rect), m_type(type), m_confidence(confidence)
    {

    }

    cv::Rect m_rect;
    std::vector<cv::Point2f> m_points;

    std::string m_type;
    float m_confidence;
};

typedef std::vector<CRegion> regions_t;

///
///
///
namespace tracking
{
///
enum Detectors
{
    Motion_VIBE,
    Motion_MOG,
    Motion_GMG,
    Motion_CNT,
    Motion_SuBSENSE,
    Motion_LOBSTER,
    Motion_MOG2,
    Face_HAAR,
    Pedestrian_HOG,
    Pedestrian_C4,
    SSD_MobileNet,
    Yolo
};

///
/// \brief The DistType enum
///
enum DistType
{
    DistCenters = 0,
    DistRects = 1,
    DistJaccard = 2
    //DistLines = 3
};

///
/// \brief The FilterGoal enum
///
enum FilterGoal
{
    FilterCenter = 0,
    FilterRect = 1
};

///
/// \brief The KalmanType enum
///
enum KalmanType
{
    KalmanLinear = 0,
    KalmanUnscented = 1,
    KalmanAugmentedUnscented
};

///
/// \brief The MatchType enum
///
enum MatchType
{
    MatchHungrian = 0,
    MatchBipart = 1
};

///
/// \brief The LostTrackType enum
///
enum LostTrackType
{
    TrackNone = 0,
    TrackKCF = 1,
    TrackMIL,
    TrackMedianFlow,
    TrackGOTURN,
    TrackMOSSE
};
}
