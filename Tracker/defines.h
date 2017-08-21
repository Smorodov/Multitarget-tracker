#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
typedef float track_t;
typedef cv::Point_<track_t> Point_t;
#define Mat_t CV_32FC

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
class CRegion
{
public:
    CRegion()
    {
    }

    CRegion(const cv::Rect& rect)
        : m_rect(rect)
    {

    }

    cv::Rect m_rect;
    std::vector<cv::Point2f> m_points;
};

typedef std::vector<CRegion> regions_t;

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
namespace tracking
{
enum DistType
{
    DistCenters = 0,
    DistRects = 1,
    DistJaccard = 2
};
enum FilterGoal
{
    FilterCenter = 0,
    FilterRect = 1
};
enum KalmanType
{
    KalmanLinear = 0,
    KalmanUnscented = 1,
    KalmanAugmentedUnscented
};
enum MatchType
{
    MatchHungrian = 0,
    MatchBipart = 1
};
enum LostTrackType
{
    TrackNone = 0,
    TrackKCF = 1,
    TrackMIL
};
}
