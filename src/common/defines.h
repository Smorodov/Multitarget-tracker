#pragma once

#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>


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
/// \brief config_t
///
typedef std::multimap<std::string, std::string> config_t;

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
        : m_brect(rect)
    {
        B2RRect();
    }

    CRegion(const cv::RotatedRect& rrect)
        : m_rrect(rrect)
    {
        R2BRect();
    }

    CRegion(const cv::Rect& rect, const std::string& type, float confidence)
        : m_type(type), m_brect(rect), m_confidence(confidence)
    {
        B2RRect();
    }

	std::string m_type;
    cv::RotatedRect m_rrect;
    cv::Rect m_brect;
	float m_confidence = -1;

private:
    ///
    /// \brief R2BRect
    /// \return
    ///
    cv::Rect R2BRect()
    {
        m_brect = m_rrect.boundingRect();
        return m_brect;
    }
    ///
    /// \brief B2RRect
    /// \return
    ///
    cv::RotatedRect B2RRect()
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
    SSD_MobileNet = 10,
    Yolo_OCV = 11,
	Yolo_Darknet = 12,
    Yolo_TensorRT = 13,
    DNN_OCV = 14
};

///
/// \brief The DistType enum
///
enum DistType
{
    DistCenters,   // Euclidean distance between centers, pixels
    DistRects,     // Euclidean distance between bounding rectangles, pixels
    DistJaccard,   // Intersection over Union, IoU, [0, 1]
	DistHist,      // Bhatacharia distance between histograms, [0, 1]
	DistsCount
};

///
/// \brief The FilterGoal enum
///
enum FilterGoal
{
    FilterCenter,
    FilterRect
};

///
/// \brief The KalmanType enum
///
enum KalmanType
{
    KalmanLinear,
    KalmanUnscented,
    KalmanAugmentedUnscented
};

///
/// \brief The MatchType enum
///
enum MatchType
{
    MatchHungrian,
    MatchBipart
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
    TrackLDES
};
}
