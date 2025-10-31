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
#include "trajectory.h"
#include "object_types.h"
#include "Kalman.h"

///
/// \brief The RegionEmbedding struct
///
struct RegionEmbedding
{
    cv::Mat m_hist;
    cv::Mat m_embedding;
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
           track_id_t trackID,
           tracking::FilterGoal filterGoal,
           tracking::LostTrackType externalTrackerForLost,
           time_point_t currTime);

    CTrack(const CRegion& region,
           const RegionEmbedding& regionEmbedding,
           tracking::KalmanType kalmanType,
           track_t deltaTime,
           track_t accelNoiseMag,
           bool useAcceleration,
           track_id_t trackID,
           tracking::FilterGoal filterGoal,
           tracking::LostTrackType externalTrackerForLost,
           time_point_t currTime);

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
    /// \brief CTrack::CalcMahalanobisDist
    /// \param reg
    /// \return
    ///
    track_t CalcMahalanobisDist(const cv::RotatedRect& rrect) const;
    ///
    /// \brief CalcDistHist
    /// Distance from 0 to 1 between objects histogramms on two N and N+1 frames
    /// \param embedding
    /// \return
    ///
    track_t CalcDistHist(const RegionEmbedding& embedding) const;
	///
	/// \brief CalcCosine
	/// Distance from 0 to 1 between objects embeddings on two N and N+1 frames
	/// \param embedding
	/// \return
	///
	std::pair<track_t, bool> CalcCosine(const RegionEmbedding& embedding) const;

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

    void Update(const CRegion& region, bool dataCorrect, double maxTraceLength, cv::UMat prevFrame, cv::UMat currFrame, int trajLen, int maxSpeedForStatic, time_point_t currTime);
    void Update(const CRegion& region, const RegionEmbedding& regionEmbedding, bool dataCorrect, double maxTraceLength, cv::UMat prevFrame, cv::UMat currFrame, int trajLen, int maxSpeedForStatic, time_point_t currTime);

    bool IsStatic() const;
    bool IsStaticTimeout(time_point_t currTime, double staticPeriod) const;
    bool IsOutOfTheFrame() const;

    cv::RotatedRect GetLastRect() const;

    const CRegion& LastRegion() const;
    objtype_t GetCurrType() const;
    double GetLostPeriod(time_point_t currTime) const;
    void ResetLostTime(time_point_t currTime);

    TrackingObject ConstructObject(time_point_t frameTime) const;
    track_id_t GetID() const;

	tracking::FilterGoal GetFilterGoal() const;
    void KalmanPredictRect();
    void KalmanPredictPoint();

private:
    TKalmanFilter m_kalman;
    CRegion m_lastRegion;
    Trace m_trace;
    cv::RotatedRect m_predictionRect;
    Point_t m_predictionPoint;

    track_id_t m_trackID;
    time_point_t m_lastDetectionTime;

    objtype_t m_currType = bad_type;
    objtype_t m_lastType = bad_type;
    size_t m_anotherTypeCounter = 0;
    static constexpr size_t m_changeTypeThreshold = 25;

    tracking::LostTrackType m_externalTrackerForLost = tracking::TrackNone;
#ifdef USE_OCV_KCF
    cv::Ptr<cv::Tracker> m_tracker;
#endif

    ///
    void RectUpdate(const CRegion& region, bool dataCorrect, cv::UMat prevFrame, cv::UMat currFrame);

    ///
    void CreateExternalTracker(int channels);

    ///
    void PointUpdate(const Point_t& pt, const cv::Size& newObjSize, float newAngle, bool dataCorrect, const cv::Size& frameSize);

    RegionEmbedding m_regionEmbedding;

    ///
    bool CheckStatic(int trajLen, cv::UMat currFrame, const CRegion& region, int maxSpeedForStatic, time_point_t currTime);
    cv::UMat m_staticFrame;
    cv::Rect m_staticRect;
    time_point_t m_staticStartTime;
    bool m_isStatic = false;

	tracking::FilterGoal m_filterGoal = tracking::FilterGoal::FilterCenter;
    bool m_outOfTheFrame = false;
};

typedef std::vector<std::unique_ptr<CTrack>> tracks_t;
