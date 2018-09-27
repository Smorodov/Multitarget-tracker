#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <array>
#include <deque>

#include "defines.h"
#include "track.h"
#include "LocalTracker.h"

// ----------------------------------------------------------------------

///
/// \brief The TrackerSettings struct
///
struct TrackerSettings
{
    ///
    /// \brief m_useLocalTracking
    /// Use local tracking for regions between two frames
    /// It was coined for tracking small and slow objects: key points on objects tracking with LK optical flow
    /// The most applications don't need this parameter
    ///
    bool m_useLocalTracking = false;

    tracking::DistType m_distType = tracking::DistCenters;
    tracking::KalmanType m_kalmanType = tracking::KalmanLinear;
    tracking::FilterGoal m_filterGoal = tracking::FilterCenter;
    tracking::LostTrackType m_lostTrackType = tracking::TrackKCF;
    tracking::MatchType m_matchType = tracking::MatchHungrian;

    ///
    /// \brief m_dt
    /// Time step for Kalman
    ///
    track_t m_dt = 1.0f;

    ///
    /// \brief m_accelNoiseMag
    /// Noise magnitude for Kalman
    ///
    track_t m_accelNoiseMag = 0.1f;

    ///
    /// \brief m_distThres
    /// Distance threshold for Assignment problem for tracking::DistCenters or for tracking::DistRects (for tracking::DistJaccard it need from 0 to 1)
    ///
    track_t m_distThres = 50;

    ///
    /// \brief m_maximumAllowedSkippedFrames
    /// If the object don't assignment more than this frames then it will be removed
    ///
    size_t m_maximumAllowedSkippedFrames = 25;

    ///
    /// \brief m_maxTraceLength
    /// The maximum trajectory length
    ///
    size_t m_maxTraceLength = 50;

    ///
    /// \brief m_useAbandonedDetection
    /// Detection abandoned objects
    ///
    bool m_useAbandonedDetection = false;

    ///
    /// \brief m_minStaticTime
    /// After this time (in seconds) the object is considered abandoned
    ///
    int m_minStaticTime = 5;
    ///
    /// \brief m_maxStaticTime
    /// After this time (in seconds) the abandoned object will be removed
    ///
    int m_maxStaticTime = 25;
};

///
/// \brief The CTracker class
///
class CTracker
{
public:
    CTracker(const TrackerSettings& settings);
	~CTracker(void);

    tracks_t tracks;
    void Update(const regions_t& regions, cv::UMat grayFrame, float fps);

    bool GrayFrameToTrack() const
    {
        return m_settings.m_lostTrackType != tracking::LostTrackType::TrackGOTURN;
    }

private:
    TrackerSettings m_settings;

    size_t m_nextTrackID;

    LocalTracker m_localTracker;

    cv::UMat m_prevFrame;

    void UpdateHungrian(const regions_t& regions, cv::UMat grayFrame, float fps);
};
