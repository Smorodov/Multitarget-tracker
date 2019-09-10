#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <array>
#include <deque>
#include <numeric>

#include "defines.h"
#include "track.h"
#include "HungarianAlg/HungarianAlg.h"

// ----------------------------------------------------------------------

///
/// \brief The TrackerSettings struct
///
struct TrackerSettings
{
    //tracking::DistType m_distType = tracking::DistCenters;
    tracking::KalmanType m_kalmanType = tracking::KalmanLinear;
    tracking::FilterGoal m_filterGoal = tracking::FilterCenter;
    tracking::LostTrackType m_lostTrackType = tracking::TrackKCF;
    tracking::MatchType m_matchType = tracking::MatchHungrian;

	std::array<track_t, tracking::DistsCount> m_distType;

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
    track_t m_distThres = 0.5f;

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

	///
	TrackerSettings()
	{
		m_distType[tracking::DistCenters] = 0.0f;
		m_distType[tracking::DistRects] = 0.0f;
		m_distType[tracking::DistJaccard] = 0.5f;
		m_distType[tracking::DistHist] = 0.5f;
		m_distType[tracking::DistHOG] = 0.0f;

		assert(CheckDistance());
	}

	///
	bool CheckDistance() const
	{
		track_t sum = std::accumulate(m_distType.begin(), m_distType.end(), 0.0f);
		track_t maxOne = std::max(1.0f, std::fabs(sum));
		return std::fabs(sum - 1.0f) <= std::numeric_limits<track_t>::epsilon() * maxOne;
	}

	///
	bool SetDistances(std::array<track_t, tracking::DistsCount> distType)
	{
		bool res = true;
		auto oldDists = m_distType;
		m_distType = distType;
		if (!CheckDistance())
		{
			m_distType = oldDists;
			res = false;
		}
		return res;
	}

	///
	bool SetDistance(tracking::DistType distType)
	{
		std::fill(m_distType.begin(), m_distType.end(), 0.0f);
		m_distType[distType] = 1.f;
		return true;
	}
};

///
/// \brief The CTracker class
///
class CTracker
{
public:
    CTracker(const TrackerSettings& settings);
	CTracker(const CTracker&) = delete;
	CTracker(CTracker&&) = delete;
	CTracker& operator=(const CTracker&) = delete;
	CTracker& operator=(CTracker&&) = delete;
	
	~CTracker(void);

    void Update(const regions_t& regions, cv::UMat currFrame, float fps);

    ///
    /// \brief CanGrayFrameToTrack
    /// \return
    ///
    bool CanGrayFrameToTrack() const
    {
		bool needColor = (m_settings.m_lostTrackType == tracking::LostTrackType::TrackGOTURN) ||
			(m_settings.m_lostTrackType == tracking::LostTrackType::TrackDAT) ||
            (m_settings.m_lostTrackType == tracking::LostTrackType::TrackSTAPLE) ||
            (m_settings.m_lostTrackType == tracking::LostTrackType::TrackLDES);
        return !needColor;
    }

	///
	/// \brief CanColorFrameToTrack
	/// \return
	///
	bool CanColorFrameToTrack() const
	{
		return true;
	}

    ///
    /// \brief GetTracksCount
    /// \return
    ///
	size_t GetTracksCount() const
	{
		return m_tracks.size();
	}
    ///
    /// \brief GetTracks
    /// \return
    ///
	std::vector<TrackingObject> GetTracks() const
	{
		std::vector<TrackingObject> tracks;
		if (!m_tracks.empty())
		{
			tracks.reserve(m_tracks.size());
			for (const auto& track : m_tracks)
			{
                tracks.push_back(track->ConstructObject());
			}
		}
		return tracks;
	}

private:
    TrackerSettings m_settings;

	tracks_t m_tracks;

    size_t m_nextTrackID;

    cv::UMat m_prevFrame;

    void CreateDistaceMatrix(const regions_t& regions, distMatrix_t& costMatrix, track_t maxPossibleCost, track_t& maxCost, cv::UMat currFrame);

    void SolveHungrian(const distMatrix_t& costMatrix, size_t N, size_t M, assignments_t& assignment);
    void SolveBipartiteGraphs(const distMatrix_t& costMatrix, size_t N, size_t M, assignments_t& assignment, track_t maxCost);

    void UpdateTrackingState(const regions_t& regions, cv::UMat currFrame, float fps);
};
