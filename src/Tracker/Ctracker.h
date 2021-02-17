#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <array>
#include <deque>
#include <map>
#include <set>

#include "defines.h"
#include "track.h"
#include "ShortPathCalculator.h"
#include "EmbeddingsCalculator.hpp"
#include "TrackerSettings.h"
// ----------------------------------------------------------------------

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
	void GetTracks(std::vector<TrackingObject>& tracks) const
	{
		tracks.clear();

		if (m_tracks.size() > tracks.capacity())
			tracks.reserve(m_tracks.size());
		for (const auto& track : m_tracks)
		{
			tracks.emplace_back(track->ConstructObject());
		}
	}

private:
    TrackerSettings m_settings;

	tracks_t m_tracks;

    size_t m_nextTrackID;

    cv::UMat m_prevFrame;

    std::unique_ptr<ShortPathCalculator> m_SPCalculator;
    std::map<objtype_t, std::shared_ptr<EmbeddingsCalculator>> m_embCalculators;

    void CreateDistaceMatrix(const regions_t& regions, std::vector<RegionEmbedding>& regionEmbeddings, distMatrix_t& costMatrix, track_t maxPossibleCost, track_t& maxCost, cv::UMat currFrame);
    void UpdateTrackingState(const regions_t& regions, cv::UMat currFrame, float fps);
};
