#pragma once

#include <vector>
#include <memory>

#include "defines.h"
#include "trajectory.h"
#include "TrackerSettings.h"

///
/// \brief The CTracker class
///
class BaseTracker
{
public:
    BaseTracker() = default;
    BaseTracker(const BaseTracker&) = delete;
    BaseTracker(BaseTracker&&) = delete;
    BaseTracker& operator=(const BaseTracker&) = delete;
    BaseTracker& operator=(BaseTracker&&) = delete;

    virtual ~BaseTracker(void) = default;

    virtual void Update(const regions_t& regions, cv::UMat currFrame, float fps) = 0;
    virtual void UpdateMat(const regions_t& regions, cv::Mat currFrame, float fps)
    {
        cv::UMat frame = currFrame.getUMat(cv::ACCESS_READ);
        Update(regions, frame, fps);
    }
    virtual bool CanGrayFrameToTrack() const = 0;
    virtual bool CanColorFrameToTrack() const = 0;
    virtual size_t GetTracksCount() const = 0;
    virtual void GetTracks(std::vector<TrackingObject>& tracks) const = 0;
    virtual std::vector<TrackingObject> GetTracksCopy() const
    {
        std::vector<TrackingObject> tracks;
        GetTracks(tracks);
        return tracks;
    }
    virtual void GetRemovedTracks(std::vector<track_id_t>& trackIDs) const = 0;

    static std::unique_ptr<BaseTracker> CreateTracker(const TrackerSettings& settings);
};
