#pragma once

#include <cstddef>
#include <opencv2/opencv.hpp>

#include "KalmanFilter.h"
#include "trajectory.h"

namespace byte_track
{
enum class STrackState {
    New = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3,
};

class STrack
{
public:
    STrack(const cv::Rect2f& rect, const float& score, time_point_t currTime);
    ~STrack() = default;

    const cv::Rect2f& getRect() const;
    const STrackState& getSTrackState() const;

    const bool& isActivated() const;
    const float& getScore() const;
    const size_t& getTrackId() const;
    const size_t& getFrameId() const;
    const size_t& getStartFrameId() const;
    const size_t& getTrackletLength() const;

    void activate(const size_t& frame_id, const size_t& track_id, time_point_t currTime);
    void reActivate(const STrack &new_track, const size_t &frame_id, const int &new_track_id, time_point_t currTime); // new_track_id = -1

    void predict();
    void update(const STrack &new_track, const size_t &frame_id, time_point_t currTime);

    void markAsLost();
    void markAsRemoved();

private:
    KalmanFilter kalman_filter_;
    KalmanFilter::StateMean mean_;
    KalmanFilter::StateCov covariance_;

    cv::Rect2f rect_;
    STrackState state_{ STrackState::New };

    bool is_activated_ = false;
    float score_ = 0.f;
    size_t track_id_ = 0;
    size_t frame_id_ = 0;
    size_t start_frame_id_ = 0;
    size_t tracklet_len_ = 0;

    Trace trace_;

    void updateRect();
};
}