#include "STrack.h"

#include <cstddef>

byte_track::STrack::STrack(const cv::Rect2f& rect, const float& score, objtype_t type, time_point_t currTime) :
    kalman_filter_(),
    mean_(),
    covariance_(),
    type_(type),
    rect_(rect),
    state_(STrackState::New),
    is_activated_(false),
    score_(score),
    track_id_(0),
    frame_id_(0),
    start_frame_id_(0),
    tracklet_len_(0)
{
    Point_t pt(rect.x + rect.width / 2.f, rect.y + rect.height);
    trace_.push_back(pt, pt, currTime);
}

const cv::Rect2f& byte_track::STrack::getRect() const
{
    return rect_;
}

const byte_track::STrackState& byte_track::STrack::getSTrackState() const
{
    return state_;
}

const bool& byte_track::STrack::isActivated() const
{
    return is_activated_;
}
const float& byte_track::STrack::getScore() const
{
    return score_;
}

const size_t& byte_track::STrack::getTrackId() const
{
    return track_id_;
}

const size_t& byte_track::STrack::getFrameId() const
{
    return frame_id_;
}

const size_t& byte_track::STrack::getStartFrameId() const
{
    return start_frame_id_;
}

const size_t& byte_track::STrack::getTrackletLength() const
{
    return tracklet_len_;
}

objtype_t byte_track::STrack::getType() const
{
    return type_;
}

const Trace& byte_track::STrack::getTrace() const
{
    return trace_;
}

cv::Vec<track_t, 2> byte_track::STrack::getVelocity() const
{
	return cv::Vec<track_t, 2>(mean_(4), mean_(5));
}

byte_track::KalmanFilter::DetectBox GetXyah(const cv::Rect2f& rect)
{
    return byte_track::KalmanFilter::DetectBox(
        rect.x + rect.width / 2.f,
        rect.y + rect.height / 2.f,
        rect.width / rect.height,
        rect.height
    );
}

void byte_track::STrack::activate(const size_t& frame_id, const size_t& track_id, time_point_t currTime)
{
    kalman_filter_.initiate(mean_, covariance_, GetXyah(rect_));

    updateRect();

    state_ = STrackState::Tracked;
    if (frame_id == 1)
        is_activated_ = true;

    track_id_ = track_id;
    frame_id_ = frame_id;
    start_frame_id_ = frame_id;
    tracklet_len_ = 0;

    Point_t pt_pr(rect_.x + rect_.width / 2.f, rect_.y + rect_.height);
    trace_.push_back(pt_pr, currTime);
}

void byte_track::STrack::reActivate(const STrack &new_track, const size_t &frame_id, const int &new_track_id, time_point_t currTime)
{
    kalman_filter_.update(mean_, covariance_, GetXyah(new_track.getRect()));

    updateRect();

    state_ = STrackState::Tracked;
    is_activated_ = true;
    score_ = new_track.getScore();
    if (0 <= new_track_id)
        track_id_ = new_track_id;

    frame_id_ = frame_id;
    tracklet_len_ = 0;

    Point_t pt_pr(rect_.x + rect_.width / 2.f, rect_.y + rect_.height);
    Point_t pt_raw(new_track.getRect().x + new_track.getRect().width / 2.f, new_track.getRect().y + new_track.getRect().height);
    trace_.push_back(pt_pr, pt_raw, currTime);
}

void byte_track::STrack::predict()
{
    if (state_ != STrackState::Tracked)
        mean_(7) = 0;

    kalman_filter_.predict(mean_, covariance_);
}

void byte_track::STrack::update(const STrack &new_track, const size_t &frame_id, time_point_t currTime)
{
    kalman_filter_.update(mean_, covariance_, GetXyah(new_track.getRect()));

    updateRect();

    state_ = STrackState::Tracked;
    is_activated_ = true;
    score_ = new_track.getScore();
    frame_id_ = frame_id;
    tracklet_len_++;

    Point_t pt_pr(rect_.x + rect_.width / 2.f, rect_.y + rect_.height);
    Point_t pt_raw(new_track.getRect().x + new_track.getRect().width / 2.f, new_track.getRect().y + new_track.getRect().height);
    trace_.push_back(pt_pr, pt_raw, currTime);
}

void byte_track::STrack::markAsLost()
{
    state_ = STrackState::Lost;
}

void byte_track::STrack::markAsRemoved()
{
    state_ = STrackState::Removed;
}

void byte_track::STrack::updateRect()
{
    rect_.width = mean_(2) * mean_(3);
    rect_.height = mean_(3);
    rect_.x = mean_(0) - rect_.width / 2.f;
    rect_.y = mean_(1) - rect_.height / 2.f;
}
