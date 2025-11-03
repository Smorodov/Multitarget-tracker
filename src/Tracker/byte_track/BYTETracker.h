#pragma once

#include "BaseTracker.h"

#include "STrack.h"
#include "lapjv.h"

namespace byte_track
{
class BYTETracker final : public BaseTracker
{
public:
    using STrackPtr = std::shared_ptr<STrack>;

    BYTETracker(const int& frame_rate,           // 30
                const int& track_buffer,         // 30
                const float& track_thresh,       // 0.5f
                const float& high_thresh,        // 0.5f
                const float& match_thresh);      // 0.8f
    ~BYTETracker() = default;

    void Update(const regions_t& regions, cv::UMat currFrame, time_point_t frameTime) override;

    void GetTracks(std::vector<TrackingObject>& tracks) const override;
    void GetRemovedTracks(std::vector<track_id_t>& trackIDs) const override;

private:
    std::vector<STrackPtr> jointStracks(const std::vector<STrackPtr> &a_tlist,
                                        const std::vector<STrackPtr> &b_tlist) const;

    std::vector<STrackPtr> subStracks(const std::vector<STrackPtr> &a_tlist,
                                      const std::vector<STrackPtr> &b_tlist) const;

    void removeDuplicateStracks(const std::vector<STrackPtr> &a_stracks,
                                const std::vector<STrackPtr> &b_stracks,
                                std::vector<STrackPtr> &a_res,
                                std::vector<STrackPtr> &b_res) const;

    void linearAssignment(const std::vector<std::vector<float>> &cost_matrix,
                          const int &cost_matrix_size,
                          const int &cost_matrix_size_size,
                          const float &thresh,
                          std::vector<std::vector<int>> &matches,
                          std::vector<int> &b_unmatched,
                          std::vector<int> &a_unmatched) const;

    std::vector<std::vector<float>> calcIouDistance(const std::vector<STrackPtr> &a_tracks,
                                                    const std::vector<STrackPtr> &b_tracks) const;

    std::vector<std::vector<float>> calcIous(const std::vector<cv::Rect2f> &a_rect,
                                             const std::vector<cv::Rect2f> &b_rect) const;

    double execLapjv(const std::vector<std::vector<float> > &cost,
                     std::vector<int> &rowsol,
                     std::vector<int> &colsol,
                     bool extend_cost = false,
                     float cost_limit = std::numeric_limits<float>::max(),
                     bool return_cost = true) const;

private:
    const float track_thresh_ = 0.5f;
    const float high_thresh_ = 0.6f;
    const float match_thresh_ = 0.8f;
    const size_t max_time_lost_ = 30;

    size_t frame_id_ = 0;
    size_t track_id_count_ = 0;

    std::vector<STrackPtr> tracked_stracks_;
    std::vector<STrackPtr> lost_stracks_;
    std::vector<STrackPtr> removed_stracks_;
    std::vector<STrackPtr> output_stracks_;
};
}