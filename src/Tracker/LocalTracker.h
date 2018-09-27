#pragma once
#include "defines.h"
#include "track.h"

// --------------------------------------------------------------------------
// Tracking only founded regions between two frames (now used LK optical flow)
// --------------------------------------------------------------------------
class LocalTracker
{
public:
    LocalTracker();
    ~LocalTracker(void);

    void Update(tracks_t& tracks, cv::UMat prevFrame, cv::UMat currFrame);
};
