#pragma once
#include "defines.h"

// --------------------------------------------------------------------------
// Tracking only founded regions between two frames (now used LK optical flow)
// --------------------------------------------------------------------------
class LocalTracker
{
public:
    LocalTracker();
    ~LocalTracker(void);

    void Update(const regions_t& srcRegions,
                regions_t& trackedRegions,
                cv::Mat grayFrame);

private:
    cv::Mat m_prevFrame;
    regions_t prevRegions;
};
