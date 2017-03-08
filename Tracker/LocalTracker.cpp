#include "LocalTracker.h"

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
LocalTracker::LocalTracker()
{
}

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
LocalTracker::~LocalTracker(void)
{
}

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
void LocalTracker::Update(
        tracks_t& tracks,
        cv::Mat grayFrame
        )
{
    if (m_prevFrame.size() != grayFrame.size())
    {
        m_prevFrame = grayFrame;
        return;
    }

    std::vector<cv::Point2f> points[2];

    points[0].reserve(8 * tracks.size());
    for (auto& track : tracks)
    {
        track->pointsCount = 0;
        for (const auto& pt : track->lastRegion.m_points)
        {
            points[0].push_back(pt);
        }
    }

    if (points[0].empty())
    {
        m_prevFrame = grayFrame;
        return;
    }

    cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
    cv::Size subPixWinSize(4, 4);
    cv::Size winSize(31, 31);

    cv::cornerSubPix(m_prevFrame, points[0], subPixWinSize, cv::Size(-1,-1), termcrit);

    std::vector<uchar> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(m_prevFrame, grayFrame, points[0], points[1], status, err, winSize, 3, termcrit, 0, 0.001);

    size_t k = 0;
    size_t i = 0;
    for (auto& track : tracks)
    {
        track->averagePoint = Point_t(0, 0);
        track->pointsCount = 0;
        for (size_t pi = 0, stop = track->lastRegion.m_points.size(); pi < stop; ++pi)
        {
            if (status[i])
            {
                ++track->pointsCount;
                track->averagePoint += points[1][i];

                points[1][k++] = points[1][i];
            }

            ++i;
        }

        if (track->pointsCount)
        {
            track->averagePoint /= track->pointsCount;
        }
        track->lastRegion.m_points.clear();
    }

    m_prevFrame = grayFrame;
}
