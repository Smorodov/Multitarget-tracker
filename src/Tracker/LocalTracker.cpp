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
        cv::UMat prevFrame,
        cv::UMat currFrame
        )
{
    std::vector<cv::Point2f> points[2];

    points[0].reserve(8 * tracks.size());
    for (auto& track : tracks)
    {
        auto pts = track->GetPoints();
        points[0].insert(points[0].end(), std::begin(pts), std::end(pts));
    }
    if (points[0].empty())
    {
        return;
    }

    cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
    cv::Size subPixWinSize(3, 3);
    cv::Size winSize(21, 21);

    cv::cornerSubPix(prevFrame, points[0], subPixWinSize, cv::Size(-1,-1), termcrit);

    std::vector<uchar> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(prevFrame, currFrame, points[0], points[1], status, err, winSize, 3, termcrit, 0, 0.001);

    size_t i = 0;
    for (auto& track : tracks)
    {
        Point_t averagePoint(0, 0);
        cv::Rect br(0, 0, 0, 0);

        const size_t count = track->GetPoints().size();
        std::vector<cv::Point2f> pts;
        pts.reserve(count);

        for (size_t j = 0; j < count; ++j)
        {
            if (status[i])
            {
                pts.push_back(points[1][i]);
                averagePoint += points[1][i];
            }
            ++i;
        }

        if (!pts.empty())
        {
            averagePoint /= static_cast<track_t>(pts.size());
            br = cv::boundingRect(pts);
        }
        track->BoundidgRect() = br;
        track->AveragePoint() = averagePoint;
    }
}
