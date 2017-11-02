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
        for (const auto& pt : track->m_lastRegion.m_points)
        {
            points[0].push_back(pt);
        }
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
        track->m_averagePoint = Point_t(0, 0);
        track->m_boundidgRect = cv::Rect(0, 0, 0, 0);

        for (auto it = track->m_lastRegion.m_points.begin(); it != track->m_lastRegion.m_points.end();)
        {
            if (status[i])
            {
                *it = points[1][i];
                track->m_averagePoint += *it;

                ++it;
            }
            else
            {
                it = track->m_lastRegion.m_points.erase(it);
            }

            ++i;
        }

        if (!track->m_lastRegion.m_points.empty())
        {
            track->m_averagePoint /= static_cast<track_t>(track->m_lastRegion.m_points.size());

            cv::Rect br = cv::boundingRect(track->m_lastRegion.m_points);
#if 0
			br.x -= subPixWinSize.width;
			br.width += 2 * subPixWinSize.width;
			if (br.x < 0)
			{
				br.width += br.x;
				br.x = 0;
			}
            if (br.x + br.width >= currFrame.cols)
			{
                br.x = currFrame.cols - br.width - 1;
			}

			br.y -= subPixWinSize.height;
			br.height += 2 * subPixWinSize.height;
			if (br.y < 0)
			{
				br.height += br.y;
				br.y = 0;
			}
            if (br.y + br.height >= currFrame.rows)
			{
                br.y = currFrame.rows - br.height - 1;
			}
#endif
            track->m_boundidgRect = br;
        }
    }
}
