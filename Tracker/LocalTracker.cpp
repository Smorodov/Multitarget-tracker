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
    cv::Size subPixWinSize(3, 3);
    cv::Size winSize(21, 21);

    cv::cornerSubPix(m_prevFrame, points[0], subPixWinSize, cv::Size(-1,-1), termcrit);

    std::vector<uchar> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(m_prevFrame, grayFrame, points[0], points[1], status, err, winSize, 3, termcrit, 0, 0.001);

    size_t i = 0;
    for (auto& track : tracks)
    {
        track->averagePoint = Point_t(0, 0);
		track->boundidgRect = cv::Rect(0, 0, 0, 0);

        for (auto it = track->lastRegion.m_points.begin(); it != track->lastRegion.m_points.end();)
        {
            if (status[i])
            {
                *it = points[1][i];
                track->averagePoint += *it;

                ++it;
            }
            else
            {
                it = track->lastRegion.m_points.erase(it);
            }

            ++i;
        }

        if (!track->lastRegion.m_points.empty())
        {
            track->averagePoint /= static_cast<track_t>(track->lastRegion.m_points.size());

            cv::Rect br = cv::boundingRect(track->lastRegion.m_points);
			br.x -= subPixWinSize.width;
			br.width += 2 * subPixWinSize.width;
			if (br.x < 0)
			{
				br.width += br.x;
				br.x = 0;
			}
			if (br.x + br.width >= grayFrame.cols)
			{
				br.x = grayFrame.cols - br.width - 1;
			}

			br.y -= subPixWinSize.height;
			br.height += 2 * subPixWinSize.height;
			if (br.y < 0)
			{
				br.height += br.y;
				br.y = 0;
			}
			if (br.y + br.height >= grayFrame.rows)
			{
				br.y = grayFrame.rows - br.height - 1;
			}

			track->boundidgRect = br;
        }
    }

    m_prevFrame = grayFrame;
}
