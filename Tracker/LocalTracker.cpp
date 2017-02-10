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
        const regions_t& srcRegions,
        regions_t& trackedRegions,
        cv::Mat grayFrame
        )
{
    if (m_prevFrame.size() != grayFrame.size())
    {
        m_prevFrame = grayFrame;
        prevRegions.assign(srcRegions.begin(), srcRegions.end());
        return;
    }

    std::vector<cv::Point2f> points[2];

    points[0].reserve(8 * srcRegions.size());
    for (const CRegion& region : srcRegions)
    {
        for (const auto& pt : region.m_points)
        {
            points[0].push_back(pt);
        }
    }

    if (points[0].empty())
    {
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
    for (size_t ri = 0; ri < srcRegions.size(); ++ri)
    {
        const CRegion& srcReg = srcRegions[ri];

        trackedRegions.push_back(CRegion(srcReg.m_rect));

        for (size_t pi = 0; pi < srcReg.m_points.size(); ++pi)
        {
            if (status[i])
            {
                trackedRegions[ri].m_points.push_back(points[1][i]);

                points[1][k++] = points[1][i];
                //circle(image, points[1][i], 3, cv::Scalar(0,255,0), -1, 8);
            }

            ++i;
        }
    }

    m_prevFrame = grayFrame;
    prevRegions.assign(srcRegions.begin(), srcRegions.end());
}
