#include "Detector.h"

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
CDetector::CDetector(
        bool collectPoints,
        cv::Mat& gray
        )
    : m_collectPoints(collectPoints)
{
	m_fg = gray.clone();
	m_bs = std::make_unique<BackgroundSubtract>(gray.channels());
	m_bs->init(gray);

	m_minObjectSize.width = std::max(5, gray.cols / 100);
	m_minObjectSize.height = m_minObjectSize.width;
}

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
CDetector::~CDetector(void)
{
}

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
void CDetector::SetMinObjectSize(cv::Size minObjectSize)
{
	m_minObjectSize = minObjectSize;
}

//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
void CDetector::DetectContour()
{
    m_regions.clear();
	m_centers.clear();
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(m_fg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point());
	if (contours.size() > 0)
	{
		for (size_t i = 0; i < contours.size(); i++)
		{
			cv::Rect r = cv::boundingRect(contours[i]);

			if (r.width >= m_minObjectSize.width &&
				r.height >= m_minObjectSize.height)
			{
                CRegion region(r);
                cv::Point2f center(r.x + 0.5f * r.width, r.y + 0.5f * r.height);

                if (m_collectPoints)
                {
                    const int yStep = 3;
                    const int xStep = 3;

                    for (int y = r.y; y < r.y + r.height; y += yStep)
                    {
                        cv::Point2f pt(0, y);
                        for (int x = r.x; x < r.x + r.width; x += xStep)
                        {
                            pt.x = x;
                            if (cv::pointPolygonTest(contours[i], pt, false) > 0)
                            {
                                region.m_points.push_back(pt);
                            }
                        }
                    }

                    if (region.m_points.empty())
                    {
                        region.m_points.push_back(center);
                    }
                }

                m_regions.push_back(region);
                m_centers.push_back(Point_t(center.x, center.y));
			}
		}
	}
}

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
const std::vector<Point_t>& CDetector::Detect(cv::Mat& gray)
{
	m_bs->subtract(gray, m_fg);

	DetectContour();
	return m_centers;
}

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
const regions_t& CDetector::GetDetects() const
{
    return m_regions;
}
