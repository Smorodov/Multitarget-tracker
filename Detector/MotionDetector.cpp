#include "MotionDetector.h"

///
/// \brief MotionDetector::MotionDetector
/// \param algType
/// \param collectPoints
/// \param gray
///
MotionDetector::MotionDetector(
	BackgroundSubtract::BGFG_ALGS algType,
	bool collectPoints,
    cv::UMat& gray
	)
    : BaseDetector(collectPoints, gray)
{
	m_fg = gray.clone();
	m_backgroundSubst = std::make_unique<BackgroundSubtract>(algType, gray.channels());
}

///
/// \brief MotionDetector::~MotionDetector
///
MotionDetector::~MotionDetector(void)
{
}

///
/// \brief MotionDetector::Init
/// \param config
/// \return
///
bool MotionDetector::Init(const config_t& config)
{
    return true;
}

///
/// \brief MotionDetector::DetectContour
///
void MotionDetector::DetectContour()
{
	m_regions.clear();
    std::vector<std::vector<cv::Point>> contours;
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
                    const int yStep = 5;
                    const int xStep = 5;

					for (int y = r.y; y < r.y + r.height; y += yStep)
					{
						cv::Point2f pt(0, static_cast<float>(y));
						for (int x = r.x; x < r.x + r.width; x += xStep)
						{
							pt.x = static_cast<float>(x);
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
			}
		}
	}
}

///
/// \brief MotionDetector::Detect
/// \param gray
///
void MotionDetector::Detect(cv::UMat& gray)
{
	m_backgroundSubst->subtract(gray, m_fg);

	DetectContour();
}

///
/// \brief MotionDetector::CalcMotionMap
/// \param frame
///
void MotionDetector::CalcMotionMap(cv::Mat frame)
{
	if (m_motionMap.size() != frame.size())
	{
		m_motionMap = cv::Mat(frame.size(), CV_32FC1, cv::Scalar(0, 0, 0));
	}

	cv::Mat normFor;
	cv::normalize(m_fg, normFor, 255, 0, cv::NORM_MINMAX, m_motionMap.type());

	double alpha = 0.95;
	cv::addWeighted(m_motionMap, alpha, normFor, 1 - alpha, 0, m_motionMap);

	const int chans = frame.channels();

	for (int y = 0; y < frame.rows; ++y)
	{
		uchar* imgPtr = frame.ptr(y);
		float* moPtr = reinterpret_cast<float*>(m_motionMap.ptr(y));
		for (int x = 0; x < frame.cols; ++x)
		{
			for (int ci = chans - 1; ci < chans; ++ci)
			{
				imgPtr[ci] = cv::saturate_cast<uchar>(imgPtr[ci] + moPtr[0]);
			}
			imgPtr += chans;
			++moPtr;
		}
	}
}
