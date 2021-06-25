#include "MotionDetector.h"

///
/// \brief MotionDetector::MotionDetector
/// \param algType
/// \param gray
///
MotionDetector::MotionDetector(
	BackgroundSubtract::BGFG_ALGS algType,
    cv::UMat& gray)
    :
      BaseDetector(gray),
      m_algType(algType)
{
	m_fg = gray.clone();
	m_backgroundSubst = std::make_unique<BackgroundSubtract>(algType, gray.channels());
}

///
/// \brief MotionDetector::Init
/// \param config
/// \return
///
bool MotionDetector::Init(const config_t& config)
{
    auto conf = config.find("useRotatedRect");
    if (conf != config.end())
        m_useRotatedRect = std::stoi(conf->second) != 0;

    return m_backgroundSubst->Init(config);
}

///
/// \brief MotionDetector::DetectContour
///
void MotionDetector::DetectContour()
{
	m_regions.clear();
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
#if (CV_VERSION_MAJOR < 4)
	cv::findContours(m_fg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point());
#else
    cv::findContours(m_fg, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point());
#endif
	for (size_t i = 0; i < contours.size(); i++)
	{
		cv::Rect br = cv::boundingRect(contours[i]);

		if (br.width >= m_minObjectSize.width &&
			br.height >= m_minObjectSize.height)
		{
			if (m_useRotatedRect)
			{
				cv::RotatedRect rr = cv::minAreaRect(contours[i]);
				m_regions.push_back(CRegion(rr));
			}
			else
			{
				m_regions.push_back(CRegion(br));
			}
		}
	}
}

///
/// \brief MotionDetector::Detect
/// \param gray
///
void MotionDetector::Detect(const cv::UMat& gray)
{
    m_backgroundSubst->Subtract(gray, m_fg);
    if (!m_ignoreMask.empty())
        cv::bitwise_and(m_fg, m_ignoreMask, m_fg);

	DetectContour();
}

///
/// \brief MotionDetector::ResetModel
/// \param img
/// \param roiRect
///
void MotionDetector::ResetModel(const cv::UMat& img, const cv::Rect& roiRect)
{
	m_backgroundSubst->ResetModel(img, roiRect);
}

///
/// \brief MotionDetector::CalcMotionMap
/// \param frame
///
void MotionDetector::CalcMotionMap(cv::Mat& frame)
{
	if (m_motionMap.size() != frame.size())
		m_motionMap = cv::Mat(frame.size(), CV_32FC1, cv::Scalar(0, 0, 0));

	cv::normalize(m_fg, m_normFor, 255, 0, cv::NORM_MINMAX, m_motionMap.type());

	double alpha = 0.95;
	cv::addWeighted(m_motionMap, alpha, m_normFor, 1 - alpha, 0, m_motionMap);

	const int chans = frame.channels();

	const int height = frame.rows;
#pragma omp parallel for
	for (int y = 0; y < height; ++y)
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

#if 0
    std::cout << "m_ignoreMask = " << m_ignoreMask.size() << std::endl;
        if (!m_ignoreMask.empty())
            cv::imshow("ignoreMask", m_ignoreMask);
#endif
}
