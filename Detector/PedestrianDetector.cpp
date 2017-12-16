#include "PedestrianDetector.h"
#include "nms.h"

///
/// \brief PedestrianDetector::PedestrianDetector
/// \param collectPoints
/// \param gray
///
PedestrianDetector::PedestrianDetector(
	bool collectPoints,
    cv::UMat& gray
	)
    :
      BaseDetector(collectPoints, gray),
      m_scannerC4(HUMAN_height, HUMAN_width, HUMAN_xdiv, HUMAN_ydiv, 256, 0.8)
{
}

///
/// \brief PedestrianDetector::~PedestrianDetector
///
PedestrianDetector::~PedestrianDetector(void)
{
}

///
/// \brief PedestrianDetector::Init
/// \param cascadeFileName
/// \return
///
bool PedestrianDetector::Init(DetectorTypes detectorType, std::string cascadeFileName1, std::string cascadeFileName2)
{
    m_detectorType = detectorType;

    switch (m_detectorType)
    {
    case HOG:
        m_hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        return true;

    case C4:
        LoadCascade(cascadeFileName1, cascadeFileName2, m_scannerC4);
        return true;

    default:
        return false;
    }

    return false;
}

///
/// \brief PedestrianDetector::Detect
/// \param gray
///
void PedestrianDetector::Detect(cv::UMat& gray)
{
    std::vector<cv::Rect> foundRects;
    std::vector<cv::Rect> filteredRects;

    int neighbors = 0;
    if (m_detectorType == HOG)
    {
        m_hog.detectMultiScale(gray, foundRects, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 4, false);
    }
    else
    {
        IntImage<double> original;
        original.Load(gray.getMat(cv::ACCESS_READ));

        m_scannerC4.FastScan(original, foundRects, 2);
        neighbors = 1;
    }

    nms(foundRects, filteredRects, 0.3f, neighbors);

    m_regions.clear();
    for (auto rect : filteredRects)
    {
        rect.x += cvRound(rect.width * 0.1f);
        rect.width = cvRound(rect.width * 0.8f);
        rect.y += cvRound(rect.height * 0.07f);
        rect.height = cvRound(rect.height * 0.8f);

        m_regions.push_back(rect);
    }
}

///
/// \brief PedestrianDetector::CalcMotionMap
/// \param frame
///
void PedestrianDetector::CalcMotionMap(cv::Mat frame)
{
	if (m_motionMap.size() != frame.size())
	{
		m_motionMap = cv::Mat(frame.size(), CV_32FC1, cv::Scalar(0, 0, 0));
	}
#if 0
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
#endif
}
