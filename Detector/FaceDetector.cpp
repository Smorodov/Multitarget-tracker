#include "FaceDetector.h"

///
/// \brief FaceDetector::FaceDetector
/// \param collectPoints
/// \param gray
///
FaceDetector::FaceDetector(
	bool collectPoints,
    cv::UMat& gray
	)
    : BaseDetector(collectPoints, gray)
{
}

///
/// \brief FaceDetector::~FaceDetector
///
FaceDetector::~FaceDetector(void)
{
}

///
/// \brief FaceDetector::Init
/// \param cascadeFileName
/// \return
///
bool FaceDetector::Init(std::string cascadeFileName)
{
    m_cascade.load(cascadeFileName);
    if (m_cascade.empty())
    {
        std::cerr << "Cascade not opened!" << std::endl;
        return false;
    }
    return true;
}

///
/// \brief FaceDetector::Detect
/// \param gray
///
void FaceDetector::Detect(cv::UMat& gray)
{
    bool findLargestObject = false;
    bool filterRects = true;
    std::vector<cv::Rect> faceRects;
    m_cascade.detectMultiScale(gray,
                             faceRects,
                             1.1,
                             (filterRects || findLargestObject) ? 3 : 0,
                             findLargestObject ? cv::CASCADE_FIND_BIGGEST_OBJECT : 0,
                             m_minObjectSize,
                             cv::Size(gray.cols / 2, gray.rows / 2));
    m_regions.clear();
    for (auto rect : faceRects)
    {
        m_regions.push_back(rect);
    }
}

///
/// \brief FaceDetector::CalcMotionMap
/// \param frame
///
void FaceDetector::CalcMotionMap(cv::Mat frame)
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
