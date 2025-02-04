#include "FaceDetector.h"

///
/// \brief FaceDetector::FaceDetector
/// \param gray
///
FaceDetector::FaceDetector(const cv::UMat& gray)
    : BaseDetector(gray)
{
}

///
/// \brief FaceDetector::FaceDetector
/// \param gray
///
FaceDetector::FaceDetector(const cv::Mat& gray)
    : BaseDetector(gray)
{
}

///
/// \brief FaceDetector::Init
/// \param cascadeFileName
/// \return
///
bool FaceDetector::Init(const config_t& config)
{
    auto cascadeFileName = config.find("cascadeFileName");
    if (cascadeFileName != config.end() &&
            (!m_cascade.load(cascadeFileName->second) || m_cascade.empty()))
    {
        std::cerr << "Cascade " << cascadeFileName->second << " not opened!" << std::endl;
        return false;
    }
    return true;
}

///
/// \brief FaceDetector::Detect
/// \param gray
///
void FaceDetector::Detect(const cv::UMat& gray)
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
