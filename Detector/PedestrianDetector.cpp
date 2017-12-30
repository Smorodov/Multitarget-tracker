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
bool PedestrianDetector::Init(const config_t& config)
{
    auto detectorType = config.find("detectorType");

    if (detectorType == config.end())
    {
        m_detectorType = HOG;
    }
    else
    {
        m_detectorType = (detectorType->second == "HOG") ? HOG : C4;
    }

    switch (m_detectorType)
    {
    case HOG:
        m_hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        return true;

    case C4:
    {
        auto cascadeFileName1 = config.find("cascadeFileName1");
        auto cascadeFileName2 = config.find("cascadeFileName2");
        if (cascadeFileName1 == config.end() || cascadeFileName2 == config.end())
        {
            return false;
        }
        else
        {
            LoadCascade(cascadeFileName1->second, cascadeFileName2->second, m_scannerC4);
            return true;
        }
    }

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

    if (m_collectPoints)
    {
        for (auto& region : m_regions)
        {
            CollectPoints(region);
        }
    }
}
