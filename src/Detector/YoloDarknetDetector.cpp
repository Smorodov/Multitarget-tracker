#include "YoloDarknetDetector.h"

///
/// \brief YoloDarknetDetector::YoloDarknetDetector
/// \param gray
///
YoloDarknetDetector::YoloDarknetDetector(
    cv::UMat& colorFrame
	)
    : BaseDetector(colorFrame)
{
    m_classNames = { "background",
                     "aeroplane", "bicycle", "bird", "boat",
                     "bottle", "bus", "car", "cat", "chair",
                     "cow", "diningtable", "dog", "horse",
                     "motorbike", "person", "pottedplant",
                     "sheep", "sofa", "train", "tvmonitor" };
}

///
/// \brief YoloDarknetDetector::~YoloDarknetDetector
///
YoloDarknetDetector::~YoloDarknetDetector(void)
{
}

///
/// \brief YoloDarknetDetector::Init
/// \return
///
bool YoloDarknetDetector::Init(const config_t& config)
{
    auto modelConfiguration = config.find("modelConfiguration");
    auto modelBinary = config.find("modelBinary");
    if (modelConfiguration != config.end() && modelBinary != config.end())
    {
        m_detector = std::make_unique<Detector>(modelConfiguration->second, modelBinary->second);
		m_detector->nms = 0.2f;
    }

    auto classNames = config.find("classNames");
    if (classNames != config.end())
    {
        std::ifstream classNamesFile(classNames->second);
        if (classNamesFile.is_open())
        {
            m_classNames.clear();
            std::string className;
            for (; std::getline(classNamesFile, className); )
            {
                m_classNames.push_back(className);
            }
        }
    }

    auto confidenceThreshold = config.find("confidenceThreshold");
    if (confidenceThreshold != config.end())
    {
        m_confidenceThreshold = std::stof(confidenceThreshold->second);
    }

    auto maxCropRatio = config.find("maxCropRatio");
    if (maxCropRatio != config.end())
    {
        m_maxCropRatio = std::stof(maxCropRatio->second);
        if (m_maxCropRatio < 1.f)
        {
            m_maxCropRatio = 1.f;
        }
    }

	bool correct = m_detector.get() != nullptr;
    return correct;
}

///
/// \brief YoloDarknetDetector::Detect
/// \param gray
///
void YoloDarknetDetector::Detect(cv::UMat& colorFrame)
{
    m_regions.clear();

    cv::Mat colorMat = colorFrame.getMat(cv::ACCESS_READ);
#if 1
	std::shared_ptr<image_t> detImage = m_detector->mat_to_image_resize(colorMat);
	std::vector<bbox_t> detects = m_detector->detect_resized(*detImage, colorMat.cols, colorMat.rows, m_confidenceThreshold, false);
#else
	std::vector<bbox_t> detects = m_detector->detect(colorMat, m_confidenceThreshold, false);
#endif
	for (const bbox_t& bbox : detects)
	{
		m_regions.emplace_back(cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), m_classNames[bbox.obj_id], bbox.prob);
	}
}
