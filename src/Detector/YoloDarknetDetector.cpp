#include "YoloDarknetDetector.h"
#include "nms.h"

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
		m_WHRatio = static_cast<float>(m_detector->get_net_width()) / static_cast<float>(m_detector->get_net_height());
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

	m_classesWhiteList.clear();
	auto whiteRange = config.equal_range("white_list");
	for (auto it = whiteRange.first; it != whiteRange.second; ++it)
	{
		m_classesWhiteList.insert(it->second);
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

	int cropHeight = cvRound(m_maxCropRatio * m_detector->get_net_height());
	int cropWidth = cvRound(m_maxCropRatio * m_detector->get_net_width());

	if (colorFrame.cols / (float)colorFrame.rows > m_WHRatio)
	{
		if (m_maxCropRatio <= 0 || cropHeight >= colorFrame.rows)
		{
			cropHeight = colorFrame.rows;
		}
		cropWidth = cvRound(cropHeight * m_WHRatio);
	}
	else
	{
		if (m_maxCropRatio <= 0 || cropWidth >= colorFrame.cols)
		{
			cropWidth = colorFrame.cols;
		}
		cropHeight = cvRound(colorFrame.cols / m_WHRatio);
	}

	//std::cout << "Frame size " << colorFrame.size() << ", crop size = " << cv::Size(cropWidth, cropHeight) << ", ratio = " << m_maxCropRatio << std::endl;

	cv::Rect crop(0, 0, cropWidth, cropHeight);
	regions_t tmpRegions;
	size_t cropsCount = 0;
	int stepX = 3 * crop.width / 4;
	int stepY = 3 * crop.height / 4;
	for (; crop.y < colorMat.rows; crop.y += stepY)
	{
		bool needBreakY = false;
		if (crop.y + crop.height >= colorMat.rows)
		{
			crop.y = colorMat.rows - crop.height;
			needBreakY = true;
		}
		for (crop.x = 0; crop.x < colorMat.cols; crop.x += stepX)
		{
			bool needBreakX = false;
			if (crop.x + crop.width >= colorMat.cols)
			{
				crop.x = colorMat.cols - crop.width;
				needBreakX = true;
			}

			//std::cout << "Crop " << cropsCount++ << ": " << crop << std::endl;
			DetectInCrop(colorMat, crop, tmpRegions);

			if (needBreakX)
			{
				break;
			}
		}
		if (needBreakY)
		{
			break;
		}
	}

	//std::cout << "nms for " << tmpRegions.size() << " objects" << std::endl;
	nms3<CRegion>(tmpRegions, m_regions, 0.4f,
		[](const CRegion& reg) -> cv::Rect { return reg.m_brect; },
		[](const CRegion& reg) -> float { return reg.m_confidence; },
		[](const CRegion& reg) -> std::string { return reg.m_type; },
		0, 0.f);

	//std::cout << "Finally " << m_regions.size() << " objects" << std::endl;
}

///
/// \brief YoloDarknetDetector::DetectInCrop
/// \param colorFrame
/// \param crop
/// \param tmpRegions
///
void YoloDarknetDetector::DetectInCrop(cv::Mat colorFrame, const cv::Rect& crop, regions_t& tmpRegions)
{
#if 1
	std::shared_ptr<image_t> detImage = m_detector->mat_to_image_resize(colorFrame(crop));
	std::vector<bbox_t> detects = m_detector->detect_resized(*detImage, crop.width, crop.height, m_confidenceThreshold, false);
#else
	std::vector<bbox_t> detects = m_detector->detect(colorFrame, m_confidenceThreshold, false);
#endif
	for (const bbox_t& bbox : detects)
	{
		if (m_classesWhiteList.empty() || m_classesWhiteList.find(m_classNames[bbox.obj_id]) != std::end(m_classesWhiteList))
		{
			tmpRegions.emplace_back(cv::Rect(bbox.x + crop.x, bbox.y + crop.y, bbox.w, bbox.h), m_classNames[bbox.obj_id], bbox.prob);
		}
	}
}
