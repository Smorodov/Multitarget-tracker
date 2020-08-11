#include <fstream>
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
	m_detector.reset();

    auto modelConfiguration = config.find("modelConfiguration");
    auto modelBinary = config.find("modelBinary");
	if (modelConfiguration == config.end() || modelBinary == config.end())
		return false;

	int currGPUID = 0;
	auto gpuId = config.find("gpuId");
	if (gpuId != config.end())
		currGPUID = std::max(0, std::stoi(gpuId->second));

	m_detector = std::make_unique<Detector>(modelConfiguration->second, modelBinary->second, currGPUID);
	m_detector->nms = 0.2f;

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
                className.erase(className.find_last_not_of(" \t\n\r\f\v") + 1);
                m_classNames.push_back(className);
            }
        }
    }

    auto confidenceThreshold = config.find("confidenceThreshold");
    if (confidenceThreshold != config.end())
        m_confidenceThreshold = std::stof(confidenceThreshold->second);

    auto maxCropRatio = config.find("maxCropRatio");
    if (maxCropRatio != config.end())
        m_maxCropRatio = std::stof(maxCropRatio->second);

	m_classesWhiteList.clear();
	auto whiteRange = config.equal_range("white_list");
	for (auto it = whiteRange.first; it != whiteRange.second; ++it)
		m_classesWhiteList.insert(it->second);

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

	if (m_maxCropRatio <= 0)
	{
		Detect(colorMat, m_regions);
	}
	else
	{
        std::vector<cv::Rect> crops = GetCrops(m_maxCropRatio, cv::Size(m_detector->get_net_width(), m_detector->get_net_height()), colorMat.size());
        regions_t tmpRegions;
        for (size_t i = 0; i < crops.size(); ++i)
        {
            const auto& crop = crops[i];
            std::cout << "Crop " << i << ": " << crop << std::endl;
            DetectInCrop(colorMat, crop, tmpRegions);
        }

		//std::cout << "nms for " << tmpRegions.size() << " objects" << std::endl;
		nms3<CRegion>(tmpRegions, m_regions, 0.4f,
			[](const CRegion& reg) { return reg.m_brect; },
			[](const CRegion& reg) { return reg.m_confidence; },
			[](const CRegion& reg) { return reg.m_type; },
			0, 0.f);
	}
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
	cv::Size netSize(m_detector->get_net_width(), m_detector->get_net_height());

	if (crop.width == netSize.width && crop.height == netSize.height)
		m_tmpImg = colorFrame(crop);
	else
		cv::resize(colorFrame(crop), m_tmpImg, netSize, 0, 0, cv::INTER_LINEAR);
	
	image_t detImage;
	FillImg(detImage);

	std::vector<bbox_t> detects = m_detector->detect_resized(detImage, crop.width, crop.height, m_confidenceThreshold, false);
	for (const bbox_t& bbox : detects)
	{
		if (m_classesWhiteList.empty() || m_classesWhiteList.find(m_classNames[bbox.obj_id]) != std::end(m_classesWhiteList))
			tmpRegions.emplace_back(cv::Rect(bbox.x + crop.x, bbox.y + crop.y, bbox.w, bbox.h), m_classNames[bbox.obj_id], bbox.prob);
	}
	std::cout << "Detected " << detects.size() << " objects" << std::endl;
}

///
/// \brief YoloDarknetDetector::Detect
/// \param colorFrame
/// \param crop
/// \param tmpRegions
///
void YoloDarknetDetector::Detect(cv::Mat colorFrame, regions_t& tmpRegions)
{
	cv::Size netSize(m_detector->get_net_width(), m_detector->get_net_height());
	if (colorFrame.cols == netSize.width && colorFrame.rows == netSize.height)
		m_tmpImg = colorFrame;
	else
		cv::resize(colorFrame, m_tmpImg, netSize, 0, 0, cv::INTER_LINEAR);

	image_t detImage;
	FillImg(detImage);

	std::vector<bbox_t> detects = m_detector->detect_resized(detImage, colorFrame.cols, colorFrame.rows, m_confidenceThreshold, false);
	for (const bbox_t& bbox : detects)
	{
		if (m_classesWhiteList.empty() || m_classesWhiteList.find(m_classNames[bbox.obj_id]) != std::end(m_classesWhiteList))
			tmpRegions.emplace_back(cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), m_classNames[bbox.obj_id], bbox.prob);
	}
	std::cout << "Detected " << detects.size() << " objects" << std::endl;
}

///
/// \brief YoloDarknetDetector::FillImg
/// \param detImage
///
void YoloDarknetDetector::FillImg(image_t& detImage)
{
	detImage.w = m_tmpImg.cols;
	detImage.h = m_tmpImg.rows;
	detImage.c = m_tmpImg.channels();
	assert(detImage.c == 3);
	if (detImage.w * detImage.h * detImage.c != m_tmpBuf.size())
		m_tmpBuf.resize(detImage.w * detImage.h * detImage.c);
	detImage.data = &m_tmpBuf[0];

	const int h = detImage.h;
	const int w = detImage.w;
	constexpr float knorm = 1.f / 255.f;
	for (int y = 0; y < h; ++y)
	{
		for (int c = 0; c < 3; ++c)
		{
			const unsigned char *data = m_tmpImg.ptr(y) + 2 - c;
			float* fdata = detImage.data + c * w * h + y * w;
			for (int x = 0; x < w; ++x)
			{
				*fdata = knorm * data[0];
				++fdata;
				data += 3;
			}
		}
	}
}
