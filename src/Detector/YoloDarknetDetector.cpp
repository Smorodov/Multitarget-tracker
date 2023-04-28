#include <fstream>
#include "YoloDarknetDetector.h"
#include "nms.h"

///
/// \brief YoloDarknetDetector::YoloDarknetDetector
/// \param colorFrame
///
YoloDarknetDetector::YoloDarknetDetector(const cv::UMat& colorFrame)
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
/// \brief YoloDarknetDetector::YoloDarknetDetector
/// \param colorFrame
///
YoloDarknetDetector::YoloDarknetDetector(const cv::Mat& colorFrame)
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

	auto maxBatch = config.find("maxBatch");
	if (maxBatch != config.end())
		m_batchSize = std::max(1, std::stoi(maxBatch->second));

	m_detector = std::make_unique<Detector>(modelConfiguration->second, modelBinary->second, currGPUID, static_cast<int>(m_batchSize));
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
			if (!FillTypesMap(m_classNames))
			{
				std::cout << "Unknown types in class names!" << std::endl;
				assert(0);
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
	{
        m_classesWhiteList.insert(TypeConverter::Str2Type(it->second));
	}

	bool correct = m_detector.get() != nullptr;
    
	m_netSize = cv::Size(m_detector->get_net_width(), m_detector->get_net_height());
	
	return correct;
}

///
/// \brief YoloDarknetDetector::Detect
/// \param gray
///
void YoloDarknetDetector::Detect(const cv::UMat& colorFrame)
{
	m_regions.clear();
	cv::Mat colorMat = colorFrame.getMat(cv::ACCESS_READ);

	if (m_maxCropRatio <= 0)
	{
		Detect(colorMat, m_regions);
	}
	else
	{
        std::vector<cv::Rect> crops = GetCrops(m_maxCropRatio, m_netSize, colorMat.size());
        std::cout << "Image on " << crops.size() << " crops with size " << crops.front().size() << ", input size " << m_netSize << ", batch " << m_batchSize << ", frame " << colorMat.size() << std::endl;
        regions_t tmpRegions;
		if (m_batchSize > 1)
		{
			std::vector<cv::Mat> batch;
			batch.reserve(m_batchSize);
				
			for (size_t i = 0; i < crops.size(); i += m_batchSize)
			{
				size_t batchSize = std::min(static_cast<size_t>(m_batchSize), crops.size() - i);
				batch.clear();
				for (size_t j = 0; j < batchSize; ++j)
				{
					batch.emplace_back(colorMat, crops[i + j]);
				}

				image_t detImage;
				FillBatchImg(batch, detImage);
				std::vector<std::vector<bbox_t>> result_vec = m_detector->detectBatch(detImage, static_cast<int>(batchSize), m_netSize.width, m_netSize.height, m_confidenceThreshold);

				const float wk = static_cast<float>(crops[i].width) / m_netSize.width;
				const float hk = static_cast<float>(crops[i].height) / m_netSize.height;
				for (size_t j = 0; j < batchSize; ++j)
				{
					for (const auto& bbox : result_vec[j])
					{
						if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(bbox.obj_id)) != std::end(m_classesWhiteList))
							tmpRegions.emplace_back(cv::Rect(crops[i + j].x + cvRound(wk * bbox.x), crops[i + j].y + cvRound(hk * bbox.y),
								                             cvRound(wk * bbox.w), cvRound(hk * bbox.h)),
								                    T2T(bbox.obj_id), bbox.prob);
					}
				}
			}
		}
		else
		{
			for (size_t i = 0; i < crops.size(); ++i)
			{
				const auto& crop = crops[i];
				//std::cout << "Crop " << i << ": " << crop << std::endl;
				DetectInCrop(colorMat, crop, tmpRegions);
			}
		}

		if (crops.size() > 1 || m_batchSize > 1)
		{
			nms3<CRegion>(tmpRegions, m_regions, static_cast<track_t>(0.4),
				[](const CRegion& reg) { return reg.m_brect; },
				[](const CRegion& reg) { return reg.m_confidence; },
				[](const CRegion& reg) { return reg.m_type; },
				0, static_cast<track_t>(0));
			//std::cout << "nms for " << tmpRegions.size() << " objects - result " << m_regions.size() << std::endl;
		}
	}
	//std::cout << "Finally " << m_regions.size() << " objects, " << colorMat.u->refcount << ", " << colorMat.u->urefcount << std::endl;
}

///
/// \brief YoloDarknetDetector::DetectInCrop
/// \param colorFrame
/// \param crop
/// \param tmpRegions
///
void YoloDarknetDetector::DetectInCrop(const cv::Mat& colorFrame, const cv::Rect& crop, regions_t& tmpRegions)
{
	if (crop.width == m_netSize.width && crop.height == m_netSize.height)
		m_tmpImg = colorFrame(crop);
	else
		cv::resize(colorFrame(crop), m_tmpImg, m_netSize, 0, 0, cv::INTER_LINEAR);
	
	image_t detImage;
	FillImg(detImage);

	std::vector<bbox_t> detects = m_detector->detect(detImage, m_confidenceThreshold, false);

	float wk = (float)crop.width / detImage.w;
	float hk = (float)crop.height / detImage.h;

	for (const bbox_t& bbox : detects)
	{
		if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(bbox.obj_id)) != std::end(m_classesWhiteList))
			tmpRegions.emplace_back(cv::Rect(cvRound(wk * bbox.x) + crop.x, cvRound(hk * bbox.y) + crop.y, cvRound(wk * bbox.w), cvRound(hk * bbox.h)), T2T(bbox.obj_id), bbox.prob);
	}
	if (crop.width == m_netSize.width && crop.height == m_netSize.height)
		m_tmpImg.release();
	//std::cout << "Detected " << detects.size() << " objects" << std::endl;
}

///
/// \brief YoloDarknetDetector::Detect
/// \param colorFrame
/// \param crop
/// \param tmpRegions
///
void YoloDarknetDetector::Detect(const cv::Mat& colorFrame, regions_t& tmpRegions)
{
	if (colorFrame.cols == m_netSize.width && colorFrame.rows == m_netSize.height)
		m_tmpImg = colorFrame;
	else
		cv::resize(colorFrame, m_tmpImg, m_netSize, 0, 0, cv::INTER_LINEAR);

	image_t detImage;
	FillImg(detImage);

	std::vector<bbox_t> detects = m_detector->detect(detImage, m_confidenceThreshold, false);

	float wk = (float)colorFrame.cols / detImage.w;
	float hk = (float)colorFrame.rows / detImage.h;

	for (const bbox_t& bbox : detects)
	{
		if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(bbox.obj_id)) != std::end(m_classesWhiteList))
			tmpRegions.emplace_back(cv::Rect(cvRound(wk * bbox.x), cvRound(hk * bbox.y), cvRound(wk * bbox.w), cvRound(hk * bbox.h)), T2T(bbox.obj_id), bbox.prob);
	}
	//std::cout << "Detected " << detects.size() << " objects" << std::endl;
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
	size_t newSize = static_cast<size_t>(detImage.w * detImage.h * detImage.c);
	if (newSize != m_tmpBuf.size())
		m_tmpBuf.resize(newSize);
	detImage.data = &m_tmpBuf[0];

	const int h = detImage.h;
	const int w = detImage.w;
	constexpr float knorm = 1.f / 255.f;
	for (int y = 0; y < h; ++y)
	{
		for (int c = 0; c < 3; ++c)
		{
			const unsigned char *data = m_tmpImg.ptr(y) + 2 - c;
			float* fdata = detImage.data + static_cast<ptrdiff_t>(c * w * h) + static_cast<ptrdiff_t>(y * w);
			for (int x = 0; x < w; ++x)
			{
				*fdata = knorm * data[0];
				++fdata;
				data += 3;
			}
		}
	}
}

///
/// \brief YoloDarknetDetector::FillBatchImg
/// \param batch
/// \param detImage
///
void YoloDarknetDetector::FillBatchImg(const std::vector<cv::Mat>& batch, image_t& detImage)
{
	detImage.w = m_netSize.width;
	detImage.h = m_netSize.height;
	detImage.c = 3;
	assert(detImage.c == 3);
	size_t imgSize = static_cast<size_t>(detImage.w * detImage.h * detImage.c);
	size_t newSize = batch.size() * imgSize;
	if (newSize > m_tmpBuf.size())
		m_tmpBuf.resize(newSize);
	detImage.data = &m_tmpBuf[0];

	for (size_t i = 0; i < batch.size(); ++i)
	{
		if (batch[i].cols == m_netSize.width && batch[i].rows == m_netSize.height)
			m_tmpImg = batch[i];
		else
			cv::resize(batch[i], m_tmpImg, m_netSize, 0, 0, cv::INTER_LINEAR);

		float* fImgStart = detImage.data + i * imgSize;

		const int h = m_tmpImg.rows;
		const int w = m_tmpImg.cols;
		constexpr float knorm = 1.f / 255.f;
		for (int y = 0; y < h; ++y)
		{
			for (int c = 0; c < 3; ++c)
			{
				const unsigned char* data = m_tmpImg.ptr(y) + 2 - c;
				float* fdata = fImgStart + static_cast<ptrdiff_t>(c * w * h) + static_cast<ptrdiff_t>(y * w);
				for (int x = 0; x < w; ++x)
				{
					*fdata = knorm * data[0];
					++fdata;
					data += 3;
				}
			}
		}
	}
}

///
/// \brief YoloDarknetDetector::Detect
/// \param frames
/// \param regions
///
void YoloDarknetDetector::Detect(const std::vector<cv::UMat>& frames, std::vector<regions_t>& regions)
{
	if (frames.size() == 1)
	{
		Detect(frames[0]);
		regions[0] = m_regions;
	}
	else
	{
		std::vector<cv::Mat> batch;
		for (const auto& frame : frames)
		{
			batch.emplace_back(frame.getMat(cv::ACCESS_READ));
		}

		image_t detImage;
		FillBatchImg(batch, detImage);
		std::vector<std::vector<bbox_t>> result_vec = m_detector->detectBatch(detImage, static_cast<int>(frames.size()), m_netSize.width, m_netSize.height, m_confidenceThreshold);

		regions_t tmpRegions;
		tmpRegions.reserve(result_vec[0].size() + 16);
		float wk = static_cast<float>(frames[0].cols) / m_netSize.width;
		float hk = static_cast<float>(frames[0].rows) / m_netSize.height;
		for (size_t i = 0; i < regions.size(); ++i)
		{
			tmpRegions.clear();
			for (const auto& bbox : result_vec[i])
			{
				if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(bbox.obj_id)) != std::end(m_classesWhiteList))
					tmpRegions.emplace_back(cv::Rect(cvRound(wk * bbox.x), cvRound(hk * bbox.y), cvRound(wk * bbox.w), cvRound(hk * bbox.h)), T2T(bbox.obj_id), bbox.prob);
			}

			nms3<CRegion>(tmpRegions, regions[i], static_cast<track_t>(0.4),
				[](const CRegion& reg) { return reg.m_brect; },
				[](const CRegion& reg) { return reg.m_confidence; },
				[](const CRegion& reg) { return reg.m_type; },
				0, static_cast<track_t>(0));
		}

		m_regions.assign(std::begin(regions.back()), std::end(regions.back()));
	}
}
