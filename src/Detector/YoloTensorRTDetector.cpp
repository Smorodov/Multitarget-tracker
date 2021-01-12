#include <fstream>
#include "YoloTensorRTDetector.h"
#include "nms.h"

///
/// \brief YoloTensorRTDetector::YoloTensorRTDetector
/// \param gray
///
YoloTensorRTDetector::YoloTensorRTDetector(const cv::UMat& colorFrame)
    : BaseDetector(colorFrame)
{
    m_classNames = { "background",
                     "aeroplane", "bicycle", "bird", "boat",
                     "bottle", "bus", "car", "cat", "chair",
                     "cow", "diningtable", "dog", "horse",
                     "motorbike", "person", "pottedplant",
                     "sheep", "sofa", "train", "tvmonitor" };

	m_localConfig.calibration_image_list_file_txt = "";
	m_localConfig.inference_precison = tensor_rt::FP32;
	m_localConfig.net_type = tensor_rt::YOLOV4;
	m_localConfig.detect_thresh = 0.5f;
	m_localConfig.gpu_id = 0;
}

///
/// \brief YoloDarknetDetector::Init
/// \return
///
bool YoloTensorRTDetector::Init(const config_t& config)
{
	m_detector.reset();

	auto modelConfiguration = config.find("modelConfiguration");
	auto modelBinary = config.find("modelBinary");
	if (modelConfiguration == config.end() || modelBinary == config.end())
		return false;

	auto confidenceThreshold = config.find("confidenceThreshold");
	if (confidenceThreshold != config.end())
		m_localConfig.detect_thresh = std::stof(confidenceThreshold->second);

	auto gpuId = config.find("gpuId");
	if (gpuId != config.end())
		m_localConfig.gpu_id = std::max(0, std::stoi(gpuId->second));

	auto maxBatch = config.find("maxBatch");
	if (maxBatch != config.end())
        m_batchSize = std::max(1, std::stoi(maxBatch->second));
	m_localConfig.batch_size = static_cast<uint32_t>(m_batchSize);
	
	m_localConfig.file_model_cfg = modelConfiguration->second;
	m_localConfig.file_model_weights = modelBinary->second;

	auto inference_precison = config.find("inference_precison");
	if (inference_precison != config.end())
	{
		std::map<std::string, tensor_rt::Precision> dictPrecison;
		dictPrecison["INT8"] = tensor_rt::INT8;
		dictPrecison["FP16"] = tensor_rt::FP16;
		dictPrecison["FP32"] = tensor_rt::FP32;
		auto precison = dictPrecison.find(inference_precison->second);
		if (precison != dictPrecison.end())
			m_localConfig.inference_precison = precison->second;
	}

	auto net_type = config.find("net_type");
	if (net_type != config.end())
	{
		std::map<std::string, tensor_rt::ModelType> dictNetType;
		dictNetType["YOLOV2"] = tensor_rt::YOLOV2;
		dictNetType["YOLOV3"] = tensor_rt::YOLOV3;
		dictNetType["YOLOV2_TINY"] = tensor_rt::YOLOV2_TINY;
		dictNetType["YOLOV3_TINY"] = tensor_rt::YOLOV3_TINY;
		dictNetType["YOLOV4"] = tensor_rt::YOLOV4;
		dictNetType["YOLOV4_TINY"] = tensor_rt::YOLOV4_TINY;
        dictNetType["YOLOV5"] = tensor_rt::YOLOV5;

		auto netType = dictNetType.find(net_type->second);
		if (netType != dictNetType.end())
			m_localConfig.net_type = netType->second;
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
			if (FillTypesMap(m_classNames))
			{
				std::cout << "Unknown types in class names!" << std::endl;
				assert(0);
			}
		}
	}

	m_classesWhiteList.clear();
	auto whiteRange = config.equal_range("white_list");
	for (auto it = whiteRange.first; it != whiteRange.second; ++it)
	{
		m_classesWhiteList.insert(std::stoi(it->second));
	}

	auto maxCropRatio = config.find("maxCropRatio");
	if (maxCropRatio != config.end())
		m_maxCropRatio = std::stof(maxCropRatio->second);

	m_detector = std::make_unique<tensor_rt::Detector>();
	m_detector->init(m_localConfig);
	return m_detector.get() != nullptr;
}

///
/// \brief YoloTensorRTDetector::Detect
/// \param gray
///
void YoloTensorRTDetector::Detect(const cv::UMat& colorFrame)
{
    m_regions.clear();
	cv::Mat colorMat = colorFrame.getMat(cv::ACCESS_READ);

    if (m_maxCropRatio <= 0)
    {
        std::vector<cv::Mat> batch = { colorMat };
        std::vector<tensor_rt::BatchResult> detects;
        m_detector->detect(batch, detects);
        for (const tensor_rt::BatchResult& dets : detects)
        {
            for (const tensor_rt::Result& bbox : dets)
            {
				if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(bbox.id)) != std::end(m_classesWhiteList))
					m_regions.emplace_back(bbox.rect, T2T(bbox.id), bbox.prob);
            }
        }
    }
    else
    {
        std::vector<cv::Rect> crops = GetCrops(m_maxCropRatio, m_detector->get_input_size(), colorMat.size());
        regions_t tmpRegions;
		std::vector<cv::Mat> batch;
		batch.reserve(m_batchSize);
        for (size_t i = 0; i < crops.size();)
        {
            size_t batchsize = std::min(static_cast<size_t>(m_batchSize), crops.size() - i);
			batch.clear();
			for (size_t j = 0; j < batchsize; ++j)
			{
				batch.emplace_back(colorMat, crops[i + j]);
			}
			std::vector<tensor_rt::BatchResult> detects;
			m_detector->detect(batch, detects);
			
			for (size_t j = 0; j < batchsize; ++j)
			{
				const auto& crop = crops[i + j];
				//std::cout << "Crop " << (i + j) << ": " << crop << std::endl;

				for (const tensor_rt::Result& bbox : detects[j])
				{
					if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(bbox.id)) != std::end(m_classesWhiteList))
						tmpRegions.emplace_back(cv::Rect(bbox.rect.x + crop.x, bbox.rect.y + crop.y, bbox.rect.width, bbox.rect.height), T2T(bbox.id), bbox.prob);
				}
			}
			i += batchsize;
        }

		if (crops.size() > 1)
		{
			nms3<CRegion>(tmpRegions, m_regions, 0.4f,
				[](const CRegion& reg) { return reg.m_brect; },
				[](const CRegion& reg) { return reg.m_confidence; },
				[](const CRegion& reg) { return reg.m_type; },
				0, 0.f);
			//std::cout << "nms for " << tmpRegions.size() << " objects - result " << m_regions.size() << std::endl;
		}
    }
}

///
/// \brief YoloTensorRTDetector::Detect
/// \param frames
/// \param regions
///
void YoloTensorRTDetector::Detect(const std::vector<cv::UMat>& frames, std::vector<regions_t>& regions)
{
	std::vector<cv::Mat> batch;
	for (const auto& frame : frames)
	{
		batch.emplace_back(frame.getMat(cv::ACCESS_READ));
	}

	std::vector<tensor_rt::BatchResult> detects;
	m_detector->detect(batch, detects);
	for (size_t i = 0; i < detects.size(); ++i)
	{
		const tensor_rt::BatchResult& dets = detects[i];
		for (const tensor_rt::Result& bbox : dets)
		{
			if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(bbox.id)) != std::end(m_classesWhiteList))
				regions[i].emplace_back(bbox.rect, T2T(bbox.id), bbox.prob);
		}
	}
	m_regions.assign(std::begin(regions.back()), std::end(regions.back()));
}
