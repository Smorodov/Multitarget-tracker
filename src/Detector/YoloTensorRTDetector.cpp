#include <fstream>
#include "YoloTensorRTDetector.h"

///
/// \brief YoloTensorRTDetector::YoloTensorRTDetector
/// \param gray
///
YoloTensorRTDetector::YoloTensorRTDetector(
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

	m_localConfig.calibration_image_list_file_txt = "";
	m_localConfig.inference_precison = tensor_rt::FP32;
	m_localConfig.net_type = tensor_rt::YOLOV4;
	m_localConfig.detect_thresh = 0.5f;
	m_localConfig.gpu_id = 0;
	m_localConfig.n_max_batch = 4;
}

///
/// \brief YoloDarknetDetector::~YoloDarknetDetector
///
YoloTensorRTDetector::~YoloTensorRTDetector(void)
{
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
		m_localConfig.n_max_batch = std::max(1, std::stoi(maxBatch->second));
	
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
		}
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
void YoloTensorRTDetector::Detect(cv::UMat& colorFrame)
{
    m_regions.clear();

	std::vector<cv::Mat> batch = { colorFrame.getMat(cv::ACCESS_READ)  };
	std::vector<tensor_rt::BatchResult> detects;
	m_detector->detect(batch, detects);
	for (const tensor_rt::BatchResult& dets : detects)
	{
		for (const tensor_rt::Result& bbox : dets)
		{
			m_regions.emplace_back(bbox.rect, m_classNames[bbox.id], bbox.prob);
		}
	}
}
