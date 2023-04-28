#include <fstream>
#include "YoloTensorRTDetector.h"
#include "nms.h"

///
/// \brief YoloTensorRTDetector::YoloTensorRTDetector
/// \param colorFrame
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
    m_localConfig.inference_precision = tensor_rt::FP32;
	m_localConfig.net_type = tensor_rt::YOLOV4;
	m_localConfig.detect_thresh = 0.5f;
	m_localConfig.gpu_id = 0;
}

///
/// \brief YoloTensorRTDetector::YoloTensorRTDetector
/// \param colorFrame
///
YoloTensorRTDetector::YoloTensorRTDetector(const cv::Mat& colorFrame)
    : BaseDetector(colorFrame)
{
    m_classNames = { "background",
                     "aeroplane", "bicycle", "bird", "boat",
                     "bottle", "bus", "car", "cat", "chair",
                     "cow", "diningtable", "dog", "horse",
                     "motorbike", "person", "pottedplant",
                     "sheep", "sofa", "train", "tvmonitor" };

	m_localConfig.calibration_image_list_file_txt = "";
    m_localConfig.inference_precision = tensor_rt::FP32;
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

    auto videoMemory = config.find("video_memory");
    if (videoMemory != config.end())
        m_localConfig.video_memory = std::max<size_t>(0, std::stoul(videoMemory->second));

    m_localConfig.file_model_cfg = modelConfiguration->second;
    m_localConfig.file_model_weights = modelBinary->second;

    auto inference_precision = config.find("inference_precision");
    if (inference_precision != config.end())
	{
        std::map<std::string, tensor_rt::Precision> dictprecision;
        dictprecision["INT8"] = tensor_rt::INT8;
        dictprecision["FP16"] = tensor_rt::FP16;
        dictprecision["FP32"] = tensor_rt::FP32;
        auto precision = dictprecision.find(inference_precision->second);
        if (precision != dictprecision.end())
            m_localConfig.inference_precision = precision->second;
	}

	auto net_type = config.find("net_type");
	if (net_type != config.end())
	{
		std::map<std::string, tensor_rt::ModelType> dictNetType;
		dictNetType["YOLOV3"] = tensor_rt::YOLOV3;
		dictNetType["YOLOV4"] = tensor_rt::YOLOV4;
		dictNetType["YOLOV4_TINY"] = tensor_rt::YOLOV4_TINY;
        dictNetType["YOLOV5"] = tensor_rt::YOLOV5;
        dictNetType["YOLOV6"] = tensor_rt::YOLOV6;
        dictNetType["YOLOV7"] = tensor_rt::YOLOV7;
		dictNetType["YOLOV7Mask"] = tensor_rt::YOLOV7Mask;
		dictNetType["YOLOV8"] = tensor_rt::YOLOV8;

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
			if (!FillTypesMap(m_classNames))
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
        m_classesWhiteList.insert(TypeConverter::Str2Type(it->second));
	}

	auto maxCropRatio = config.find("maxCropRatio");
	if (maxCropRatio != config.end())
		m_maxCropRatio = std::stof(maxCropRatio->second);

	m_detector = std::make_unique<tensor_rt::Detector>();
	if (m_detector)
        m_detector->Init(m_localConfig);

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

#define DRAW_MASK 0
#if DRAW_MASK
	cv::Mat img = colorMat.clone();
	std::vector<cv::Scalar> color;
	srand(time(0));
	for (int i = 0; i < m_classNames.size(); i++)
	{
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.emplace_back(b, g, r);
	}
	cv::Mat mask = img.clone();
#endif

    if (m_maxCropRatio <= 0)
    {
        std::vector<cv::Mat> batch = { colorMat };
        std::vector<tensor_rt::BatchResult> detects;
        m_detector->Detect(batch, detects);
        for (const tensor_rt::BatchResult& dets : detects)
        {
            for (const tensor_rt::Result& bbox : dets)
            {
				if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(bbox.m_id)) != std::end(m_classesWhiteList))
				{
					m_regions.emplace_back(bbox.m_rrect, bbox.m_brect, T2T(bbox.m_id), bbox.m_prob, bbox.m_boxMask);

					//std::cout << "YoloTensorRTDetector::Detect: bbox.m_rrect " << bbox.m_rrect.center << ", " << bbox.m_rrect.angle << ", " << bbox.m_rrect.size << std::endl;
					//std::cout << "YoloTensorRTDetector::Detect: m_regions.back().m_rrect " << m_regions.back().m_rrect.center << ", " << m_regions.back().m_rrect.angle << ", " << m_regions.back().m_rrect.size << std::endl;
#if DRAW_MASK
					rectangle(img, bbox.m_brect, color[bbox.m_id], 2, 8);
					mask(bbox.m_brect).setTo(color[bbox.m_id], bbox.m_boxMask);
#endif
				}
            }
        }
#if DRAW_MASK
		cv::addWeighted(img, 0.5, mask, 0.5, 0, img);
		cv::imshow("mask", mask);
		cv::waitKey(1);
#endif
    }
    else
    {
        std::vector<cv::Rect> crops = GetCrops(m_maxCropRatio, m_detector->GetInputSize(), colorMat.size());
        //std::cout << "Image on " << crops.size() << " crops with size " << crops.front().size() << ", input size " << m_detector->GetInputSize() << ", batch " << m_batchSize << ", frame " << colorMat.size() << std::endl;
        regions_t tmpRegions;
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
			std::vector<tensor_rt::BatchResult> detects;
			m_detector->Detect(batch, detects);
			
			for (size_t j = 0; j < batchSize; ++j)
			{
				const auto& crop = crops[i + j];
				//std::cout << "batch " << (i / batchSize) << ", crop " << (i + j) << ": " << crop << std::endl;

				for (const tensor_rt::Result& bbox : detects[j])
				{
					if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(bbox.m_id)) != std::end(m_classesWhiteList))
						tmpRegions.emplace_back(cv::Rect(bbox.m_brect.x + crop.x, bbox.m_brect.y + crop.y, bbox.m_brect.width, bbox.m_brect.height), T2T(bbox.m_id), bbox.m_prob);
				}
			}
        }

		if (crops.size() > 1)
		{
			nms3<CRegion>(tmpRegions, m_regions, static_cast<track_t>(0.4),
				[](const CRegion& reg) { return reg.m_brect; },
				[](const CRegion& reg) { return reg.m_confidence; },
				[](const CRegion& reg) { return reg.m_type; },
				0, static_cast<track_t>(0));
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
    if (frames.size() == 1)
    {
        Detect(frames.front());
        regions[0].assign(std::begin(m_regions), std::end(m_regions));
    }
    else
    {
        std::vector<cv::Mat> batch;
        for (const auto& frame : frames)
        {
            batch.emplace_back(frame.getMat(cv::ACCESS_READ));
        }

        std::vector<tensor_rt::BatchResult> detects;
        m_detector->Detect(batch, detects);
        for (size_t i = 0; i < detects.size(); ++i)
        {
            const tensor_rt::BatchResult& dets = detects[i];
            for (const tensor_rt::Result& bbox : dets)
            {
                if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(bbox.m_id)) != std::end(m_classesWhiteList))
                    regions[i].emplace_back(bbox.m_brect, T2T(bbox.m_id), bbox.m_prob);
            }
        }
        m_regions.assign(std::begin(regions.back()), std::end(regions.back()));
    }
}

///
/// \brief CalcMotionMap
/// \param frame
///
void YoloTensorRTDetector::CalcMotionMap(cv::Mat& frame)
{
	if (m_localConfig.net_type == tensor_rt::YOLOV7Mask)
	{
		static std::vector<cv::Scalar> color;
		if (color.empty())
		{
			srand((unsigned int)time(0));
			for (int i = 0; i < m_classNames.size(); i++)
			{
				int b = rand() % 256;
				int g = rand() % 256;
				int r = rand() % 256;
				color.emplace_back(b, g, r);
			}
		}
		cv::Mat mask = frame.clone();

		for (const auto& region : m_regions)
		{
			//cv::rectangle(frame, region.m_brect, color[region.m_type], 2, 8);
			if (!region.m_boxMask.empty())
				mask(region.m_brect).setTo(color[region.m_type], region.m_boxMask);
		}
		cv::addWeighted(frame, 0.5, mask, 0.5, 0, frame);
	}
	else
	{
		BaseDetector::CalcMotionMap(frame);
	}
}
