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
    auto modelConfiguration = config.find("modelConfiguration");
    auto modelBinary = config.find("modelBinary");
    if (modelConfiguration != config.end() && modelBinary != config.end())
    {
        m_detector = std::make_unique<tensor_rt::Detector>();
		tensor_rt::Config localConfig;
		localConfig.file_model_cfg = modelConfiguration->second;
		localConfig.file_model_weights = modelBinary->second;
		localConfig.calibration_image_list_file_txt = "";
		localConfig.inference_precison = tensor_rt::FP32;
		m_detector->init(localConfig);
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
/// \brief YoloTensorRTDetector::Detect
/// \param gray
///
void YoloTensorRTDetector::Detect(cv::UMat& colorFrame)
{
    m_regions.clear();

    cv::Mat colorMat = colorFrame.getMat(cv::ACCESS_READ);
	std::vector<tensor_rt::Result> detects;
	m_detector->detect(colorMat, detects);
	for (const tensor_rt::Result& bbox : detects)
	{
		m_regions.emplace_back(bbox.rect, m_classNames[bbox.id], bbox.prob);
	}
}
