#include <fstream>
#include "SSDMobileNetDetector.h"
#include "nms.h"

///
/// \brief SSDMobileNetDetector::SSDMobileNetDetector
/// \param gray
///
SSDMobileNetDetector::SSDMobileNetDetector(const cv::UMat& colorFrame)
    :
      BaseDetector(colorFrame)
{
    m_classNames = { "background",
                     "aeroplane", "bicycle", "bird", "boat",
                     "bottle", "bus", "car", "cat", "chair",
                     "cow", "diningtable", "dog", "horse",
                     "motorbike", "person", "pottedplant",
                     "sheep", "sofa", "train", "tvmonitor" };
}

///
/// \brief SSDMobileNetDetector::~SSDMobileNetDetector
///
SSDMobileNetDetector::~SSDMobileNetDetector(void)
{
}

///
/// \brief SSDMobileNetDetector::Init
/// \return
///
bool SSDMobileNetDetector::Init(const config_t& config)
{
    auto modelConfiguration = config.find("modelConfiguration");
    auto modelBinary = config.find("modelBinary");
    if (modelConfiguration != config.end() && modelBinary != config.end())
    {
        m_net = cv::dnn::readNetFromCaffe(modelConfiguration->second, modelBinary->second);
    }

    auto dnnTarget = config.find("dnnTarget");
    if (dnnTarget != config.end())
    {
        std::map<std::string, cv::dnn::Target> targets;
        targets["DNN_TARGET_CPU"] = cv::dnn::DNN_TARGET_CPU;
        targets["DNN_TARGET_OPENCL"] = cv::dnn::DNN_TARGET_OPENCL;
#if (CV_VERSION_MAJOR >= 4)
        targets["DNN_TARGET_OPENCL_FP16"] = cv::dnn::DNN_TARGET_OPENCL_FP16;
        targets["DNN_TARGET_MYRIAD"] = cv::dnn::DNN_TARGET_MYRIAD;
#endif
#if (((CV_VERSION_MAJOR == 4) && (CV_VERSION_MINOR >= 2)) || (CV_VERSION_MAJOR > 4))
		targets["DNN_TARGET_CUDA"] = cv::dnn::DNN_TARGET_CUDA;
		targets["DNN_TARGET_CUDA_FP16"] = cv::dnn::DNN_TARGET_CUDA_FP16;
#endif
		std::cout << "Trying to set target " << dnnTarget->second << "... ";
		auto target = targets.find(dnnTarget->second);
		if (target != std::end(targets))
		{
			std::cout << "Succeded!" << std::endl;
			m_net.setPreferableTarget(target->second);
		}
		else
		{
			std::cout << "Failed" << std::endl;
		}
    }

#if (CV_VERSION_MAJOR >= 4)
    auto dnnBackend = config.find("dnnBackend");
    if (dnnBackend != config.end())
    {
        std::map<std::string, cv::dnn::Backend> backends;
        backends["DNN_BACKEND_DEFAULT"] = cv::dnn::DNN_BACKEND_DEFAULT;
        backends["DNN_BACKEND_HALIDE"] = cv::dnn::DNN_BACKEND_HALIDE;
        backends["DNN_BACKEND_INFERENCE_ENGINE"] = cv::dnn::DNN_BACKEND_INFERENCE_ENGINE;
        backends["DNN_BACKEND_OPENCV"] = cv::dnn::DNN_BACKEND_OPENCV;
        backends["DNN_BACKEND_VKCOM"] = cv::dnn::DNN_BACKEND_VKCOM;
#if (((CV_VERSION_MAJOR == 4) && (CV_VERSION_MINOR >= 2)) || (CV_VERSION_MAJOR > 4))
		backends["DNN_BACKEND_CUDA"] = cv::dnn::DNN_BACKEND_CUDA;
#endif
		std::cout << "Trying to set backend " << dnnBackend->second << "... ";
		auto backend = backends.find(dnnBackend->second);
		if (backend != std::end(backends))
		{
			std::cout << "Succeded!" << std::endl;
			m_net.setPreferableBackend(backend->second);
		}
		else
		{
			std::cout << "Failed" << std::endl;
		}
    }
#endif

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

    return !m_net.empty();
}

///
/// \brief SSDMobileNetDetector::Detect
/// \param gray
///
void SSDMobileNetDetector::Detect(const cv::UMat& colorFrame)
{
    m_regions.clear();

	std::vector<cv::Rect> crops = GetCrops(m_maxCropRatio, cv::Size(m_inWidth, m_inHeight), colorFrame.size());
	regions_t tmpRegions;
	for (size_t i = 0; i < crops.size(); ++i)
	{
		const auto& crop = crops[i];
		//std::cout << "Crop " << i << ": " << crop << std::endl;
		DetectInCrop(colorFrame, crop, tmpRegions);
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

///
/// \brief SSDMobileNetDetector::DetectInCrop
/// \param colorFrame
/// \param crop
/// \param tmpRegions
///
void SSDMobileNetDetector::DetectInCrop(const cv::UMat& colorFrame, const cv::Rect& crop, regions_t& tmpRegions)
{
    cv::dnn::blobFromImage(cv::UMat(colorFrame, crop), m_inputBlob, m_inScaleFactor, cv::Size(m_inWidth, m_inHeight), m_meanVal, false, true);

    m_net.setInput(m_inputBlob, "data"); //set the network input

    cv::Mat detection = m_net.forward("detection_out"); //compute output

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for (int i = 0; i < detectionMat.rows; ++i)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > m_confidenceThreshold)
        {
            size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

            int xLeftBottom = cvRound(detectionMat.at<float>(i, 3) * crop.width) + crop.x;
            int yLeftBottom = cvRound(detectionMat.at<float>(i, 4) * crop.height) + crop.y;
            int xRightTop = cvRound(detectionMat.at<float>(i, 5) * crop.width) + crop.x;
            int yRightTop = cvRound(detectionMat.at<float>(i, 6) * crop.height) + crop.y;

            cv::Rect object(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom);

            tmpRegions.emplace_back(object, (objectClass < m_classNames.size()) ? m_classNames[objectClass] : "", confidence);
        }
    }
}
