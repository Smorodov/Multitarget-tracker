#include <fstream>
#include "OCVDNNDetector.h"
#include "nms.h"

///
/// \brief OCVDNNDetector::OCVDNNDetector
/// \param colorFrame
///
OCVDNNDetector::OCVDNNDetector(const cv::UMat& colorFrame)
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
/// \brief OCVDNNDetector::OCVDNNDetector
/// \param colorFrame
///
OCVDNNDetector::OCVDNNDetector(const cv::Mat& colorFrame)
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
/// \brief OCVDNNDetector::Init
/// \return
///
bool OCVDNNDetector::Init(const config_t& config)
{
#if (((CV_VERSION_MAJOR == 4) && (CV_VERSION_MINOR >= 2)) || (CV_VERSION_MAJOR > 4))
    std::map<cv::dnn::Target, std::string> dictTargets;
    dictTargets[cv::dnn::DNN_TARGET_CPU] = "DNN_TARGET_CPU";
    dictTargets[cv::dnn::DNN_TARGET_OPENCL] = "DNN_TARGET_OPENCL";
    dictTargets[cv::dnn::DNN_TARGET_OPENCL_FP16] = "DNN_TARGET_OPENCL_FP16";
    dictTargets[cv::dnn::DNN_TARGET_MYRIAD] = "DNN_TARGET_MYRIAD";
    dictTargets[cv::dnn::DNN_TARGET_CUDA] = "DNN_TARGET_CUDA";
    dictTargets[cv::dnn::DNN_TARGET_CUDA_FP16] = "DNN_TARGET_CUDA_FP16";
#if (((CV_VERSION_MAJOR == 4) && (CV_VERSION_MINOR >= 10)) || (CV_VERSION_MAJOR > 4))
    dictTargets[cv::dnn::DNN_TARGET_HDDL] = "DNN_TARGET_HDDL";
    dictTargets[cv::dnn::DNN_TARGET_NPU] = "DNN_TARGET_NPU";
    dictTargets[cv::dnn::DNN_TARGET_CPU_FP16] = "DNN_TARGET_CPU_FP16";
#endif

    std::map<int, std::string> dictBackends;
    dictBackends[cv::dnn::DNN_BACKEND_DEFAULT] = "DNN_BACKEND_DEFAULT";
    dictBackends[cv::dnn::DNN_BACKEND_INFERENCE_ENGINE] = "DNN_BACKEND_INFERENCE_ENGINE";
    dictBackends[cv::dnn::DNN_BACKEND_OPENCV] = "DNN_BACKEND_OPENCV";
    dictBackends[cv::dnn::DNN_BACKEND_VKCOM] = "DNN_BACKEND_VKCOM";
    dictBackends[cv::dnn::DNN_BACKEND_CUDA] = "DNN_BACKEND_CUDA";
#if (((CV_VERSION_MAJOR == 4) && (CV_VERSION_MINOR >= 10)) || (CV_VERSION_MAJOR > 4))
    dictBackends[cv::dnn::DNN_BACKEND_WEBNN] = "DNN_BACKEND_WEBNN";
    dictBackends[cv::dnn::DNN_BACKEND_TIMVX] = "DNN_BACKEND_TIMVX";
    dictBackends[cv::dnn::DNN_BACKEND_CANN] = "DNN_BACKEND_CANN";
#endif
    dictBackends[1000000] = "DNN_BACKEND_INFERENCE_ENGINE_NGRAPH";
    dictBackends[1000000 + 1] = "DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019";

    std::cout << "Avaible pairs for Target - backend:" << std::endl;
    std::vector<std::pair<cv::dnn::Backend, cv::dnn::Target>> pairs = cv::dnn::getAvailableBackends();
    for (auto p : pairs)
    {
        std::cout << dictBackends[p.first] << " (" << p.first << ") - " << dictTargets[p.second] << " (" << p.second << ")" << std::endl;
    }
#endif

    auto modelConfiguration = config.find("modelConfiguration");
    auto modelBinary = config.find("modelBinary");
    if (modelConfiguration != config.end() && modelBinary != config.end())
        m_net = cv::dnn::readNet(modelConfiguration->second, modelBinary->second, "");

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
#if (CV_VERSION_MAJOR > 4)
        targets["DNN_TARGET_HDDL"] = cv::dnn::DNN_TARGET_HDDL;
        targets["DNN_TARGET_NPU"] = cv::dnn::DNN_TARGET_NPU;
        targets["DNN_TARGET_CPU_FP16"] = cv::dnn::DNN_TARGET_CPU_FP16;
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
        backends["DNN_BACKEND_INFERENCE_ENGINE"] = cv::dnn::DNN_BACKEND_INFERENCE_ENGINE;
        backends["DNN_BACKEND_OPENCV"] = cv::dnn::DNN_BACKEND_OPENCV;
        backends["DNN_BACKEND_VKCOM"] = cv::dnn::DNN_BACKEND_VKCOM;
#if (((CV_VERSION_MAJOR == 4) && (CV_VERSION_MINOR >= 2)) || (CV_VERSION_MAJOR > 4))
        backends["DNN_BACKEND_CUDA"] = cv::dnn::DNN_BACKEND_CUDA;
#endif
#if (CV_VERSION_MAJOR > 4)
        backends["DNN_BACKEND_WEBNN"] = cv::dnn::DNN_BACKEND_WEBNN;
        backends["DNN_BACKEND_TIMVX"] = cv::dnn::DNN_BACKEND_TIMVX;
        backends["DNN_BACKEND_CANN"] = cv::dnn::DNN_BACKEND_CANN;
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

    auto net_type = config.find("net_type");
    if (net_type != config.end())
    {
        std::map<std::string, ModelType> dictNetType;
        dictNetType["YOLOV3"] = ModelType::YOLOV3;
        dictNetType["YOLOV4"] = ModelType::YOLOV4;
        dictNetType["YOLOV4_TINY"] = ModelType::YOLOV4_TINY;
        dictNetType["YOLOV5"] = ModelType::YOLOV5;
        dictNetType["YOLOV5_OBB"] = ModelType::YOLOV5_OBB;
        dictNetType["YOLOV5Mask"] = ModelType::YOLOV5Mask;
        dictNetType["YOLOV6"] = ModelType::YOLOV6;
        dictNetType["YOLOV7"] = ModelType::YOLOV7;
        dictNetType["YOLOV7Mask"] = ModelType::YOLOV7Mask;
        dictNetType["YOLOV8"] = ModelType::YOLOV8;
        dictNetType["YOLOV8_OBB"] = ModelType::YOLOV8_OBB;
        dictNetType["YOLOV8Mask"] = ModelType::YOLOV8Mask;
        dictNetType["YOLOV9"] = ModelType::YOLOV9;
        dictNetType["YOLOV10"] = ModelType::YOLOV10;
        dictNetType["YOLOV11"] = ModelType::YOLOV11;
        dictNetType["YOLOV11_OBB"] = ModelType::YOLOV11_OBB;
        dictNetType["YOLOV11Mask"] = ModelType::YOLOV11Mask;
        dictNetType["YOLOV12"] = ModelType::YOLOV12;
        dictNetType["RFDETR"] = ModelType::RFDETR;
        dictNetType["DFINE"] = ModelType::DFINE;

        auto netType = dictNetType.find(net_type->second);
        if (netType != dictNetType.end())
            m_netType = netType->second;
        else
        {
            assert(netType == dictNetType.end());
            std::cerr << "net_type = " << net_type->second << ", " << (int)m_netType << std::endl;
        }

        std::cout << "net_type = " << net_type->second << ", " << (int)m_netType << std::endl;
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

	m_classesWhiteList.clear();
	auto whiteRange = config.equal_range("white_list");
	for (auto it = whiteRange.first; it != whiteRange.second; ++it)
	{
        m_classesWhiteList.insert(TypeConverter::Str2Type(it->second));
	}

    auto confidenceThreshold = config.find("confidenceThreshold");
    if (confidenceThreshold != config.end())
        m_confidenceThreshold = std::stof(confidenceThreshold->second);

    auto nmsThreshold = config.find("nmsThreshold");
    if (nmsThreshold != config.end())
        m_nmsThreshold = std::stof(nmsThreshold->second);

    auto swapRB = config.find("swapRB");
    if (swapRB != config.end())
        m_swapRB = std::stoi(swapRB->second) != 0;

    auto maxCropRatio = config.find("maxCropRatio");
    if (maxCropRatio != config.end())
        m_maxCropRatio = std::stof(maxCropRatio->second);

    auto inWidth = config.find("inWidth");
    if (inWidth != config.end())
        m_inWidth = std::stoi(inWidth->second);

    auto inHeight = config.find("inHeight");
    if (inHeight != config.end())
        m_inHeight = std::stoi(inHeight->second);

    if (!m_net.empty())
    {
        m_outNames = m_net.getUnconnectedOutLayersNames();
        m_outLayers = m_net.getUnconnectedOutLayers();
        assert(!m_outLayers.empty());

        m_outLayerTypes.clear();
        for (auto it : m_outLayers)
        {
            m_outLayerTypes.push_back(m_net.getLayer(it)->type);
        }

        std::cout << "outNames: ";
        for (auto it : m_outNames)
        {
            std::cout << it << " | ";
        }
        std::cout << std::endl;

        std::cout << "outLayerType: ";
        for (auto it : m_outLayerTypes)
        {
            std::cout << it << " | ";
        }
        std::cout << std::endl;

#if (CV_VERSION_MAJOR < 5)
        std::vector<cv::dnn::MatShape> outputs;
        std::vector<cv::dnn::MatShape> internals;
        m_net.getLayerShapes(cv::dnn::MatShape(), 0, outputs, internals);
#else
        std::vector<cv::MatShape> outputs;
        std::vector<cv::MatShape> internals;
        m_net.getLayerShapes(cv::MatShape(), CV_32F, 0, outputs, internals);
#endif
        std::cout << "getLayerShapes: outputs (" << outputs.size() << ") = " << (outputs.size() > 0 ? outputs[0].size() : 0) << ", internals (" << internals.size() << ") = " << (internals.size() > 0 ? internals[0].size() : 0) << std::endl;
        if (outputs.size() && outputs[0].size() > 3)
        {
            std::cout << "outputs: ";
            for (size_t i = 0; i < outputs.size(); ++i)
            {
#if (CV_VERSION_MAJOR < 5)
                std::cout << i << ": [";
                for (size_t j = 0; j < outputs[i].size(); ++j)
                {
                    std::cout << outputs[i][j] << " ";
                }
                std::cout << "]";
#else
                std::cout << i << ": " << outputs[i].str();
#endif
            }
            std::cout << std::endl;

            std::cout << "internals: ";
            for (size_t i = 0; i < internals.size(); ++i)
            {
#if (CV_VERSION_MAJOR < 5)
                std::cout << i << ": [";
                for (size_t j = 0; j < internals[i].size(); ++j)
                {
                    std::cout << internals[i][j] << " ";
                }
                std::cout << "]";
#else
                std::cout << i << ": " << internals[i].str();
#endif
            }
            std::cout << std::endl;

            if (!m_inWidth || !m_inHeight)
            {
                m_inWidth = outputs[0][2];
                m_inHeight = outputs[0][3];
            }
        }
    }
    if (!m_inWidth || !m_inHeight)
    {
        m_inWidth = 608;
        m_inHeight = 608;
    }
    m_WHRatio = static_cast<float>(m_inWidth) / static_cast<float>(m_inHeight);

    std::cout << "input size: " << cv::Size(m_inWidth, m_inHeight) << ", m_WHRatio = " << m_WHRatio << std::endl;

    return !m_net.empty();
}

///
/// \brief OCVDNNDetector::Detect
/// \param gray
///
void OCVDNNDetector::Detect(const cv::UMat& colorFrame)
{
    m_regions.clear();

    regions_t tmpRegions;
    if (m_maxCropRatio <= 0)
    {
        DetectInCrop(colorFrame, cv::Rect(0, 0, colorFrame.cols, colorFrame.rows), tmpRegions);
    }
    else
    {
		std::vector<cv::Rect> crops = GetCrops(m_maxCropRatio, cv::Size(m_inWidth, m_inHeight), colorFrame.size());
		for (size_t i = 0; i < crops.size(); ++i)
		{
			const auto& crop = crops[i];
			//std::cout << "Crop " << i << ": " << crop << std::endl;
			DetectInCrop(colorFrame, crop, tmpRegions);
		}
    }
    nms3<CRegion>(tmpRegions, m_regions, m_nmsThreshold,
        [](const CRegion& reg) { return reg.m_brect; },
        [](const CRegion& reg) { return reg.m_confidence; },
        [](const CRegion& reg) { return reg.m_type; },
        0, static_cast<track_t>(0));
}

///
/// \brief OCVDNNDetector::DetectInCrop
/// \param colorFrame
/// \param crop
/// \param tmpRegions
///
void OCVDNNDetector::DetectInCrop(const cv::UMat& colorFrame, const cv::Rect& crop, regions_t& tmpRegions)
{
    //Convert Mat to batch of images
    cv::dnn::blobFromImage(cv::UMat(colorFrame, crop), m_inputBlob, 1.0, cv::Size(m_inWidth, m_inHeight), m_meanVal, m_swapRB, false, CV_8U);

    m_net.setInput(m_inputBlob, "", m_inScaleFactor, m_meanVal); //set the network input

    if (m_net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
    {
        //cv::resize(frame, frame, cv::Size(m_inWidth, m_inHeight));
        cv::Mat imInfo = (cv::Mat_<float>(1, 3) << m_inHeight, m_inWidth, 1.6f);
        m_net.setInput(imInfo, "im_info");
    }

    std::vector<cv::Mat> detections;
    m_net.forward(detections, m_outNames); //compute output

	switch (m_netType)
	{
	case ModelType::YOLOV5:
        ParseYOLOv5(crop, detections, tmpRegions);
        break;
	case ModelType::YOLOV8:
        ParseYOLOv8(crop, detections, tmpRegions);
        break;
	case ModelType::YOLOV9:
        ParseYOLOv9(crop, detections, tmpRegions);
        break;
	case ModelType::YOLOV10:
        ParseYOLOv10(crop, detections, tmpRegions);
        break;
	case ModelType::YOLOV11:
        ParseYOLOv11(crop, detections, tmpRegions);
        break;
    case ModelType::YOLOV12:
        ParseYOLOv11(crop, detections, tmpRegions);
        break;

    case ModelType::YOLOV5_OBB:
    case ModelType::YOLOV8_OBB:
    case ModelType::YOLOV11_OBB:
        ParseYOLOv5_8_11_obb(crop, detections, tmpRegions);
        break;

    case ModelType::YOLOV5Mask:
    case ModelType::YOLOV8Mask:
    case ModelType::YOLOV11Mask:
        ParseYOLOv5_8_11_seg(crop, detections, tmpRegions);
        break;

    case ModelType::RFDETR:
        ParseRFDETR(crop, detections, tmpRegions);
        break;

    case ModelType::DFINE:
        ParseDFINE(crop, detections, tmpRegions);
        break;

	default:
        ParseOldYOLO(crop, detections, tmpRegions);
        break;
	}
}

///
/// \brief OCVDNNDetector::ParseOldYOLO
/// \param crop
/// \param detections
/// \param tmpRegions
///
void OCVDNNDetector::ParseOldYOLO(const cv::Rect& crop, const std::vector<cv::Mat>& detections, regions_t& tmpRegions)
{
    if (m_outLayerTypes[0] == "DetectionOutput")
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(detections.size() > 0);
        for (size_t k = 0; k < detections.size(); ++k)
        {
            const float* data = reinterpret_cast<float*>(detections[k].data);
            for (size_t i = 0; i < detections[k].total(); i += 7)
            {
                float confidence = data[i + 2];
                if (confidence > m_confidenceThreshold)
                {
                    int left = (int)data[i + 3];
                    int top = (int)data[i + 4];
                    int right = (int)data[i + 5];
                    int bottom = (int)data[i + 6];
                    int width = right - left + 1;
                    int height = bottom - top + 1;
                    if (width <= 2 || height <= 2)
                    {
                        left = cvRound(data[i + 3] * crop.width);
                        top = cvRound(data[i + 4] * crop.height);
                        right = cvRound(data[i + 5] * crop.width);
                        bottom = cvRound(data[i + 6] * crop.height);
                        width = right - left + 1;
                        height = bottom - top + 1;
                    }
                    size_t objectClass = (int)(data[i + 1]) - 1;
                    if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(objectClass)) != std::end(m_classesWhiteList))
                        tmpRegions.emplace_back(cv::Rect(left + crop.x, top + crop.y, width, height), T2T(objectClass), confidence);
                }
            }
        }
    }
    else if (m_outLayerTypes[0] == "Region")
    {
        for (size_t i = 0; i < detections.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            const float* data = reinterpret_cast<float*>(detections[i].data);
            for (int j = 0; j < detections[i].rows; ++j, data += detections[i].cols)
            {
                cv::Mat scores = detections[i].row(j).colRange(5, detections[i].cols);
                cv::Point classIdPoint;
                double confidence = 0;
                cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > m_confidenceThreshold)
                {
                    int centerX = cvRound(data[0] * crop.width);
                    int centerY = cvRound(data[1] * crop.height);
                    int width = cvRound(data[2] * crop.width);
                    int height = cvRound(data[3] * crop.height);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(classIdPoint.x)) != std::end(m_classesWhiteList))
                        tmpRegions.emplace_back(cv::Rect(left + crop.x, top + crop.y, width, height), T2T(classIdPoint.x), static_cast<float>(confidence));
                }
            }
        }
    }
    else
    {
        CV_Error(cv::Error::StsNotImplemented, "OCVDNNDetector::ParseOldYOLO: Unknown output layer type: " + m_outLayerTypes[0] + ", net type " + std::to_string((int)m_netType));
    }
}

///
/// \brief OCVDNNDetector::ParseYOLOv5
/// \param crop
/// \param detections
/// \param tmpRegions
///
void OCVDNNDetector::ParseYOLOv5(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions)
{
	int rows = detections[0].size[1];
	int dimensions = detections[0].size[2];

	// yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
	// yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
	if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
	{
		rows = detections[0].size[2];
		dimensions = detections[0].size[1];

		detections[0] = detections[0].reshape(1, dimensions);
		cv::transpose(detections[0], detections[0]);
	}
	float* data = (float*)detections[0].data;

	float x_factor = crop.width / static_cast<float>(m_inWidth);
	float y_factor = crop.height / static_cast<float>(m_inHeight);

	for (int i = 0; i < rows; ++i)
	{
		float confidence = data[4];

		if (confidence >= m_confidenceThreshold)
		{
			float* classes_scores = data + 5;

			cv::Mat scores(1, static_cast<int>(m_classNames.size()), CV_32FC1, classes_scores);
			cv::Point class_id;
			double maxClassScore = 0;
			cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

			if (maxClassScore > m_confidenceThreshold)
			{
				float x = data[0];
				float y = data[1];
				float w = data[2];
				float h = data[3];

				int left = cvRound((x - 0.5f * w) * x_factor);
				int top = cvRound((y - 0.5f * h) * y_factor);

				int width = cvRound(w * x_factor);
				int height = cvRound(h * y_factor);

				if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(class_id.x)) != std::end(m_classesWhiteList))
					tmpRegions.emplace_back(cv::Rect(left + crop.x, top + crop.y, width, height), T2T(class_id.x), static_cast<float>(maxClassScore));
			}
		}
		data += dimensions;
	}
}

///
/// \brief OCVDNNDetector::ParseYOLOv8
/// \param crop
/// \param detections
/// \param tmpRegions
///
void OCVDNNDetector::ParseYOLOv8(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions)
{
	int rows = detections[0].size[1];
	int dimensions = detections[0].size[2];

	// yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
	// yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
	if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
	{
		rows = detections[0].size[2];
		dimensions = detections[0].size[1];

		detections[0] = detections[0].reshape(1, dimensions);
		cv::transpose(detections[0], detections[0]);
	}
	float* data = (float*)detections[0].data;

	float x_factor = crop.width / static_cast<float>(m_inWidth);
	float y_factor = crop.height / static_cast<float>(m_inHeight);

	for (int i = 0; i < rows; ++i)
	{
		float* classes_scores = data + 4;

		cv::Mat scores(1, static_cast<int>(m_classNames.size()), CV_32FC1, classes_scores);
		cv::Point class_id;
		double maxClassScore = 0;
		cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

		if (maxClassScore > m_confidenceThreshold)
		{
			float x = data[0];
			float y = data[1];
			float w = data[2];
			float h = data[3];

			int left = cvRound((x - 0.5f * w) * x_factor);
			int top = cvRound((y - 0.5f * h) * y_factor);

			int width = cvRound(w * x_factor);
			int height = cvRound(h * y_factor);

			if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(class_id.x)) != std::end(m_classesWhiteList))
				tmpRegions.emplace_back(cv::Rect(left + crop.x, top + crop.y, width, height), T2T(class_id.x), static_cast<float>(maxClassScore));
		}
		data += dimensions;
	}
}

///
/// \brief OCVDNNDetector::ParseYOLOv9
/// \param crop
/// \param detections
/// \param tmpRegions
///
void OCVDNNDetector::ParseYOLOv9(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions)
{
	int rows = detections[0].size[1];
	int dimensions = detections[0].size[2];

	// yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
	// yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
	if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
	{
		rows = detections[0].size[2];
		dimensions = detections[0].size[1];

		detections[0] = detections[0].reshape(1, dimensions);
		cv::transpose(detections[0], detections[0]);
	}
	float* data = (float*)detections[0].data;

	float x_factor = crop.width / static_cast<float>(m_inWidth);
	float y_factor = crop.height / static_cast<float>(m_inHeight);

	for (int i = 0; i < rows; ++i)
	{
		float* classes_scores = data + 4;

		cv::Mat scores(1, static_cast<int>(m_classNames.size()), CV_32FC1, classes_scores);
		cv::Point class_id;
		double maxClassScore = 0;
		cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

		if (maxClassScore > m_confidenceThreshold)
		{
			float x = data[0];
			float y = data[1];
			float w = data[2];
			float h = data[3];

			int left = cvRound((x - 0.5f * w) * x_factor);
			int top = cvRound((y - 0.5f * h) * y_factor);

			int width = cvRound(w * x_factor);
			int height = cvRound(h * y_factor);

			if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(class_id.x)) != std::end(m_classesWhiteList))
				tmpRegions.emplace_back(cv::Rect(left + crop.x, top + crop.y, width, height), T2T(class_id.x), static_cast<float>(maxClassScore));
		}
		data += dimensions;
	}
}

///
/// \brief OCVDNNDetector::ParseYOLOv10
/// \param crop
/// \param detections
/// \param tmpRegions
///
void OCVDNNDetector::ParseYOLOv10(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions)
{
	int rows = detections[0].size[1];
	int dimensions = detections[0].size[2];

	// yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
	// yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
	if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
	{
		rows = detections[0].size[2];
		dimensions = detections[0].size[1];

		detections[0] = detections[0].reshape(1, dimensions);
		cv::transpose(detections[0], detections[0]);
	}
	float* data = (float*)detections[0].data;

	float x_factor = crop.width / static_cast<float>(m_inWidth);
	float y_factor = crop.height / static_cast<float>(m_inHeight);

	for (int i = 0; i < rows; ++i)
	{
		int left = cvRound(x_factor * data[0]);
		int top = cvRound(y_factor * data[1]);
		int width = cvRound(x_factor * (data[2] - data[0]));
		int height = cvRound(y_factor * (data[3] - data[1]));
		float confidence = data[4];
		int classId = cvRound(data[5]);

		if (confidence >= m_confidenceThreshold)
		{
			if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(classId)) != std::end(m_classesWhiteList))
				tmpRegions.emplace_back(cv::Rect(left + crop.x, top + crop.y, width, height), T2T(classId), confidence);
		}
		data += dimensions;
	}
}

///
/// \brief OCVDNNDetector::ParseYOLOv11
/// \param crop
/// \param detections
/// \param tmpRegions
///
void OCVDNNDetector::ParseYOLOv11(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions)
{
	int rows = detections[0].size[1];
	int dimensions = detections[0].size[2];

	// yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
	// yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
	if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
	{
		rows = detections[0].size[2];
		dimensions = detections[0].size[1];

		detections[0] = detections[0].reshape(1, dimensions);
		cv::transpose(detections[0], detections[0]);
	}
	float* data = (float*)detections[0].data;

	float x_factor = crop.width / static_cast<float>(m_inWidth);
	float y_factor = crop.height / static_cast<float>(m_inHeight);

	for (int i = 0; i < rows; ++i)
	{
		float* classes_scores = data + 4;

		cv::Mat scores(1, static_cast<int>(m_classNames.size()), CV_32FC1, classes_scores);
		cv::Point class_id;
		double maxClassScore = 0;
		cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

		if (maxClassScore > m_confidenceThreshold)
		{
			float x = data[0];
			float y = data[1];
			float w = data[2];
			float h = data[3];

			int left = cvRound((x - 0.5f * w) * x_factor);
			int top = cvRound((y - 0.5f * h) * y_factor);

			int width = cvRound(w * x_factor);
			int height = cvRound(h * y_factor);

			if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(class_id.x)) != std::end(m_classesWhiteList))
				tmpRegions.emplace_back(cv::Rect(left + crop.x, top + crop.y, width, height), T2T(class_id.x), static_cast<float>(maxClassScore));
		}
		data += dimensions;
	}
}

///
/// \brief OCVDNNDetector::ParseYOLOv5_8_11_obb
/// \param crop
/// \param detections
/// \param tmpRegions
///
void OCVDNNDetector::ParseYOLOv5_8_11_obb(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions)
{
    int rows = detections[0].size[1];
    int dimensions = detections[0].size[2];

    // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
    // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
    if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
    {
        rows = detections[0].size[2];
        dimensions = detections[0].size[1];

        detections[0] = detections[0].reshape(1, dimensions);
        cv::transpose(detections[0], detections[0]);
    }
    float* data = (float*)detections[0].data;

    float x_factor = crop.width / static_cast<float>(m_inWidth);
    float y_factor = crop.height / static_cast<float>(m_inHeight);

    for (int i = 0; i < rows; ++i)
    {
        float* classes_scores = data + 4;

        cv::Mat scores(1, static_cast<int>(m_classNames.size()), CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore = 0;
        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > m_confidenceThreshold)
        {
            float x = data[0] * x_factor + crop.x;
            float y = data[1] * y_factor + crop.y;
            float w = data[2] * x_factor;
            float h = data[3] * y_factor;
            float angle = 180.f * data[4 + scores.cols] / static_cast<float>(M_PI);

            if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(class_id.x)) != std::end(m_classesWhiteList))
                tmpRegions.emplace_back(cv::RotatedRect(cv::Point2f(x, y), cv::Size2f(w, h), angle), T2T(class_id.x), static_cast<float>(maxClassScore));
        }
        data += dimensions;
    }
}

///
/// \brief OCVDNNDetector::ParseYOLOv5_8_11_seg
/// \param crop
/// \param detections
/// \param tmpRegions
///
void OCVDNNDetector::ParseYOLOv5_8_11_seg(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions)
{
    int rows = detections[0].size[1];
    int dimensions = detections[0].size[2];

    // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
    // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
    if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
    {
        rows = detections[0].size[2];
        dimensions = detections[0].size[1];

        detections[0] = detections[0].reshape(1, dimensions);
        cv::transpose(detections[0], detections[0]);
    }
    float* data = (float*)detections[0].data;

    float x_factor = crop.width / static_cast<float>(m_inWidth);
    float y_factor = crop.height / static_cast<float>(m_inHeight);

    for (int i = 0; i < rows; ++i)
    {
        float* classes_scores = data + 4;

        cv::Mat scores(1, static_cast<int>(m_classNames.size()), CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore = 0;
        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > m_confidenceThreshold)
        {
            float x = data[0] * x_factor + crop.x;
            float y = data[1] * y_factor + crop.y;
            float w = data[2] * x_factor;
            float h = data[3] * y_factor;
            //float angle = 180.f * data[4 + scores.cols] / M_PI;

            if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(class_id.x)) != std::end(m_classesWhiteList))
                tmpRegions.emplace_back(cv::RotatedRect(cv::Point2f(x, y), cv::Size2f(w, h), 0), T2T(class_id.x), static_cast<float>(maxClassScore));
        }
        data += dimensions;
    }
}

///
/// \brief OCVDNNDetector::ParseRFDETR
/// \param crop
/// \param detections
/// \param tmpRegions
///
void OCVDNNDetector::ParseRFDETR(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions)
{
    int rows = detections[0].size[1];
    int dimensionsDets = detections[0].size[2];
    int dimensionsLabels = detections[1].size[2];

    //0: name: input, size : 1x3x560x560
    //1: name: dets, size : 1x300x4
    //2: name: labels, size : 1x300x91

    float* dets = (float*)detections[0].data;
    float* labels = (float*)detections[1].data;

    float x_factor = crop.width / static_cast<float>(m_inWidth);
    float y_factor = crop.height / static_cast<float>(m_inHeight);

    auto L2Conf = [](float v)
    {
        return 1.f / (1.f + std::exp(-v));
    };

    for (int i = 0; i < rows; ++i)
    {
        float maxClassScore = L2Conf(labels[0]);
        size_t classId = 0;
        for (size_t cli = 1; cli < static_cast<size_t>(dimensionsLabels); ++cli)
        {
            auto conf = L2Conf(labels[cli]);
            if (maxClassScore < conf)
            {
                maxClassScore = conf;
                classId = cli;
            }
        }
        if (classId > 0)
            --classId;

        if (maxClassScore > m_confidenceThreshold)
        {
            float x = dets[0];
            float y = dets[1];
            float w = dets[2];
            float h = dets[3];

            int left = cvRound((x - 0.5f * w) * x_factor);
            int top = cvRound((y - 0.5f * h) * y_factor);

            int width = cvRound(w * x_factor);
            int height = cvRound(h * y_factor);

            if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(classId)) != std::end(m_classesWhiteList))
                tmpRegions.emplace_back(cv::Rect(left + crop.x, top + crop.y, width, height), T2T(classId), static_cast<float>(maxClassScore));
        }
        dets += dimensionsDets;
        labels += dimensionsLabels;
    }
}

///
/// \brief OCVDNNDetector::ParseDFINE
/// \param crop
/// \param detections
/// \param tmpRegions
///
void OCVDNNDetector::ParseDFINE(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions)
{
    int rows = detections[0].size[1];

    //0: name: images, size : 1x3x640x640
    //1: name: orig_target_sizes, size : 1x2
    //2: name: labels, size : 1x300
    //3: name: boxes, size : 1x300x4
    //4: name: scores, size : 1x300

    int64_t* labels = (int64_t*)detections[0].data;
    float* dets = (float*)detections[1].data;
    float* scores = (float*)detections[2].data;

    float x_factor = crop.width / static_cast<float>(m_inWidth);
    float y_factor = crop.height / static_cast<float>(m_inHeight);

    for (int i = 0; i < rows; ++i)
    {
        float maxClassScore = scores[i];
        size_t classId = labels[i];

        if (maxClassScore > m_confidenceThreshold)
        {
            float x = dets[4 * i + 0];
            float y = dets[4 * i + 1];
            float w = dets[4 * i + 2] - x;
            float h = dets[4 * i + 3] - y;

            int left = cvRound(x * x_factor);
            int top = cvRound(y * y_factor);

            int width = cvRound(w * x_factor);
            int height = cvRound(h * y_factor);

            if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(classId)) != std::end(m_classesWhiteList))
                tmpRegions.emplace_back(cv::Rect(left + crop.x, top + crop.y, width, height), T2T(classId), static_cast<float>(maxClassScore));
        }
    }
}

