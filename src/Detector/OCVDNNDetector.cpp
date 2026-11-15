#include <algorithm>
#include <cctype>
#include <fstream>
#include <utility>
#include "OCVDNNDetector.h"
#include "mtracking/nms.h"

namespace
{
///
/// \brief Safe query of the parser-consumed shape of a forward output.
///
/// A valid layout is rank 2 (rows x dimensions) or rank 3 (1 x rows x
/// dimensions). Returns false and leaves the outputs untouched for an
/// index outside the vector, an empty output and any other rank: the
/// parsers must not index size[] beyond the actual dims and must not
/// treat a wrong-model output as data (roadmap H19).
///
bool OutShape(const std::vector<cv::Mat>& detections, size_t index, int& rows, int& dimensions)
{
    if (index >= detections.size())
        return false;
    const cv::Mat& out = detections[index];
    if (out.total() == 0)
        return false;
    if (out.dims == 2)
    {
        rows = out.size[0];
        dimensions = out.size[1];
    }
    else if (out.dims == 3 && out.size[0] == 1)
    {
        rows = out.size[1];
        dimensions = out.size[2];
    }
    else
        return false;
    return rows > 0 && dimensions > 0;
}

///
/// \brief Find a forward output by a substring of its name
///        (case-insensitive).
/// \param names Output blob names (m_outNames).
/// \param keys Substrings to search for.
/// \return The first matching index or -1 if nothing matches.
///
/// The parsers must not rely on a hard-coded output order: the same model
/// family is exported with different output orders (e.g. D-FINE provides
/// [labels, boxes, scores] in the OpenCV DNN export and
/// [scores, labels, boxes] in the TensorRT one, see DFINE_bb.hpp).
///
int FindOut(const std::vector<cv::String>& names, std::initializer_list<const char*> keys)
{
    for (size_t i = 0; i < names.size(); ++i)
    {
        std::string name = names[i];
        std::transform(name.begin(), name.end(), name.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        for (const char* key : keys)
        {
            if (name.find(key) != std::string::npos)
                return static_cast<int>(i);
        }
    }
    return -1;
}
} // namespace

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
        dictNetType["RFDETR_IS"] = ModelType::RFDETR_IS;
        dictNetType["DFINE"] = ModelType::DFINE;
        dictNetType["YOLOV13"] = ModelType::YOLOV13;
        dictNetType["DFINE_IS"] = ModelType::DFINE_IS;
        dictNetType["YOLOV26"] = ModelType::YOLOV26;
        dictNetType["YOLOV26_OBB"] = ModelType::YOLOV26_OBB;
        dictNetType["YOLOV26Mask"] = ModelType::YOLOV26Mask;
        dictNetType["YOLOE"] = ModelType::YOLOE;
        dictNetType["YOLOEMask"] = ModelType::YOLOEMask;

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
            // A trailing blank line must not shift class indices or fail
            // validation; internal blank lines stay and are rejected below.
            while (!m_classNames.empty() && m_classNames.back().empty())
                m_classNames.pop_back();

			if (!FillTypesMap(m_classNames))
			{
				std::cerr << "OCVDNNDetector: unknown types in class names file "
					<< classNames->second << std::endl;
				return false;
			}
        }
    }

	m_classesWhiteList.clear();
	auto whiteRange = config.equal_range("white_list");
	for (auto it = whiteRange.first; it != whiteRange.second; ++it)
	{
		const objtype_t type = TypeConverter::Str2Type(it->second);
		if (type != bad_type)
			m_classesWhiteList.insert(type);
		else
			std::cerr << "OCVDNNDetector: unknown white_list type \""
				<< it->second << "\" (ignored)" << std::endl;
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
/// \brief OCVDNNDetector::ExpectedOutCount
///
size_t OCVDNNDetector::ExpectedOutCount(ModelType netType)
{
    switch (netType)
    {
    case ModelType::RFDETR:     // dets + labels
    case ModelType::RFDETR_IS:
        return 2;
    case ModelType::DFINE:      // labels + boxes + scores
        return 3;
    default:
        // The YOLO* families index only detections[0]; the old YOLO
        // parsers iterate all outputs and tolerate any count.
        return 1;
    }
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
    cv::dnn::blobFromImage(colorFrame(crop), m_inputBlob, 1.0, cv::Size(m_inWidth, m_inHeight), m_meanVal, m_swapRB, false, CV_8U);

    m_net.setInput(m_inputBlob, "", m_inScaleFactor, m_meanVal); //set the network input

    if (m_net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
    {
        //cv::resize(frame, frame, cv::Size(m_inWidth, m_inHeight));
        cv::Mat imInfo = cv::Mat_<float>({ 1, 3 }, { (float)m_inHeight, (float)m_inWidth, 1.6f });
        m_net.setInput(imInfo, "im_info");
    }

    std::vector<cv::Mat> detections;
    m_net.forward(detections, m_outNames); //compute output

#if 0
    std::cout << "Out layers:\n";
    for (size_t i = 0; i < detections.size(); ++i)
    {
        std::cout << i << ": " << detections[i].size() << "\n";
    }
#endif

    // A model that produced fewer outputs than the parser for the
    // configured net_type indexes is a model/net_type mismatch: without
    // the check the parser did an out-of-bounds std::vector access
    // (roadmap H19). Log once, skip the detections.
    const size_t expectedOutCount = ExpectedOutCount(m_netType);
    if (detections.size() < expectedOutCount)
    {
        if (!m_netMismatchLogged)
        {
            m_netMismatchLogged = true;
            std::cout << "OCVDNNDetector: the net produced " << detections.size()
                      << " outputs, but the configured net_type needs at least "
                      << expectedOutCount << " - the model does not match net_type"
                      << std::endl;
        }
        return;
    }

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

    case ModelType::RFDETR_IS:
        ParseRFDETR_IS(crop, detections, tmpRegions);
        break;

    case ModelType::DFINE:
        ParseDFINE(crop, detections, tmpRegions);
        break;

    case ModelType::YOLOV13:
        ParseYOLOv11(crop, detections, tmpRegions);
        break;

    case ModelType::DFINE_IS:
        ParseDFINE_IS(crop, detections, tmpRegions);
        break;

    case ModelType::YOLOV26:
        ParseYOLOv26(crop, detections, tmpRegions);
        break;

    case ModelType::YOLOV26_OBB:
        ParseYOLOv26_obb(crop, detections, tmpRegions);
        break;

    case ModelType::YOLOV26Mask:
        ParseYOLOv26_seg(crop, detections, tmpRegions);
        break;

    case ModelType::YOLOE:
        ParseYOLOE(crop, detections, tmpRegions);
        break;

    case ModelType::YOLOEMask:
        ParseYOLOEMask(crop, detections, tmpRegions);
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
	int rows = 0;
	int dimensions = 0;

	// yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
	// yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
	// Safe shape query: reject an empty/wrong-rank output (roadmap H19).
	if (!OutShape(detections, 0, rows, dimensions))
		return; // unexpected export: not (1 x N x F) / (N x F)
	if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
	{
		detections[0] = detections[0].reshape(1, rows);
		cv::transpose(detections[0], detections[0]);
		std::swap(rows, dimensions);
	}
	float* data = (float*)detections[0].data;

	if (dimensions < 5 + static_cast<int>(m_classNames.size()))
		return; // not enough columns for box + confidence + class scores

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
	int rows = 0;
	int dimensions = 0;

	// yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
	// yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
	// Safe shape query: reject an empty/wrong-rank output (roadmap H19).
	if (!OutShape(detections, 0, rows, dimensions))
		return; // unexpected export: not (1 x N x F) / (N x F)
	if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
	{
		detections[0] = detections[0].reshape(1, rows);
		cv::transpose(detections[0], detections[0]);
		std::swap(rows, dimensions);
	}
	float* data = (float*)detections[0].data;

	if (dimensions < 4 + static_cast<int>(m_classNames.size()))
		return; // not enough columns for box + class scores

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
	int rows = 0;
	int dimensions = 0;

	// yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
	// yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
	// Safe shape query: reject an empty/wrong-rank output (roadmap H19).
	if (!OutShape(detections, 0, rows, dimensions))
		return; // unexpected export: not (1 x N x F) / (N x F)
	if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
	{
		detections[0] = detections[0].reshape(1, rows);
		cv::transpose(detections[0], detections[0]);
		std::swap(rows, dimensions);
	}
	float* data = (float*)detections[0].data;

	if (dimensions < 4 + static_cast<int>(m_classNames.size()))
		return; // not enough columns for box + class scores

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
	int rows = 0;
	int dimensions = 0;

	// yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
	// yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
	// Safe shape query: reject an empty/wrong-rank output (roadmap H19).
	if (!OutShape(detections, 0, rows, dimensions))
		return; // unexpected export: not (1 x N x F) / (N x F)
	if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
	{
		detections[0] = detections[0].reshape(1, rows);
		cv::transpose(detections[0], detections[0]);
		std::swap(rows, dimensions);
	}
	float* data = (float*)detections[0].data;

	if (dimensions < 6)
		return; // not enough columns for box + confidence + class

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
	int rows = 0;
	int dimensions = 0;

	// yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
	// yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
	// Safe shape query: reject an empty/wrong-rank output (roadmap H19).
	if (!OutShape(detections, 0, rows, dimensions))
		return; // unexpected export: not (1 x N x F) / (N x F)
	if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
	{
		detections[0] = detections[0].reshape(1, rows);
		cv::transpose(detections[0], detections[0]);
		std::swap(rows, dimensions);
	}
	float* data = (float*)detections[0].data;

	if (dimensions < 4 + static_cast<int>(m_classNames.size()))
		return; // not enough columns for box + class scores

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
    int rows = 0;
    int dimensions = 0;

    // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
    // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
    // Safe shape query: reject an empty/wrong-rank output (roadmap H19).
    if (!OutShape(detections, 0, rows, dimensions))
        return; // unexpected export: not (1 x N x F) / (N x F)
    if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
    {
        detections[0] = detections[0].reshape(1, rows);
        cv::transpose(detections[0], detections[0]);
        std::swap(rows, dimensions);
    }
    float* data = (float*)detections[0].data;

    if (dimensions < 5 + static_cast<int>(m_classNames.size()))
        return; // not enough columns for box + class scores + angle

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
    int rows = 0;
    int dimensions = 0;

    // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
    // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
    // Safe shape query: reject an empty/wrong-rank output (roadmap H19).
    if (!OutShape(detections, 0, rows, dimensions))
        return; // unexpected export: not (1 x N x F) / (N x F)
    if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
    {
        detections[0] = detections[0].reshape(1, rows);
        cv::transpose(detections[0], detections[0]);
        std::swap(rows, dimensions);
    }
    float* data = (float*)detections[0].data;

    if (dimensions < 4 + static_cast<int>(m_classNames.size()))
        return; // not enough columns for box + class scores

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
    int rowsDets = 0, dimensionsDets = 0, rowsLabels = 0, dimensionsLabels = 0;
    if (!OutShape(detections, 0, rowsDets, dimensionsDets) ||
        !OutShape(detections, 1, rowsLabels, dimensionsLabels))
        return; // unexpected export: the outputs are not (1 x N x F) / (N x F) (roadmap H19)
    if (dimensionsDets < 4 || dimensionsLabels < 1)
        return; // not enough columns for box / labels
    const int rows = rowsDets < rowsLabels ? rowsDets : rowsLabels;

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
/// \brief OCVDNNDetector::ParseRFDETR_IS
/// \param crop
/// \param detections
/// \param tmpRegions
///
void OCVDNNDetector::ParseRFDETR_IS(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions)
{
    int rowsDets = 0, dimensionsDets = 0, rowsLabels = 0, dimensionsLabels = 0;
    if (!OutShape(detections, 0, rowsDets, dimensionsDets) ||
        !OutShape(detections, 1, rowsLabels, dimensionsLabels))
        return; // unexpected export: the outputs are not (1 x N x F) / (N x F) (roadmap H19)
    if (dimensionsDets < 4 || dimensionsLabels < 1)
        return; // not enough columns for box / labels
    const int rows = rowsDets < rowsLabels ? rowsDets : rowsLabels;

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
    // The post-processed D-FINE export provides three outputs (see
    // DFINE_bb.hpp): labels (1xN, integer), boxes (1xNx4, xyxy) and
    // scores (1xN). The export order may differ between backends, so
    // select the tensors by name and fall back to [labels, boxes,
    // scores] with a warning (roadmap C6).
    int labelsInd = FindOut(m_outNames, { "label" });
    int boxesInd = FindOut(m_outNames, { "box", "det" });
    int scoresInd = FindOut(m_outNames, { "score" });
    if (labelsInd < 0 || boxesInd < 0 || scoresInd < 0)
    {
        labelsInd = 0;
        boxesInd = 1;
        scoresInd = 2;
        std::cout << "OCVDNNDetector: ParseDFINE cannot select outputs by name, using the [labels, boxes, scores] order" << std::endl;
    }
    if (labelsInd >= static_cast<int>(detections.size()) ||
        boxesInd >= static_cast<int>(detections.size()) ||
        scoresInd >= static_cast<int>(detections.size()))
        return; // fewer output blobs than expected (roadmap H19)

    int rowsBoxes = 0, dimBoxes = 0;
    if (!OutShape(detections, boxesInd, rowsBoxes, dimBoxes))
        return; // unexpected export: the outputs are not (1 x N x F) / (N x F) (roadmap H19)
    if (dimBoxes < 4)
        return; // not enough columns for boxes
    if (detections[boxesInd].total() < static_cast<size_t>(rowsBoxes) * 4)
        return; // boxes shorter than expected
    if (detections[boxesInd].depth() != CV_32F || detections[scoresInd].depth() != CV_32F)
        return; // unexpected boxes/scores data type (roadmap C6)
    // Rank sanity for the flat outputs (roadmap H19); the detection count
    // is taken from total(), not from OutShape rows: a (1xN) tensor yields
    // rows=1, which would clamp the number of parsed detections to 1.
    int rowsLabels = 0, dimLabels = 0, rowsScores = 0, dimScores = 0;
    if (!OutShape(detections, labelsInd, rowsLabels, dimLabels) ||
        !OutShape(detections, scoresInd, rowsScores, dimScores))
        return; // unexpected labels/scores shape (roadmap H19)
    (void)dimLabels;
    (void)dimScores;

    // OpenCV DNN has no int64 blob type: integer outputs arrive as CV_32S.
    // The old code cast labels to int64_t* and read 8 bytes per element out
    // of a 4-byte buffer (roadmap C6).
    cv::Mat labelsFlat;
    if (detections[labelsInd].depth() == CV_32S)
        labelsFlat = detections[labelsInd].reshape(1, 1); // share the buffer, no copy
    else if (detections[labelsInd].depth() == CV_32F || detections[labelsInd].depth() == CV_64F)
        detections[labelsInd].reshape(1, 1).convertTo(labelsFlat, CV_32S);
    else
        return; // unexpected labels data type (roadmap C6)
    cv::Mat boxesFlat = detections[boxesInd].reshape(1, 1);
    cv::Mat scoresFlat = detections[scoresInd].reshape(1, 1);

    int rows = std::min(rowsBoxes, static_cast<int>(labelsFlat.total()));
    rows = std::min(rows, static_cast<int>(scoresFlat.total()));

    const float* dets = boxesFlat.ptr<float>(0);
    const float* scores = scoresFlat.ptr<float>(0);
    const int* labels = labelsFlat.ptr<int>(0);

    float x_factor = crop.width / static_cast<float>(m_inWidth);
    float y_factor = crop.height / static_cast<float>(m_inHeight);

    for (int i = 0; i < rows; ++i)
    {
        const float maxClassScore = scores[i];
        const int label = labels[i];
        if (label < 0)
            continue; // invalid class id
        const size_t classId = static_cast<size_t>(label);

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

///
/// \brief OCVDNNDetector::ParseDFINE_IS
/// \param crop
/// \param detections
/// \param tmpRegions
///
void OCVDNNDetector::ParseDFINE_IS(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions)
{
    // The D-FINE instance segmentation export (see DFINE_is.hpp) provides
    // logits (1xNxC), boxes (1xNx4, cxcywh in normalized input
    // coordinates) and mask_probs (1xNxmhxmw). The post-processed export
    // provides integer labels (1xN) and scores (1xN) instead of logits.
    // Masks are not consumed by the tracking pipeline (the same as in
    // ParseYOLOv5_8_11_seg and ParseRFDETR_IS), so only boxes and classes
    // are parsed here (roadmap C6).
    int boxesInd = FindOut(m_outNames, { "box", "det" });
    int labelsInd = FindOut(m_outNames, { "logit", "label" });
    if (boxesInd < 0 || labelsInd < 0 || boxesInd == labelsInd)
    {
        // Fallback: the raw export order [logits, boxes, masks].
        labelsInd = 0;
        boxesInd = 1;
        std::cout << "OCVDNNDetector: ParseDFINE_IS cannot select outputs by name, using the [logits, boxes, masks] order" << std::endl;
    }
    if (boxesInd >= static_cast<int>(detections.size()) ||
        labelsInd >= static_cast<int>(detections.size()))
        return; // fewer output blobs than expected (roadmap H19)

    int rowsBoxes = 0, dimBoxes = 0;
    if (!OutShape(detections, boxesInd, rowsBoxes, dimBoxes))
        return; // unexpected export: the outputs are not (1 x N x F) / (N x F) (roadmap H19)
    if (dimBoxes < 4)
        return; // not enough columns for boxes
    if (detections[boxesInd].total() < static_cast<size_t>(rowsBoxes) * 4)
        return; // boxes shorter than expected
    if (detections[boxesInd].depth() != CV_32F)
        return; // unexpected boxes data type (roadmap C6)
    // Rank sanity for the labels/logits output (roadmap H19).
    int rowsLogits = 0, dimLogits = 0;
    if (!OutShape(detections, labelsInd, rowsLogits, dimLogits))
        return; // unexpected labels/logits shape (roadmap H19)
    (void)rowsLogits;
    (void)dimLogits;
    cv::Mat boxesFlat = detections[boxesInd].reshape(1, 1);
    const float* dets = boxesFlat.ptr<float>(0);

    float x_factor = crop.width / static_cast<float>(m_inWidth);
    float y_factor = crop.height / static_cast<float>(m_inHeight);

    auto L2Conf = [](float v)
    {
        return 1.f / (1.f + std::exp(-v));
    };

    if (detections[labelsInd].depth() == CV_32S)
    {
        // Post-processed export: integer labels + float scores. Never
        // reinterpret an integer blob through a wider type (roadmap C6).
        int scoresInd = FindOut(m_outNames, { "score" });
        if (scoresInd < 0 || scoresInd >= static_cast<int>(detections.size()))
        {
            std::cout << "OCVDNNDetector: ParseDFINE_IS cannot find the scores output" << std::endl;
            return;
        }
        if (detections[scoresInd].depth() != CV_32F)
            return; // unexpected scores data type (roadmap C6)
        // Rank sanity for the scores output (roadmap H19).
        int rowsScores = 0, dimScores = 0;
        if (!OutShape(detections, scoresInd, rowsScores, dimScores))
            return; // unexpected scores shape (roadmap H19)
        (void)rowsScores;
        (void)dimScores;
        cv::Mat scoresFlat = detections[scoresInd].reshape(1, 1);
        cv::Mat labelsFlat = detections[labelsInd].reshape(1, 1);

        // Labels and scores are flat (1xN): count elements, not rows.
        int rows = std::min(rowsBoxes, static_cast<int>(labelsFlat.total()));
        rows = std::min(rows, static_cast<int>(scoresFlat.total()));

        const float* scores = scoresFlat.ptr<float>(0);
        const int* labels = labelsFlat.ptr<int>(0);

        for (int i = 0; i < rows; ++i)
        {
            const float maxClassScore = scores[i];
            const int label = labels[i];
            if (label < 0)
                continue; // invalid class id
            const size_t classId = static_cast<size_t>(label);

            if (maxClassScore > m_confidenceThreshold)
            {
                // Boxes are xyxy in the post-processed export (the same as ParseDFINE).
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
    else if (detections[labelsInd].depth() == CV_32F)
    {
        // Raw export: per-class logits; the class is argmax + sigmoid.
        const cv::Mat& logitsMat = detections[labelsInd];
        int nc = 0;
        if (logitsMat.dims == 3 && logitsMat.size[0] == 1)
            nc = logitsMat.size[2];
        else if (logitsMat.dims == 2)
            nc = logitsMat.size[1];
        else
            return; // unexpected logits shape
        if (nc < 2)
            return; // argmax over classes makes no sense
        if (logitsMat.total() < static_cast<size_t>(rowsBoxes) * static_cast<size_t>(nc))
            return; // logits shorter than boxes

        cv::Mat logitsFlat = logitsMat.reshape(1, 1);
        const float* logits = logitsFlat.ptr<float>(0);

        for (int i = 0; i < rowsBoxes; ++i)
        {
            const float* curLogits = logits + static_cast<size_t>(i) * nc;
            float classConf = L2Conf(curLogits[0]);
            size_t classId = 0;
            for (int cli = 1; cli < nc; ++cli)
            {
                const float conf = L2Conf(curLogits[cli]);
                if (classConf < conf)
                {
                    classConf = conf;
                    classId = static_cast<size_t>(cli);
                }
            }

            if (classConf > m_confidenceThreshold)
            {
                // Boxes are cxcywh in normalized input coordinates (see DFINE_is.hpp).
                const float cx = dets[4 * i + 0];
                const float cy = dets[4 * i + 1];
                const float bw = dets[4 * i + 2];
                const float bh = dets[4 * i + 3];
                const float x = (cx - 0.5f * bw) * static_cast<float>(m_inWidth);
                const float y = (cy - 0.5f * bh) * static_cast<float>(m_inHeight);
                const float w = bw * static_cast<float>(m_inWidth);
                const float h = bh * static_cast<float>(m_inHeight);

                int left = cvRound(x * x_factor);
                int top = cvRound(y * y_factor);

                int width = cvRound(w * x_factor);
                int height = cvRound(h * y_factor);

                if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(classId)) != std::end(m_classesWhiteList))
                    tmpRegions.emplace_back(cv::Rect(left + crop.x, top + crop.y, width, height), T2T(classId), classConf);
            }
        }
    }
    else
        return; // unexpected labels/logits data type (roadmap C6)
}

///
/// \brief OCVDNNDetector::ParseYOLOv26
/// \param crop
/// \param detections
/// \param tmpRegions
///
void OCVDNNDetector::ParseYOLOv26(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions)
{
    int rows = 0;
    int dimensions = 0;
    if (!OutShape(detections, 0, rows, dimensions))
        return; // unexpected export (roadmap H19)
    if (dimensions < 6)
        return; // not enough columns for box + confidence + class

    //0: name: images, size: 1x3x640x640
    //1: name: output0, size: 1x300x6

    float* dets = (float*)detections[0].data;

    float x_factor = crop.width / static_cast<float>(m_inWidth);
    float y_factor = crop.height / static_cast<float>(m_inHeight);

    //std::cout << "detections: " << rows << std::endl;

    for (int i = 0; i < rows; ++i)
    {
        auto ind = 6 * i;

        float maxClassScore = dets[ind + 4];
        size_t classId = static_cast<size_t>(dets[ind + 5]);

        if (maxClassScore > m_confidenceThreshold)
        {
            float x = dets[ind + 0];
            float y = dets[ind + 1];
            float w = dets[ind + 2] - x;
            float h = dets[ind + 3] - y;

            int left = cvRound(x * x_factor);
            int top = cvRound(y * y_factor);

            int width = cvRound(w * x_factor);
            int height = cvRound(h * y_factor);

            //std::cout << "ind: " << ind << ", score = " << maxClassScore << ", class = " << classId << ", rect = " << cv::Rect(left, top, width, height) << std::endl;

            if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(classId)) != std::end(m_classesWhiteList))
                tmpRegions.emplace_back(cv::Rect(left + crop.x, top + crop.y, width, height), T2T(classId), static_cast<float>(maxClassScore));
        }
    }
}

///
/// \brief OCVDNNDetector::ParseYOLOv26_obb
/// \param crop
/// \param detections
/// \param tmpRegions
///
void OCVDNNDetector::ParseYOLOv26_obb(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions)
{
    int rows = 0;
    int dimensions = 0;
    if (!OutShape(detections, 0, rows, dimensions))
        return; // unexpected export (roadmap H19)
    if (dimensions < 7)
        return; // not enough columns for box + confidence + class + angle

    //0: name: images, size: 1x3x1024x1024
    //1: name: output0, size: 1x300x7

    float* dets = (float*)detections[0].data;

    float x_factor = crop.width / static_cast<float>(m_inWidth);
    float y_factor = crop.height / static_cast<float>(m_inHeight);

    //std::cout << "detections: " << rows << std::endl;

    for (int i = 0; i < rows; ++i)
    {
        auto ind = 7 * i;

        float maxClassScore = dets[ind + 4];
        size_t classId = static_cast<size_t>(dets[ind + 5]);

        if (maxClassScore > m_confidenceThreshold)
        {
            float x = dets[ind + 0] * x_factor;
            float y = dets[ind + 1] * y_factor;
            float w = dets[ind + 2] * x_factor;
            float h = dets[ind + 3] * y_factor;
            float angle = 180.f * dets[ind + 6] / static_cast<float>(M_PI);

            //std::cout << "ind: " << ind << ", score = " << maxClassScore << ", class = " << classId << ", rect = " << cv::Rect(left, top, width, height) << std::endl;

            if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(classId)) != std::end(m_classesWhiteList))
                tmpRegions.emplace_back(cv::RotatedRect(cv::Point2f(x + crop.x, y + crop.y), cv::Size2f(w, h), angle), T2T(classId), static_cast<float>(maxClassScore));
        }
    }
}

///
/// \brief OCVDNNDetector::ParseYOLOv26_seg
/// \param crop
/// \param detections
/// \param tmpRegions
///
void OCVDNNDetector::ParseYOLOv26_seg(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions)
{
    int rows = 0;
    int dimensions = 0;
    if (!OutShape(detections, 0, rows, dimensions))
        return; // unexpected export (roadmap H19)
    if (dimensions < 38)
        return; // not enough columns for box + confidence + class + mask coefficients

    //0: name: images, size: 1x3x640x640
    //1: name: output0, size: 1x300x38
    //2: name: output1, size: 1x32x160x160

    float* dets = (float*)detections[0].data;

    float x_factor = crop.width / static_cast<float>(m_inWidth);
    float y_factor = crop.height / static_cast<float>(m_inHeight);

    //std::cout << "detections: " << rows << std::endl;

    for (int i = 0; i < rows; ++i)
    {
        auto ind = 38 * i;

        float maxClassScore = dets[ind + 4];
        size_t classId = static_cast<size_t>(dets[ind + 5]);

        if (maxClassScore > m_confidenceThreshold)
        {
            float x = dets[ind + 0];
            float y = dets[ind + 1];
            float w = dets[ind + 2] - x;
            float h = dets[ind + 3] - y;

            int left = cvRound(x * x_factor);
            int top = cvRound(y * y_factor);

            int width = cvRound(w * x_factor);
            int height = cvRound(h * y_factor);

            //std::cout << "ind: " << ind << ", score = " << maxClassScore << ", class = " << classId << ", rect = " << cv::Rect(left, top, width, height) << std::endl;

            if (m_classesWhiteList.empty() || m_classesWhiteList.find(T2T(classId)) != std::end(m_classesWhiteList))
                tmpRegions.emplace_back(cv::Rect(left + crop.x, top + crop.y, width, height), T2T(classId), static_cast<float>(maxClassScore));
        }
    }
}

///
/// \brief OCVDNNDetector::ParseYOLOE
/// \param crop
/// \param detections
/// \param tmpRegions
///
void OCVDNNDetector::ParseYOLOE(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions)
{
    int rows = 0;
    int dimensions = 0;
    if (!OutShape(detections, 0, rows, dimensions))
        return; // unexpected export (roadmap H19)
    if (rows < dimensions) // features are in the first axis (e.g. 1x54x8400)
    {
        detections[0] = detections[0].reshape(1, rows);
        cv::transpose(detections[0], detections[0]);
        std::swap(rows, dimensions);
    }
    if (dimensions < 4 + static_cast<int>(m_classNames.size()))
        return; // not enough columns for box + class scores

    float x_factor = crop.width / static_cast<float>(m_inWidth);
    float y_factor = crop.height / static_cast<float>(m_inHeight);

    float* data = (float*)detections[0].data;

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
/// \brief OCVDNNDetector::ParseYOLOEMask
/// \param crop
/// \param detections
/// \param tmpRegions
///
void OCVDNNDetector::ParseYOLOEMask(const cv::Rect& crop, std::vector<cv::Mat>& detections, regions_t& tmpRegions)
{
    int rows = 0;
    int dimensions = 0;
    if (!OutShape(detections, 0, rows, dimensions))
        return; // unexpected export (roadmap H19)
    if (rows < dimensions) // features are in the first axis (e.g. 1x54x8400)
    {
        detections[0] = detections[0].reshape(1, rows);
        cv::transpose(detections[0], detections[0]);
        std::swap(rows, dimensions);
    }
    if (dimensions < 4 + static_cast<int>(m_classNames.size()))
        return; // not enough columns for box + class scores

    float x_factor = crop.width / static_cast<float>(m_inWidth);
    float y_factor = crop.height / static_cast<float>(m_inHeight);

    float* data = (float*)detections[0].data;

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
