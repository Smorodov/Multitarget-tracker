#include <fstream>
#include "OCVDNNDetector.h"
#include "nms.h"

///
/// \brief OCVDNNDetector::OCVDNNDetector
/// \param gray
///
OCVDNNDetector::OCVDNNDetector(cv::UMat& colorFrame)
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
/// \brief OCVDNNDetector::~OCVDNNDetector
///
OCVDNNDetector::~OCVDNNDetector(void)
{
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

    std::map<int, std::string> dictBackends;
    dictBackends[cv::dnn::DNN_BACKEND_DEFAULT] = "DNN_BACKEND_DEFAULT";
    dictBackends[cv::dnn::DNN_BACKEND_HALIDE] = "DNN_BACKEND_HALIDE";
    dictBackends[cv::dnn::DNN_BACKEND_INFERENCE_ENGINE] = "DNN_BACKEND_INFERENCE_ENGINE";
    dictBackends[cv::dnn::DNN_BACKEND_OPENCV] = "DNN_BACKEND_OPENCV";
    dictBackends[cv::dnn::DNN_BACKEND_VKCOM] = "DNN_BACKEND_VKCOM";
    dictBackends[cv::dnn::DNN_BACKEND_CUDA] = "DNN_BACKEND_CUDA";
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
        m_outLayerType = m_net.getLayer(m_outLayers[0])->type;

        std::vector<cv::dnn::MatShape> outputs;
        std::vector<cv::dnn::MatShape> internals;
        m_net.getLayerShapes(cv::dnn::MatShape(), 0, outputs, internals);
        std::cout << "getLayerShapes: outputs (" << outputs.size() << ") = " << (outputs.size() > 0 ? outputs[0].size() : 0) << ", internals (" << internals.size() << ") = " << (internals.size() > 0 ? internals[0].size() : 0) << std::endl;
        if (outputs.size() && outputs[0].size() > 3)
        {
            std::cout << "outputs = [" << outputs[0][0] << ", " << outputs[0][1] << ", " << outputs[0][2]  << ", " << outputs[0][3] << "], internals = [" << internals[0][0] << ", " << internals[0][1] << ", " << internals[0][2]  << ", " << internals[0][3] << "]" << std::endl;

            m_inWidth = outputs[0][2];
            m_inHeight = outputs[0][3];
        }
    }
    m_WHRatio = static_cast<float>(m_inWidth) / static_cast<float>(m_inHeight);

    return !m_net.empty();
}

///
/// \brief OCVDNNDetector::Detect
/// \param gray
///
void OCVDNNDetector::Detect(cv::UMat& colorFrame)
{
    m_regions.clear();

    cv::Mat colorMat = colorFrame.getMat(cv::ACCESS_READ);

    if (m_maxCropRatio <= 0)
    {
        DetectInCrop(colorMat, cv::Rect(0, 0, colorMat.cols, colorMat.rows), m_regions);
    }
    else
    {
        int cropHeight = cvRound(m_maxCropRatio * m_inHeight);
        int cropWidth = cvRound(m_maxCropRatio * m_inWidth);

        if (colorFrame.cols / (float)colorFrame.rows > m_WHRatio)
        {
            if (cropHeight >= colorFrame.rows)
                cropHeight = colorFrame.rows;

            cropWidth = cvRound(cropHeight * m_WHRatio);
        }
        else
        {
            if (cropWidth >= colorFrame.cols)
                cropWidth = colorFrame.cols;

            cropHeight = cvRound(colorFrame.cols / m_WHRatio);
        }

        cv::Rect crop(0, 0, cropWidth, cropHeight);
        regions_t tmpRegions;
        size_t cropsCount = 0;
        for (; crop.y < colorMat.rows; crop.y += crop.height / 2)
        {
            bool needBreakY = false;
            if (crop.y + crop.height >= colorMat.rows)
            {
                crop.y = colorMat.rows - crop.height;
                needBreakY = true;
            }
            for (crop.x = 0; crop.x < colorMat.cols; crop.x += crop.width / 2)
            {
                bool needBreakX = false;
                if (crop.x + crop.width >= colorMat.cols)
                {
                    crop.x = colorMat.cols - crop.width;
                    needBreakX = true;
                }

                DetectInCrop(colorMat, crop, tmpRegions);
                ++cropsCount;

                if (needBreakX)
                    break;
            }
            if (needBreakY)
                break;
        }

        std::cout << "cropsCount = " << cropsCount << std::endl;
        if (cropsCount > 1)
            nms3<CRegion>(tmpRegions, m_regions, m_nmsThreshold,
                          [](const CRegion& reg) -> cv::Rect { return reg.m_brect; },
            [](const CRegion& reg) -> float { return reg.m_confidence; },
            [](const CRegion& reg) -> std::string { return reg.m_type; },
            0, 0.f);
    }
}

///
/// \brief OCVDNNDetector::DetectInCrop
/// \param colorFrame
/// \param crop
/// \param tmpRegions
///
void OCVDNNDetector::DetectInCrop(cv::Mat colorFrame, const cv::Rect& crop, regions_t& tmpRegions)
{
    //Convert Mat to batch of images
    cv::dnn::blobFromImage(cv::Mat(colorFrame, crop), m_inputBlob, 1.0, cv::Size(m_inWidth, m_inHeight), m_meanVal, m_swapRB, false, CV_8U);

    m_net.setInput(m_inputBlob, "", m_inScaleFactor, m_meanVal); //set the network input

    if (m_net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
    {
        //cv::resize(frame, frame, cv::Size(m_inWidth, m_inHeight));
        cv::Mat imInfo = (cv::Mat_<float>(1, 3) << m_inHeight, m_inWidth, 1.6f);
        m_net.setInput(imInfo, "im_info");
    }

    std::vector<cv::Mat> detections;
    m_net.forward(detections, m_outNames); //compute output

    if (m_outLayerType == "DetectionOutput")
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
                        left = (int)(data[i + 3] * crop.width);
                        top = (int)(data[i + 4] * crop.height);
                        right = (int)(data[i + 5] * crop.width);
                        bottom = (int)(data[i + 6] * crop.height);
                        width = right - left + 1;
                        height = bottom - top + 1;
                    }
                    size_t objectClass = (int)(data[i + 1]) - 1;
                    tmpRegions.emplace_back(cv::Rect(left + crop.x, top + crop.y, width, height),
                                            (objectClass < m_classNames.size()) ? m_classNames[objectClass] : "", confidence);
                }
            }
        }
    }
    else if (m_outLayerType == "Region")
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
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > m_confidenceThreshold)
                {
                    int centerX = (int)(data[0] * crop.width);
                    int centerY = (int)(data[1] * crop.height);
                    int width = (int)(data[2] * crop.width);
                    int height = (int)(data[3] * crop.height);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    tmpRegions.emplace_back(cv::Rect(left + crop.x, top + crop.y, width, height),
                                            (classIdPoint.x < static_cast<int>(m_classNames.size())) ? m_classNames[classIdPoint.x] : "", static_cast<float>(confidence));
                }
            }
        }
    }
    else
        CV_Error(cv::Error::StsNotImplemented, "Unknown output layer type: " + m_outLayerType);
}
