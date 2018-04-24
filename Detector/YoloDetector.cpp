#include "YoloDetector.h"
#include "nms.h"

///
/// \brief YoloDetector::YoloDetector
/// \param collectPoints
/// \param gray
///
YoloDetector::YoloDetector(
	bool collectPoints,
    cv::UMat& colorFrame
	)
    :
      BaseDetector(collectPoints, colorFrame),
      m_WHRatio(InWidth / (float)InHeight),
      m_inScaleFactor(0.003921f),
      m_meanVal(0),
      m_confidenceThreshold(0.24f),
      m_maxCropRatio(2.0f)
{
    m_classNames = { "background",
                     "aeroplane", "bicycle", "bird", "boat",
                     "bottle", "bus", "car", "cat", "chair",
                     "cow", "diningtable", "dog", "horse",
                     "motorbike", "person", "pottedplant",
                     "sheep", "sofa", "train", "tvmonitor" };
}

///
/// \brief YoloDetector::~YoloDetector
///
YoloDetector::~YoloDetector(void)
{
}

///
/// \brief YoloDetector::Init
/// \return
///
bool YoloDetector::Init(const config_t& config)
{
    auto modelConfiguration = config.find("modelConfiguration");
    auto modelBinary = config.find("modelBinary");
    if (modelConfiguration != config.end() && modelBinary != config.end())
    {
        m_net = cv::dnn::readNetFromDarknet(modelConfiguration->second, modelBinary->second);
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

    return !m_net.empty();
}

///
/// \brief YoloDetector::Detect
/// \param gray
///
void YoloDetector::Detect(cv::UMat& colorFrame)
{
    m_regions.clear();

    regions_t tmpRegions;

    cv::Mat colorMat = colorFrame.getMat(cv::ACCESS_READ);

    int cropHeight = cvRound(m_maxCropRatio * InHeight);
    int cropWidth = cvRound(m_maxCropRatio * InWidth);

    if (colorFrame.cols / (float)colorFrame.rows > m_WHRatio)
    {
        if (m_maxCropRatio <= 0 || cropHeight >= colorFrame.rows)
        {
            cropHeight = colorFrame.rows;
        }
        cropWidth = cvRound(cropHeight * m_WHRatio);
    }
    else
    {
        if (m_maxCropRatio <= 0 || cropWidth >= colorFrame.cols)
        {
            cropWidth = colorFrame.cols;
        }
        cropHeight = cvRound(colorFrame.cols / m_WHRatio);
    }

    cv::Rect crop(0, 0, cropWidth, cropHeight);

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

            if (needBreakX)
            {
                break;
            }
        }
        if (needBreakY)
        {
            break;
        }
    }

    nms3<CRegion>(tmpRegions, m_regions, 0.4f,
         [](const CRegion& reg) -> cv::Rect { return reg.m_rect; },
    [](const CRegion& reg) -> float { return reg.m_confidence; },
    [](const CRegion& reg) -> std::string { return reg.m_type; },
    0, 0.f);

    if (m_collectPoints)
    {
        for (auto& region : m_regions)
        {
            CollectPoints(region);
        }
    }
}

///
/// \brief YoloDetector::DetectInCrop
/// \param colorFrame
/// \param crop
/// \param tmpRegions
///
void YoloDetector::DetectInCrop(cv::Mat colorFrame, const cv::Rect& crop, regions_t& tmpRegions)
{
    //Convert Mat to batch of images
    cv::Mat inputBlob = cv::dnn::blobFromImage(cv::Mat(colorFrame, crop), m_inScaleFactor, cv::Size(InWidth, InHeight), m_meanVal, false, true);

    m_net.setInput(inputBlob, "data"); //set the network input

    cv::Mat detectionMat = m_net.forward("detection_out"); //compute output

    for (int i = 0; i < detectionMat.rows; ++i)
    {
        const int probability_index = 5;
        const int probability_size = detectionMat.cols - probability_index;
        float* prob_array_ptr = &detectionMat.at<float>(i, probability_index);

        size_t objectClass = std::max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
        float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

        if (confidence > m_confidenceThreshold)
        {
            float x_center = detectionMat.at<float>(i, 0) * crop.width + crop.x;
            float y_center = detectionMat.at<float>(i, 1) * crop.height + crop.y;
            float width = detectionMat.at<float>(i, 2) * crop.width;
            float height = detectionMat.at<float>(i, 3) * crop.height;
            cv::Point p1(cvRound(x_center - width / 2), cvRound(y_center - height / 2));
            cv::Point p2(cvRound(x_center + width / 2), cvRound(y_center + height / 2));
            cv::Rect object(p1, p2);

            tmpRegions.push_back(CRegion(object, m_classNames[objectClass], confidence));
        }
    }
}
