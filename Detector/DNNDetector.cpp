#include "DNNDetector.h"

///
/// \brief DNNDetector::DNNDetector
/// \param collectPoints
/// \param gray
///
DNNDetector::DNNDetector(
	bool collectPoints,
    cv::UMat& colorFrame
	)
    :
      BaseDetector(collectPoints, colorFrame)
{
}

///
/// \brief DNNDetector::~DNNDetector
///
DNNDetector::~DNNDetector(void)
{
}

///
/// \brief DNNDetector::Init
/// \return
///
bool DNNDetector::Init(std::string modelConfiguration, std::string modelBinary)
{
    m_net = cv::dnn::readNetFromCaffe(modelConfiguration, modelBinary);

    return !m_net.empty();
}

///
/// \brief DNNDetector::Detect
/// \param gray
///
void DNNDetector::Detect(cv::UMat& colorFrame)
{
    const int inWidth = 300;
    const int inHeight = 300;
    const float WHRatio = inWidth / (float)inHeight;
    const float inScaleFactor = 0.007843f;
    const float meanVal = 127.5;
    const float confidenceThreshold = 0.2f;
    std::string classNames[] = {"background",
                                "aeroplane", "bicycle", "bird", "boat",
                                "bottle", "bus", "car", "cat", "chair",
                                "cow", "diningtable", "dog", "horse",
                                "motorbike", "person", "pottedplant",
                                "sheep", "sofa", "train", "tvmonitor"};

    cv::Size cropSize;
    if (colorFrame.cols / (float)colorFrame.rows > WHRatio)
    {
        cropSize = cv::Size(cvRound(colorFrame.rows * WHRatio), colorFrame.rows);
    }
    else
    {
        cropSize = cv::Size(colorFrame.cols, cvRound(colorFrame.cols / WHRatio));
    }

    cv::Rect crop(cv::Point((colorFrame.cols - cropSize.width) / 2, (colorFrame.rows - cropSize.height) / 2), cropSize);

    cv::Mat inputBlob = cv::dnn::blobFromImage(colorFrame.getMat(cv::ACCESS_READ), inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false); //Convert Mat to batch of images

    m_net.setInput(inputBlob, "data"); //set the network input

    cv::Mat detection = m_net.forward("detection_out"); //compute output

    std::vector<double> layersTimings;
    //double freq = cv::getTickFrequency() / 1000;
    //double time = m_net.getPerfProfile(layersTimings) / freq;

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    //cv::Mat frame = colorFrame(crop);

    //ss << "FPS: " << 1000/time << " ; time: " << time << " ms";
    //putText(frame, ss.str(), Point(20,20), 0, 0.5, Scalar(0,0,255));
    //std::cout << "Inference time, ms: " << time << endl;

    m_regions.clear();

    for (int i = 0; i < detectionMat.rows; ++i)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > confidenceThreshold)
        {
            size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

            int xLeftBottom = cvRound(detectionMat.at<float>(i, 3) * crop.width) + crop.x;
            int yLeftBottom = cvRound(detectionMat.at<float>(i, 4) * crop.height) + crop.y;
            int xRightTop = cvRound(detectionMat.at<float>(i, 5) * crop.width) + crop.x;
            int yRightTop = cvRound(detectionMat.at<float>(i, 6) * crop.height) + crop.y;

            cv::Rect object(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom);

            m_regions.push_back(CRegion(object, classNames[objectClass], confidence));

            //cv::rectangle(frame, object, Scalar(0, 255, 0));
            //std::string label = classNames[objectClass] + ": " + std::to_string(confidence);
            //int baseLine = 0;
            //cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            //cv::rectangle(frame, cv::Rect(cv::Point(xLeftBottom, yLeftBottom - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 255, 255), CV_FILLED);
            //cv::putText(frame, label, Point(xLeftBottom, yLeftBottom), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0));
        }
    }
}
