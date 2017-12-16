#include "BaseDetector.h"
#include "MotionDetector.h"
#include "FaceDetector.h"
#include "PedestrianDetector.h"
#include "DNNDetector.h"

///
/// \brief CreateDetector
/// \param detectorType
/// \param collectPoints
/// \param gray
/// \return
///
BaseDetector* CreateDetector(
        tracking::Detectors detectorType,
        bool collectPoints,
        cv::UMat& gray
        )
{
    switch (detectorType)
    {
    case tracking::Motion_VIBE:
        return new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_VIBE, collectPoints, gray);

    case tracking::Motion_MOG:
        return new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_MOG, collectPoints, gray);

    case tracking::Motion_GMG:
        return new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_GMG, collectPoints, gray);

    case tracking::Motion_CNT:
        return new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_CNT, collectPoints, gray);

    case tracking::Motion_SuBSENSE:
        return new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_SuBSENSE, collectPoints, gray);

    case tracking::Motion_LOBSTER:
        return new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_LOBSTER, collectPoints, gray);

    case tracking::Motion_MOG2:
        return new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_MOG2, collectPoints, gray);

    case tracking::Face_HAAR:
    {
        FaceDetector* detector = new FaceDetector(collectPoints, gray);
        if (!detector->Init("../data/haarcascade_frontalface_alt2.xml"))
        {
            delete detector;
            detector = nullptr;
        }
        return detector;
    }

    case tracking::Pedestrian_HOG:
    case tracking::Pedestrian_C4:
    {
        PedestrianDetector* detector = new PedestrianDetector(collectPoints, gray);
        if (!detector->Init((detectorType == tracking::Pedestrian_HOG) ? PedestrianDetector::HOG : PedestrianDetector::C4,
                            "../data/combined.txt.model", "../data/combined.txt.model_"))
        {
            delete detector;
            detector = nullptr;
        }
        return detector;
    }

    case tracking::DNN:
    {
        DNNDetector* detector = new DNNDetector(collectPoints, gray);
        if (!detector->Init("../data/MobileNetSSD_deploy.prototxt", "../data/MobileNetSSD_deploy.caffemodel"))
        {
            delete detector;
            detector = nullptr;
        }
        return detector;
    }


    default:
        return nullptr;
    }
    return nullptr;
}
