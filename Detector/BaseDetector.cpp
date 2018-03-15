#include "BaseDetector.h"
#include "MotionDetector.h"
#include "FaceDetector.h"
#include "PedestrianDetector.h"
#include "SSDMobileNetDetector.h"
#include "YoloDetector.h"

///
/// \brief CreateDetector
/// \param detectorType
/// \param collectPoints
/// \param gray
/// \return
///
BaseDetector* CreateDetector(
        tracking::Detectors detectorType,
        const BaseDetector::config_t& config,
        bool collectPoints,
        cv::UMat& gray
        )
{
    BaseDetector* detector = nullptr;

    switch (detectorType)
    {
    case tracking::Motion_VIBE:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_VIBE, collectPoints, gray);
        break;

    case tracking::Motion_MOG:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_MOG, collectPoints, gray);
        break;

    case tracking::Motion_GMG:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_GMG, collectPoints, gray);
        break;

    case tracking::Motion_CNT:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_CNT, collectPoints, gray);
        break;

    case tracking::Motion_SuBSENSE:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_SuBSENSE, collectPoints, gray);
        break;

    case tracking::Motion_LOBSTER:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_LOBSTER, collectPoints, gray);
        break;

    case tracking::Motion_MOG2:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_MOG2, collectPoints, gray);
        break;

    case tracking::Face_HAAR:
        detector = new FaceDetector(collectPoints, gray);
        break;

    case tracking::Pedestrian_HOG:
    case tracking::Pedestrian_C4:
        detector = new PedestrianDetector(collectPoints, gray);
        break;

    case tracking::SSD_MobileNet:
        detector = new SSDMobileNetDetector(collectPoints, gray);
        break;

    case tracking::Yolo:
        detector = new YoloDetector(collectPoints, gray);
        break;

    default:
        break;
    }

    if (!detector->Init(config))
    {
        delete detector;
        detector = nullptr;
    }
    return detector;
}
