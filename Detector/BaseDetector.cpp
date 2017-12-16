#include "BaseDetector.h"
#include "MotionDetector.h"
#include "FaceDetector.h"
#include "PedestrianDetector.h"

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
        FaceDetector* faceDetector = new FaceDetector(collectPoints, gray);
        if (!faceDetector->Init("../data/haarcascade_frontalface_alt2.xml"))
        {
            delete faceDetector;
            faceDetector = nullptr;
        }
        return faceDetector;
    }

    case tracking::Pedestrian_HOG:
    case tracking::Pedestrian_C4:
    {
        PedestrianDetector* pedestrianDetector = new PedestrianDetector(collectPoints, gray);
        if (!pedestrianDetector->Init((detectorType == tracking::Pedestrian_HOG) ? PedestrianDetector::HOG : PedestrianDetector::C4,
                                      "../data/combined.txt.model", "../data/combined.txt.model_"))
        {
            delete pedestrianDetector;
            pedestrianDetector = nullptr;
        }
        return pedestrianDetector;
    }

    default:
        return nullptr;
    }
    return nullptr;
}
