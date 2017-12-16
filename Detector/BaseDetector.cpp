#include "BaseDetector.h"
#include "MotionDetector.h"

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
        return new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_VIBE, collectPoints, gray);

    case tracking::Pedestrian_HOG:
        return new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_VIBE, collectPoints, gray);

    case tracking::Pedestrian_C4:
        return new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_VIBE, collectPoints, gray);

    default:
        return nullptr;
    }
    return nullptr;
}
