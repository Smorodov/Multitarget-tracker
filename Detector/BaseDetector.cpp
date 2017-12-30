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
    {
        MotionDetector* detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_VIBE, collectPoints, gray);
        BaseDetector::config_t config;
        if (!detector->Init(config))
        {
            delete detector;
            detector = nullptr;
        }
        return detector;
    }

    case tracking::Motion_MOG:
    {
        MotionDetector* detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_MOG, collectPoints, gray);
        BaseDetector::config_t config;
        if (!detector->Init(config))
        {
            delete detector;
            detector = nullptr;
        }
        return detector;
    }

    case tracking::Motion_GMG:
    {
        MotionDetector* detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_GMG, collectPoints, gray);
        BaseDetector::config_t config;
        if (!detector->Init(config))
        {
            delete detector;
            detector = nullptr;
        }
        return detector;
    }

    case tracking::Motion_CNT:
    {
        MotionDetector* detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_CNT, collectPoints, gray);
        BaseDetector::config_t config;
        if (!detector->Init(config))
        {
            delete detector;
            detector = nullptr;
        }
        return detector;
    }

    case tracking::Motion_SuBSENSE:
    {
        MotionDetector* detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_SuBSENSE, collectPoints, gray);
        BaseDetector::config_t config;
        if (!detector->Init(config))
        {
            delete detector;
            detector = nullptr;
        }
        return detector;
    }

    case tracking::Motion_LOBSTER:
    {
        MotionDetector* detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_LOBSTER, collectPoints, gray);
        BaseDetector::config_t config;
        if (!detector->Init(config))
        {
            delete detector;
            detector = nullptr;
        }
        return detector;
    }

    case tracking::Motion_MOG2:
    {
        MotionDetector* detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_MOG2, collectPoints, gray);
        BaseDetector::config_t config;
        if (!detector->Init(config))
        {
            delete detector;
            detector = nullptr;
        }
        return detector;
    }

    case tracking::Face_HAAR:
    {
        FaceDetector* detector = new FaceDetector(collectPoints, gray);
        BaseDetector::config_t config;
        config["cascadeFileName"] = "../data/haarcascade_frontalface_alt2.xml";
        if (!detector->Init(config))
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
        BaseDetector::config_t config;
        config["detectorType"] = (detectorType == tracking::Pedestrian_HOG) ?
                    std::to_string(PedestrianDetector::HOG) : std::to_string(PedestrianDetector::C4);
        config["cascadeFileName1"] = "../data/combined.txt.model";
        config["cascadeFileName2"] = "../data/combined.txt.model_";
        if (!detector->Init(config))
        {
            delete detector;
            detector = nullptr;
        }
        return detector;
    }

    case tracking::DNN:
    {
        DNNDetector* detector = new DNNDetector(collectPoints, gray);
        BaseDetector::config_t config;
        config["modelConfiguration"] = "../data/MobileNetSSD_deploy.prototxt";
        config["modelBinary"] = "../data/MobileNetSSD_deploy.caffemodel";
        config["confidenceThreshold"] = "0.2";
        if (!detector->Init(config))
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
