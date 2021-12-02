#include "BaseDetector.h"
#include "MotionDetector.h"
#include "FaceDetector.h"
#include "PedestrianDetector.h"
#include "OCVDNNDetector.h"

#ifdef BUILD_YOLO_LIB
#include "YoloDarknetDetector.h"
#endif
#ifdef BUILD_YOLO_TENSORRT
#include "YoloTensorRTDetector.h"
#endif

///
/// \brief CreateDetector
/// \param detectorType
/// \param gray
/// \return
///
std::unique_ptr<BaseDetector> BaseDetector::CreateDetector(tracking::Detectors detectorType,
                                                           const config_t& config,
                                                           const cv::UMat& frame)
{
    std::unique_ptr<BaseDetector> detector;

    switch (detectorType)
    {
    case tracking::Motion_VIBE:
        detector = std::make_unique<MotionDetector>(BackgroundSubtract::BGFG_ALGS::ALG_VIBE, frame);
        break;

    case tracking::Motion_MOG:
        detector = std::make_unique<MotionDetector>(BackgroundSubtract::BGFG_ALGS::ALG_MOG, frame);
        break;

    case tracking::Motion_GMG:
        detector = std::make_unique<MotionDetector>(BackgroundSubtract::BGFG_ALGS::ALG_GMG, frame);
        break;

    case tracking::Motion_CNT:
        detector = std::make_unique<MotionDetector>(BackgroundSubtract::BGFG_ALGS::ALG_CNT, frame);
        break;

    case tracking::Motion_SuBSENSE:
        detector = std::make_unique<MotionDetector>(BackgroundSubtract::BGFG_ALGS::ALG_SuBSENSE, frame);
        break;

    case tracking::Motion_LOBSTER:
        detector = std::make_unique<MotionDetector>(BackgroundSubtract::BGFG_ALGS::ALG_LOBSTER, frame);
        break;

    case tracking::Motion_MOG2:
        detector = std::make_unique<MotionDetector>(BackgroundSubtract::BGFG_ALGS::ALG_MOG2, frame);
        break;

    case tracking::Face_HAAR:
        detector = std::make_unique<FaceDetector>(frame);
        break;

    case tracking::Pedestrian_HOG:
    case tracking::Pedestrian_C4:
        detector = std::make_unique<PedestrianDetector>(frame);
        break;

#ifdef USE_OCV_DNN
    case tracking::DNN_OCV:
        detector = std::make_unique<OCVDNNDetector>(frame);
        break;
#endif

	case tracking::Yolo_Darknet:
#ifdef BUILD_YOLO_LIB
        detector = std::make_unique<YoloDarknetDetector>(frame);
#else
		std::cerr << "Darknet inference engine was not configured in CMake" << std::endl;
#endif
		break;

	case tracking::Yolo_TensorRT:
#ifdef BUILD_YOLO_TENSORRT
		detector = std::make_unique<YoloTensorRTDetector>(frame);
#else
		std::cerr << "TensorRT inference engine was not configured in CMake" << std::endl;
#endif
		break;

    default:
        break;
    }

    if (!detector->Init(config))
        detector.reset();
    return detector;
}

///
std::unique_ptr<BaseDetector> BaseDetector::CreateDetectorKV(tracking::Detectors detectorType, const KeyVal& config, const cv::Mat& gray)
{
    config_t mconfig;
    for (auto kv : config.m_config)
    {
        mconfig.emplace(kv.first, kv.second);
    }
    cv::UMat uframe = gray.getUMat(cv::ACCESS_READ);
    return CreateDetector(detectorType, mconfig, uframe);
}
