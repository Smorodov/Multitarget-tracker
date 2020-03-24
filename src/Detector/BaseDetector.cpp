#include "BaseDetector.h"
#include "MotionDetector.h"
#include "FaceDetector.h"
#include "PedestrianDetector.h"
#include "SSDMobileNetDetector.h"
#include "YoloDetector.h"

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
BaseDetector* CreateDetector(
        tracking::Detectors detectorType,
        const config_t& config,
        cv::UMat& frame
        )
{
    BaseDetector* detector = nullptr;

    switch (detectorType)
    {
    case tracking::Motion_VIBE:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_VIBE, frame);
        break;

    case tracking::Motion_MOG:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_MOG, frame);
        break;

    case tracking::Motion_GMG:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_GMG, frame);
        break;

    case tracking::Motion_CNT:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_CNT, frame);
        break;

    case tracking::Motion_SuBSENSE:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_SuBSENSE, frame);
        break;

    case tracking::Motion_LOBSTER:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_LOBSTER, frame);
        break;

    case tracking::Motion_MOG2:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_MOG2, frame);
        break;

    case tracking::Face_HAAR:
        detector = new FaceDetector(frame);
        break;

    case tracking::Pedestrian_HOG:
    case tracking::Pedestrian_C4:
        detector = new PedestrianDetector(frame);
        break;

    case tracking::SSD_MobileNet:
        detector = new SSDMobileNetDetector(frame);
        break;

    case tracking::Yolo_OCV:
        detector = new YoloOCVDetector(frame);
        break;

	case tracking::Yolo_Darknet:
#ifdef BUILD_YOLO_LIB
        detector = new YoloDarknetDetector(frame);
#else
		std::cerr << "Darknet inference engine was not configured in CMake" << std::endl;
#endif
		break;

	case tracking::Yolo_TensorRT:
#ifdef BUILD_YOLO_TENSORRT
		detector = new YoloTensorRTDetector(frame);
#else
		std::cerr << "TensorRT inference engine was not configured in CMake" << std::endl;
#endif
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
