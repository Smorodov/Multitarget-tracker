#include "MouseExample.h"
#include "examples.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

// ----------------------------------------------------------------------

static void Help()
{
    printf("\nExamples of the Multitarget tracking algorithm\n"
           "Usage: \n"
           "          ./MultitargetTracker <path to movie file> [--example]=<number of example 0..6> [--start_frame]=<start a video from this position> [--end_frame]=<play a video to this position> [--end_delay]=<delay in milliseconds after video ending> [--out]=<name of result video file> [--show_logs]=<show logs> [--async]=<async pipeline> \n\n"
           "Press:\n"
           "\'m\' key for change mode: play|pause. When video is paused you can press any key for get next frame. \n\n"
           "Press Esc to exit from video \n\n"
           );
}

const char* keys =
{
    "{ @1             |../data/atrium.avi  | movie file | }"
    "{ e  example     |1                   | number of example 0 - MouseTracking, 1 - MotionDetector, 2 - FaceDetector, 3 - PedestrianDetector, 4 - MobileNet SSD detector, 5 - YOLO OpenCV detector, 6 - YOLO Darknet detector, 7 - YOLO TensorRT Detector | }"
    "{ sf start_frame |0                   | Start a video from this position | }"
    "{ ef end_frame   |0                   | Play a video to this position (if 0 then played to the end of file) | }"
    "{ ed end_delay   |0                   | Delay in milliseconds after video ending | }"
    "{ o  out         |                    | Name of result video file | }"
    "{ sl show_logs   |1                   | Show Trackers logs | }"
    "{ g gpu          |0                   | Use OpenCL acceleration | }"
    "{ a async        |1                   | Use 2 theads for processing pipeline | }"
};

// ----------------------------------------------------------------------

int main(int argc, char** argv)
{
    Help();

    cv::CommandLineParser parser(argc, argv, keys);

    bool useOCL = parser.get<int>("gpu") ? 1 : 0;
    cv::ocl::setUseOpenCL(useOCL);
    std::cout << (cv::ocl::useOpenCL() ? "OpenCL is enabled" : "OpenCL not used") << std::endl;

    int exampleNum = parser.get<int>("example");
    int asyncPipeline = parser.get<int>("async");

    switch (exampleNum)
    {
    case 0:
        MouseTracking(parser);
        break;

    case 1:
    {
        MotionDetectorExample mdetector(parser);
        asyncPipeline ? mdetector.AsyncProcess() : mdetector.SyncProcess();
        break;
    }

    case 2:
    {
        FaceDetectorExample face_detector(parser);
        asyncPipeline ? face_detector.AsyncProcess() : face_detector.SyncProcess();
        break;
    }

    case 3:
    {
        PedestrianDetectorExample ped_detector(parser);
        asyncPipeline ? ped_detector.AsyncProcess() : ped_detector.SyncProcess();
        break;
    }

    case 4:
    {
        SSDMobileNetExample dnn_detector(parser);
        asyncPipeline ? dnn_detector.AsyncProcess() : dnn_detector.SyncProcess();
        break;
    }

    case 5:
    {
        YoloExample yolo_detector(parser);
        asyncPipeline ? yolo_detector.AsyncProcess() : yolo_detector.SyncProcess();
        break;
    }

#ifdef BUILD_YOLO_LIB
	case 6:
	{
		YoloDarknetExample yolo_detector(parser);
        asyncPipeline ? yolo_detector.AsyncProcess() : yolo_detector.SyncProcess();
		break;
	}
#endif

#ifdef BUILD_YOLO_TENSORRT
	case 7:
	{
		YoloTensorRTExample yolo_detector(parser);
		asyncPipeline ? yolo_detector.AsyncProcess() : yolo_detector.SyncProcess();
		break;
	}
#endif

    default:
        std::cerr << "Wrong example number: " << exampleNum << std::endl;
        break;
    }

#ifndef SILENT_WORK
    cv::destroyAllWindows();
#endif
    return 0;
}
