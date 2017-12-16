#include "MouseExample.h"
#include "VideoExample.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

// ----------------------------------------------------------------------

static void Help()
{
    printf("\nExamples of the Multitarget tracking algorithm\n"
           "Usage: \n"
           "          ./MultitargetTracker <path to movie file> [--example]=<number of example 0..3> [--start_frame]=<start a video from this position> [--end_frame]=<play a video to this position> [--end_delay]=<delay in milliseconds after video ending> [--out]=<name of result video file> [--show_logs]=<show logs> \n\n"
           "Press:\n"
           "\'m\' key for change mode: play|pause. When video is paused you can press any key for get next frame. \n\n"
           "Press Esc to exit from video \n\n"
           );
}

const char* keys =
{
    "{ @1             |../data/atrium.avi  | movie file | }"
    "{ e  example     |1                   | number of example 0 - MouseTracking, 1 - MotionDetector, 2 - FaceDetector, 3 - PedestrianDetector, 4 - Hybrid face and motion detectors, 5 - MobileNet SSD detector | }"
    "{ sf start_frame |0                   | Start a video from this position | }"
    "{ ef end_frame   |0                   | Play a video to this position (if 0 then played to the end of file) | }"
    "{ ed end_delay   |0                   | Delay in milliseconds after video ending | }"
    "{ o  out         |                    | Name of result video file | }"
    "{ sl show_logs   |1                   | Show Trackers logs | }"
    "{ g gpu          |0                   | Use OpenCL acceleration | }"
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

    switch (exampleNum)
    {
    case 0:
        MouseTracking(parser);
        break;

    case 1:
    {
        MotionDetectorExample mdetector(parser);
        mdetector.Process();
        break;
    }

    case 2:
    {
        FaceDetectorExample face_detector(parser);
        face_detector.Process();
        break;
    }

    case 3:
    {
        PedestrianDetectorExample ped_detector(parser);
        ped_detector.Process();
        break;
    }

    case 4:
    {
        HybridFaceDetectorExample face_detector(parser);
        face_detector.Process();
        break;
    }

    case 5:
    {
        DNNDetectorExample dnn_detector(parser);
        dnn_detector.Process();
        break;
    }
    }


    cv::destroyAllWindows();
    return 0;
}
