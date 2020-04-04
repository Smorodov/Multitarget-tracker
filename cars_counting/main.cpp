#include "CarsCounting.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

// ----------------------------------------------------------------------

static void Help()
{
    printf("\nExamples of the CarsCounting\n"
           "Usage: \n"
           "          ./CarsCounting <path to movie file> [--start_frame]=<start a video from this position> [--end_frame]=<play a video to this position> [--end_delay]=<delay in milliseconds after video ending> [--out]=<name of result video file> [--show_logs]=<show logs> \n\n"
           "Press:\n"
           "\'m\' key for change mode: play|pause. When video is paused you can press any key for get next frame. \n\n"
           "Press Esc to exit from video \n\n"
           );
}

const char* keys =
{
    "{ @1             |../data/atrium.avi  | movie file | }"
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

    CarsCounting cars_counting(parser);

	cars_counting.AddLine(RoadLine(cv::Point2f(0.02f, 0.45f), cv::Point2f(0.28f, 0.55f), 0));
    cars_counting.AddLine(RoadLine(cv::Point2f(0.3f, 0.6f), cv::Point2f(0.8f, 0.6f), 1));

    cars_counting.Process();

#ifndef SILENT_WORK
    cv::destroyAllWindows();
#endif

    return 0;
}
