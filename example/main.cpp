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
    "{ e  example     |1                   | number of example 0 - MouseTracking, 1 - MotionDetector, 2 - FaceDetector, 3 - PedestrianDetector, 4 - MobileNet SSD detector, 5 - Yolo OpenCV detector, 6 - Yolo Darknet detector | }"
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
#if 0
    TKalmanFilter kf(tracking::KalmanLinear, 0.2f, 0.5f);
    cv::Mat img(1080, 1920, CV_8UC3, cv::Scalar(255, 255, 255));
    Point_t lastPt(10, img.rows / 2);
    kf.Update(lastPt, true);
	int xStep = 5;
	srand(12345);

	cv::circle(img, lastPt, 2, cv::Scalar(0, 255, 0), 1);
    cv::namedWindow("img", cv::WINDOW_AUTOSIZE);
    track_t distR = 0;
    track_t distF = 0;
    std::vector<cv::Vec<track_t, 2>> dists;
    int stop_i = 200;
    float step = CV_2PI / stop_i;
    for (int i = 0; i < 2 * stop_i; ++i)
	{
        int rx = rand() % 20;
        int ry = rand() % 30;
        lastPt.x += xStep;
        lastPt.y = img.rows / 2 + sin(i * step) * img.rows / 4;
        Point_t pt(lastPt.x + ((rx % 2) ? -1 : 1) * rx,
                   lastPt.y + ((ry % 2) ? 1 : -1) * ry);

		Point_t upd = kf.Update(pt, true);
		Point_t pred = kf.GetPointPrediction();

		distR += cv::norm(cv::Vec2f(lastPt.x, lastPt.y), cv::Vec2f(pt.x, pt.y), cv::NORM_L2);
		distF += cv::norm(cv::Vec2f(lastPt.x, lastPt.y), cv::Vec2f(upd.x, upd.y), cv::NORM_L2);

        dists.push_back(cv::Vec<track_t, 2>(std::abs<track_t>(pt.x - upd.x), std::abs<track_t>(pt.y - upd.y)));
        if (dists.size() > 50)
		{
            dists.erase(dists.begin());
		}

        cv::Scalar mean;
        cv::Scalar var;
        cv::meanStdDev(dists, mean, var);
        std::cout << "mean = " << mean << ", var = " << var << std::endl;

#if 0
        float angle = atan2(var[0], var[1]);
        cv::RotatedRect rr(pred,
                           cv::Size2f(std::max(10.f, static_cast<track_t>(20.f * var[0])),
                           std::max(10.f, static_cast<track_t>(20.f * var[1]))),
                180.f * angle / CV_PI);
#else
        auto vel = kf.GetVelocity();
        std::cout << "velocity = " << vel << std::endl;
        float angle = atan2(vel[0], vel[1]);
        cv::RotatedRect rr(pred,
                           cv::Size2f(std::max(img.rows / 20.f, static_cast<track_t>(3.f * fabs(vel[0]))),
                           std::max(img.cols / 20.f, static_cast<track_t>(3.f * fabs(vel[1])))),
                180.f * angle / CV_PI);
#endif
        cv::ellipse(img, rr, cv::Scalar(100, 100, 100), 1);

		std::cout << "orig = " << lastPt << ", noise = " << pt << ", filtered = " << upd << std::endl;

		cv::circle(img, lastPt, 2, cv::Scalar(0, 255, 0), 1);
		cv::circle(img, pt, 3, cv::Scalar(0, 0, 255), -1);
		cv::circle(img, upd, 4, cv::Scalar(255, 0, 0), 1);
        cv::imshow("img", img);
        cv::waitKey(0);
	}
	std::cout << "err rand = " << distR << ", err fiiltered = " << distF << std::endl;
    cv::waitKey(0);
	return 0;
#endif
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

    default:
        std::cerr << "Wrong example number: " << exampleNum << std::endl;
        break;
    }

#ifndef SILENT_WORK
    cv::destroyAllWindows();
#endif
    return 0;
}
