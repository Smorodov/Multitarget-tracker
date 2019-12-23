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
#if 1
	Point_t lastPt(10, 10);
	TKalmanFilter kf(tracking::KalmanLinear, lastPt, 0.2f, 0.5f);

	int xStep = 5;
	int yStep = 4;
	srand(12345);
	cv::Mat img(1080, 1920, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::circle(img, lastPt, 2, cv::Scalar(0, 255, 0), 1);
	cv::namedWindow("img", cv::WINDOW_AUTOSIZE);
	double distR = 0;
	double distF = 0;
	std::vector<double> dists;
	std::vector<double> distsX;
	std::vector<double> distsY;
	for (int i = 0; i < 100; ++i)
	{
		int r = rand() % 6;
		lastPt.x += xStep;
		lastPt.y += yStep;
		Point_t pt(lastPt.x + ((r % 2) ? -1 : 1) * r, lastPt.y + ((r % 2) ? 1 : -1) * r);

		Point_t upd = kf.Update(pt, true);
		Point_t pred = kf.GetPointPrediction();

		distR += cv::norm(cv::Vec2f(lastPt.x, lastPt.y), cv::Vec2f(pt.x, pt.y), cv::NORM_L2);
		distF += cv::norm(cv::Vec2f(lastPt.x, lastPt.y), cv::Vec2f(upd.x, upd.y), cv::NORM_L2);

		dists.push_back(cv::norm(cv::Vec2f(pt.x, pt.y), cv::Vec2f(upd.x, upd.y), cv::NORM_L2));
		distsX.push_back(std::abs<double>(pt.x - upd.x));
		distsY.push_back(std::abs<double>(pt.y - upd.y));
		if (dists.size() > 50)
		{
			dists.erase(dists.begin());
			distsX.erase(distsX.begin());
			distsY.erase(distsY.begin());
		}
		cv::Scalar mean;
		cv::Scalar var;
		cv::meanStdDev(dists, mean, var);
		std::cout << "mean = " << mean << ", var = " << var << std::endl;
		cv::circle(img, upd, std::max(10, cvRound(3 * var[0])), cv::Scalar(255, 0, 255), 1);

		cv::Scalar meanX;
		cv::Scalar varX;
		cv::meanStdDev(distsX, meanX, varX);
		std::cout << "meanX = " << meanX << ", varX = " << varX << std::endl;
		cv::Scalar meanY;
		cv::Scalar varY;
		cv::meanStdDev(distsY, meanY, varY);
		std::cout << "meanY = " << meanY << ", varY = " << varY << std::endl;
		//cv::circle(img, upd, std::max(10, cvRound(3 * var[0])), cv::Scalar(255, 0, 255), 1);

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
