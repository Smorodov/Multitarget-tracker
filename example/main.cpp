#include "MouseExample.h"
#include "examples.h"

#ifdef BUILD_CARS_COUNTING
#include "CarsCounting.h"
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

// ----------------------------------------------------------------------

static void Help()
{
    printf("\nExamples of the Multitarget tracking algorithm\n"
           "Usage: \n"
           "          ./MultitargetTracker <path to movie file> [--example]=<number of example 0..7> [--start_frame]=<start a video from this position> [--end_frame]=<play a video to this position> [--end_delay]=<delay in milliseconds after video ending> [--out]=<name of result video file> [--show_logs]=<show logs> [--async]=<async pipeline> [--res]=<csv log file> [--settings]=<ini file> [--batch_size=<number of frames>] \n\n"
           "Press:\n"
           "\'m\' key for change mode: play|pause. When video is paused you can press any key for get next frame. \n\n"
           "Press Esc to exit from video \n\n"
           );
}

const char* keys =
{
    "{ @1               |../data/atrium.avi  | movie file | }"
    "{ e  example       |1                   | number of example 0 - MouseTracking, 1 - MotionDetector, 2 - FaceDetector, 3 - PedestrianDetector, 4 - OpenCV dnn objects detector, 5 - YOLO Darknet detector, 6 - YOLO TensorRT Detector, 7 - Cars counting | }"
    "{ sf start_frame   |0                   | Start a video from this position | }"
    "{ ef end_frame     |0                   | Play a video to this position (if 0 then played to the end of file) | }"
    "{ ed end_delay     |0                   | Delay in milliseconds after video ending | }"
    "{ o  out           |                    | Name of result video file | }"
    "{ sl show_logs     |1                   | Show Trackers logs | }"
    "{ g gpu            |0                   | Use OpenCL acceleration | }"
    "{ a async          |1                   | Use 2 theads for processing pipeline | }"
    "{ r log_res        |                    | Path to the csv file with tracking result | }"
    "{ cvat_res         |                    | Path to the xml file in cvat format with tracking result | }"
    "{ s settings       |                    | Path to the ini file with tracking settings | }"
    "{ bs batch_size    |1                   | Batch size - frames count for processing | }"
    "{ wf write_n_frame |1                   | Write logs on each N frame: 1 for writing each frame | }"
    "{ hm heat_map      |0                   | For CarsCounting: Draw heat map | }"
    "{ geo_bind         |geo_bind.ini        | For CarsCounting: ini file with geographical binding | }"
};

// ----------------------------------------------------------------------

#pragma once

///
/// \brief The EmbeddingsCalculator class
///
class EmbeddingsCalculatorSimple
{
public:
	EmbeddingsCalculatorSimple() = default;
	virtual ~EmbeddingsCalculatorSimple() = default;

	///
	bool Initialize(const std::string& cfgName, const std::string& weightsName, const cv::Size& inputLayer)
	{
		m_inputLayer = inputLayer;

#if 1
		m_net = cv::dnn::readNet(weightsName);
#else
		m_net = cv::dnn::readNetFromTorch(weightsName);
#endif
		if (!m_net.empty())
		{
			//m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
			//m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

			auto outNames = m_net.getUnconnectedOutLayersNames();
			auto outLayers = m_net.getUnconnectedOutLayers();
			auto outLayerType = m_net.getLayer(outLayers[0])->type;

			std::vector<cv::dnn::MatShape> outputs;
			std::vector<cv::dnn::MatShape> internals;
			m_net.getLayerShapes(cv::dnn::MatShape(), 0, outputs, internals);
			std::cout << "REID: getLayerShapes: outputs (" << outputs.size() << ") = " << (outputs.size() > 0 ? outputs[0].size() : 0) << ", internals (" << internals.size() << ") = " << (internals.size() > 0 ? internals[0].size() : 0) << std::endl;
			if (outputs.size() && outputs[0].size() > 3)
				std::cout << "outputs = [" << outputs[0][0] << ", " << outputs[0][1] << ", " << outputs[0][2] << ", " << outputs[0][3] << "], internals = [" << internals[0][0] << ", " << internals[0][1] << ", " << internals[0][2] << ", " << internals[0][3] << "]" << std::endl;
		}
		return !m_net.empty();
	}

	///
	bool IsInitialized() const
	{
		return !m_net.empty();
	}

	///
	cv::Mat Calc(const cv::Mat& img, cv::Rect rect)
	{
		auto Clamp = [](int& v, int& size, int hi) -> int
		{
			int res = 0;
			if (v < 0)
			{
				res = v;
				v = 0;
				return res;
			}
			else if (v + size > hi - 1)
			{
				res = v;
				v = hi - 1 - size;
				if (v < 0)
				{
					size += v;
					v = 0;
				}
				res -= v;
				return res;
			}
			return res;
		};
		Clamp(rect.x, rect.width, img.cols);
		Clamp(rect.y, rect.height, img.rows);

		cv::Mat obj;
		cv::resize(img(rect), obj, m_inputLayer, 0., 0., cv::INTER_CUBIC);
		cv::Mat blob = cv::dnn::blobFromImage(obj, 1.0 / 255.0, cv::Size(), cv::Scalar(), false, false, CV_32F);

		m_net.setInput(blob);
		cv::Mat embedding;
		std::cout << "embedding: " << embedding.size() << ", chans = " << embedding.channels() << std::endl;
		//std::cout << "orig: " << embedding << std::endl;
		cv::normalize(m_net.forward(), embedding);
		//std::cout << "normalized: " << embedding << std::endl;
		return embedding;
	}

private:
	cv::dnn::Net m_net;
	cv::Size m_inputLayer{ 128, 256 };
};


int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);

    Help();
    parser.printMessage();

    bool useOCL = parser.get<int>("gpu") != 0;
    cv::ocl::setUseOpenCL(useOCL);
    std::cout << (cv::ocl::useOpenCL() ? "OpenCL is enabled" : "OpenCL not used") << std::endl;

#if 0
	EmbeddingsCalculatorSimple ec;
	ec.Initialize("C:/work/home/mtracker/tmp/reid/models/osnet_x0_25_msmt17.onnx",
		"C:/work/home/mtracker/tmp/reid/models/osnet_x0_25_msmt17.onnx",
		cv::Size(128, 256));
	std::cout << "ec.IsInitialized(): " << ec.IsInitialized() << std::endl;

	cv::Mat img = cv::imread("C:/work/home/mtracker/Multitarget-tracker/build/Release/vlcsnap-2023-10-06-17h31m54s413.png");
	cv::Rect r1(564, 526, 124, 260);
	//cv::Rect r2(860, 180, 48, 160);
	cv::Rect r2(560, 522, 132, 264);

	cv::Mat e1 = ec.Calc(img, r1);
	cv::Mat e2 = ec.Calc(img, r2);

	//cv::Mat mul = e1 * e2.t();
	std::cout << "e1: " << e1 << std::endl;
	std::cout << "e2: " << e2 << std::endl;
	cv::Mat diff;
	cv::absdiff(e1, e2, diff);
	cv::Scalar ss = cv::sum(diff);
	cv::Mat mul = e1 * e2.t();
	float res = static_cast<float>(1.f - mul.at<float>(0, 0));
	std::cout << "mul = " << mul << ", sum = " << ss << ", res = " << res << std::endl;

	cv::rectangle(img, r1, cv::Scalar(255, 0, 255));
	cv::rectangle(img, r2, cv::Scalar(255, 0, 0));
	cv::imshow("img", img);
	cv::waitKey(0);

	return 0;
#endif

    int exampleNum = parser.get<int>("example");
    int asyncPipeline = parser.get<int>("async");

	std::unique_ptr<VideoExample> detector;

    switch (exampleNum)
    {
    case 0:
        MouseTracking(parser);
        break;

    case 1:
        detector = std::make_unique<MotionDetectorExample>(parser);
        break;

    case 2:
		detector = std::make_unique<FaceDetectorExample>(parser);
        break;

    case 3:
		detector = std::make_unique<PedestrianDetectorExample>(parser);
        break;

	case 4:
		detector = std::make_unique<OpenCVDNNExample>(parser);
		break;

#ifdef BUILD_YOLO_LIB
	case 5:
		detector = std::make_unique<YoloDarknetExample>(parser);
		break;
#endif

#ifdef BUILD_YOLO_TENSORRT
	case 6:
		detector = std::make_unique<YoloTensorRTExample>(parser);
		break;
#endif

#ifdef BUILD_CARS_COUNTING
    case 7:
    {
        auto carsCounting = new CarsCounting(parser);
        detector = std::unique_ptr<CarsCounting>(carsCounting);
        break;
    }
#endif

    default:
        std::cerr << "Wrong example number: " << exampleNum << std::endl;
        break;
    }

	if (detector.get())
		asyncPipeline ? detector->AsyncProcess() : detector->SyncProcess();

#ifndef SILENT_WORK
    cv::destroyAllWindows();
#endif
    return 0;
}
