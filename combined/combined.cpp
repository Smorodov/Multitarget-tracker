#include <iomanip>
#include <ctime>
#include <future>

#include "combined.h"

///
/// \brief DrawFilledRect
///
void DrawFilledRect(cv::Mat& frame, const cv::Rect& rect, cv::Scalar cl, int alpha)
{
	if (alpha)
	{
		const int alpha_1 = 255 - alpha;
		const int nchans = frame.channels();
		int color[3] = { cv::saturate_cast<int>(cl[0]), cv::saturate_cast<int>(cl[1]), cv::saturate_cast<int>(cl[2]) };
		for (int y = rect.y; y < rect.y + rect.height; ++y)
		{
			uchar* ptr = frame.ptr(y) + nchans * rect.x;
			for (int x = rect.x; x < rect.x + rect.width; ++x)
			{
				for (int i = 0; i < nchans; ++i)
				{
					ptr[i] = cv::saturate_cast<uchar>((alpha_1 * ptr[i] + alpha * color[i]) / 255);
				}
				ptr += nchans;
			}
		}
	}
	else
	{
		cv::rectangle(frame, rect, cl, cv::FILLED);
	}
}

///
/// \brief CombinedDetector::CombinedDetector
/// \param parser
///
CombinedDetector::CombinedDetector(const cv::CommandLineParser& parser)
{
    m_inFile = parser.get<std::string>(0);
    m_outFile = parser.get<std::string>("out");
    m_showLogs = parser.get<int>("show_logs") != 0;
    m_startFrame = parser.get<int>("start_frame");
    m_endFrame = parser.get<int>("end_frame");
    m_finishDelay = parser.get<int>("end_delay");
	m_flipV = parser.get<int>("flipv") != 0;

    m_colors.push_back(cv::Scalar(255, 0, 0));
    m_colors.push_back(cv::Scalar(0, 255, 0));
    m_colors.push_back(cv::Scalar(0, 0, 255));
    m_colors.push_back(cv::Scalar(255, 255, 0));
    m_colors.push_back(cv::Scalar(0, 255, 255));
    m_colors.push_back(cv::Scalar(255, 0, 255));
    m_colors.push_back(cv::Scalar(255, 127, 255));
    m_colors.push_back(cv::Scalar(127, 0, 255));
    m_colors.push_back(cv::Scalar(127, 0, 127));
}

///
/// \brief CombinedDetector::SyncProcess
///
void CombinedDetector::SyncProcess()
{
    cv::VideoWriter writer;

#ifndef SILENT_WORK
    cv::namedWindow("Video", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    bool manualMode = false;
#endif

    cv::Mat frame;

    double freq = cv::getTickFrequency();
    int64 allTime = 0;

    int framesCounter = m_startFrame + 1;

    cv::VideoCapture capture;
    if (!OpenCapture(capture))
    {
        std::cerr << "Can't open " << m_inFile << std::endl;
        return;
    }

    int64 startLoopTime = cv::getTickCount();

    for (;;)
    {
        capture >> frame;
		if (frame.empty())
			break;
		else if (m_flipV)
			cv::flip(frame, frame, 0);

		if (!m_isDetectorInitialized || !m_isTrackerInitialized)
		{
			cv::UMat ufirst = frame.getUMat(cv::ACCESS_READ);
			if (!m_isDetectorInitialized)
			{
				m_isDetectorInitialized = InitDetector(ufirst);
				if (!m_isDetectorInitialized)
				{
					std::cerr << "CaptureAndDetect: Detector initialize error!!!" << std::endl;
					break;
				}
			}
			if (!m_isTrackerInitialized)
			{
				m_isTrackerInitialized = InitTracker(ufirst);
				if (!m_isTrackerInitialized)
				{
					std::cerr << "CaptureAndDetect: Tracker initialize error!!!" << std::endl;
					break;
				}
			}
		}

        int64 t1 = cv::getTickCount();

        DetectAndTrack(frame);

        int64 t2 = cv::getTickCount();

        allTime += t2 - t1;
        int currTime = cvRound(1000 * (t2 - t1) / freq);

        DrawData(frame, framesCounter, currTime);

#ifndef SILENT_WORK
        cv::imshow("Video", frame);

        int waitTime = manualMode ? 0 : 1;// std::max<int>(1, cvRound(1000 / m_fps - currTime));
        int k = cv::waitKey(waitTime);
        if (k == 27)
            break;
        else if (k == 'm' || k == 'M')
            manualMode = !manualMode;
#else
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif

        WriteFrame(writer, frame);

        ++framesCounter;
        if (m_endFrame && framesCounter > m_endFrame)
        {
            std::cout << "Process: riched last " << m_endFrame << " frame" << std::endl;
            break;
        }
    }

    int64 stopLoopTime = cv::getTickCount();

    std::cout << "algorithms time = " << (allTime / freq) << ", work time = " << ((stopLoopTime - startLoopTime) / freq) << std::endl;
#ifndef SILENT_WORK
    cv::waitKey(m_finishDelay);
#endif
}

///
/// \brief CombinedDetector::DetectAndTrack
/// \param frame
///
void CombinedDetector::DetectAndTrack(cv::Mat frame)
{
    cv::UMat uframe = frame.getUMat(cv::ACCESS_READ);

	// YOLO detection
	m_detectorDNN->Detect(uframe);
	const regions_t& regsDNN = m_detectorDNN->GetDetects();
	m_trackerDNN->Update(regsDNN, uframe, m_fps);
	m_trackerDNN->GetTracks(m_tracksDNN);

    m_detectorBGFG->ResetIgnoreMask();
	for (const auto& track : m_tracksDNN)
	{
		if (track.IsRobust(2, 0.5f, cv::Size2f(0.1f, 8.0f)))
        {
			AddBbox(track.m_rrect.boundingRect());
            m_detectorBGFG->UpdateIgnoreMask(uframe, track.m_rrect.boundingRect());
        }
	}

	// BGFG detection
	cv::UMat uGray;
	cv::cvtColor(uframe, uGray, cv::COLOR_BGR2GRAY);
	for (const auto& track : m_tracksBGFG)
	{
		if (track.m_isStatic)
        {
			m_detectorBGFG->UpdateIgnoreMask(uGray, track.m_rrect.boundingRect());
        }
	}

    m_detectorBGFG->Detect(uGray);

    const regions_t& regsBGFG = m_detectorBGFG->GetDetects();

	//m_trackerBGFG->Update(regsBGFG, uGray, m_fps);
	m_trackerBGFG->Update(regions_t(), uGray, m_fps);

	m_trackerBGFG->GetTracks(m_tracksBGFG);
	for (const auto& bbox : m_oldBoxes)
	{
		//std::cout << "ResetModel: " << bbox.m_rect << ", " << bbox.m_lifeTime << std::endl;
		//m_detectorBGFG->UpdateIgnoreMask(uGray, bbox.m_rect);
	}
	CleanBboxes();
}

///
/// \brief CombinedDetector::AddBbox
/// \param rect
/// \return
///
bool CombinedDetector::AddBbox(const cv::Rect& rect)
{
	//std::cout << "AddBbox: " << rect << std::endl;
	bool founded = false;
	for (auto& bbox : m_oldBoxes)
	{
		if ((bbox.m_rect & rect).area() / static_cast<float>((bbox.m_rect | rect).area()) > m_bboxIoUThresh)
		{
			bbox.m_lifeTime = m_maxLifeTime;
			founded = true;
		}
	}
	if (!founded)
		m_oldBoxes.emplace_back(rect, m_maxLifeTime);
	//std::cout << "Add res = " << founded << std::endl;
	return !founded;
}

///
/// \brief CombinedDetector::CleanBboxes
/// \return
///
void CombinedDetector::CleanBboxes()
{
	//std::cout << "Clean bboxes..." << std::endl;
	for (auto it = std::begin(m_oldBoxes); it != std::end(m_oldBoxes);)
	{
		it->m_lifeTime--;
		if (it->m_lifeTime < 1)
		{
			//std::cout << "Erase " << it->m_rect << std::endl;
			it = m_oldBoxes.erase(it);
		}
		else
			++it;
	}
	//std::cout << "Clean bboxes finished" << std::endl;
}

///
/// \brief CombinedDetector::InitDetector
/// \param frame
/// \return
///
bool CombinedDetector::InitDetector(cv::UMat frame)
{
	// Create DGFG detector
	config_t configBGFG;
	configBGFG.emplace("useRotatedRect", "0");

	tracking::Detectors detectorType = tracking::Detectors::Motion_MOG2;

	switch (detectorType)
	{
	case tracking::Detectors::Motion_VIBE:
		configBGFG.emplace("samples", "20");
		configBGFG.emplace("pixelNeighbor", "2");
		configBGFG.emplace("distanceThreshold", "15");
		configBGFG.emplace("matchingThreshold", "3");
		configBGFG.emplace("updateFactor", "16");
		break;
	case tracking::Detectors::Motion_MOG:
		configBGFG.emplace("history", std::to_string(cvRound(50 * m_minStaticTime * m_fps)));
		configBGFG.emplace("nmixtures", "3");
		configBGFG.emplace("backgroundRatio", "0.7");
		configBGFG.emplace("noiseSigma", "0");
		break;
	case tracking::Detectors::Motion_GMG:
		configBGFG.emplace("initializationFrames", "50");
		configBGFG.emplace("decisionThreshold", "0.7");
		break;
	case tracking::Detectors::Motion_CNT:
		configBGFG.emplace("minPixelStability", "15");
		configBGFG.emplace("maxPixelStability", std::to_string(cvRound(20 * m_minStaticTime * m_fps)));
		configBGFG.emplace("useHistory", "1");
		configBGFG.emplace("isParallel", "1");
		break;
	case tracking::Detectors::Motion_SuBSENSE:
		break;
	case tracking::Detectors::Motion_LOBSTER:
		break;
	case tracking::Detectors::Motion_MOG2:
		configBGFG.emplace("history", std::to_string(cvRound(20 * m_minStaticTime * m_fps)));
		configBGFG.emplace("varThreshold", "10");
		configBGFG.emplace("detectShadows", "1");
		break;
	}
	m_detectorBGFG = BaseDetector::CreateDetector(detectorType, configBGFG, frame);

	if (m_detectorBGFG.get())
		m_detectorBGFG->SetMinObjectSize(cv::Size(frame.cols / 100, frame.cols / 100));

	// Create DNN based (YOLO) detector
	config_t configDNN;

#ifdef _WIN32
	std::string pathToModel = "../../data/";
#else
	std::string pathToModel = "../data/";
#endif
	size_t maxBatch = 1;
	enum class YOLOModels
	{
		TinyYOLOv3 = 0,
		YOLOv3,
		YOLOv4,
		TinyYOLOv4,
		YOLOv5,
		YOLOv6,
		YOLOv7,
		YOLOv7Mask,
		YOLOv8
	};
	YOLOModels usedModel = YOLOModels::YOLOv8;
	switch (usedModel)
	{
	case YOLOModels::TinyYOLOv3:
		configDNN.emplace("modelConfiguration", pathToModel + "yolov3-tiny.cfg");
		configDNN.emplace("modelBinary", pathToModel + "yolov3-tiny.weights");
		configDNN.emplace("confidenceThreshold", "0.5");
		configDNN.emplace("inference_precision", "FP32");
		configDNN.emplace("net_type", "YOLOV3");
		maxBatch = 4;
		configDNN.emplace("maxCropRatio", "2");
		break;

	case YOLOModels::YOLOv3:
		configDNN.emplace("modelConfiguration", pathToModel + "yolov3.cfg");
		configDNN.emplace("modelBinary", pathToModel + "yolov3.weights");
		configDNN.emplace("confidenceThreshold", "0.7");
		configDNN.emplace("inference_precision", "FP32");
		configDNN.emplace("net_type", "YOLOV3");
		maxBatch = 2;
		configDNN.emplace("maxCropRatio", "-1");
		break;

	case YOLOModels::YOLOv4:
		configDNN.emplace("modelConfiguration", pathToModel + "yolov4.cfg");
		configDNN.emplace("modelBinary", pathToModel + "yolov4.weights");
		configDNN.emplace("confidenceThreshold", "0.4");
		configDNN.emplace("inference_precision", "FP32");
		configDNN.emplace("net_type", "YOLOV4");
		maxBatch = 1;
		configDNN.emplace("maxCropRatio", "-1");
		break;

	case YOLOModels::TinyYOLOv4:
		configDNN.emplace("modelConfiguration", pathToModel + "yolov4-tiny.cfg");
		configDNN.emplace("modelBinary", pathToModel + "yolov4-tiny.weights");
		configDNN.emplace("confidenceThreshold", "0.5");
		configDNN.emplace("inference_precision", "FP32");
		configDNN.emplace("net_type", "YOLOV4_TINY");
		maxBatch = 1;
		configDNN.emplace("maxCropRatio", "-1");
		break;

	case YOLOModels::YOLOv5:
		configDNN.emplace("modelConfiguration", pathToModel + "yolov5s.cfg");
		configDNN.emplace("modelBinary", pathToModel + "yolov5s.weights");
		configDNN.emplace("confidenceThreshold", "0.5");
		configDNN.emplace("inference_precision", "FP32");
		configDNN.emplace("net_type", "YOLOV5");
		maxBatch = 1;
		configDNN.emplace("maxCropRatio", "-1");
		break;

	case YOLOModels::YOLOv6:
		configDNN.emplace("modelConfiguration", pathToModel + "yolov6s.onnx");
		configDNN.emplace("modelBinary", pathToModel + "yolov6s.onnx");
		configDNN.emplace("confidenceThreshold", "0.5");
		configDNN.emplace("inference_precision", "FP32");
		configDNN.emplace("net_type", "YOLOV6");
		maxBatch = 1;
		configDNN.emplace("maxCropRatio", "-1");
		break;

	case YOLOModels::YOLOv7:
	{
		//std::string modelName = "yolov7.onnx";
		std::string modelName = "yolov7x.onnx";
		//std::string modelName = "yolov7-w6.onnx";
		configDNN.emplace("modelConfiguration", pathToModel + modelName);
		configDNN.emplace("modelBinary", pathToModel + modelName);
		configDNN.emplace("confidenceThreshold", "0.2");
		configDNN.emplace("inference_precision", "FP16");
		configDNN.emplace("net_type", "YOLOV7");
		maxBatch = 1;
		configDNN.emplace("maxCropRatio", "-1");
		break;
	}
	case YOLOModels::YOLOv7Mask:
		configDNN.emplace("modelConfiguration", pathToModel + "yolov7-seg_orig.onnx");
		configDNN.emplace("modelBinary", pathToModel + "yolov7-seg_orig.onnx");
		configDNN.emplace("confidenceThreshold", "0.2");
		configDNN.emplace("inference_precision", "FP16");
		configDNN.emplace("net_type", "YOLOV7Mask");
		maxBatch = 1;
		configDNN.emplace("maxCropRatio", "-1");
		break;

	case YOLOModels::YOLOv8:
		//configDNN.emplace("modelConfiguration", pathToModel + "yolov8s.onnx");
		//configDNN.emplace("modelBinary", pathToModel + "yolov8s.onnx");
		configDNN.emplace("modelConfiguration", "C:/work/mtracking/Nuzhny007/Multitarget-tracker/data/yolov8x.onnx");
		configDNN.emplace("modelBinary", "C:/work/mtracking/Nuzhny007/Multitarget-tracker/data/yolov8x.onnx");
		configDNN.emplace("confidenceThreshold", "0.2");
		configDNN.emplace("inference_precision", "FP16");
		configDNN.emplace("net_type", "YOLOV8");
		configDNN.emplace("inWidth", "640");
		configDNN.emplace("inHeight", "640");
		maxBatch = 1;
		configDNN.emplace("maxCropRatio", "-1");
		break;
	}
	configDNN.emplace("maxBatch", std::to_string(maxBatch));
	configDNN.emplace("classNames", pathToModel + "coco.names");
	configDNN.emplace("maxCropRatio", "-1");

    configDNN.emplace("white_list", "person");
    configDNN.emplace("white_list", "backpack");
    configDNN.emplace("white_list", "handbag");
    configDNN.emplace("white_list", "suitcase");

#if 1
	configDNN.emplace("dnnTarget", "DNN_TARGET_CPU");
	configDNN.emplace("dnnBackend", "DNN_BACKEND_DEFAULT");
#else
	configDNN.emplace("dnnTarget", "DNN_TARGET_CUDA");
	configDNN.emplace("dnnBackend", "DNN_BACKEND_CUDA");
#endif

	m_detectorDNN = BaseDetector::CreateDetector(tracking::Detectors::Yolo_TensorRT, configDNN, frame);
	//m_detectorDNN = BaseDetector::CreateDetector(tracking::Detectors::DNN_OCV, configDNN, frame);

	return m_detectorBGFG.get() && m_detectorDNN.get();
}
///
/// \brief CombinedDetector::InitTracker
/// \param frame
/// \return
///
bool CombinedDetector::InitTracker(cv::UMat frame)
{
	// Create BGFG tracker
	TrackerSettings settingsBGFG;
	settingsBGFG.SetDistance(tracking::DistCenters);
	settingsBGFG.m_kalmanType = tracking::KalmanLinear;
	settingsBGFG.m_filterGoal = tracking::FilterCenter;
	settingsBGFG.m_lostTrackType = tracking::TrackNone;       // Use visual objects tracker for collisions resolving
	settingsBGFG.m_matchType = tracking::MatchHungrian;
	settingsBGFG.m_useAcceleration = false;                   // Use constant acceleration motion model
	settingsBGFG.m_dt = settingsBGFG.m_useAcceleration ? 0.05f : 0.2f; // Delta time for Kalman filter
	settingsBGFG.m_accelNoiseMag = 0.2f;                  // Accel noise magnitude for Kalman filter
	settingsBGFG.m_distThres = 0.95f;                    // Distance threshold between region and object on two frames
#if 0
	settingsBGFG.m_minAreaRadiusPix = frame.rows / 20.f;
#else
	settingsBGFG.m_minAreaRadiusPix = -1.f;
#endif
	settingsBGFG.m_minAreaRadiusK = 0.8f;

	settingsBGFG.m_useAbandonedDetection = true;
	if (settingsBGFG.m_useAbandonedDetection)
	{
		settingsBGFG.m_minStaticTime = m_minStaticTime;
		settingsBGFG.m_maxStaticTime = 30;
		settingsBGFG.m_maximumAllowedSkippedFrames = cvRound(settingsBGFG.m_minStaticTime * m_fps); // Maximum allowed skipped frames
		settingsBGFG.m_maxTraceLength = 2 * settingsBGFG.m_maximumAllowedSkippedFrames;        // Maximum trace length
	}
	else
	{
		settingsBGFG.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
		settingsBGFG.m_maxTraceLength = cvRound(4 * m_fps);              // Maximum trace length
	}

	m_trackerBGFG = BaseTracker::CreateTracker(settingsBGFG);

	// Create tracker for DNN based detector
	TrackerSettings settingsDNN;
	settingsDNN.SetDistance(tracking::DistCenters);
	settingsDNN.m_kalmanType = tracking::KalmanLinear;
	settingsDNN.m_filterGoal = tracking::FilterCenter;
	settingsDNN.m_lostTrackType = tracking::TrackNone;      // Use visual objects tracker for collisions resolving
	settingsDNN.m_matchType = tracking::MatchHungrian;
	settingsDNN.m_useAcceleration = false;                   // Use constant acceleration motion model
	settingsDNN.m_dt = settingsDNN.m_useAcceleration ? 0.05f : 0.4f; // Delta time for Kalman filter
	settingsDNN.m_accelNoiseMag = 0.2f;                     // Accel noise magnitude for Kalman filter
	settingsDNN.m_distThres = 0.8f;                         // Distance threshold between region and object on two frames
#if 0
	settingsDNN.m_minAreaRadiusPix = frame.rows / 20.f;
#else
	settingsDNN.m_minAreaRadiusPix = -1.f;
#endif
	settingsDNN.m_minAreaRadiusK = 0.8f;

	settingsDNN.m_useAbandonedDetection = true;
	if (settingsDNN.m_useAbandonedDetection)
	{
		settingsDNN.m_minStaticTime = m_minStaticTime;
		settingsDNN.m_maxStaticTime = 30;
		settingsDNN.m_maximumAllowedSkippedFrames = cvRound(settingsDNN.m_minStaticTime * m_fps); // Maximum allowed skipped frames
		settingsDNN.m_maxTraceLength = 2 * settingsDNN.m_maximumAllowedSkippedFrames;        // Maximum trace length
	}
	else
	{
		settingsDNN.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
		settingsDNN.m_maxTraceLength = cvRound(4 * m_fps);              // Maximum trace length
	}

	settingsDNN.AddNearTypes(TypeConverter::Str2Type("backpack"), TypeConverter::Str2Type("handbag"), true);
	settingsDNN.AddNearTypes(TypeConverter::Str2Type("backpack"), TypeConverter::Str2Type("suitcase"), true);
	settingsDNN.AddNearTypes(TypeConverter::Str2Type("suitcase"), TypeConverter::Str2Type("handbag"), true);

	m_trackerDNN = BaseTracker::CreateTracker(settingsDNN);

	return true;
}

///
/// \brief CombinedDetector::DrawData
/// \param frame
/// \param framesCounter
/// \param currTime
///
void CombinedDetector::DrawData(cv::Mat frame, int framesCounter, int currTime)
{
	if (m_showLogs)
		std::cout << "Frame " << framesCounter << ": tracks = " << (m_tracksBGFG.size() + m_tracksDNN.size()) << ", time = " << currTime << std::endl;

	for (const auto& track : m_tracksBGFG)
	{
		if (track.IsRobust(cvRound(4),          // Minimal trajectory size
			0.2f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
			cv::Size2f(0.1f, 8.0f)))      // Min and max ratio: width / height
		{
			int staticSeconds = cvRound(track.m_staticTime / m_fps);
			if (track.m_isStatic && m_minStaticTime < staticSeconds)
			{
				DrawTrack(frame, track, false);

				std::string label = "abandoned " + std::to_string(staticSeconds) + " s";
				int baseLine = 0;
				cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

				cv::Rect brect = track.m_rrect.boundingRect();
				if (brect.x < 0)
				{
					brect.width = std::min(brect.width, frame.cols - 1);
					brect.x = 0;
				}
				else if (brect.x + brect.width >= frame.cols)
				{
					brect.x = std::max(0, frame.cols - brect.width - 1);
					brect.width = std::min(brect.width, frame.cols - 1);
				}
				if (brect.y - labelSize.height < 0)
				{
					brect.height = std::min(brect.height, frame.rows - 1);
					brect.y = labelSize.height;
				}
				else if (brect.y + brect.height >= frame.rows)
				{
					brect.y = std::max(0, frame.rows - brect.height - 1);
					brect.height = std::min(brect.height, frame.rows - 1);
				}
				DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 0, 255), 150);
				cv::putText(frame, label, brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
			}
			else
			{
#if 0
				DrawTrack(frame, 1, track, true);
#endif
			}
		}
	}

	//m_detectorBGFG->CalcMotionMap(frame);
	//m_detectorDNN->CalcMotionMap(frame);

	for (const auto& track : m_tracksDNN)
	{
		if (track.IsRobust(4,            // Minimal trajectory size
			0.85f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
			cv::Size2f(0.1f, 10.0f),      // Min and max ratio: width / height
			4))
		{
			DrawTrack(frame, track);

			std::stringstream label;
			int staticSeconds = cvRound(track.m_staticTime / m_fps);
			if (track.m_isStatic && m_minStaticTime < staticSeconds && track.m_type != TypeConverter::Str2Type("person"))
				label << "abandoned ";
			label << TypeConverter::Type2Str(track.m_type) << " " << std::to_string(staticSeconds) + "s " << std::fixed << std::setw(2) << std::setprecision(2) << track.m_confidence;
			int baseLine = 0;
			cv::Size labelSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

			cv::Rect brect = track.m_rrect.boundingRect();
			if (brect.x < 0)
			{
				brect.width = std::min(brect.width, frame.cols - 1);
				brect.x = 0;
			}
			else if (brect.x + brect.width >= frame.cols)
			{
				brect.x = std::max(0, frame.cols - brect.width - 1);
				brect.width = std::min(brect.width, frame.cols - 1);
			}
			if (brect.y - labelSize.height < 0)
			{
				brect.height = std::min(brect.height, frame.rows - 1);
				brect.y = labelSize.height;
			}
			else if (brect.y + brect.height >= frame.rows)
			{
				brect.y = std::max(0, frame.rows - brect.height - 1);
				brect.height = std::min(brect.height, frame.rows - 1);
			}
			DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), track.m_isStatic ? cv::Scalar(200, 0, 200) : cv::Scalar(200, 200, 200), 150);
			cv::putText(frame, label.str(), brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		}
	}

	for (const auto& bbox : m_oldBoxes)
	{
		break;
		cv::rectangle(frame, bbox.m_rect, cv::Scalar(0, 255, 255), 1);

		std::string label = std::to_string(bbox.m_lifeTime);
		int baseLine = 0;
		cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

		cv::Rect brect = bbox.m_rect;
		if (brect.x < 0)
		{
			brect.width = std::min(brect.width, frame.cols - 1);
			brect.x = 0;
		}
		else if (brect.x + brect.width >= frame.cols)
		{
			brect.x = std::max(0, frame.cols - brect.width - 1);
			brect.width = std::min(brect.width, frame.cols - 1);
		}
		if (brect.y - labelSize.height < 0)
		{
			brect.height = std::min(brect.height, frame.rows - 1);
			brect.y = labelSize.height;
		}
		else if (brect.y + brect.height >= frame.rows)
		{
			brect.y = std::max(0, frame.rows - brect.height - 1);
			brect.height = std::min(brect.height, frame.rows - 1);
		}

		DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(200, 200, 200), 150);
		cv::putText(frame, label, brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
	}
}

///
/// \brief CombinedDetector::DrawTrack
/// \param frame
/// \param resizeCoeff
/// \param track
/// \param drawTrajectory
///
void CombinedDetector::DrawTrack(cv::Mat frame, const TrackingObject& track, bool drawTrajectory)
{
    cv::Scalar color = track.m_isStatic ? cv::Scalar(255, 0, 255) : cv::Scalar(0, 255, 0);
    cv::Point2f rectPoints[4];
    track.m_rrect.points(rectPoints);
    for (int i = 0; i < 4; ++i)
    {
        cv::line(frame, rectPoints[i], rectPoints[(i+1) % 4], color);
    }

    if (drawTrajectory)
    {
        cv::Scalar cl = m_colors[track.m_ID.ID2Module(m_colors.size())];

        for (size_t j = 0; j < track.m_trace.size() - 1; ++j)
        {
            const TrajectoryPoint& pt1 = track.m_trace.at(j);
            const TrajectoryPoint& pt2 = track.m_trace.at(j + 1);
            cv::line(frame, pt1.m_prediction, pt2.m_prediction, cl, 1, cv::LINE_AA);
            if (!pt2.m_hasRaw)
                cv::circle(frame, pt2.m_prediction, 4, cl, 1, cv::LINE_AA);
        }
    }
}

///
/// \brief CombinedDetector::OpenCapture
/// \param capture
/// \return
///
bool CombinedDetector::OpenCapture(cv::VideoCapture& capture)
{
	if (m_inFile.size() == 1)
	{
#ifdef _WIN32
		capture.open(atoi(m_inFile.c_str()), cv::CAP_DSHOW);
#else
		capture.open(atoi(m_inFile.c_str()));
#endif
		if (capture.isOpened())
			capture.set(cv::CAP_PROP_SETTINGS, 1);
	}
    else
        capture.open(m_inFile);

    if (capture.isOpened())
    {
        capture.set(cv::CAP_PROP_POS_FRAMES, m_startFrame);

        m_fps = std::max(30.f, (float)capture.get(cv::CAP_PROP_FPS));

		std::cout << "Video " << m_inFile << " was started from " << m_startFrame << " frame with " << m_fps << " fps" << std::endl;

        return true;
    }
    return false;
}

///
/// \brief CombinedDetector::WriteFrame
/// \param writer
/// \param frame
/// \return
///
bool CombinedDetector::WriteFrame(cv::VideoWriter& writer, const cv::Mat& frame)
{
    if (!m_outFile.empty())
    {
        if (!writer.isOpened())
            writer.open(m_outFile, m_fourcc, m_fps, frame.size(), true);

        if (writer.isOpened())
        {
            writer << frame;
            return true;
        }
    }
    return false;
}
