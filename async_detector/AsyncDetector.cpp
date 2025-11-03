#include "AsyncDetector.h"

///
/// \brief AsyncDetector::AsyncDetector
/// \param parser
///
AsyncDetector::AsyncDetector(const cv::CommandLineParser& parser)
{
    m_inFile = parser.get<std::string>(0);
    m_outFile = parser.get<std::string>("out");
    m_showLogsLevel = parser.get<std::string>("show_logs");
    m_startFrame = parser.get<int>("start_frame");
    m_endFrame = parser.get<int>("end_frame");
    m_finishDelay = parser.get<int>("end_delay");

    m_colors.emplace_back(255, 0, 0);
    m_colors.emplace_back(0, 255, 0);
    m_colors.emplace_back(0, 0, 255);
    m_colors.emplace_back(255, 255, 0);
    m_colors.emplace_back(0, 255, 255);
    m_colors.emplace_back(255, 0, 255);
    m_colors.emplace_back(255, 127, 255);
    m_colors.emplace_back(127, 0, 255);
    m_colors.emplace_back(127, 0, 127);

    // Create loggers
    m_consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    m_consoleSink->set_level(spdlog::level::from_str(m_showLogsLevel));
    m_consoleSink->set_pattern("[%^%l%$] %v");

    auto currentTime = std::chrono::system_clock::now();
    auto transformed = currentTime.time_since_epoch().count() / 1000000;
    std::time_t tt = std::chrono::system_clock::to_time_t(currentTime);
    char buffer[80];
#ifdef WIN32
    tm timeInfo;
    localtime_s(&timeInfo, &tt);
    strftime(buffer, 80, "%G%m%d_%H%M%S", &timeInfo);
#else
    auto timeInfo = localtime(&tt);
    strftime(buffer, 80, "%G%m%d_%H%M%S", timeInfo);
#endif

    size_t max_size = 1024 * 1024 * 5;
    size_t max_files = 3;
    m_fileSink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>("logs/" + std::string(buffer) + std::to_string(transformed % 1000) + ".txt", max_size, max_files);
    m_fileSink->set_level(spdlog::level::from_str(m_showLogsLevel));

    m_logger = std::shared_ptr<spdlog::logger>(new spdlog::logger("traffic", { m_consoleSink, m_fileSink }));
    m_logger->set_level(spdlog::level::from_str(m_showLogsLevel));
    m_logger->info("Start service");
}

///
/// \brief AsyncDetector::Process
///
void AsyncDetector::Process()
{
    double freq = cv::getTickFrequency();
    int64 allTime = 0;

    bool stopFlag = false;

    std::thread thCapture(CaptureThread, m_inFile, m_startFrame, &m_framesCount, &m_fps, &m_framesQue, &stopFlag);

#ifndef SILENT_WORK
    cv::namedWindow("Video", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::waitKey(1);
#endif

	cv::VideoWriter writer;

    int framesCounter = m_startFrame + 1;

    for (; !stopFlag;)
    {
		int64 t1 = cv::getTickCount();

        // Show frame after detecting and tracking
        frame_ptr processedFrame = m_framesQue.GetFirstProcessedFrame();
        if (!processedFrame)
        {
            stopFlag = true;
            break;
        }

        int64 t2 = cv::getTickCount();

        allTime += t2 - processedFrame->m_dt;
        int currTime = cvRound(1000 * (t2 - t1) / freq);

        DrawData(processedFrame, framesCounter, currTime);

		if (!m_outFile.empty())
		{
			if (!writer.isOpened())
				writer.open(m_outFile, cv::VideoWriter::fourcc('H', 'F', 'Y', 'U'), m_fps, processedFrame->m_frame.size(), true);
			if (writer.isOpened())
				writer << processedFrame->m_frame;
		}

#ifndef SILENT_WORK
        cv::imshow("Video", processedFrame->m_frame);

		int waitTime = 1;// std::max<int>(1, cvRound(1000 / m_fps - currTime));
        int k = cv::waitKey(waitTime);
        if (k == 27)
            break;
#else
        //std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif

        ++framesCounter;
        if (m_endFrame && framesCounter > m_endFrame)
        {
            m_logger->info("Process: riched last {} frame", m_endFrame);
            break;
        }
    }

    m_logger->info("Stopping threads...");
    stopFlag = true;
    m_framesQue.SetBreak(true);

    if (thCapture.joinable())
        thCapture.join();

    m_logger->info("work time = {}", allTime / freq);
#ifndef SILENT_WORK
	cv::waitKey(m_finishDelay);
#endif
}

///
/// \brief AsyncDetector::DrawTrack
/// \param frame
/// \param track
/// \param drawTrajectory
/// \param isStatic
///
void AsyncDetector::DrawTrack(cv::Mat frame,
                             const TrackingObject& track,
                             bool drawTrajectory)
{
    if (track.m_isStatic)
    {
#if (CV_VERSION_MAJOR >= 4)
        cv::rectangle(frame, track.m_rrect.boundingRect(), cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
#else
        cv::rectangle(frame, track.m_rrect.boundingRect(), cv::Scalar(255, 0, 255), 2, CV_AA);
#endif
    }
    else
    {
#if (CV_VERSION_MAJOR >= 4)
        cv::rectangle(frame, track.m_rrect.boundingRect(), cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
#else
        cv::rectangle(frame, track.m_rrect.boundingRect(), cv::Scalar(0, 255, 0), 1, CV_AA);
#endif
    }

    if (drawTrajectory)
    {
        cv::Scalar cl = m_colors[track.m_ID.ID2Module(m_colors.size())];

        for (size_t j = 0; j < track.m_trace.size() - 1; ++j)
        {
            const TrajectoryPoint& pt1 = track.m_trace.at(j);
            const TrajectoryPoint& pt2 = track.m_trace.at(j + 1);
#if (CV_VERSION_MAJOR >= 4)
            cv::line(frame, pt1.m_prediction, pt2.m_prediction, cl, 1, cv::LINE_AA);
#else
            cv::line(frame, pt1.m_prediction, pt2.m_prediction, cl, 1, CV_AA);
#endif
            if (pt2.m_hasRaw)
            {
#if (CV_VERSION_MAJOR >= 4)
                cv::circle(frame, pt2.m_prediction, 4, cl, 4, cv::LINE_AA);
#else
                cv::circle(frame, pt2.m_prediction, 4, cl, 4, CV_AA);
#endif
            }
        }
    }
}

///
/// \brief AsyncDetector::DrawData
/// \param frameinfo
///
void AsyncDetector::DrawData(frame_ptr frameInfo, int framesCounter, int currTime)
{
    int id = frameInfo->m_inDetector.load();
    if (id != FrameInfo::StateNotProcessed && id != FrameInfo::StateSkipped)
        m_logger->info("Frame {0} ({1}): ({2}) detects= {3}, tracks = {4}, time = {5}", framesCounter, m_framesCount, id, frameInfo->m_regions.size(), frameInfo->m_tracks.size(), currTime);
    else
        m_logger->info("Frame {0} ({1}): tracks = {2}, time = {3}", framesCounter, m_framesCount, frameInfo->m_tracks.size(), currTime);

    for (const auto& track : frameInfo->m_tracks)
    {
        if (track.m_isStatic)
        {
            DrawTrack(frameInfo->m_frame, track, true);
        }
        else
        {
            if (track.IsRobust(1,          // Minimal trajectory size
                               0.3f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                               cv::Size2f(0.1f, 8.0f)))      // Min and max ratio: width / height
            {
				//std::cout << TypeConverter::Type2Str(track.m_type) << " - " << track.m_rect << std::endl;

                DrawTrack(frameInfo->m_frame, track, true);

				std::string label = TypeConverter::Type2Str(track.m_type);// +": " + std::to_string(track.m_confidence);
				int baseLine = 0;
				cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                cv::Rect brect = track.m_rrect.boundingRect();
				if (brect.x < 0)
				{
					brect.width = std::min(brect.width, frameInfo->m_frame.cols - 1);
					brect.x = 0;
				}
				else if (brect.x + brect.width >= frameInfo->m_frame.cols)
				{
					brect.x = std::max(0, frameInfo->m_frame.cols - brect.width - 1);
					brect.width = std::min(brect.width, frameInfo->m_frame.cols - 1);
				}
				if (brect.y - labelSize.height < 0)
				{
					brect.height = std::min(brect.height, frameInfo->m_frame.rows - 1);
					brect.y = labelSize.height;
				}
				else if (brect.y + brect.height >= frameInfo->m_frame.rows)
				{
					brect.y = std::max(0, frameInfo->m_frame.rows - brect.height - 1);
					brect.height = std::min(brect.height, frameInfo->m_frame.rows - 1);
				}
				DrawFilledRect(frameInfo->m_frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(200, 200, 200), 150);
                cv::putText(frameInfo->m_frame, label, brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }
    }
}

///
/// \brief AsyncDetector::CaptureThread
/// \param fileName
/// \param framesQue
/// \param stopFlag
///
void AsyncDetector::CaptureThread(std::string fileName, int startFrame, int* framesCount, float* fps, FramesQueue* framesQue, bool* stopFlag)
{
    cv::VideoCapture capture;
    if (fileName.size() == 1)
        capture.open(atoi(fileName.c_str()));
    else
        capture.open(fileName);

    if (!capture.isOpened())
    {
        *stopFlag = true;
        std::cerr << "Can't open " << fileName << std::endl;
        return;
    }
    *framesCount = cvRound(capture.get(cv::CAP_PROP_FRAME_COUNT));
    capture.set(cv::CAP_PROP_POS_FRAMES, startFrame);

    time_point_t startTimeStamp = std::chrono::system_clock::now();

    *fps = std::max(1.f, (float)capture.get(cv::CAP_PROP_FPS));
	int frameHeight = cvRound(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

    // Detector
    config_t detectorConfig;

#ifdef _WIN32
    std::string pathToModel = "../../data/";
#else
    std::string pathToModel = "../data/";
#endif

#if 0
	detectorConfig.emplace("modelConfiguration", pathToModel + "yolov3-tiny.cfg");
	detectorConfig.emplace("modelBinary", pathToModel + "yolov3-tiny.weights");
#else
    detectorConfig.emplace("modelConfiguration", pathToModel + "yolov4-csp.cfg");
    detectorConfig.emplace("modelBinary", pathToModel + "yolov4-csp.weights");
#endif
    detectorConfig.emplace("classNames", pathToModel + "coco.names");
    detectorConfig.emplace("confidenceThreshold", "0.3");
    detectorConfig.emplace("maxCropRatio", "3.0");
	
#if 1
    detectorConfig.emplace("white_list", "person");
    detectorConfig.emplace("white_list", "car");
    detectorConfig.emplace("white_list", "bicycle");
    detectorConfig.emplace("white_list", "motorbike");
    detectorConfig.emplace("white_list", "bus");
    detectorConfig.emplace("white_list", "truck");
#endif

    // Tracker
    const int minStaticTime = 5;

    TrackerSettings trackerSettings;
	trackerSettings.SetDistance(tracking::DistCenters);
    trackerSettings.m_kalmanType = tracking::KalmanLinear;
    trackerSettings.m_filterGoal = tracking::FilterRect;
    trackerSettings.m_lostTrackType = tracking::TrackCSRT; // Use KCF tracker for collisions resolving. Used if m_filterGoal == tracking::FilterRect
    trackerSettings.m_matchType = tracking::MatchHungrian;
    trackerSettings.m_dt = 0.2f;                           // Delta time for Kalman filter
    trackerSettings.m_accelNoiseMag = 0.3f;                // Accel noise magnitude for Kalman filter
    trackerSettings.m_distThres = 0.8f;                    // Distance threshold between region and object on two frames
    trackerSettings.m_minAreaRadiusPix = frameHeight / 20.f;

    trackerSettings.m_useAbandonedDetection = false;
    if (trackerSettings.m_useAbandonedDetection)
    {
        trackerSettings.m_minStaticTime = minStaticTime;
        trackerSettings.m_maxStaticTime = 60;
        trackerSettings.m_maximumAllowedLostTime = trackerSettings.m_minStaticTime;      // Maximum allowed lost time
        trackerSettings.m_maxTraceLength = 2 * trackerSettings.m_maximumAllowedLostTime; // Maximum trace length
    }
    else
    {
        trackerSettings.m_maximumAllowedLostTime = 2.; // Maximum allowed lost time
        trackerSettings.m_maxTraceLength = 4.;         // Maximum trace length
    }

    // Capture the first frame
	size_t frameInd = startFrame;
    cv::Mat firstFrame;
    capture >> firstFrame;
	++frameInd;

    std::thread thDetection(DetectThread, detectorConfig, firstFrame, framesQue, stopFlag);
    std::thread thTracking(TrackingThread, trackerSettings, framesQue, *fps, stopFlag);

    // Capture frame
    for (; !(*stopFlag);)
    {
        frame_ptr frameInfo(new FrameInfo(frameInd));
        frameInfo->m_dt = cv::getTickCount();
        frameInfo->m_frameTimeStamp = startTimeStamp + std::chrono::milliseconds(cvRound(frameInd * (1000.f / (*fps))));
        capture >> frameInfo->m_frame;
        if (frameInfo->m_frame.empty())
        {
            std::cerr << "Frame is empty!" << std::endl;
            *stopFlag = true;
            framesQue->SetBreak(true);
            break;
        }
		if (frameInfo->m_clFrame.empty())
			frameInfo->m_clFrame = frameInfo->m_frame.getUMat(cv::ACCESS_READ);

        framesQue->AddNewFrame(frameInfo, 15);

#if 1
        //std::this_thread::sleep_for(std::chrono::milliseconds(1000 / cvRound(*fps) - 1));
#else
        //std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif

		++frameInd;
    }

    framesQue->SetBreak(true);
    if (thTracking.joinable())
        thTracking.join();

    framesQue->SetBreak(true);
    if (thDetection.joinable())
        thDetection.join();

    framesQue->SetBreak(true);
}

///
/// \brief AsyncDetector::DetectThread
/// \param
///
void AsyncDetector::DetectThread(const config_t& config, cv::Mat firstFrame, FramesQueue* framesQue, bool* stopFlag)
{
	cv::UMat ufirst = firstFrame.getUMat(cv::ACCESS_READ);
    std::unique_ptr<BaseDetector> detector = BaseDetector::CreateDetector(tracking::Detectors::ONNX_TensorRT, config, ufirst);

    for (; !(*stopFlag);)
    {
        frame_ptr frameInfo = framesQue->GetLastUndetectedFrame();
        if (frameInfo)
        {
            detector->Detect(frameInfo->m_clFrame);

            const regions_t& regions = detector->GetDetects();
            frameInfo->m_regions.assign(regions.begin(), regions.end());

            frameInfo->m_inDetector.store(FrameInfo::StateCompleted);
            framesQue->Signal(frameInfo->m_dt);
        }
    }
}

///
/// \brief AsyncDetector::TrackingThread
/// \param
///
void AsyncDetector::TrackingThread(const TrackerSettings& settings, FramesQueue* framesQue, float fps, bool* stopFlag)
{
    std::unique_ptr<BaseTracker> tracker = BaseTracker::CreateTracker(settings, fps);

    for (; !(*stopFlag);)
    {
        frame_ptr frameInfo = framesQue->GetFirstDetectedFrame();
        if (frameInfo)
        {
            tracker->Update(frameInfo->m_regions, frameInfo->m_clFrame, frameInfo->m_frameTimeStamp);

            tracker->GetTracks(frameInfo->m_tracks);
            frameInfo->m_inTracker.store(FrameInfo::StateCompleted);
            framesQue->Signal(frameInfo->m_dt);
        }
    }
}
