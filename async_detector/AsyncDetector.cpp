#include "AsyncDetector.h"

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
/// \brief AsyncDetector::AsyncDetector
/// \param parser
///
AsyncDetector::AsyncDetector(const cv::CommandLineParser& parser)
    :
      m_showLogs(true),
      m_fps(25),
      m_startFrame(0),
      m_endFrame(0),
      m_finishDelay(0)
{
    m_inFile = parser.get<std::string>(0);
    m_outFile = parser.get<std::string>("out");
    m_showLogs = parser.get<int>("show_logs") != 0;
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
}

///
/// \brief AsyncDetector::Process
///
void AsyncDetector::Process()
{
    double freq = cv::getTickFrequency();
    int64 allTime = 0;

    bool stopFlag = false;

    std::thread thCapture(CaptureThread, m_inFile, m_startFrame, &m_fps, &m_framesQue, &stopFlag);

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
            std::cout << "Process: riched last " << m_endFrame << " frame" << std::endl;
            break;
        }
    }

    std::cout << "Stopping threads..." << std::endl;
    stopFlag = true;
    m_framesQue.SetBreak(true);

    if (thCapture.joinable())
        thCapture.join();

    std::cout << "work time = " << (allTime / freq) << std::endl;
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
    if (m_showLogs)
    {
		std::cout << "Frame " << framesCounter << ": ";
        int id = frameInfo->m_inDetector.load();
        if (id != FrameInfo::StateNotProcessed && id != FrameInfo::StateSkipped)
            std::cout << "(" << id << ") detects = " << frameInfo->m_regions.size() << ", ";
		std::cout << "tracks = " << frameInfo->m_tracks.size() << ", time = " << currTime << std::endl;
    }

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
void AsyncDetector::CaptureThread(std::string fileName, int startFrame, float* fps, FramesQueue* framesQue, bool* stopFlag)
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
    capture.set(cv::CAP_PROP_POS_FRAMES, startFrame);

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
        trackerSettings.m_maximumAllowedSkippedFrames = cvRound(trackerSettings.m_minStaticTime * (*fps)); // Maximum allowed skipped frames
        trackerSettings.m_maxTraceLength = 2 * trackerSettings.m_maximumAllowedSkippedFrames;        // Maximum trace length
    }
    else
    {
        trackerSettings.m_maximumAllowedSkippedFrames = cvRound(2 * (*fps)); // Maximum allowed skipped frames
        trackerSettings.m_maxTraceLength = cvRound(4 * (*fps));              // Maximum trace length
    }

    // Capture the first frame
	size_t frameInd = startFrame;
    cv::Mat firstFrame;
    capture >> firstFrame;
	++frameInd;

    std::thread thDetection(DetectThread, detectorConfig, firstFrame, framesQue, stopFlag);
    std::thread thTracking(TrackingThread, trackerSettings, framesQue, stopFlag);

    // Capture frame
    for (; !(*stopFlag);)
    {
        frame_ptr frameInfo(new FrameInfo(frameInd));
        frameInfo->m_dt = cv::getTickCount();
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
    std::unique_ptr<BaseDetector> detector = BaseDetector::CreateDetector(tracking::Detectors::Yolo_Darknet, config, ufirst);

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
void AsyncDetector::TrackingThread(const TrackerSettings& settings, FramesQueue* framesQue, bool* stopFlag)
{
    std::unique_ptr<BaseTracker> tracker = BaseTracker::CreateTracker(settings);

    for (; !(*stopFlag);)
    {
        frame_ptr frameInfo = framesQue->GetFirstDetectedFrame();
        if (frameInfo)
        {
            tracker->Update(frameInfo->m_regions, frameInfo->m_clFrame, frameInfo->m_fps);

            tracker->GetTracks(frameInfo->m_tracks);
            frameInfo->m_inTracker.store(FrameInfo::StateCompleted);
            framesQue->Signal(frameInfo->m_dt);
        }
    }
}
