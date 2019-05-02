#include "AsyncDetector.h"

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
/// \brief AsyncDetector::~AsyncDetector
///
AsyncDetector::~AsyncDetector()
{

}

///
/// \brief AsyncDetector::Process
///
void AsyncDetector::Process()
{
    int k = 0;

    double freq = cv::getTickFrequency();
    int64 allTime = 0;

    bool stopFlag = false;

    std::thread thCapture(CaptureThread, m_inFile, m_startFrame, &m_fps, &m_framesQue, &stopFlag);
    thCapture.detach();

    cv::namedWindow("Video", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);

    cv::VideoWriter writer;

    cv::waitKey(1);

    int framesCounter = m_startFrame + 1;

	FrameInfo frames[2];
	size_t frameInd = 0;
    for (;;)
    {
        // Show frame after detecting and tracking
		frames[frameInd] = m_framesQue.GetFirstProcessedFrame();
        FrameInfo& processedFrame = frames[frameInd];

        int64 t2 = cv::getTickCount();

        allTime += t2 - processedFrame.m_dt;
        int currTime = cvRound(1000 * (t2 - processedFrame.m_dt) / freq);

        DrawData(&processedFrame, framesCounter, currTime);

        if (!writer.isOpened())
        {
            writer.open(m_outFile, cv::VideoWriter::fourcc('H', 'F', 'Y', 'U'), m_fps, processedFrame.m_frame.size(), true);
        }
        if (writer.isOpened())
        {
            writer << processedFrame.m_frame;
        }

        cv::imshow("Video", processedFrame.m_frame);

        int waitTime = std::max<int>(1, cvRound(1000 / m_fps - currTime));
        k = cv::waitKey(waitTime);
        if (k == 27)
        {
            break;
        }

        ++framesCounter;
        if (m_endFrame && framesCounter > m_endFrame)
        {
            std::cout << "Process: riched last " << m_endFrame << " frame" << std::endl;
            break;
        }

		frameInd = !frameInd;
    }

    std::cout << "Stopping threads..." << std::endl;
    stopFlag = true;

    if (thCapture.joinable())
    {
        thCapture.join();
    }

    std::cout << "work time = " << (allTime / freq) << std::endl;
    cv::waitKey(m_finishDelay);
}

///
/// \brief AsyncDetector::DrawTrack
/// \param frame
/// \param resizeCoeff
/// \param track
/// \param drawTrajectory
/// \param isStatic
///
void AsyncDetector::DrawTrack(cv::Mat frame,
                             int resizeCoeff,
                             const TrackingObject& track,
                             bool drawTrajectory
                             )
{
    auto ResizeRect = [&](const cv::Rect& r) -> cv::Rect
    {
        return cv::Rect(resizeCoeff * r.x, resizeCoeff * r.y, resizeCoeff * r.width, resizeCoeff * r.height);
    };
    auto ResizePoint = [&](const cv::Point& pt) -> cv::Point
    {
        return cv::Point(resizeCoeff * pt.x, resizeCoeff * pt.y);
    };

    if (track.m_isStatic)
    {
#if (CV_VERSION_MAJOR >= 4)
        cv::rectangle(frame, ResizeRect(track.m_rect), cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
#else
        cv::rectangle(frame, ResizeRect(track.m_rect), cv::Scalar(255, 0, 255), 2, CV_AA);
#endif
    }
    else
    {
#if (CV_VERSION_MAJOR >= 4)
        cv::rectangle(frame, ResizeRect(track.m_rect), cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
#else
        cv::rectangle(frame, ResizeRect(track.m_rect), cv::Scalar(0, 255, 0), 1, CV_AA);
#endif
    }

    if (drawTrajectory)
    {
        cv::Scalar cl = m_colors[track.m_ID % m_colors.size()];

        for (size_t j = 0; j < track.m_trace.size() - 1; ++j)
        {
            const TrajectoryPoint& pt1 = track.m_trace.at(j);
            const TrajectoryPoint& pt2 = track.m_trace.at(j + 1);
#if (CV_VERSION_MAJOR >= 4)
            cv::line(frame, ResizePoint(pt1.m_prediction), ResizePoint(pt2.m_prediction), cl, 1, cv::LINE_AA);
#else
            cv::line(frame, ResizePoint(pt1.m_prediction), ResizePoint(pt2.m_prediction), cl, 1, CV_AA);
#endif
            if (pt2.m_hasRaw)
            {
#if (CV_VERSION_MAJOR >= 4)
                cv::circle(frame, ResizePoint(pt2.m_prediction), 4, cl, 4, cv::LINE_AA);
#else
                cv::circle(frame, ResizePoint(pt2.m_prediction), 4, cl, 4, CV_AA);
#endif
            }
        }
    }
}

///
/// \brief AsyncDetector::DrawData
/// \param frameinfo
///
void AsyncDetector::DrawData(FrameInfo* frameInfo, int framesCounter, int currTime)
{
    if (m_showLogs)
    {
        std::cout << "Frame " << framesCounter << ": tracks = " << frameInfo->m_tracks.size() << ", time = " << currTime << std::endl;
    }


    for (const auto& track : frameInfo->m_tracks)
    {
        if (track.m_isStatic)
        {
            DrawTrack(frameInfo->m_frame, 1, track, true);
        }
        else
        {
            if (track.IsRobust(5,          // Minimal trajectory size
                               -1.f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                               cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
				//std::cout << track.m_type << " - " << track.m_rect << std::endl;

                DrawTrack(frameInfo->m_frame, 1, track, true);

				std::string label = track.m_type + ": " + std::to_string(track.m_confidence);
				int baseLine = 0;
				cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

#if (CV_VERSION_MAJOR >= 4)
				cv::rectangle(frameInfo->m_frame, cv::Rect(cv::Point(track.m_rect.x, track.m_rect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 255, 255), cv::FILLED);
#else
				cv::rectangle(frameInfo->m_frame, cv::Rect(cv::Point(track.m_rect.x, track.m_rect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 255, 255), CV_FILLED);
#endif
				cv::putText(frameInfo->m_frame, label, cv::Point(track.m_rect.x, track.m_rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
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
    {
        capture.open(atoi(fileName.c_str()));
    }
    else
    {
        capture.open(fileName);
    }
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

    detectorConfig["modelConfiguration"] = pathToModel + "yolov3-tiny.cfg";
    detectorConfig["modelBinary"] = pathToModel + "yolov3-tiny.weights";
    detectorConfig["classNames"] = pathToModel + "coco.names";
    detectorConfig["confidenceThreshold"] = "0.1";
    detectorConfig["maxCropRatio"] = "2.0";

    // Tracker
    const int minStaticTime = 5;

    TrackerSettings trackerSettings;
    trackerSettings.m_useLocalTracking = false;
    trackerSettings.m_distType = tracking::DistCenters;
    trackerSettings.m_kalmanType = tracking::KalmanLinear;
    trackerSettings.m_filterGoal = tracking::FilterRect;
    trackerSettings.m_lostTrackType = tracking::TrackSTAPLE; // Use KCF tracker for collisions resolving
    trackerSettings.m_matchType = tracking::MatchHungrian;
    trackerSettings.m_dt = 0.5f;                             // Delta time for Kalman filter
    trackerSettings.m_accelNoiseMag = 0.5f;                  // Accel noise magnitude for Kalman filter
    trackerSettings.m_distThres = frameHeight / 10.f;         // Distance threshold between region and object on two frames

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
    cv::Mat firstFrame;
    cv::UMat firstGray;
    capture >> firstFrame;
    cv::cvtColor(firstFrame, firstGray, cv::COLOR_BGR2GRAY);

    std::thread thDetection(DetectThread, detectorConfig, firstGray, framesQue, stopFlag);
    thDetection.detach();
    std::thread thTracking(TrackingThread, trackerSettings, framesQue, stopFlag);
    thTracking.detach();

    // Capture frame
    for (; !(*stopFlag);)
    {
        FrameInfo frameInfo;
        frameInfo.m_dt = cv::getTickCount();;
        capture >> frameInfo.m_frame;
        if (frameInfo.m_frame.empty())
        {
            std::cerr << "Frame is empty!" << std::endl;
            *stopFlag = true;
            break;
        }
		if (frameInfo.m_clFrame.empty())
		{
			frameInfo.m_clFrame = frameInfo.m_frame.getUMat(cv::ACCESS_READ);
		}
        cv::cvtColor(frameInfo.m_frame, frameInfo.m_gray, cv::COLOR_BGR2GRAY);

        framesQue->AddNewFrame(frameInfo);

        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / cvRound(*fps)));
    }

    if (thDetection.joinable())
    {
        thDetection.join();
    }
    if (thTracking.joinable())
    {
        thTracking.join();
    }
}

///
/// \brief AsyncDetector::DetectThread
/// \param
///
void AsyncDetector::DetectThread(const config_t& config, cv::UMat firstGray, FramesQueue* framesQue, bool* stopFlag)
{
    std::unique_ptr<BaseDetector> detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Yolo_Darknet, config, false, firstGray));
    detector->SetMinObjectSize(cv::Size(firstGray.cols / 50, firstGray.cols / 50));

    for (; !(*stopFlag);)
    {
        FrameInfo& frameInfo = framesQue->GetLastUndetectedFrame();

        detector->Detect(frameInfo.m_clFrame);
        //std::this_thread::sleep_for(std::chrono::milliseconds(100));

        const regions_t& regions = detector->GetDetects();
        frameInfo.m_regions.assign(regions.begin(), regions.end());

        frameInfo.m_inDetector = 2;
        framesQue->Signal(frameInfo.m_dt);
    }
}

///
/// \brief AsyncDetector::TrackingThread
/// \param
///
void AsyncDetector::TrackingThread(const TrackerSettings& settings, FramesQueue* framesQue, bool* stopFlag)
{
    std::unique_ptr<CTracker> tracker = std::make_unique<CTracker>(settings);

    for (; !(*stopFlag);)
    {
        FrameInfo& frameInfo = framesQue->GetFirstDetectedFrame();
        if (tracker->GrayFrameToTrack())
        {
            tracker->Update(frameInfo.m_regions, frameInfo.m_gray, frameInfo.m_fps);
        }
        else
        {
            tracker->Update(frameInfo.m_regions, frameInfo.m_clFrame, frameInfo.m_fps);
        }

        frameInfo.m_tracks = tracker->GetTracks();
        frameInfo.m_inTracker = 2;
        framesQue->Signal(frameInfo.m_dt);
    }
}
