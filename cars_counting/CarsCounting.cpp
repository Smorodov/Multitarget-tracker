#include "CarsCounting.h"

///
/// \brief CarsCounting::CarsCounting
/// \param parser
///
CarsCounting::CarsCounting(const cv::CommandLineParser& parser)
    :
      m_showLogs(true),
      m_fps(25),
      m_useLocalTracking(false),
      m_captureTimeOut(60000),
      m_trackingTimeOut(60000),
      m_isTrackerInitialized(false),
      m_startFrame(0),
      m_endFrame(0),
      m_finishDelay(0),
      m_currFrame(0)
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
/// \brief CarsCounting::~CarsCounting
///
CarsCounting::~CarsCounting()
{

}

///
/// \brief CarsCounting::Process
///
void CarsCounting::Process()
{
    m_currFrame = 1;
    bool stopCapture = false;
    std::mutex frameLock;
    std::condition_variable frameCond;

    std::mutex trackLock;
    std::condition_variable trackCond;
    std::thread thCapDet(CaptureAndDetect, this, &stopCapture, &frameLock, &frameCond, &trackLock, &trackCond);
    thCapDet.detach();

    {
        std::unique_lock<std::mutex> lock(frameLock);
        auto now = std::chrono::system_clock::now();
        if (frameCond.wait_until(lock, now + std::chrono::milliseconds(m_captureTimeOut)) == std::cv_status::timeout)
        {
            std::cerr << "Process: Init capture timeout" << std::endl;
            stopCapture = true;

            if (thCapDet.joinable())
            {
                thCapDet.join();
            }
            return;
        }
    }

    cv::VideoWriter writer;

    cv::namedWindow("Video", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);

    int k = 0;

    double freq = cv::getTickFrequency();

    int64 allTime = 0;

    bool manualMode = false;
    int framesCounter = m_startFrame + 1;

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    trackCond.notify_all();

    int currFrame = 0;
    for (; !stopCapture && k != 27; )
    {
        {
            std::unique_lock<std::mutex> lock(frameLock);
            auto now = std::chrono::system_clock::now();
            if (frameCond.wait_until(lock, now + std::chrono::milliseconds(m_captureTimeOut)) == std::cv_status::timeout)
            {
                std::cerr << "Process: Frame capture timeout" << std::endl;
                break;
            }
        }
        if (stopCapture)
        {
            break;
        }

        frameLock.lock();
        currFrame = m_currFrame;
        frameLock.unlock();

        if (!writer.isOpened())
        {
            writer.open(m_outFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), m_fps, m_frameInfo[currFrame].m_frame.size(), true);
        }

        int64 t1 = cv::getTickCount();

        Tracking(m_frameInfo[currFrame].m_frame, m_frameInfo[currFrame].m_gray, m_frameInfo[currFrame].m_regions);

        int64 t2 = cv::getTickCount();

        allTime += t2 - t1 + m_frameInfo[currFrame].m_dt;
        int currTime = cvRound(1000 * (t2 - t1 + m_frameInfo[currFrame].m_dt) / freq);

        DrawData(m_frameInfo[currFrame].m_frame, framesCounter, currTime);

        cv::imshow("Video", m_frameInfo[currFrame].m_frame);

        int waitTime = manualMode ? 0 : std::max<int>(1, cvRound(1000 / m_fps - currTime));
        k = cv::waitKey(waitTime);
        if (k == 'm' || k == 'M')
        {
            manualMode = !manualMode;
        }

        if (writer.isOpened())
        {
            writer << m_frameInfo[currFrame].m_frame;
        }

        trackCond.notify_all();

        ++framesCounter;
        if (m_endFrame && framesCounter > m_endFrame)
        {
            std::cout << "Process: riched last " << m_endFrame << " frame" << std::endl;
            break;
        }
    }
    stopCapture = true;

    if (thCapDet.joinable())
    {
        thCapDet.join();
    }

    std::cout << "work time = " << (allTime / freq) << std::endl;
    cv::waitKey(m_finishDelay);
}

///
/// \brief CarsCounting::CaptureAndDetect
/// \param thisPtr
/// \param stopCapture
/// \param frameLock
/// \param frameCond
/// \param trackLock
/// \param trackCond
///
void CarsCounting::CaptureAndDetect(CarsCounting* thisPtr,
                                    bool* stopCapture,
                                    std::mutex* frameLock,
                                    std::condition_variable* frameCond,
                                    std::mutex* trackLock,
                                    std::condition_variable* trackCond)
{
    cv::VideoCapture capture;

    if (thisPtr->m_inFile.size() == 1)
    {
        capture.open(atoi(thisPtr->m_inFile.c_str()));
    }
    else
    {
        capture.open(thisPtr->m_inFile);
    }

    if (!capture.isOpened())
    {
        std::cerr << "Can't open " << thisPtr->m_inFile << std::endl;
        return;
    }

    capture.set(cv::CAP_PROP_POS_FRAMES, thisPtr->m_startFrame);

    thisPtr->m_fps = std::max(1.f, (float)capture.get(cv::CAP_PROP_FPS));

    frameCond->notify_all();

    int currFrame = 0;
    for (; !(*stopCapture);)
    {
        {
            std::unique_lock<std::mutex> lock(*trackLock);
            auto now = std::chrono::system_clock::now();
            if (trackCond->wait_until(lock, now + std::chrono::milliseconds(thisPtr->m_trackingTimeOut)) == std::cv_status::timeout)
            {
                std::cerr << "CaptureAndDetect: Tracking timeout!" << std::endl;
                break;
            }
        }
        frameLock->lock();
        currFrame = thisPtr->m_currFrame ? 0 : 1;
        frameLock->unlock();

        capture >> thisPtr->m_frameInfo[currFrame].m_frame;
        if (thisPtr->m_frameInfo[currFrame].m_frame.empty())
        {
            std::cerr << "CaptureAndDetect: frame is empty!" << std::endl;
            break;
        }
        cv::cvtColor(thisPtr->m_frameInfo[currFrame].m_frame, thisPtr->m_frameInfo[currFrame].m_gray, cv::COLOR_BGR2GRAY);

        if (!thisPtr->m_isTrackerInitialized)
        {
            thisPtr->m_isTrackerInitialized = thisPtr->InitTracker(thisPtr->m_frameInfo[currFrame].m_gray);
            if (!thisPtr->m_isTrackerInitialized)
            {
                std::cerr << "CaptureAndDetect: Tracker initilize error!!!" << std::endl;
                break;
            }
        }

        int64 t1 = cv::getTickCount();
        thisPtr->Detection(thisPtr->m_frameInfo[currFrame].m_frame, thisPtr->m_frameInfo[currFrame].m_gray, thisPtr->m_frameInfo[currFrame].m_regions);
        int64 t2 = cv::getTickCount();
        thisPtr->m_frameInfo[currFrame].m_dt = t2 - t1;

        frameLock->lock();
        thisPtr->m_currFrame = thisPtr->m_currFrame ? 0 : 1;
        frameLock->unlock();
        frameCond->notify_all();
    }

    *stopCapture = true;
    frameCond->notify_all();
}

///
/// \brief CarsCounting::GrayProcessing
/// \return
///
bool CarsCounting::GrayProcessing() const
{
    return true;
}

///
/// \brief CarsCounting::Detection
/// \param frame
/// \param grayFrame
/// \param regions
///
void CarsCounting::Detection(cv::Mat frame, cv::UMat grayFrame, regions_t& regions)
{
    cv::UMat clFrame;
    if (!GrayProcessing() || !m_tracker->GrayFrameToTrack())
    {
        clFrame = frame.getUMat(cv::ACCESS_READ);
    }

    m_detector->Detect(GrayProcessing() ? grayFrame : clFrame);

    const regions_t& regs = m_detector->GetDetects();

    regions.assign(std::begin(regs), std::end(regs));
}

///
/// \brief CarsCounting::Tracking
/// \param frame
/// \param grayFrame
/// \param regions
///
void CarsCounting::Tracking(cv::Mat frame, cv::UMat grayFrame, const regions_t& regions)
{
    cv::UMat clFrame;
    if (!GrayProcessing() || !m_tracker->GrayFrameToTrack())
    {
        clFrame = frame.getUMat(cv::ACCESS_READ);
    }

    m_tracker->Update(regions, m_tracker->GrayFrameToTrack() ? grayFrame : clFrame, m_fps);
}

///
/// \brief CarsCounting::DrawTrack
/// \param frame
/// \param resizeCoeff
/// \param track
/// \param drawTrajectory
/// \param isStatic
///
void CarsCounting::DrawTrack(cv::Mat frame,
                             int resizeCoeff,
                             const CTrack& track,
                             bool drawTrajectory,
                             bool isStatic
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

    if (isStatic)
    {
        cv::rectangle(frame, ResizeRect(track.GetLastRect()), cv::Scalar(255, 0, 255), 2, CV_AA);
    }
    else
    {
        cv::rectangle(frame, ResizeRect(track.GetLastRect()), cv::Scalar(0, 255, 0), 1, CV_AA);
    }

    if (drawTrajectory)
    {
        cv::Scalar cl = m_colors[track.m_trackID % m_colors.size()];

        for (size_t j = 0; j < track.m_trace.size() - 1; ++j)
        {
            const TrajectoryPoint& pt1 = track.m_trace.at(j);
            const TrajectoryPoint& pt2 = track.m_trace.at(j + 1);

            cv::line(frame, ResizePoint(pt1.m_prediction), ResizePoint(pt2.m_prediction), cl, 1, CV_AA);
            if (!pt2.m_hasRaw)
            {
                cv::circle(frame, ResizePoint(pt2.m_prediction), 4, cl, 1, CV_AA);
            }
        }
    }

    if (m_useLocalTracking)
    {
        cv::Scalar cl = m_colors[track.m_trackID % m_colors.size()];

        for (auto pt : track.m_lastRegion.m_points)
        {
            cv::circle(frame, cv::Point(cvRound(pt.x), cvRound(pt.y)), 1, cl, -1, CV_AA);
        }
    }
}

///
/// \brief CarsCounting::InitTracker
/// \param grayFrame
///
bool CarsCounting::InitTracker(cv::UMat frame)
{
    m_useLocalTracking = false;

    m_minObjWidth = frame.cols / 50;

    const int minStaticTime = 5;

    config_t config;
#if 1
    config["history"] = std::to_string(cvRound(10 * minStaticTime * m_fps));
    config["varThreshold"] = "16";
    config["detectShadows"] = "1";
    m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Motion_MOG2, config, m_useLocalTracking, frame));
#else
    config["minPixelStability"] = "15";
    config["maxPixelStability"] = "900";
    config["useHistory"] = "1";
    config["isParallel"] = "1";
    m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Motion_CNT, config, m_useLocalTracking, frame));
#endif
    m_detector->SetMinObjectSize(cv::Size(m_minObjWidth, m_minObjWidth));

    TrackerSettings settings;
    settings.m_useLocalTracking = m_useLocalTracking;
    settings.m_distType = tracking::DistCenters;
    settings.m_kalmanType = tracking::KalmanLinear;
    settings.m_filterGoal = tracking::FilterRect;
    settings.m_lostTrackType = tracking::TrackKCF;    // Use KCF tracker for collisions resolving
    settings.m_matchType = tracking::MatchHungrian;
    settings.m_dt = 0.4f;                             // Delta time for Kalman filter
    settings.m_accelNoiseMag = 0.5f;                  // Accel noise magnitude for Kalman filter
    settings.m_distThres = frame.rows / 20;           // Distance threshold between region and object on two frames

    settings.m_useAbandonedDetection = false;
    if (settings.m_useAbandonedDetection)
    {
        settings.m_minStaticTime = minStaticTime;
        settings.m_maxStaticTime = 60;
        settings.m_maximumAllowedSkippedFrames = settings.m_minStaticTime * m_fps; // Maximum allowed skipped frames
        settings.m_maxTraceLength = 2 * settings.m_maximumAllowedSkippedFrames;        // Maximum trace length
    }
    else
    {
        settings.m_maximumAllowedSkippedFrames = 2 * m_fps; // Maximum allowed skipped frames
        settings.m_maxTraceLength = 4 * m_fps;              // Maximum trace length
    }

    m_tracker = std::make_unique<CTracker>(settings);

    return true;
}

///
/// \brief CarsCounting::DrawData
/// \param frame
///
void CarsCounting::DrawData(cv::Mat frame, int framesCounter, int currTime)
{
    if (m_showLogs)
    {
        std::cout << "Frame " << framesCounter << ": tracks = " << m_tracker->tracks.size() << ", time = " << currTime << std::endl;
    }

    for (const auto& track : m_tracker->tracks)
    {
        if (track->IsStatic())
        {
            DrawTrack(frame, 1, *track, true, true);
        }
        else
        {
            if (track->IsRobust(cvRound(m_fps / 4),          // Minimal trajectory size
                                0.7f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, *track, true);
            }
        }
    }

    m_detector->CalcMotionMap(frame);
}
