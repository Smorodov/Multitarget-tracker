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
      m_isTrackerInitialized(false),
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
    cv::VideoWriter writer;

    cv::namedWindow("Video", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);

    int k = 0;

    double freq = cv::getTickFrequency();

    int64 allTime = 0;

    bool manualMode = false;
    int framesCounter = m_startFrame + 1;

    cv::VideoCapture capture;
    if (m_inFile.size() == 1)
    {
        capture.open(atoi(m_inFile.c_str()));
    }
    else
    {
        capture.open(m_inFile);
    }
    if (!capture.isOpened())
    {
        std::cerr << "Can't open " << m_inFile << std::endl;
        return;
    }
    capture.set(cv::CAP_PROP_POS_FRAMES, m_startFrame);

    m_fps = std::max(1.f, (float)capture.get(cv::CAP_PROP_FPS));

    m_fps = std::max(1.f, (float)capture.get(cv::CAP_PROP_FPS));

    cv::Mat colorFrame;
    cv::UMat grayFrame;
    for (;;)
    {
        capture >> colorFrame;
        if (colorFrame.empty())
        {
            std::cerr << "Frame is empty!" << std::endl;
            break;
        }
        cv::cvtColor(colorFrame, grayFrame, cv::COLOR_BGR2GRAY);

        if (!m_isTrackerInitialized)
        {
            m_isTrackerInitialized = InitTracker(grayFrame);
            if (!m_isTrackerInitialized)
            {
                std::cerr << "Tracker initilize error!!!" << std::endl;
                break;
            }
        }

        if (!writer.isOpened())
        {
            writer.open(m_outFile, cv::VideoWriter::fourcc('H', 'F', 'Y', 'U'), m_fps, colorFrame.size(), true);
        }

        int64 t1 = cv::getTickCount();

        cv::UMat clFrame;
        if (!GrayProcessing() || !m_tracker->GrayFrameToTrack())
        {
            clFrame = colorFrame.getUMat(cv::ACCESS_READ);
        }

        m_detector->Detect(GrayProcessing() ? grayFrame : clFrame);

        const regions_t& regions = m_detector->GetDetects();

        m_tracker->Update(regions, m_tracker->GrayFrameToTrack() ? grayFrame : clFrame, m_fps);

        int64 t2 = cv::getTickCount();

        allTime += t2 - t1;
        int currTime = cvRound(1000 * (t2 - t1) / freq);

        DrawData(colorFrame, framesCounter, currTime);

        cv::imshow("Video", colorFrame);

        int waitTime = manualMode ? 0 : std::max<int>(1, cvRound(1000 / m_fps - currTime));
        k = cv::waitKey(waitTime);
        if (k == 'm' || k == 'M')
        {
            manualMode = !manualMode;
        }
        else if (k == 27)
        {
            break;
        }

        if (writer.isOpened())
        {
            writer << colorFrame;
        }

        ++framesCounter;
        if (m_endFrame && framesCounter > m_endFrame)
        {
            std::cout << "Process: riched last " << m_endFrame << " frame" << std::endl;
            break;
        }
    }

    std::cout << "work time = " << (allTime / freq) << std::endl;
    cv::waitKey(m_finishDelay);
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
#if (CV_VERSION_MAJOR >= 4)
        cv::rectangle(frame, ResizeRect(track.GetLastRect()), cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
#else
		cv::rectangle(frame, ResizeRect(track.GetLastRect()), cv::Scalar(255, 0, 255), 2, CV_AA);
#endif
    }
    else
    {
#if (CV_VERSION_MAJOR >= 4)
        cv::rectangle(frame, ResizeRect(track.GetLastRect()), cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
#else
		cv::rectangle(frame, ResizeRect(track.GetLastRect()), cv::Scalar(0, 255, 0), 1, CV_AA);
#endif
    }

    if (drawTrajectory)
    {
        cv::Scalar cl = m_colors[track.m_trackID % m_colors.size()];

        for (size_t j = 0; j < track.m_trace.size() - 1; ++j)
        {
            const TrajectoryPoint& pt1 = track.m_trace.at(j);
            const TrajectoryPoint& pt2 = track.m_trace.at(j + 1);
#if (CV_VERSION_MAJOR >= 4)
            cv::line(frame, ResizePoint(pt1.m_prediction), ResizePoint(pt2.m_prediction), cl, 1, cv::LINE_AA);
#else
			cv::line(frame, ResizePoint(pt1.m_prediction), ResizePoint(pt2.m_prediction), cl, 1, CV_AA);
#endif
            if (!pt2.m_hasRaw)
            {
#if (CV_VERSION_MAJOR >= 4)
                cv::circle(frame, ResizePoint(pt2.m_prediction), 4, cl, 1, cv::LINE_AA);
#else
				cv::circle(frame, ResizePoint(pt2.m_prediction), 4, cl, 1, CV_AA);
#endif
            }
        }
    }

    if (m_useLocalTracking)
    {
        cv::Scalar cl = m_colors[track.m_trackID % m_colors.size()];

        for (auto pt : track.m_lastRegion.m_points)
        {
#if (CV_VERSION_MAJOR >= 4)
            cv::circle(frame, cv::Point(cvRound(pt.x), cvRound(pt.y)), 1, cl, -1, cv::LINE_AA);
#else
			cv::circle(frame, cv::Point(cvRound(pt.x), cvRound(pt.y)), 1, cl, -1, CV_AA);
#endif
        }
    }
}

///
/// \brief CarsCounting::InitTracker
/// \param grayFrame
///
bool CarsCounting::InitTracker(cv::UMat frame)
{
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
    settings.m_lostTrackType = tracking::TrackCSRT; // Use KCF tracker for collisions resolving
    settings.m_matchType = tracking::MatchHungrian;
    settings.m_dt = 0.5f;                             // Delta time for Kalman filter
    settings.m_accelNoiseMag = 0.5f;                  // Accel noise magnitude for Kalman filter
    settings.m_distThres = frame.rows / 15.f;         // Distance threshold between region and object on two frames

    settings.m_useAbandonedDetection = false;
    if (settings.m_useAbandonedDetection)
    {
        settings.m_minStaticTime = minStaticTime;
        settings.m_maxStaticTime = 60;
        settings.m_maximumAllowedSkippedFrames = cvRound(settings.m_minStaticTime * m_fps); // Maximum allowed skipped frames
        settings.m_maxTraceLength = 2 * settings.m_maximumAllowedSkippedFrames;        // Maximum trace length
    }
    else
    {
        settings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
        settings.m_maxTraceLength = cvRound(4 * m_fps);              // Maximum trace length
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

    std::set<size_t> currIntersections;

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

                CheckLinesIntersection(*track, static_cast<float>(frame.cols), static_cast<float>(frame.rows), currIntersections);
            }
        }
    }

    m_lastIntersections.clear();
    m_lastIntersections = currIntersections;

    //m_detector->CalcMotionMap(frame);

    for (const auto& rl : m_lines)
    {
        rl.Draw(frame);
    }
}

///
/// \brief CarsCounting::AddLine
/// \param newLine
///
void CarsCounting::AddLine(const RoadLine& newLine)
{
    m_lines.push_back(newLine);
}

///
/// \brief CarsCounting::GetLine
/// \param lineUid
/// \return
///
bool CarsCounting::GetLine(unsigned int lineUid, RoadLine& line)
{
    for (const auto& rl : m_lines)
    {
        if (rl.m_uid == lineUid)
        {
            line = rl;
            return true;
        }
    }
    return false;
}

///
/// \brief CarsCounting::RemoveLine
/// \param lineUid
/// \return
///
bool CarsCounting::RemoveLine(unsigned int lineUid)
{
    for (auto it = std::begin(m_lines); it != std::end(m_lines);)
    {
        if (it->m_uid == lineUid)
        {
            it = m_lines.erase(it);
        }
        else
        {
            ++it;
        }
    }
    return false;
}

///
/// \brief CarsCounting::CheckLinesIntersection
/// \param track
///
void CarsCounting::CheckLinesIntersection(const CTrack& track, float xMax, float yMax, std::set<size_t>& currIntersections)
{
    auto Pti2f = [&](cv::Point pt) -> cv::Point2f
    {
        return cv::Point2f(pt.x / xMax, pt.y / yMax);
    };

    for (auto& rl : m_lines)
    {
        if (m_lastIntersections.find(track.m_trackID) == m_lastIntersections.end())
        {
            if (rl.IsIntersect(Pti2f(track.m_trace[track.m_trace.size() - 3]), Pti2f(track.m_trace[track.m_trace.size() - 1])))
            {
                currIntersections.insert(track.m_trackID);
            }
        }
    }
}
