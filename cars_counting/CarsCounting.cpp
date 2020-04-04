#include "CarsCounting.h"

///
/// \brief CarsCounting::CarsCounting
/// \param parser
///
CarsCounting::CarsCounting(const cv::CommandLineParser& parser)
    :
      m_showLogs(true),
      m_fps(25),
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

#ifndef SILENT_WORK
    cv::namedWindow("Video", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
#endif

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

    cv::Mat colorFrame;
	capture >> colorFrame;
	if (colorFrame.empty())
	{
		std::cerr << "Frame is empty!" << std::endl;
		return;
	}
	if (!m_isTrackerInitialized)
	{
		cv::UMat uframe = colorFrame.getUMat(cv::ACCESS_READ);
		m_isTrackerInitialized = InitTracker(uframe);
		if (!m_isTrackerInitialized)
		{
			std::cerr << "Tracker initialize error!!!" << std::endl;
			return;
		}
	}

    for (;;)
    {
        capture >> colorFrame;
        if (colorFrame.empty())
        {
            std::cerr << "Frame is empty!" << std::endl;
            break;
        }

        int64 t1 = cv::getTickCount();

        cv::UMat uframe;
        if (!m_detector->CanGrayProcessing() || m_tracker->CanColorFrameToTrack())
        {
            uframe = colorFrame.getUMat(cv::ACCESS_READ);
        }
		else
		{
			cv::cvtColor(colorFrame, uframe, cv::COLOR_BGR2GRAY);
		}

        m_detector->Detect(uframe);

        const regions_t& regions = m_detector->GetDetects();

        m_tracker->Update(regions, uframe, m_fps);

        int64 t2 = cv::getTickCount();

        allTime += t2 - t1;
        int currTime = cvRound(1000 * (t2 - t1) / freq);

        DrawData(colorFrame, framesCounter, currTime);

#ifndef SILENT_WORK
        cv::imshow("Video", colorFrame);

		int waitTime = manualMode ? 0 : 1;// std::max<int>(1, cvRound(1000 / m_fps - currTime));
        k = cv::waitKey(waitTime);
        if (k == 'm' || k == 'M')
        {
            manualMode = !manualMode;
        }
        else if (k == 27)
        {
            break;
        }
#else
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif

		if (!m_outFile.empty() && !writer.isOpened())
		{
			writer.open(m_outFile, cv::VideoWriter::fourcc('H', 'F', 'Y', 'U'), m_fps, colorFrame.size(), true);
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
#ifndef SILENT_WORK
	cv::waitKey(m_finishDelay);
#endif
}

///
/// \brief CarsCounting::DrawTrack
/// \param frame
/// \param resizeCoeff
/// \param track
/// \param drawTrajectory
///
void CarsCounting::DrawTrack(cv::Mat frame,
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
        cv::rectangle(frame, ResizeRect(track.m_rrect.boundingRect()), cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
#else
        cv::rectangle(frame, ResizeRect(track.m_rrect.boundingRect()), cv::Scalar(255, 0, 255), 2, CV_AA);
#endif
    }
    else
    {
#if (CV_VERSION_MAJOR >= 4)
        cv::rectangle(frame, ResizeRect(track.m_rrect.boundingRect()), cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
#else
        cv::rectangle(frame, ResizeRect(track.m_rrect.boundingRect()), cv::Scalar(0, 255, 0), 1, CV_AA);
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

#if 1 // YOLO
#ifdef _WIN32
	std::string pathToModel = "../../data/";
#else
	std::string pathToModel = "../data/";
#endif

	config.emplace("modelConfiguration", pathToModel + "yolov3.cfg");
	config.emplace("modelBinary", pathToModel + "yolov3.weights");
	config.emplace("confidenceThreshold", "0.7");
	config.emplace("classNames", pathToModel + "coco.names");
	config.emplace("maxCropRatio", "-1");

	config.emplace("white_list", "person");
	config.emplace("white_list", "car");
	config.emplace("white_list", "bicycle");
	config.emplace("white_list", "motorbike");
	config.emplace("white_list", "bus");
	config.emplace("white_list", "truck");
	//config.emplace("white_list", "traffic light");
	//config.emplace("white_list", "stop sign");

	m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Yolo_Darknet, config, frame));

#else // Background subtraction

#if 1
    config.emplace("history", std::to_string(cvRound(10 * minStaticTime * m_fps)));
    config.emplace("varThreshold", "16");
    config.emplace("detectShadows", "1");
    m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Motion_MOG2, config, frame));
#else
    config.emplace("minPixelStability", "15");
    config.emplace("maxPixelStability", "900");
    config.emplace("useHistory", "1");
    config.emplace("isParallel", "1");
    m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Motion_CNT, config, m_useLocalTracking, frame));
#endif

#endif

    m_detector->SetMinObjectSize(cv::Size(m_minObjWidth, m_minObjWidth));

    TrackerSettings settings;
	settings.SetDistance(tracking::DistJaccard);
    settings.m_kalmanType = tracking::KalmanLinear;
    settings.m_filterGoal = tracking::FilterCenter;
    settings.m_lostTrackType = tracking::TrackCSRT; // Use KCF tracker for collisions resolving
    settings.m_matchType = tracking::MatchHungrian;
    settings.m_dt = 0.3f;                           // Delta time for Kalman filter
    settings.m_accelNoiseMag = 0.2f;                // Accel noise magnitude for Kalman filter
    settings.m_distThres = 0.7f;                    // Distance threshold between region and object on two frames
    settings.m_minAreaRadius = frame.rows / 20.f;
	settings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
	settings.m_maxTraceLength = cvRound(3 * m_fps);      // Maximum trace length

	settings.AddNearTypes("car", "bus", false);
	settings.AddNearTypes("car", "truck", false);
	settings.AddNearTypes("person", "bicycle", true);
	settings.AddNearTypes("person", "motorbike", true);


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
	auto tracks = m_tracker->GetTracks();

    if (m_showLogs)
    {
        std::cout << "Frame " << framesCounter << ": tracks = " << tracks.size() << ", time = " << currTime << std::endl;
    }

    std::set<size_t> currIntersections;
	
    for (const auto& track : tracks)
    {
        if (track.m_isStatic)
        {
            DrawTrack(frame, 1, track, true);
        }
        else
        {
            if (track.IsRobust(cvRound(m_fps / 4),          // Minimal trajectory size
                                0.8f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, track, true);

                CheckLinesIntersection(track, static_cast<float>(frame.cols), static_cast<float>(frame.rows), currIntersections);
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
void CarsCounting::CheckLinesIntersection(const TrackingObject& track, float xMax, float yMax, std::set<size_t>& currIntersections)
{
    auto Pti2f = [&](cv::Point pt) -> cv::Point2f
    {
        return cv::Point2f(pt.x / xMax, pt.y / yMax);
    };

    for (auto& rl : m_lines)
    {
        if (m_lastIntersections.find(track.m_ID) == m_lastIntersections.end())
        {
            if (rl.IsIntersect(Pti2f(track.m_trace[track.m_trace.size() - 3]), Pti2f(track.m_trace[track.m_trace.size() - 1])))
            {
                currIntersections.insert(track.m_ID);
            }
        }
    }
}
