#include <iomanip>
#include <ctime>

#include "VideoExample.h"

///
/// \brief VideoExample::VideoExample
/// \param parser
///
VideoExample::VideoExample(const cv::CommandLineParser& parser)
    : m_resultsLog(parser.get<std::string>("log_res"), parser.get<int>("write_n_frame")),
	m_cvatAnnotationsGenerator(parser.get<std::string>("cvat_res"))
{
    m_inFile = parser.get<std::string>(0);
    m_outFile = parser.get<std::string>("out");
    m_showLogsLevel = parser.get<std::string>("show_logs");
    m_startFrame = parser.get<int>("start_frame");
    m_endFrame = parser.get<int>("end_frame");
    m_finishDelay = parser.get<int>("end_delay");
	m_batchSize = std::max(1, parser.get<int>("batch_size"));
    m_useContrastAdjustment = parser.get<int>("contrast_adjustment") != 0;

    m_colors.emplace_back(255, 0, 0);
    m_colors.emplace_back(0, 255, 0);
    m_colors.emplace_back(0, 0, 255);
    m_colors.emplace_back(255, 255, 0);
    m_colors.emplace_back(0, 255, 255);
    m_colors.emplace_back(255, 0, 255);
    m_colors.emplace_back(255, 127, 255);
    m_colors.emplace_back(127, 0, 255);
    m_colors.emplace_back(127, 0, 127);

    m_resultsLog.Open();

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

    std::string settingsFile = parser.get<std::string>("settings");
    m_trackerSettingsLoaded = ParseTrackerSettings(settingsFile, m_trackerSettings);

	if (m_batchSize > 1)
	{
		m_frameInfo[0].SetBatchSize(m_batchSize);
		m_frameInfo[1].SetBatchSize(m_batchSize);
	}
    for (auto& fr : m_frameInfo[0].m_frames)
    {
        fr.SetUseAdjust(m_useContrastAdjustment);
    }
    for (auto& fr : m_frameInfo[1].m_frames)
    {
        fr.SetUseAdjust(m_useContrastAdjustment);
    }

    m_startTimeStamp = currentTime;
}

///
/// \brief VideoExample::SyncProcess
///
void VideoExample::SyncProcess()
{
    cv::VideoWriter writer;

#ifndef SILENT_WORK
    cv::namedWindow("Video", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    bool manualMode = false;
#endif

    double freq = cv::getTickFrequency();
    int64 allTime = 0;

    int framesCounter = m_startFrame + 1;

    cv::VideoCapture capture;
    if (!OpenCapture(capture))
    {
        m_logger->critical("Can't open {}", m_inFile);
        return;
    }

#if 0
	// Write preview
	cv::Mat prFrame;
	capture >> prFrame;
	cv::Mat textFrame(prFrame.size(), CV_8UC3);
	textFrame = cv::Scalar(0, 0, 0);
	std::string label{ "Original video" };
	int baseLine = 0;
	double fontScale = (textFrame.cols < 1920) ? 2.0 : 3.0;
	int thickness = 2;
	int lineType = cv::LINE_AA;
	int fontFace = cv::FONT_HERSHEY_TRIPLEX;
	cv::Size labelSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseLine);
	cv::putText(textFrame, label, cv::Point(textFrame.cols / 2 - labelSize.width / 2, textFrame.rows / 2 - labelSize.height / 2), fontFace, fontScale, cv::Scalar(255, 255, 255), thickness, lineType);
	for (size_t fi = 0; fi < cvRound(2 * m_fps); ++fi)
	{
		WriteFrame(writer, textFrame);
	}
	WriteFrame(writer, prFrame);
	for (;;)
	{
		capture >> prFrame;
		if (prFrame.empty())
			break;
		WriteFrame(writer, prFrame);
	}
	textFrame = cv::Scalar(0, 0, 0);
	label = "Detection result";
	labelSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseLine);
	cv::putText(textFrame, label, cv::Point(textFrame.cols / 2 - labelSize.width / 2, textFrame.rows / 2 - labelSize.height / 2), fontFace, fontScale, cv::Scalar(255, 255, 255), thickness, lineType);
	for (size_t fi = 0; fi < cvRound(2 * m_fps); ++fi)
	{
		WriteFrame(writer, textFrame);
	}
	capture.release();
	OpenCapture(capture);
#endif

	FrameInfo frameInfo(m_batchSize);
	frameInfo.m_frames.resize(frameInfo.m_batchSize);
	frameInfo.m_frameInds.resize(frameInfo.m_batchSize);
    frameInfo.m_frameTimeStamps.resize(frameInfo.m_batchSize);

    for (auto& fr : frameInfo.m_frames)
    {
        fr.SetUseAdjust(m_useContrastAdjustment);
    }

    int64 startLoopTime = cv::getTickCount();

    for (;;)
    {
		size_t i = 0;
		for (; i < m_batchSize; ++i)
		{
			capture >> frameInfo.m_frames[i].GetMatBGRWrite();
			if (frameInfo.m_frames[i].empty())
				break;
			frameInfo.m_frameInds[i] = framesCounter;
            frameInfo.m_frameTimeStamps[i] = GetNextTimeStamp(framesCounter);
            frameInfo.m_frames[i].AdjustMatBGR();

			++framesCounter;
			if (m_endFrame && framesCounter > m_endFrame)
			{
                m_logger->info("Process: riched last {} frame", m_endFrame);
				break;
			}

            m_logger->debug("VideoExample::SyncProcess: Capture {0} frame", framesCounter);
		}
		if (i < m_batchSize)
			break;

		if (!m_isDetectorInitialized || !m_isTrackerInitialized)
		{
			cv::UMat ufirst = frameInfo.m_frames[0].GetUMatBGR();
			if (!m_isDetectorInitialized)
			{
				m_isDetectorInitialized = InitDetector(ufirst);
				if (!m_isDetectorInitialized)
				{
                    m_logger->critical("CaptureAndDetect: Detector initialize error!!!");
					break;
				}
			}
			if (!m_isTrackerInitialized)
			{
				m_isTrackerInitialized = InitTracker(ufirst);
				if (!m_isTrackerInitialized)
				{
                    m_logger->critical("CaptureAndDetect: Tracker initialize error!!!");
					break;
				}
			}
		}

        int64 t1 = cv::getTickCount();

        Detection(frameInfo);
        Tracking(frameInfo);
        int64 t2 = cv::getTickCount();

        allTime += t2 - t1;
        int currTime = cvRound(1000 * (t2 - t1) / freq);

		for (i = 0; i < m_batchSize; ++i)
		{
			DrawData(frameInfo.m_frames[i].GetMatBGR(), frameInfo.m_tracks[i], frameInfo.m_frameInds[i], currTime);

#ifndef SILENT_WORK
			cv::imshow("Video", frameInfo.m_frames[i].GetMatBGR());

			int waitTime = manualMode ? 0 : 1;// std::max<int>(1, cvRound(1000 / m_fps - currTime));
			int k = cv::waitKey(waitTime);
			if (k == 27)
				break;
			else if (k == 'm' || k == 'M')
				manualMode = !manualMode;
#else
            //std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif

			WriteFrame(writer, frameInfo.m_frames[i].GetMatBGR());
		}
        if (framesCounter % 100 == 0)
            m_resultsLog.Flush();
    }

	m_cvatAnnotationsGenerator.Save(m_inFile, m_framesCount, m_frameSize);

    int64 stopLoopTime = cv::getTickCount();

    m_logger->info("algorithms time = {0}, work time = {1}", allTime / freq, (stopLoopTime - startLoopTime) / freq);
#ifndef SILENT_WORK
    cv::waitKey(m_finishDelay);
#endif
}

///
/// \brief VideoExample::AsyncProcess
///
void VideoExample::AsyncProcess()
{
    std::atomic<bool> stopCapture(false);

    std::thread thCapDet(CaptureAndDetect, this, std::ref(stopCapture));

    cv::VideoWriter writer;

#ifndef SILENT_WORK
    cv::namedWindow("Video", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    bool manualMode = false;
#endif

    double freq = cv::getTickFrequency();

    int64 allTime = 0;
    int64 startLoopTime = cv::getTickCount();
    size_t processCounter = 0;
    for (; !stopCapture.load(); )
    {
        FrameInfo& frameInfo = m_frameInfo[processCounter % 2];
        m_logger->debug("--- waiting tracking from {0} ind = {1}", processCounter % 2, processCounter);
        {
            std::unique_lock<std::mutex> lock(frameInfo.m_mutex);
            if (!frameInfo.m_cond.wait_for(lock, std::chrono::milliseconds(m_captureTimeOut), [&frameInfo] { return frameInfo.m_captured.load(); }))
            {
                m_logger->info("--- Wait frame timeout!");
                break;
            }
        }
        m_logger->debug("--- tracking from {} in progress...", processCounter % 2);
        if (!m_isTrackerInitialized)
        {
            m_isTrackerInitialized = InitTracker(frameInfo.m_frames[0].GetUMatBGR());
            if (!m_isTrackerInitialized)
            {
                m_logger->critical("--- AsyncProcess: Tracker initialize error!!!");
                frameInfo.m_cond.notify_one();
                break;
            }
        }

        int64 t1 = cv::getTickCount();

        Tracking(frameInfo);

        int64 t2 = cv::getTickCount();

        allTime += t2 - t1 + frameInfo.m_dt;
        int currTime = cvRound(1000 * (t2 - t1 + frameInfo.m_dt) / freq);

        m_logger->debug("--- Frame {0}: td = {1}, tt = {2}", frameInfo.m_frameInds[0], 1000 * frameInfo.m_dt / freq, 1000 * (t2 - t1) / freq);

		int key = 0;
		for (size_t i = 0; i < m_batchSize; ++i)
		{
			DrawData(frameInfo.m_frames[i].GetMatBGR(), frameInfo.m_tracks[i], frameInfo.m_frameInds[i], currTime);

			WriteFrame(writer, frameInfo.m_frames[i].GetMatBGR());

#ifndef SILENT_WORK
			cv::imshow("Video", frameInfo.m_frames[i].GetMatBGR());

			int waitTime = manualMode ? 0 : 1;// std::max<int>(1, cvRound(1000 / m_fps - currTime));
			key = cv::waitKey(waitTime);
			if (key == 'm' || key == 'M')
				manualMode = !manualMode;
			else
				break;
#else
			//std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif
		}

        {
            std::unique_lock<std::mutex> lock(frameInfo.m_mutex);
            m_logger->debug("--- tracking m_captured {0} - captured still {1}", processCounter % 2, frameInfo.m_captured.load());
            assert(frameInfo.m_captured.load());
            frameInfo.m_captured = false;
        }
        frameInfo.m_cond.notify_one();

        if (key == 27)
            break;

        ++processCounter;

        if (processCounter % 100 == 0)
            m_resultsLog.Flush();
    }
    stopCapture = true;

    if (thCapDet.joinable())
        thCapDet.join();

	m_cvatAnnotationsGenerator.Save(m_inFile, m_framesCount, m_frameSize);

    int64 stopLoopTime = cv::getTickCount();

    m_logger->info("--- algorithms time = {0}, work time = {1}", allTime / freq, (stopLoopTime - startLoopTime) / freq);

#ifndef SILENT_WORK
    cv::waitKey(m_finishDelay);
#endif
}

///
/// \brief VideoExample::CaptureAndDetect
/// \param thisPtr
/// \param stopCapture
///
void VideoExample::CaptureAndDetect(VideoExample* thisPtr, std::atomic<bool>& stopCapture)
{
    cv::VideoCapture capture;
    if (!thisPtr->OpenCapture(capture))
    {
        thisPtr->m_logger->critical("+++ Can't open {}", thisPtr->m_inFile);
        stopCapture = true;
        return;
    }

	int framesCounter = 0;

    const auto localEndFrame = thisPtr->m_endFrame;
    auto localIsDetectorInitialized = thisPtr->m_isDetectorInitialized;
    auto localTrackingTimeOut = thisPtr->m_trackingTimeOut;
    size_t processCounter = 0;
    for (; !stopCapture.load();)
    {
        FrameInfo& frameInfo = thisPtr->m_frameInfo[processCounter % 2];
        thisPtr->m_logger->debug("+++ waiting capture to {0}, ind = {1}", processCounter % 2, processCounter);
        {
            std::unique_lock<std::mutex> lock(frameInfo.m_mutex);
            if (!frameInfo.m_cond.wait_for(lock, std::chrono::milliseconds(localTrackingTimeOut), [&frameInfo] { return !frameInfo.m_captured.load(); }))
            {
                thisPtr->m_logger->info("+++ Wait tracking timeout!");
                frameInfo.m_cond.notify_one();
                break;
            }
        }
        thisPtr->m_logger->debug("+++ capture to {0} in progress...", processCounter % 2);
		if (frameInfo.m_frames.size() < frameInfo.m_batchSize)
		{
			frameInfo.m_frames.resize(frameInfo.m_batchSize);
			frameInfo.m_frameInds.resize(frameInfo.m_batchSize);
            frameInfo.m_frameTimeStamps.resize(frameInfo.m_batchSize);
		}

        cv::Mat frame;
		size_t i = 0;
		for (; i < frameInfo.m_batchSize; ++i)
		{
			capture >> frame;
			if (frame.empty())
			{
                thisPtr->m_logger->error("+++ CaptureAndDetect: frame is empty!");
				frameInfo.m_cond.notify_one();
				break;
			}
            frameInfo.m_frames[i].GetMatBGRWrite() = frame;
            frameInfo.m_frames[i].AdjustMatBGR();
            frameInfo.m_frameInds[i] = framesCounter;
            frameInfo.m_frameTimeStamps[i] = thisPtr->GetNextTimeStamp(framesCounter);
			++framesCounter;

            if (localEndFrame && framesCounter > localEndFrame)
            {
                thisPtr->m_logger->info("+++ Process: riched last {} frame", localEndFrame);
                break;
            }
        }
        if (i < frameInfo.m_batchSize)
            break;

        if (!localIsDetectorInitialized)
        {
            thisPtr->m_isDetectorInitialized = thisPtr->InitDetector(frameInfo.m_frames[0].GetUMatBGR());
            localIsDetectorInitialized = thisPtr->m_isDetectorInitialized;
            if (!thisPtr->m_isDetectorInitialized)
            {
                thisPtr->m_logger->critical("+++ CaptureAndDetect: Detector initialize error!!!");
                frameInfo.m_cond.notify_one();
                break;
            }
        }

        int64 t1 = cv::getTickCount();
        thisPtr->Detection(frameInfo);
        int64 t2 = cv::getTickCount();
        frameInfo.m_dt = t2 - t1;

        {
            std::unique_lock<std::mutex> lock(frameInfo.m_mutex);
            thisPtr->m_logger->debug("+++ capture m_captured {0} - captured still {1}", processCounter % 2, frameInfo.m_captured.load());
            assert(!frameInfo.m_captured.load());
            frameInfo.m_captured = true;
        }
        frameInfo.m_cond.notify_one();

		++processCounter;
    }
    stopCapture = true;
}

///
/// \brief VideoExample::GetNextTimeStamp
/// \param framesCounter
/// \return
///
time_point_t VideoExample::GetNextTimeStamp(int framesCounter) const
{
    if (m_useArchieveTime)
        return m_startTimeStamp + std::chrono::milliseconds(cvRound(framesCounter * (1000.f / m_fps)));
    else
        return std::chrono::system_clock::now();
}

///
/// \brief VideoExample::Detection
/// \param frame
/// \param regions
///
void VideoExample::Detection(FrameInfo& frame)
{
	if (m_trackerSettings.m_useAbandonedDetection)
	{
		for (const auto& track : m_tracks)
		{
			if (track.m_isStatic)
				m_detector->ResetModel(frame.m_frames[0].GetUMatBGR(), track.m_rrect.boundingRect());
		}
	}

    std::vector<cv::UMat> frames;
	for (size_t i = 0; i < frame.m_frames.size(); ++i)
	{
        if (m_detector->CanGrayProcessing())
            frames.emplace_back(frame.m_frames[i].GetUMatGray());
        else
            frames.emplace_back(frame.m_frames[i].GetUMatBGR());
	}
	frame.CleanRegions();
    m_detector->Detect(frames, frame.m_regions);
}

///
/// \brief VideoExample::Tracking
/// \param frame
/// \param regions
///
void VideoExample::Tracking(FrameInfo& frame)
{
	assert(frame.m_regions.size() == frame.m_frames.size());

	frame.CleanTracks();
	for (size_t i = 0; i < frame.m_frames.size(); ++i)
	{
		m_tracker->Update(frame.m_regions[i], frame.m_frames[i].GetUMatBGR(), frame.m_frameTimeStamps[i]);
		m_tracker->GetTracks(frame.m_tracks[i]);

		m_cvatAnnotationsGenerator.NewDetects(frame.m_frameInds[i], frame.m_tracks[i], 0);
	}
	if (m_trackerSettings.m_useAbandonedDetection)
		m_tracker->GetTracks(m_tracks);
}

///
/// \brief VideoExample::DrawTrack
/// \param frame
/// \param track
/// \param drawTrajectory
///
void VideoExample::DrawTrack(cv::Mat frame,
                             const TrackingObject& track,
                             bool drawTrajectory,
                             int framesCounter,
                             const std::string& userLabel)
{
    cv::Scalar color = track.m_isStatic ? cv::Scalar(255, 0, 255) : cv::Scalar(0, 255, 0);
    cv::Point2f rectPoints[4];
    track.m_rrect.points(rectPoints);
    //std::cout << "track: rrect [" << track.m_rrect.size << " from " << track.m_rrect.center << ", " << track.m_rrect.angle << "]" << std::endl;
    for (int i = 0; i < 4; ++i)
    {
        cv::line(frame, rectPoints[i], rectPoints[(i+1) % 4], color);
    }

#if 0
#if 0
	track_t minAreaRadiusPix = frame.rows / 20.f;
#else
	track_t minAreaRadiusPix = -1.f;
#endif
	track_t minAreaRadiusK = 0.5f;
	cv::Size_<track_t> minRadius(minAreaRadiusPix, minAreaRadiusPix);
	if (minAreaRadiusPix < 0)
	{
		minRadius.width = minAreaRadiusK * track.m_rrect.size.width;
		minRadius.height = minAreaRadiusK * track.m_rrect.size.height;
	}

	Point_t d(3.f * track.m_velocity[0], 3.f * track.m_velocity[1]);
	cv::Size2f els(std::max(minRadius.width, fabs(d.x)), std::max(minRadius.height, fabs(d.y)));
	Point_t p1 = track.m_rrect.center;
	Point_t p2(p1.x + d.x, p1.y + d.y);
	float angle = 0;
	Point_t nc = p1;
	Point_t p2_(p2.x - p1.x, p2.y - p1.y);
	if (fabs(p2_.x - p2_.y) > 5) // pix
	{
		if (fabs(p2_.x) > 0.0001f)
		{
			track_t l = std::min(els.width, els.height) / 3;

			track_t p2_l = sqrt(sqr(p2_.x) + sqr(p2_.y));
			nc.x = l * p2_.x / p2_l + p1.x;
			nc.y = l * p2_.y / p2_l + p1.y;

			angle = atan(p2_.y / p2_.x);
		}
		else
		{
			nc.y += d.y / 3;
			angle = CV_PI / 2.f;
		}
	}

	cv::RotatedRect rr(nc, els, 180.f * angle / CV_PI);
    cv::ellipse(frame, rr, cv::Scalar(100, 0, 100), 1);
#endif
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
            if (!pt2.m_hasRaw)
            {
#if (CV_VERSION_MAJOR >= 4)
                cv::circle(frame, pt2.m_prediction, 4, cl, 1, cv::LINE_AA);
#else
                cv::circle(frame, pt2.m_prediction, 4, cl, 1, CV_AA);
#endif
            }
        }
    }

    cv::Rect brect = track.m_rrect.boundingRect();
    std::stringstream label;
    label << track.m_ID.ID2Str();
    if (track.m_type != bad_type)
        label << ": " << TypeConverter::Type2Str(track.m_type);
    else if (!userLabel.empty())
        label << ": " << userLabel;
    if (track.m_confidence > 0)
        label << ", " << std::fixed << std::setw(2) << std::setprecision(2) << track.m_confidence;
#if 0
    track_t mean = 0;
    track_t stddev = 0;
	TrackingObject::LSParams lsParams;
	if (track.LeastSquares2(10, mean, stddev, lsParams))
	{
		std::cout << "LSParams: " << lsParams << std::endl;
		cv::Scalar cl(255, 0, 255);
		label += ", [" + std::to_string(cvRound(mean)) + ", " + std::to_string(cvRound(stddev)) + "]";
		for (size_t j = 0; j < track.m_trace.size() - 1; ++j)
		{
			track_t t1 = j;
			track_t t2 = j + 1;
			cv::Point pt1(lsParams.m_ax * sqr(t1) + lsParams.m_v0x * t1 + lsParams.m_x0, lsParams.m_ay * sqr(t1) + lsParams.m_v0y * t1 + lsParams.m_y0);
			cv::Point pt2(lsParams.m_ax * sqr(t2) + lsParams.m_v0x * t2 + lsParams.m_x0, lsParams.m_ay * sqr(t2) + lsParams.m_v0y * t2 + lsParams.m_y0);
			//std::cout << pt1 << " - " << pt2 << std::endl;
#if (CV_VERSION_MAJOR >= 4)
			cv::line(frame, pt1, pt2, cl, 1, cv::LINE_AA);
#else
			cv::line(frame, pt1, pt2, cl, 1, CV_AA);
#endif
		}
	}
    label += ", " + std::to_string(cvRound(sqrt(sqr(track.m_velocity[0]) + sqr(track.m_velocity[1]))));
#endif
    int baseLine = 0;
    double fontScale = (frame.cols < 1920) ? 0.5 : 0.7;
    cv::Size labelSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_TRIPLEX, fontScale, 1, &baseLine);
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
    cv::putText(frame, label.str(), brect.tl(), cv::FONT_HERSHEY_TRIPLEX, fontScale, cv::Scalar(0, 0, 0));

	m_resultsLog.AddTrack(framesCounter, track.m_ID, brect, track.m_type, track.m_confidence);
	m_resultsLog.AddRobustTrack(track.m_ID);
}

///
/// \brief VideoExample::OpenCapture
/// \param capture
/// \return
///
bool VideoExample::OpenCapture(cv::VideoCapture& capture)
{
	if (m_inFile.size() == 1)
	{
#ifdef _WIN32
		capture.open(atoi(m_inFile.c_str()), cv::CAP_DSHOW);
#else
		capture.open(atoi(m_inFile.c_str()));
#endif
		//if (capture.isOpened())
		//	capture.set(cv::CAP_PROP_SETTINGS, 1);
	}
    else
        capture.open(m_inFile);

    if (capture.isOpened())
    {
        capture.set(cv::CAP_PROP_POS_FRAMES, m_startFrame);

        m_fps = std::max(1.f, (float)capture.get(cv::CAP_PROP_FPS));

		m_frameSize.width = cvRound(capture.get(cv::CAP_PROP_FRAME_WIDTH));
		m_frameSize.height = cvRound(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
		m_framesCount = cvRound(capture.get(cv::CAP_PROP_FRAME_COUNT));

		std::cout << "Video " << m_inFile << " was started from " << m_startFrame << " frame with " << m_fps << " fps, frame size " << m_frameSize << " and length " << m_framesCount << std::endl;

        return true;
    }
    return false;
}

///
/// \brief VideoExample::WriteFrame
/// \param writer
/// \param frame
/// \return
///
bool VideoExample::WriteFrame(cv::VideoWriter& writer, const cv::Mat& frame)
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
