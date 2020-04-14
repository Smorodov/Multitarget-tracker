#include <iomanip>
#include <ctime>
#include <future>

#include "VideoExample.h"

///
/// \brief VideoExample::VideoExample
/// \param parser
///
VideoExample::VideoExample(const cv::CommandLineParser& parser)
    :
      m_showLogs(true),
      m_fps(25),
      m_captureTimeOut(60000),
      m_trackingTimeOut(60000),
      m_isTrackerInitialized(false),
      m_isDetectorInitialized(false),
      m_fourcc(cv::VideoWriter::fourcc('M', 'J', 'P', 'G')),
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
/// \brief VideoExample::SyncProcess
///
void VideoExample::SyncProcess()
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
        {
            break;
        }

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

        regions_t regions;
        Detection(frame, regions);
        Tracking(frame, regions);

        int64 t2 = cv::getTickCount();

        allTime += t2 - t1;
        int currTime = cvRound(1000 * (t2 - t1) / freq);

        DrawData(frame, framesCounter, currTime);

#ifndef SILENT_WORK
        cv::imshow("Video", frame);

		int waitTime = manualMode ? 0 : 1;// std::max<int>(1, cvRound(1000 / m_fps - currTime));
        int k = cv::waitKey(waitTime);
        if (k == 27)
        {
            break;
        }
        else if (k == 'm' || k == 'M')
        {
            manualMode = !manualMode;
        }
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

    int framesCounter = m_startFrame + 1;

    int64 allTime = 0;
    int64 startLoopTime = cv::getTickCount();
	size_t processCounter = 0;
    for (; !stopCapture.load(); )
    {
        FrameInfo& frameInfo = m_frameInfo[processCounter % 2];
        {
            std::unique_lock<std::mutex> lock(frameInfo.m_mutex);
            if (!frameInfo.m_cond.wait_for(lock, std::chrono::milliseconds(m_captureTimeOut), [&frameInfo]{ return frameInfo.m_captured; }))
            {
                std::cout << "Wait frame timeout!" << std::endl;
                break;
            }
        }

        if (!m_isTrackerInitialized)
        {
			cv::UMat ufirst = frameInfo.m_frame.getUMat(cv::ACCESS_READ);
            m_isTrackerInitialized = InitTracker(ufirst);
            if (!m_isTrackerInitialized)
            {
                std::cerr << "CaptureAndDetect: Tracker initialize error!!!" << std::endl;
                frameInfo.m_cond.notify_one();
                break;
            }
        }

        int64 t1 = cv::getTickCount();

        Tracking(frameInfo.m_frame, frameInfo.m_regions);

        int64 t2 = cv::getTickCount();

        allTime += t2 - t1 + frameInfo.m_dt;
        int currTime = cvRound(1000 * (t2 - t1 + frameInfo.m_dt) / freq);

        //std::cout << "Frame " << framesCounter << ": td = " << (1000 * frameInfo.m_dt / freq) << ", tt = " << (1000 * (t2 - t1) / freq) << std::endl;

        DrawData(frameInfo.m_frame, framesCounter, currTime);

        WriteFrame(writer, frameInfo.m_frame);

        int k = 0;
#ifndef SILENT_WORK
        cv::imshow("Video", frameInfo.m_frame);

		int waitTime = manualMode ? 0 : 1;// std::max<int>(1, cvRound(1000 / m_fps - currTime));
        k = cv::waitKey(waitTime);
        if (k == 'm' || k == 'M')
        {
            manualMode = !manualMode;
        }
#else
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif

		{
			std::unique_lock<std::mutex> lock(frameInfo.m_mutex);
			frameInfo.m_captured = false;
		}
        frameInfo.m_cond.notify_one();

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
		++processCounter;
    }
    stopCapture = true;

    if (thCapDet.joinable())
    {
        thCapDet.join();
    }

    int64 stopLoopTime = cv::getTickCount();

    std::cout << "algorithms time = " << (allTime / freq) << ", work time = " << ((stopLoopTime - startLoopTime) / freq) << std::endl;

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
        std::cerr << "Can't open " << thisPtr->m_inFile << std::endl;
        stopCapture = true;
        return;
    }

    int trackingTimeOut = thisPtr->m_trackingTimeOut;
	size_t processCounter = 0;
    for (; !stopCapture.load();)
    {
        FrameInfo& frameInfo = thisPtr->m_frameInfo[processCounter % 2];

        {
            std::unique_lock<std::mutex> lock(frameInfo.m_mutex);
            if (!frameInfo.m_cond.wait_for(lock, std::chrono::milliseconds(trackingTimeOut), [&frameInfo]{ return !frameInfo.m_captured; }))
            {
                std::cout << "Wait tracking timeout!" << std::endl;
                frameInfo.m_cond.notify_one();
                break;
            }
        }

        capture >> frameInfo.m_frame;
        if (frameInfo.m_frame.empty())
        {
            std::cerr << "CaptureAndDetect: frame is empty!" << std::endl;
            frameInfo.m_cond.notify_one();
            break;
        }

        if (!thisPtr->m_isDetectorInitialized)
        {
			cv::UMat ufirst = frameInfo.m_frame.getUMat(cv::ACCESS_READ);
            thisPtr->m_isDetectorInitialized = thisPtr->InitDetector(ufirst);
            if (!thisPtr->m_isDetectorInitialized)
            {
                std::cerr << "CaptureAndDetect: Detector initialize error!!!" << std::endl;
                frameInfo.m_cond.notify_one();
                break;
            }
        }

        int64 t1 = cv::getTickCount();
        thisPtr->Detection(frameInfo.m_frame, frameInfo.m_regions);
        int64 t2 = cv::getTickCount();
        frameInfo.m_dt = t2 - t1;

		{
			std::unique_lock<std::mutex> lock(frameInfo.m_mutex);
			frameInfo.m_captured = true;
		}
		frameInfo.m_cond.notify_one();

		++processCounter;
    }
    stopCapture = true;
}

///
/// \brief VideoExample::Detection
/// \param frame
/// \param regions
///
void VideoExample::Detection(cv::Mat frame, regions_t& regions)
{
    cv::UMat uframe;
    if (!m_detector->CanGrayProcessing())
    {
        uframe = frame.getUMat(cv::ACCESS_READ);
    }
	else
	{
		cv::cvtColor(frame, uframe, cv::COLOR_BGR2GRAY);
	}

    m_detector->Detect(uframe);

    const regions_t& regs = m_detector->GetDetects();

    regions.assign(std::begin(regs), std::end(regs));
}

///
/// \brief VideoExample::Tracking
/// \param frame
/// \param regions
///
void VideoExample::Tracking(cv::Mat frame, const regions_t& regions)
{
 	cv::UMat uframe;
	if (m_tracker->CanColorFrameToTrack())
	{
		uframe = frame.getUMat(cv::ACCESS_READ);
	}
	else
	{
		cv::cvtColor(frame, uframe, cv::COLOR_BGR2GRAY);
	}

    m_tracker->Update(regions, uframe, m_fps);
}

///
/// \brief VideoExample::DrawTrack
/// \param frame
/// \param resizeCoeff
/// \param track
/// \param drawTrajectory
///
void VideoExample::DrawTrack(cv::Mat frame,
                             int resizeCoeff,
                             const TrackingObject& track,
                             bool drawTrajectory
                             )
{
    auto ResizePoint = [resizeCoeff](const cv::Point& pt) -> cv::Point
    {
        return cv::Point(resizeCoeff * pt.x, resizeCoeff * pt.y);
    };

    cv::Scalar color = track.m_isStatic ? cv::Scalar(255, 0, 255) : cv::Scalar(0, 255, 0);
    cv::Point2f rectPoints[4];
    track.m_rrect.points(rectPoints);
    for (int i = 0; i < 4; ++i)
    {
        cv::line(frame, ResizePoint(rectPoints[i]), ResizePoint(rectPoints[(i+1) % 4]), color);
    }
#if 0
    float angle = 0;//atan2(track.m_velocity[0], track.m_velocity[1]);
    cv::RotatedRect rr(track.m_rrect.center,
                       cv::Size2f(std::max(frame.rows / 20.f, static_cast<track_t>(3.f * fabs(track.m_velocity[0]))),
                       std::max(frame.cols / 20.f, static_cast<track_t>(3.f * fabs(track.m_velocity[1])))),
            180.f * angle / CV_PI);
    cv::ellipse(frame, rr, cv::Scalar(100, 100, 100), 1);
#endif
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
/// \brief VideoExample::OpenCapture
/// \param capture
/// \return
///
bool VideoExample::OpenCapture(cv::VideoCapture& capture)
{
    if (m_inFile.size() == 1)
        capture.open(atoi(m_inFile.c_str()));
    else
        capture.open(m_inFile);

    if (capture.isOpened())
    {
        capture.set(cv::CAP_PROP_POS_FRAMES, m_startFrame);

        m_fps = std::max(1.f, (float)capture.get(cv::CAP_PROP_FPS));
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
        {
            writer.open(m_outFile, m_fourcc, m_fps, frame.size(), true);
        }
        if (writer.isOpened())
        {
            writer << frame;
            return true;
        }
    }
    return false;
}
