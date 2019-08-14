#include <iomanip>
#include <ctime>
#include <future>

#include "VideoExample.h"

#define MULTITHREADING_LOGS 0

#if MULTITHREADING_LOGS
///
/// \brief currTime
/// \return
///
std::string CurrTime()
{
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d.%m.%Y %H:%M:%S:");
    return oss.str();
}

#define LOG_TIME std::cout << CurrTime()
#define LOG_ERR_TIME (std::cerr << CurrTime())

#else

class NullBuffer : public std::streambuf
{
public:
    int overflow(int c) { return c; }
};
NullBuffer NullBuffer;
std::ostream NullStream(&NullBuffer);
#define LOG_TIME NullStream
#define LOG_ERR_TIME std::cerr

#endif

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
      m_fourcc(cv::VideoWriter::fourcc('M', 'J', 'P', 'G')),
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
/// \brief VideoExample::~VideoExample
///
VideoExample::~VideoExample()
{

}

///
/// \brief VideoExample::SyncProcess
///
void VideoExample::SyncProcess()
{
    m_currFrame = 1;

    cv::VideoWriter writer;

#ifndef SILENT_WORK
    cv::namedWindow("Video", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    bool manualMode = false;
#endif

    cv::Mat frame;
    cv::UMat gray;

    int k = 0;

    double freq = cv::getTickFrequency();

    int64 allTime = 0;

    int framesCounter = m_startFrame + 1;

    cv::VideoCapture capture;
    if (!OpenCapture(capture))
    {
        LOG_ERR_TIME << "Can't open " << m_inFile << std::endl;
        return;
    }

    int64 startLoopTime = cv::getTickCount();

    for (; k != 27; )
    {
        capture >> frame;
        if (frame.empty())
        {
            break;
        }
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        if (!m_isTrackerInitialized)
        {
            m_isTrackerInitialized = InitTracker(gray);
            if (!m_isTrackerInitialized)
            {
                std::cerr << "CaptureAndDetect: Tracker initilize error!!!" << std::endl;
                break;
            }
        }

        int64 t1 = cv::getTickCount();

        regions_t regions;
        Detection(frame, gray, regions);
        Tracking(frame, gray, regions);

        int64 t2 = cv::getTickCount();

        allTime += t2 - t1;
        int currTime = cvRound(1000 * (t2 - t1) / freq);

        DrawData(frame, framesCounter, currTime);

#ifndef SILENT_WORK
        cv::imshow("Video", frame);

		int waitTime = manualMode ? 0 : 1;// std::max<int>(1, cvRound(1000 / m_fps - currTime));
        k = cv::waitKey(waitTime);
        if (k == 'm' || k == 'M')
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
    m_currFrame = 1;
    std::atomic<bool> stopCapture(false);
    Gate frameLock;

    Gate trackLock;
    std::thread thCapDet(CaptureAndDetect, this, std::ref(stopCapture), &frameLock, &trackLock);

    cv::VideoWriter writer;

#ifndef SILENT_WORK
    cv::namedWindow("Video", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    bool manualMode = false;
#endif

    int k = 0;

    double freq = cv::getTickFrequency();

    int64 allTime = 0;

    int framesCounter = m_startFrame + 1;

    if (!frameLock.WaitAtGateFor(2000))
    {
        LOG_ERR_TIME << "Process: Init capture timeout" << std::endl;
        stopCapture = true;

        if (thCapDet.joinable())
        {
            thCapDet.join();
        }
        return;
    }
    trackLock.OpenGate();

    int64 startLoopTime = cv::getTickCount();

    for (; !stopCapture && k != 27; )
    {
        LOG_TIME << "Process:: lock(frameLock);" << std::endl;

        if (!frameLock.WaitAtGateFor(m_captureTimeOut))
        {
            std::cerr << "Process: Frame capture timeout" << std::endl;
            break;
        }
        LOG_TIME << "Process:: if (stopCapture)" << std::endl;
        if (stopCapture)
        {
            break;
        }

        LOG_TIME << "Process:: frameLock.lock();" << std::endl;
        frameLock.Lock();
        FrameInfo& frameInfo = m_frameInfo[m_currFrame];
        frameLock.Unlock();
        LOG_TIME << "Process:: frameLock.unlock();" << std::endl;

        int64 t1 = cv::getTickCount();

        Tracking(frameInfo.m_frame, frameInfo.m_gray, frameInfo.m_regions);

        int64 t2 = cv::getTickCount();

        allTime += t2 - t1 + frameInfo.m_dt;
        int currTime = cvRound(1000 * (t2 - t1 + frameInfo.m_dt) / freq);

        //std::cout << "Frame " << framesCounter << ": td = " << (1000 * frameInfo.m_dt / freq) << ", tt = " << (1000 * (t2 - t1) / freq) << std::endl;

        DrawData(frameInfo.m_frame, framesCounter, currTime);

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

        WriteFrame(writer, frameInfo.m_frame);

        trackLock.OpenGate();
        LOG_TIME << "Process:: trackCond.notify_all();" << std::endl;

        ++framesCounter;
        if (m_endFrame && framesCounter > m_endFrame)
        {
            LOG_TIME << "Process: riched last " << m_endFrame << " frame" << std::endl;
            break;
        }
    }
    stopCapture = true;

    frameLock.OpenGate();
    trackLock.OpenGate();

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
/// \param frameLock
/// \param trackLock
///
void VideoExample::CaptureAndDetect(VideoExample* thisPtr,
                                    std::atomic<bool>& stopCapture,
                                    Gate* frameLock,
                                    Gate* trackLock)
{
    cv::VideoCapture capture;
    if (!thisPtr->OpenCapture(capture))
    {
        LOG_ERR_TIME << "Can't open " << thisPtr->m_inFile << std::endl;
        stopCapture = true;
        return;
    }

    frameLock->OpenGate();
    LOG_TIME << "CaptureAndDetect:: frameCond->notify_all();" << std::endl;

    for (; !stopCapture;)
    {
        {
            LOG_TIME << "CaptureAndDetect:: lock(*trackLock);" << std::endl;

            if (!trackLock->WaitAtGateFor(thisPtr->m_trackingTimeOut))
            {
                LOG_ERR_TIME << "CaptureAndDetect: Tracking timeout!" << std::endl;
                break;
            }
        }
        LOG_TIME << "CaptureAndDetect:: frameLock->lock();" << std::endl;
        frameLock->Lock();
        FrameInfo& frameInfo = thisPtr->m_frameInfo[(thisPtr->m_currFrame ? 0 : 1)];
        frameLock->Unlock();
        LOG_TIME << "CaptureAndDetect:: frameLock->unlock();" << std::endl;

        capture >> frameInfo.m_frame;
        if (frameInfo.m_frame.empty())
        {
            LOG_ERR_TIME << "CaptureAndDetect: frame is empty!" << std::endl;
            break;
        }
        cv::cvtColor(frameInfo.m_frame, frameInfo.m_gray, cv::COLOR_BGR2GRAY);

        if (!thisPtr->m_isTrackerInitialized)
        {
            thisPtr->m_isTrackerInitialized = thisPtr->InitTracker(frameInfo.m_gray);
            if (!thisPtr->m_isTrackerInitialized)
            {
                LOG_ERR_TIME << "CaptureAndDetect: Tracker initilize error!!!" << std::endl;
                break;
            }
        }

        int64 t1 = cv::getTickCount();
        thisPtr->Detection(frameInfo.m_frame, frameInfo.m_gray, frameInfo.m_regions);
        int64 t2 = cv::getTickCount();
        frameInfo.m_dt = t2 - t1;

        LOG_TIME << "CaptureAndDetect:: frameLock->lock(); 2" << std::endl;
        frameLock->Lock();
        thisPtr->m_currFrame = thisPtr->m_currFrame ? 0 : 1;
        frameLock->Unlock();
        LOG_TIME << "CaptureAndDetect:: frameLock->unlock(); 2" << std::endl;
        frameLock->OpenGate();
        LOG_TIME << "CaptureAndDetect:: frameCond->notify_all(); 2" << std::endl;
    }

    stopCapture = true;
    frameLock->OpenGate();
    LOG_TIME << "CaptureAndDetect:: frameCond->notify_all();; 3" << std::endl;
}

///
/// \brief VideoExample::GrayProcessing
/// \return
///
bool VideoExample::GrayProcessing() const
{
    return true;
}

///
/// \brief VideoExample::Detection
/// \param frame
/// \param grayFrame
/// \param regions
///
void VideoExample::Detection(cv::Mat frame, cv::UMat grayFrame, regions_t& regions)
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
/// \brief VideoExample::Tracking
/// \param frame
/// \param grayFrame
/// \param regions
///
void VideoExample::Tracking(cv::Mat frame, cv::UMat grayFrame, const regions_t& regions)
{
    cv::UMat clFrame;
    if (!GrayProcessing() || !m_tracker->GrayFrameToTrack())
    {
        clFrame = frame.getUMat(cv::ACCESS_READ);
    }

    m_tracker->Update(regions, m_tracker->GrayFrameToTrack() ? grayFrame : clFrame, m_fps);
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
    {
        capture.open(atoi(m_inFile.c_str()));
    }
    else
    {
        capture.open(m_inFile);
    }
    if (capture.isOpened())
    {
        capture.set(cv::CAP_PROP_POS_FRAMES, m_startFrame);

        m_fps = std::max(1.f, (float)capture.get(cv::CAP_PROP_FPS));
        return true;
    }
    return true;
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
