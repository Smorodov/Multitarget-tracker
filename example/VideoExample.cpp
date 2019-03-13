#include "VideoExample.h"
#include <iomanip>
#include <ctime>

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
/// \brief VideoExample::~VideoExample
///
VideoExample::~VideoExample()
{

}

///
/// \brief VideoExample::Process
///
void VideoExample::Process()
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
/// \brief VideoExample::CaptureAndDetect
/// \param thisPtr
/// \param stopCapture
/// \param frameLock
/// \param frameCond
/// \param trackLock
/// \param trackCond
///
void VideoExample::CaptureAndDetect(VideoExample* thisPtr,
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
/// \param isStatic
///
void VideoExample::DrawTrack(cv::Mat frame,
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
