#pragma once

#include "BaseDetector.h"

#include "Ctracker.h"
#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>

#include "nms.h"

// ----------------------------------------------------------------------

///
/// \brief The VideoExample class
///
class VideoExample
{
public:
    VideoExample(const cv::CommandLineParser& parser)
        :
          m_showLogs(true),
          m_fps(25),
          m_useLocalTracking(false),
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
    virtual ~VideoExample()
    {

    }

    ///
    /// \brief Process
    ///
    void Process()
    {
        m_currFrame = 1;
        bool stopCapture = false;
        std::mutex frameLock;
        std::condition_variable frameCond;

        std::mutex trackLock;
        std::condition_variable trackCond;
        std::thread thCapDet(CaptureAndDetect, this, &stopCapture, &frameLock, &frameCond, &trackLock, &trackCond);
        thCapDet.detach();

        const int captureTimeOut = 10000;
        {
            std::unique_lock<std::mutex> lock(frameLock);
            auto now = std::chrono::system_clock::now();
            if (frameCond.wait_until(lock, now + std::chrono::milliseconds(captureTimeOut)) == std::cv_status::timeout)
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
                if (frameCond.wait_until(lock, now + std::chrono::milliseconds(captureTimeOut)) == std::cv_status::timeout)
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

protected:
    std::unique_ptr<BaseDetector> m_detector;
    std::unique_ptr<CTracker> m_tracker;

    bool m_showLogs;
    float m_fps;
    bool m_useLocalTracking;


    ///
    /// \brief CaptureAndDetect
    /// \param stopCapture
    /// \param frameLock
    /// \param frameCond
    ///
    static void CaptureAndDetect(VideoExample* thisPtr,
                                 bool* stopCapture,
                                 std::mutex* frameLock,
                                 std::condition_variable* frameCond,
                                 std::mutex* trackLock,
                                 std::condition_variable* trackCond)
    {
        cv::VideoCapture capture(thisPtr->m_inFile);

        if (!capture.isOpened())
        {
            std::cerr << "Can't open " << thisPtr->m_inFile << std::endl;
            return;
        }

        capture.set(cv::CAP_PROP_POS_FRAMES, thisPtr->m_startFrame);

        thisPtr->m_fps = std::max(1.f, (float)capture.get(cv::CAP_PROP_FPS));

        const int trackingTimeOut = 10000;

        frameCond->notify_all();

        int currFrame = 0;
        for (; !(*stopCapture);)
        {
            {
                std::unique_lock<std::mutex> lock(*trackLock);
                auto now = std::chrono::system_clock::now();
                if (trackCond->wait_until(lock, now + std::chrono::milliseconds(trackingTimeOut)) == std::cv_status::timeout)
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
    /// \brief GrayProcessing
    /// \return
    ///
    virtual bool GrayProcessing() const
    {
        return true;
    }

    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    virtual bool InitTracker(cv::UMat frame) = 0;

    ///
    /// \brief Detection
    /// \param frame
    /// \param grayFrame
    ///
    void Detection(cv::Mat frame, cv::UMat grayFrame, regions_t& regions)
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
    /// \brief Tracking
    /// \param frame
    /// \param grayFrame
    ///
    void Tracking(cv::Mat frame, cv::UMat grayFrame, const regions_t& regions)
    {
        cv::UMat clFrame;
        if (!GrayProcessing() || !m_tracker->GrayFrameToTrack())
        {
            clFrame = frame.getUMat(cv::ACCESS_READ);
        }

        m_tracker->Update(regions, m_tracker->GrayFrameToTrack() ? grayFrame : clFrame, m_fps);
    }

    ///
    /// \brief DrawData
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
    virtual void DrawData(cv::Mat frame, int framesCounter, int currTime) = 0;

    ///
    /// \brief DrawTrack
    /// \param frame
    /// \param resizeCoeff
    /// \param track
    ///
    void DrawTrack(cv::Mat frame,
                   int resizeCoeff,
                   const CTrack& track,
                   bool drawTrajectory = true,
                   bool isStatic = false
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

private:
    bool m_isTrackerInitialized;
    std::string m_inFile;
    std::string m_outFile;
    int m_startFrame;
    int m_endFrame;
    int m_finishDelay;
    std::vector<cv::Scalar> m_colors;

    struct FrameInfo
    {
        cv::Mat m_frame;
        cv::UMat m_gray;
        regions_t m_regions;
        int64 m_dt;

        FrameInfo()
            : m_dt(0)
        {

        }
    };
    FrameInfo m_frameInfo[2];

    int m_currFrame;
};

// ----------------------------------------------------------------------

///
/// \brief The MotionDetectorExample class
///
class MotionDetectorExample : public VideoExample
{
public:
    MotionDetectorExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser),
          m_minObjWidth(10)
    {
    }

protected:
    ///
    /// \brief InitTracker
    /// \param grayFrame
    ///
    bool InitTracker(cv::UMat frame)
    {
        m_useLocalTracking = false;

        m_minObjWidth = frame.cols / 50;

        const int minStaticTime = 5;

        config_t config;
#if 0
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
        settings.m_dt = 0.5f;                             // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                  // Accel noise magnitude for Kalman filter
        settings.m_distThres = frame.rows / 20;           // Distance threshold between region and object on two frames

        settings.m_useAbandonedDetection = true;
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
    /// \brief DrawData
    /// \param frame
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
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

        //m_detector->CalcMotionMap(frame);
    }

private:
    int m_minObjWidth;
};

// ----------------------------------------------------------------------

///
/// \brief The FaceDetectorExample class
///
class FaceDetectorExample : public VideoExample
{
public:
    FaceDetectorExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitTracker
    /// \param grayFrame
    ///
    bool InitTracker(cv::UMat frame)
    {
        config_t config;
        config["cascadeFileName"] = "../data/haarcascade_frontalface_alt2.xml";
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Face_HAAR, config, m_useLocalTracking, frame));
        if (!m_detector.get())
        {
            return false;
        }
        m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));

        TrackerSettings settings;
        settings.m_useLocalTracking = m_useLocalTracking;
        settings.m_distType = tracking::DistJaccard;
        settings.m_kalmanType = tracking::KalmanUnscented;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackKCF;    // Use KCF tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                             // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                  // Accel noise magnitude for Kalman filter
        settings.m_distThres = 0.8f;           // Distance threshold between region and object on two frames
        settings.m_maximumAllowedSkippedFrames = m_fps / 2;   // Maximum allowed skipped frames
        settings.m_maxTraceLength = 5 * m_fps;            // Maximum trace length

        m_tracker = std::make_unique<CTracker>(settings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << m_tracker->tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : m_tracker->tracks)
        {
            if (track->IsRobust(8,                           // Minimal trajectory size
                                0.4f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, *track);
            }
        }

        m_detector->CalcMotionMap(frame);
    }
};

// ----------------------------------------------------------------------

///
/// \brief The PedestrianDetectorExample class
///
class PedestrianDetectorExample : public VideoExample
{
public:
    PedestrianDetectorExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitTracker
    /// \param grayFrame
    ///
    bool InitTracker(cv::UMat frame)
    {
        tracking::Detectors detectorType = tracking::Detectors::Pedestrian_C4; // tracking::Detectors::Pedestrian_HOG;

        config_t config;
        config["detectorType"] = (detectorType == tracking::Pedestrian_HOG) ? "HOG" : "C4";
        config["cascadeFileName1"] = "../data/combined.txt.model";
        config["cascadeFileName2"] = "../data/combined.txt.model_";
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(detectorType, config, m_useLocalTracking, frame));
        if (!m_detector.get())
        {
            return false;
        }
        m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));


        TrackerSettings settings;
        settings.m_useLocalTracking = m_useLocalTracking;
        settings.m_distType = tracking::DistRects;
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackKCF;    // Use KCF tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                             // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                  // Accel noise magnitude for Kalman filter
        settings.m_distThres = frame.rows / 10;           // Distance threshold between region and object on two frames
        settings.m_maximumAllowedSkippedFrames = m_fps;   // Maximum allowed skipped frames
        settings.m_maxTraceLength = 5 * m_fps;            // Maximum trace length

        m_tracker = std::make_unique<CTracker>(settings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << m_tracker->tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : m_tracker->tracks)
        {
			if (track->IsRobust(cvRound(m_fps / 2),          // Minimal trajectory size
                                0.4f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, *track);
            }
        }

        m_detector->CalcMotionMap(frame);
    }
};

// ----------------------------------------------------------------------

///
/// \brief The SSDMobileNetExample class
///
class SSDMobileNetExample : public VideoExample
{
public:
    SSDMobileNetExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitTracker
    /// \param grayFrame
    ///
    bool InitTracker(cv::UMat frame)
    {
        config_t config;
        config["modelConfiguration"] = "../data/MobileNetSSD_deploy.prototxt";
        config["modelBinary"] = "../data/MobileNetSSD_deploy.caffemodel";
        config["confidenceThreshold"] = "0.5";
        config["maxCropRatio"] = "3.0";
        config["dnnTarget"] = "DNN_TARGET_OPENCL_FP16";
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::SSD_MobileNet, config, m_useLocalTracking, frame));
        if (!m_detector.get())
        {
            return false;
        }
        m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));

        TrackerSettings settings;
        settings.m_useLocalTracking = m_useLocalTracking;
        settings.m_distType = tracking::DistRects;
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackKCF;       // Use KCF tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                                // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                     // Accel noise magnitude for Kalman filter
        settings.m_distThres = frame.rows / 10;              // Distance threshold between region and object on two frames
        settings.m_maximumAllowedSkippedFrames = 2 * m_fps;  // Maximum allowed skipped frames
        settings.m_maxTraceLength = 5 * m_fps;               // Maximum trace length

        m_tracker = std::make_unique<CTracker>(settings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << m_tracker->tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : m_tracker->tracks)
        {
            if (track->IsRobust(5,                           // Minimal trajectory size
                                0.2f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, *track);

                std::string label = track->m_lastRegion.m_type + ": " + std::to_string(track->m_lastRegion.m_confidence);
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                auto rect(track->GetLastRect());
                cv::rectangle(frame, cv::Rect(cv::Point(rect.x, rect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 255, 255), CV_FILLED);
                cv::putText(frame, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }

        m_detector->CalcMotionMap(frame);
    }

    ///
    /// \brief GrayProcessing
    /// \return
    ///
    bool GrayProcessing() const
    {
        return false;
    }
};

// ----------------------------------------------------------------------

///
/// \brief The YoloExample class
///
class YoloExample : public VideoExample
{
public:
    YoloExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitTracker
    /// \param grayFrame
    ///
    bool InitTracker(cv::UMat frame)
    {
        m_useLocalTracking = false;

        config_t config;
        //config["modelConfiguration"] = "../data/tiny-yolo.cfg";
        //config["modelBinary"] = "../data/tiny-yolo.weights";
        config["modelConfiguration"] = "../data/yolov3-tiny.cfg";
        config["modelBinary"] = "../data/yolov3-tiny.weights";
        config["classNames"] = "../data/coco.names";
        config["confidenceThreshold"] = "0.5";
        config["maxCropRatio"] = "3.0";
        config["dnnTarget"] = "DNN_TARGET_OPENCL_FP16";

        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Yolo, config, m_useLocalTracking, frame));
        if (!m_detector.get())
        {
            return false;
        }
        m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));

        TrackerSettings settings;
        settings.m_useLocalTracking = m_useLocalTracking;
        settings.m_distType = tracking::DistRects;
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackKCF;       // Use KCF tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                                // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                     // Accel noise magnitude for Kalman filter
        settings.m_distThres = frame.rows / 10;              // Distance threshold between region and object on two frames
        settings.m_maximumAllowedSkippedFrames = 2 * m_fps;  // Maximum allowed skipped frames
        settings.m_maxTraceLength = 5 * m_fps;               // Maximum trace length

        m_tracker = std::make_unique<CTracker>(settings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << m_tracker->tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : m_tracker->tracks)
        {
            if (track->IsRobust(5,                           // Minimal trajectory size
                                0.2f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, *track);

                std::string label = track->m_lastRegion.m_type + ": " + std::to_string(track->m_lastRegion.m_confidence);
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                auto rect(track->GetLastRect());
                cv::rectangle(frame, cv::Rect(cv::Point(rect.x, rect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 255, 255), CV_FILLED);
                cv::putText(frame, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }

        m_detector->CalcMotionMap(frame);
    }

    ///
    /// \brief GrayProcessing
    /// \return
    ///
    bool GrayProcessing() const
    {
        return false;
    }
};
