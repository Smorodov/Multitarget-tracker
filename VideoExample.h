#pragma once

#include "BaseDetector.h"

#include "Ctracker.h"
#include <iostream>
#include <vector>
#include <map>

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
          m_fps(25),
          m_useLocalTracking(false)
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
        cv::VideoWriter writer;

        cv::VideoCapture capture(m_inFile);
        if (!capture.isOpened())
        {
            std::cerr << "Can't open " << m_inFile << std::endl;
            return;
        }
        cv::namedWindow("Video");
        cv::Mat frame;
        cv::UMat gray;

        capture.set(cv::CAP_PROP_POS_FRAMES, m_startFrame);

        m_fps = std::max(1.f, (float)capture.get(cv::CAP_PROP_FPS));

        capture >> frame;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        if (!InitTracker(gray))
        {
            return;
        }

        int k = 0;

        double freq = cv::getTickFrequency();

        int64 allTime = 0;

        bool manualMode = false;
        int framesCounter = m_startFrame + 1;
        while (k != 27)
        {
            capture >> frame;
            if (frame.empty())
            {
                break;
            }
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            if (!writer.isOpened())
            {
                writer.open(m_outFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), capture.get(cv::CAP_PROP_FPS), frame.size(), true);
            }

            int64 t1 = cv::getTickCount();

            ProcessFrame(frame, gray);

            int64 t2 = cv::getTickCount();

            allTime += t2 - t1;
            int currTime = cvRound(1000 * (t2 - t1) / freq);

            DrawData(frame, framesCounter, currTime);

            cv::imshow("Video", frame);

            int waitTime = manualMode ? 0 : std::max<int>(1, cvRound(1000 / m_fps - currTime));
            k = cv::waitKey(waitTime);

            if (k == 'm' || k == 'M')
            {
                manualMode = !manualMode;
            }

            if (writer.isOpened())
            {
                writer << frame;
            }

            ++framesCounter;
            if (m_endFrame && framesCounter > m_endFrame)
            {
                break;
            }
        }

        std::cout << "work time = " << (allTime / freq) << std::endl;
        cv::waitKey(m_finishDelay);
    }

protected:
    std::unique_ptr<BaseDetector> m_detector;
    std::unique_ptr<CTracker> m_tracker;

    virtual bool GrayProcessing() const
    {
        return true;
    }

    virtual bool InitTracker(cv::UMat frame) = 0;

    ///
    /// \brief ProcessFrame
    /// \param grayFrame
    ///
    virtual void ProcessFrame(cv::Mat frame, cv::UMat grayFrame)
    {
        cv::UMat clFrame;
        if (!GrayProcessing() || !m_tracker->GrayFrameToTrack())
        {
            clFrame = frame.getUMat(cv::ACCESS_READ);
        }

        m_detector->Detect(GrayProcessing() ? grayFrame : clFrame);

        const regions_t& regions = m_detector->GetDetects();

        m_tracker->Update(regions, m_tracker->GrayFrameToTrack() ? grayFrame : clFrame, m_fps);
    }

    virtual void DrawData(cv::Mat frame, int framesCounter, int currTime) = 0;

    bool m_showLogs;
    float m_fps;
    bool m_useLocalTracking;

    ///
    /// \brief DrawTrack
    /// \param frame
    /// \param resizeCoeff
    /// \param track
    ///
    void DrawTrack(cv::Mat frame,
                   int resizeCoeff,
                   const CTrack& track,
                   bool drawTrajectory = true
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

        cv::rectangle(frame, ResizeRect(track.GetLastRect()), cv::Scalar(0, 255, 0), 1, CV_AA);

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
    }

private:
    std::string m_inFile;
    std::string m_outFile;
    int m_startFrame;
    int m_endFrame;
    int m_finishDelay;
    std::vector<cv::Scalar> m_colors;
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
        m_minObjWidth = frame.cols / 50;

        BaseDetector::config_t config;
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Motion_MOG2, config, m_useLocalTracking, frame));
        m_detector->SetMinObjectSize(cv::Size(m_minObjWidth, m_minObjWidth));

        m_tracker = std::make_unique<CTracker>(m_useLocalTracking,
                                               tracking::DistCenters,
                                               tracking::KalmanLinear,
                                               tracking::FilterRect,
                                               tracking::TrackKCF,       // Use KCF tracker for collisions resolving
                                               tracking::MatchHungrian,
                                               0.2f,                     // Delta time for Kalman filter
                                               0.1f,                     // Accel noise magnitude for Kalman filter
                                               frame.rows / 10,          // Distance threshold between region and object on two frames
                                               m_fps,                    // Maximum allowed skipped frames
                                               3 * m_fps                 // Maximum trace length
                                               );

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
                                0.6f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, *track);
            }
        }

        m_detector->CalcMotionMap(frame);
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
        BaseDetector::config_t config;
        config["cascadeFileName"] = "../data/haarcascade_frontalface_alt2.xml";
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Face_HAAR, config, m_useLocalTracking, frame));
        if (!m_detector.get())
        {
            return false;
        }
        m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));

        m_tracker = std::make_unique<CTracker>(m_useLocalTracking,
                                               tracking::DistJaccard,
                                               tracking::KalmanUnscented,
                                               tracking::FilterRect,
                                               tracking::TrackKCF,      // Use KCF tracker for collisions resolving
                                               tracking::MatchHungrian,
                                               0.3f,                     // Delta time for Kalman filter
                                               0.1f,                     // Accel noise magnitude for Kalman filter
                                               0.8f,                     // Distance threshold between region and object on two frames
                                               m_fps / 2,                // Maximum allowed skipped frames
                                               5 * m_fps                 // Maximum trace length
                                               );

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

        BaseDetector::config_t config;
        config["detectorType"] = (detectorType == tracking::Pedestrian_HOG) ? "HOG" : "C4";
        config["cascadeFileName1"] = "../data/combined.txt.model";
        config["cascadeFileName2"] = "../data/combined.txt.model_";
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(detectorType, config, m_useLocalTracking, frame));
        if (!m_detector.get())
        {
            return false;
        }
        m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));

        m_tracker = std::make_unique<CTracker>(m_useLocalTracking,
                                               tracking::DistRects,
                                               tracking::KalmanLinear,
                                               tracking::FilterRect,
                                               tracking::TrackKCF,      // Use KCF tracker for collisions resolving
                                               tracking::MatchHungrian,
                                               0.3f,                     // Delta time for Kalman filter
                                               0.1f,                     // Accel noise magnitude for Kalman filter
                                               frame.rows / 10,          // Distance threshold between region and object on two frames
                                               1 * m_fps,                // Maximum allowed skipped frames
                                               5 * m_fps                 // Maximum trace length
                                               );

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
/// \brief The HybridFaceDetectorExample class
///
class HybridFaceDetectorExample : public VideoExample
{
public:
    HybridFaceDetectorExample(const cv::CommandLineParser& parser)
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
        std::string fileName = "../data/haarcascade_frontalface_alt2.xml";
        m_cascade.load(fileName);
        if (m_cascade.empty())
        {
            std::cerr << "Cascade not opened!" << std::endl;
            return false;
        }

        m_tracker = std::make_unique<CTracker>(m_useLocalTracking,
                                               tracking::DistCenters,
                                               tracking::KalmanUnscented,
                                               tracking::FilterRect,
                                               tracking::TrackKCF,      // Use KCF tracker for collisions resolving
                                               tracking::MatchHungrian,
                                               0.3f,                     // Delta time for Kalman filter
                                               0.1f,                     // Accel noise magnitude for Kalman filter
                                               frame.cols / 10,      // Distance threshold between region and object on two frames
                                               2 * m_fps,                // Maximum allowed skipped frames
                                               5 * m_fps                 // Maximum trace length
                                               );

        BaseDetector::config_t config;
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Motion_MOG2, config, m_useLocalTracking, frame));
        m_detector->SetMinObjectSize(cv::Size(frame.cols / 50, frame.rows / 50));

        return true;
    }

    ///
    /// \brief ProcessFrame
    /// \param grayFrame
    ///
    void ProcessFrame(cv::Mat /*frame*/, cv::UMat grayFrame)
    {
        bool findLargestObject = false;
        bool filterRects = true;
        std::vector<cv::Rect> faceRects;
        m_cascade.detectMultiScale(grayFrame,
                                 faceRects,
                                 1.1,
                                 (filterRects || findLargestObject) ? 3 : 0,
                                 findLargestObject ? cv::CASCADE_FIND_BIGGEST_OBJECT : 0,
                                 cv::Size(grayFrame.cols / 20, grayFrame.rows / 20),
                                 cv::Size(grayFrame.cols / 2, grayFrame.rows / 2));

        m_detector->Detect(grayFrame);
        const regions_t& regions1 = m_detector->GetDetects();

        std::vector<float> scores(faceRects.size(), 0.5f);
        for (const auto& reg : regions1)
        {
            faceRects.push_back(reg.m_rect);
            scores.push_back(0.4f);
        }
        faceRects.insert(faceRects.end(), m_prevRects.begin(), m_prevRects.end());
        scores.insert(scores.end(), m_prevRects.size(), 0.4f);
        std::vector<cv::Rect> allRects;
        nms2(faceRects, scores, allRects, 0.3f, 1, 0.7f);

        regions_t regions;
        for (auto rect : allRects)
        {
            regions.push_back(rect);
        }

        m_tracker->Update(regions, grayFrame, m_fps);
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

        m_prevRects.clear();

        for (const auto& track : m_tracker->tracks)
        {
            if (track->IsRobust(4,                           // Minimal trajectory size
                                0.4f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.4f, 3.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, *track);
                m_prevRects.push_back(track->GetLastRect());
            }
        }

        m_detector->CalcMotionMap(frame);
    }

private:
    std::vector<cv::Rect> m_prevRects;
    cv::CascadeClassifier m_cascade;
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
        BaseDetector::config_t config;
        config["modelConfiguration"] = "../data/MobileNetSSD_deploy.prototxt";
        config["modelBinary"] = "../data/MobileNetSSD_deploy.caffemodel";
        config["confidenceThreshold"] = "0.5";
        config["maxCropRatio"] = "3.0";
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::SSD_MobileNet, config, m_useLocalTracking, frame));
        if (!m_detector.get())
        {
            return false;
        }
        m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));

        m_tracker = std::make_unique<CTracker>(m_useLocalTracking,
                                               tracking::DistRects,
                                               tracking::KalmanLinear,
                                               tracking::FilterRect,
                                               tracking::TrackKCF,      // Use KCF tracker for collisions resolving
                                               tracking::MatchHungrian,
                                               0.3f,                     // Delta time for Kalman filter
                                               0.1f,                     // Accel noise magnitude for Kalman filter
                                               frame.rows / 10,          // Distance threshold between region and object on two frames
                                               2 * m_fps,                // Maximum allowed skipped frames
                                               5 * m_fps                 // Maximum trace length
                                               );

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

        //m_detector->CalcMotionMap(frame);
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
        BaseDetector::config_t config;
        config["modelConfiguration"] = "../data/tiny-yolo.cfg";
        config["modelBinary"] = "../data/tiny-yolo.weights";
        config["classNames"] = "../data/coco.names";
        config["confidenceThreshold"] = "0.5";
        config["maxCropRatio"] = "3.0";
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Yolo, config, m_useLocalTracking, frame));
        if (!m_detector.get())
        {
            return false;
        }
        m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));

        m_tracker = std::make_unique<CTracker>(m_useLocalTracking,
                                               tracking::DistRects,
                                               tracking::KalmanLinear,
                                               tracking::FilterRect,
                                               tracking::TrackKCF,      // Use KCF tracker for collisions resolving
                                               tracking::MatchHungrian,
                                               0.3f,                     // Delta time for Kalman filter
                                               0.1f,                     // Accel noise magnitude for Kalman filter
                                               frame.rows / 10,          // Distance threshold between region and object on two frames
                                               2 * m_fps,                // Maximum allowed skipped frames
                                               5 * m_fps                 // Maximum trace length
                                               );

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

        //m_detector->CalcMotionMap(frame);
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
