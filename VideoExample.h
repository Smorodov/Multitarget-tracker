#pragma once

#include "BackgroundSubtract.h"
#include "Detector.h"

#include "Ctracker.h"
#include <iostream>
#include <vector>
#include <map>

#include "pedestrians/c4-pedestrian-detector.h"
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

        m_fps = std::max(1, cvRound(capture.get(cv::CAP_PROP_FPS)));

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

            ProcessFrame(gray);

            int64 t2 = cv::getTickCount();

            allTime += t2 - t1;
            int currTime = cvRound(1000 * (t2 - t1) / freq);

            DrawData(frame, framesCounter, currTime);

            cv::imshow("Video", frame);

            int waitTime = manualMode ? 0 : std::max<int>(1, 1000 / m_fps - currTime);
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
    virtual bool InitTracker(cv::UMat grayFrame) = 0;
    virtual void ProcessFrame(cv::UMat grayFrame) = 0;
    virtual void DrawData(cv::Mat frame, int framesCounter, int currTime) = 0;

    bool m_showLogs;
    int m_fps;
    bool m_useLocalTracking;

    ///
    /// \brief DrawTrack
    /// \param frame
    /// \param resizeCoeff
    /// \param track
    ///
    void DrawTrack(cv::Mat frame,
                   int resizeCoeff,
                   const CTrack& track
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
/// \brief The MotionDetector class
///
class MotionDetector : public VideoExample
{
public:
    MotionDetector(const cv::CommandLineParser& parser)
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
    bool InitTracker(cv::UMat grayFrame)
    {
        m_minObjWidth = grayFrame.cols / 50;

        m_detector = std::make_unique<CDetector>(BackgroundSubtract::ALG_MOG2, m_useLocalTracking, grayFrame);
        m_detector->SetMinObjectSize(cv::Size(m_minObjWidth, m_minObjWidth));

        m_tracker = std::make_unique<CTracker>(m_useLocalTracking,
                                               tracking::DistCenters,
                                               tracking::KalmanLinear,
                                               tracking::FilterRect,
                                               tracking::TrackKCF,      // Use KCF tracker for collisions resolving
                                               tracking::MatchHungrian,
                                               0.2f,                     // Delta time for Kalman filter
                                               0.1f,                     // Accel noise magnitude for Kalman filter
                                               grayFrame.rows / 10,      // Distance threshold between region and object on two frames
                                               m_fps,                    // Maximum allowed skipped frames
                                               3 * m_fps                 // Maximum trace length
                                               );

        return true;
    }

    ///
    /// \brief ProcessFrame
    /// \param grayFrame
    ///
    void ProcessFrame(cv::UMat grayFrame)
    {
        const std::vector<Point_t>& centers = m_detector->Detect(grayFrame);
        const regions_t& regions = m_detector->GetDetects();

        m_tracker->Update(centers, regions, grayFrame);
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
            if (track->IsRobust(m_fps / 2,                         // Minimal trajectory size
                                0.6f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, *track);
            }
        }

        //detector.CalcMotionMap(frame);
    }

private:

    int m_minObjWidth;
    std::unique_ptr<CDetector> m_detector;
    std::unique_ptr<CTracker> m_tracker;
};

// ----------------------------------------------------------------------

///
/// \brief The FaceDetector class
///
class FaceDetector : public VideoExample
{
public:
    FaceDetector(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitTracker
    /// \param grayFrame
    ///
    bool InitTracker(cv::UMat /*grayFrame*/)
    {
        std::string fileName = "../data/haarcascade_frontalface_alt2.xml";
        m_cascade.load(fileName);
        if (m_cascade.empty())
        {
            std::cerr << "Cascade not opened!" << std::endl;
            return false;
        }

        m_tracker = std::make_unique<CTracker>(m_useLocalTracking,
                                               tracking::DistJaccard,
                                               tracking::KalmanUnscented,
                                               tracking::FilterRect,
                                               tracking::TrackKCF,      // Use KCF tracker for collisions resolving
                                               tracking::MatchHungrian,
                                               0.3f,                     // Delta time for Kalman filter
                                               0.1f,                     // Accel noise magnitude for Kalman filter
                                               0.8f,                     // Distance threshold between region and object on two frames
                                               2 * m_fps,                // Maximum allowed skipped frames
                                               5 * m_fps                 // Maximum trace length
                                               );

        return true;
    }

    ///
    /// \brief ProcessFrame
    /// \param grayFrame
    ///
    void ProcessFrame(cv::UMat grayFrame)
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
        std::vector<Point_t> centers;
        regions_t regions;
        for (auto rect : faceRects)
        {
            centers.push_back((rect.tl() + rect.br()) / 2);
            regions.push_back(rect);
        }

        m_tracker->Update(centers, regions, grayFrame);
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
    }

private:

    cv::CascadeClassifier m_cascade;
    std::unique_ptr<CTracker> m_tracker;
};

// ----------------------------------------------------------------------

#define USE_HOG 1
///
/// \brief The PedestrianDetector class
///
class PedestrianDetector : public VideoExample
{
public:
    PedestrianDetector(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    #if !USE_HOG
          , m_scanner(HUMAN_height, HUMAN_width, HUMAN_xdiv, HUMAN_ydiv, 256, 0.8)
    #endif
    {
    }

protected:
    ///
    /// \brief InitTracker
    /// \param grayFrame
    ///
    bool InitTracker(cv::UMat /*grayFrame*/)
    {
#if USE_HOG
        m_hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
#else
        std::string cascade1 = "../data/combined.txt.model";
        std::string cascade2 = "../data/combined.txt.model_";
        LoadCascade(cascade1, cascade2, m_scanner);
#endif

        m_tracker = std::make_unique<CTracker>(m_useLocalTracking,
                                               tracking::DistJaccard,
                                               tracking::KalmanUnscented,
                                               tracking::FilterRect,
                                               tracking::TrackKCF,      // Use KCF tracker for collisions resolving
                                               tracking::MatchHungrian,
                                               0.3f,                     // Delta time for Kalman filter
                                               0.1f,                     // Accel noise magnitude for Kalman filter
                                               0.8f,                     // Distance threshold between region and object on two frames
                                               1 * m_fps,                // Maximum allowed skipped frames
                                               5 * m_fps                 // Maximum trace length
                                               );

        return true;
    }

    ///
    /// \brief ProcessFrame
    /// \param grayFrame
    ///
    void ProcessFrame(cv::UMat grayFrame)
    {
        std::vector<Point_t> centers;
        regions_t regions;

        std::vector<cv::Rect> foundRects;
        std::vector<cv::Rect> filteredRects;

        int neighbors = 0;
#if USE_HOG
        m_hog.detectMultiScale(grayFrame, foundRects, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 4, false);
#else
        IntImage<double> original;
        original.Load(grayFrame);

        m_scanner.FastScan(original, foundRects, 2);
        neighbors = 1;
#endif

        nms(foundRects, filteredRects, 0.3f, neighbors);

        for (auto rect : filteredRects)
        {
            rect.x += cvRound(rect.width * 0.1f);
            rect.width = cvRound(rect.width * 0.8f);
            rect.y += cvRound(rect.height * 0.07f);
            rect.height = cvRound(rect.height * 0.8f);

            centers.push_back((rect.tl() + rect.br()) / 2);
            regions.push_back(rect);
        }

        m_tracker->Update(centers, regions, grayFrame);
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
            if (track->IsRobust(m_fps / 2,                   // Minimal trajectory size
                                0.7f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, *track);
            }
        }
    }

private:

#if USE_HOG
    cv::HOGDescriptor m_hog;
#else
    static const int HUMAN_height = 108;
    static const int HUMAN_width = 36;
    static const int HUMAN_xdiv = 9;
    static const int HUMAN_ydiv = 4;

    DetectionScanner m_scanner;
#endif
    std::unique_ptr<CTracker> m_tracker;
};

// ----------------------------------------------------------------------

///
/// \brief The HybridFaceDetector class
///
class HybridFaceDetector : public VideoExample
{
public:
    HybridFaceDetector(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitTracker
    /// \param grayFrame
    ///
    bool InitTracker(cv::UMat grayFrame)
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
                                               grayFrame.cols / 10,      // Distance threshold between region and object on two frames
                                               2 * m_fps,                // Maximum allowed skipped frames
                                               5 * m_fps                 // Maximum trace length
                                               );

        m_detector = std::make_unique<CDetector>(BackgroundSubtract::ALG_MOG, m_useLocalTracking, grayFrame);
        m_detector->SetMinObjectSize(cv::Size(grayFrame.cols / 50, grayFrame.rows / 50));

        return true;
    }

    ///
    /// \brief ProcessFrame
    /// \param grayFrame
    ///
    void ProcessFrame(cv::UMat grayFrame)
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
        nms2(faceRects, scores, allRects, 0.3f, 1, 0.7);

        std::vector<Point_t> centers;
        regions_t regions;
        for (auto rect : allRects)
        {
            centers.push_back((rect.tl() + rect.br()) / 2);
            regions.push_back(rect);
        }

        m_tracker->Update(centers, regions, grayFrame);
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

        //detector.CalcMotionMap(frame);
    }

private:
    cv::CascadeClassifier m_cascade;
    std::unique_ptr<CDetector> m_detector;
    std::unique_ptr<CTracker> m_tracker;

    std::vector<cv::Rect> m_prevRects;
};
