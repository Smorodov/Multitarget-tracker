#include "opencv2/opencv.hpp"
#include "BackgroundSubtract.h"
#include "Detector.h"

#include <opencv2/highgui/highgui_c.h>
#include "Ctracker.h"
#include <iostream>
#include <vector>

#include "pedestrians/c4-pedestrian-detector.h"
#include "nms.h"

//------------------------------------------------------------------------
// Mouse callbacks
//------------------------------------------------------------------------
void mv_MouseCallback(int event, int x, int y, int /*flags*/, void* param)
{
    if (event == cv::EVENT_MOUSEMOVE)
    {
        cv::Point2f* p = (cv::Point2f*)param;
        if (p)
        {
            p->x = static_cast<float>(x);
            p->y = static_cast<float>(y);
        }
    }
}

// ----------------------------------------------------------------------
void MouseTracking(cv::CommandLineParser parser)
{
    std::string outFile = parser.get<std::string>("out");

    cv::VideoWriter writer;

    int k = 0;
    std::vector<cv::Scalar> colors = { cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 127, 255), cv::Scalar(127, 0, 255), cv::Scalar(127, 0, 127) };
    cv::namedWindow("Video");
    cv::Mat frame = cv::Mat(800, 800, CV_8UC3);

    if (!writer.isOpened())
    {
        writer.open(outFile, cv::VideoWriter::fourcc('P', 'I', 'M', '1'), 20, frame.size(), true);
    }

    // Set mouse callback
    cv::Point2f pointXY;
    cv::setMouseCallback("Video", mv_MouseCallback, (void*)&pointXY);

    bool useLocalTracking = false;

    CTracker tracker(useLocalTracking,
                     CTracker::DistCenters,
                     CTracker::KalmanLinear,
                     CTracker::FilterCenter,
                     CTracker::TrackNone,
                     CTracker::MatchHungrian,
                     0.2f,
                     0.5f,
                     100.0f,
                     25,
                     25);
    track_t alpha = 0;
    cv::RNG rng;
    while (k != 27)
    {
        frame = cv::Scalar::all(0);

        // Noise addition (measurements/detections simulation )
        float Xmeasured = pointXY.x + static_cast<float>(rng.gaussian(2.0));
        float Ymeasured = pointXY.y + static_cast<float>(rng.gaussian(2.0));

        // Append circulating around mouse cv::Points (frequently intersecting)
        std::vector<Point_t> pts;
        pts.push_back(Point_t(Xmeasured + 100.0f*sin(-alpha), Ymeasured + 100.0f*cos(-alpha)));
        pts.push_back(Point_t(Xmeasured + 100.0f*sin(alpha), Ymeasured + 100.0f*cos(alpha)));
        pts.push_back(Point_t(Xmeasured + 100.0f*sin(alpha / 2.0f), Ymeasured + 100.0f*cos(alpha / 2.0f)));
        pts.push_back(Point_t(Xmeasured + 100.0f*sin(alpha / 3.0f), Ymeasured + 100.0f*cos(alpha / 1.0f)));
        alpha += 0.05f;

        regions_t regions;
        for (auto p : pts)
        {
            regions.push_back(CRegion(cv::Rect(static_cast<int>(p.x - 1), static_cast<int>(p.y - 1), 3, 3)));
        }


        for (size_t i = 0; i < pts.size(); i++)
        {
            cv::circle(frame, pts[i], 3, cv::Scalar(0, 255, 0), 1, CV_AA);
        }

        tracker.Update(pts, regions, cv::Mat());

        std::cout << tracker.tracks.size() << std::endl;

        for (size_t i = 0; i < tracker.tracks.size(); i++)
        {
            const auto& track = tracker.tracks[i];

            if (track->m_trace.size() > 1)
            {
                for (size_t j = 0; j < track->m_trace.size() - 1; j++)
                {
                    cv::line(frame, track->m_trace[j], track->m_trace[j + 1], colors[i % colors.size()], 2, CV_AA);
                }
            }
        }

        cv::imshow("Video", frame);

        if (writer.isOpened())
        {
            writer << frame;
        }

        k = cv::waitKey(10);
    }
}

// ----------------------------------------------------------------------
void DrawTrack(cv::Mat frame,
               const CTrack& track,
               const std::vector<cv::Scalar>& colors
               )
{
    cv::rectangle(frame, track.GetLastRect(), cv::Scalar(0, 255, 0), 1, CV_AA);

    cv::Scalar cl = colors[track.m_trackID % colors.size()];

    for (size_t j = 0; j < track.m_trace.size() - 1; ++j)
    {
        const TrajectoryPoint& pt1 = track.m_trace.at(j);
        const TrajectoryPoint& pt2 = track.m_trace.at(j + 1);

        cv::line(frame, pt1.m_prediction, pt2.m_prediction, cl, 1, CV_AA);
        if (!pt2.m_hasRaw)
        {
            cv::circle(frame, pt2.m_prediction, 4, cl, 1, CV_AA);
        }
    }
}

// ----------------------------------------------------------------------
void MotionDetector(cv::CommandLineParser parser)
{
    std::string inFile = parser.get<std::string>(0);
    std::string outFile = parser.get<std::string>("out");

    cv::VideoWriter writer;

    std::vector<cv::Scalar> colors = { cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 127, 255), cv::Scalar(127, 0, 255), cv::Scalar(127, 0, 127) };
    cv::VideoCapture capture(inFile);
    if (!capture.isOpened())
    {
        return;
    }
    cv::namedWindow("Video");
    cv::Mat frame;
    cv::Mat gray;

    bool showLogs = parser.get<int>("show_logs") != 0;

    const int StartFrame = parser.get<int>("start_frame");
    const int EndFrame = parser.get<int>("end_frame");
    capture.set(cv::CAP_PROP_POS_FRAMES, StartFrame);

    const int fps = std::max(1, cvRound(capture.get(cv::CAP_PROP_FPS)));

    capture >> frame;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // If true then trajectories will be more smooth and accurate
    // But on high resolution videos with many objects may be to slow
    bool useLocalTracking = false;

    CDetector detector(BackgroundSubtract::ALG_MOG, useLocalTracking, gray);
    detector.SetMinObjectSize(cv::Size(gray.cols / 50, gray.rows / 50));
    //detector.SetMinObjectSize(cv::Size(2, 2));

    CTracker tracker(useLocalTracking,
                     CTracker::DistRects,
                     CTracker::KalmanLinear,
                     CTracker::FilterRect,
                     CTracker::TrackKCF,      // Use KCF tracker for collisions resolving
                     CTracker::MatchBipart,
                     0.3f,                    // Delta time for Kalman filter
                     0.1f,                    // Accel noise magnitude for Kalman filter
                     gray.cols / 20.0f,       // Distance threshold between two frames
                     fps,                     // Maximum allowed skipped frames
                     5 * fps                 // Maximum trace length
                     );

    int k = 0;

    double freq = cv::getTickFrequency();

    int64 allTime = 0;

    bool manualMode = false;
    int framesCounter = StartFrame + 1;
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
            writer.open(outFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), capture.get(cv::CAP_PROP_FPS), frame.size(), true);
        }

        int64 t1 = cv::getTickCount();

        const std::vector<Point_t>& centers = detector.Detect(gray);
        const regions_t& regions = detector.GetDetects();

        tracker.Update(centers, regions, gray);

        int64 t2 = cv::getTickCount();

        allTime += t2 - t1;
        int currTime = cvRound(1000 * (t2 - t1) / freq);

        if (showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << tracker.tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : tracker.tracks)
        {
            if (track->IsRobust(fps / 2,                         // Minimal trajectory size
                                0.5f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, *track, colors);
            }
        }

        detector.CalcMotionMap(frame);

        cv::imshow("Video", frame);

        int waitTime = manualMode ? 0 : std::max<int>(1, 1000 / fps - currTime);
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
        if (EndFrame && framesCounter > EndFrame)
        {
            break;
        }
    }

    std::cout << "work time = " << (allTime / freq) << std::endl;
    cv::waitKey(parser.get<int>("end_delay"));
}

// ----------------------------------------------------------------------
void FaceDetector(cv::CommandLineParser parser)
{
    std::string inFile = parser.get<std::string>(0);
    std::string outFile = parser.get<std::string>("out");

    cv::VideoWriter writer;

    std::vector<cv::Scalar> colors = { cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 127, 255), cv::Scalar(127, 0, 255), cv::Scalar(127, 0, 127) };
    cv::VideoCapture capture(inFile);
    if (!capture.isOpened())
    {
        return;
    }
    cv::namedWindow("Video");
    cv::Mat frame;
    cv::Mat gray;

    bool showLogs = parser.get<int>("show_logs") != 0;

    const int StartFrame = parser.get<int>("start_frame");
    const int EndFrame = parser.get<int>("end_frame");
    capture.set(cv::CAP_PROP_POS_FRAMES, StartFrame);

    const int fps = std::max(1, cvRound(capture.get(cv::CAP_PROP_FPS)));

    capture >> frame;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // If true then trajectories will be more smooth and accurate
    // But on high resolution videos with many objects may be to slow
    bool useLocalTracking = false;

    cv::CascadeClassifier cascade;
    std::string fileName = "../data/haarcascade_frontalface_alt2.xml";
    cascade.load(fileName);
    if (cascade.empty())
    {
        std::cerr << "Cascade not opened!" << std::endl;
        return;
    }

    CTracker tracker(useLocalTracking,
                     CTracker::DistJaccard,   // For this distance type threshold must be from 0 to 1
                     CTracker::KalmanUnscented,
                     CTracker::FilterRect,
                     CTracker::TrackKCF,      // Use KCF tracker for collisions resolving
                     CTracker::MatchHungrian,
                     0.3f,                    // Delta time for Kalman filter
                     0.1f,                    // Accel noise magnitude for Kalman filter
                     0.8f,                    // Distance threshold between two frames
                     2 * fps,                 // Maximum allowed skipped frames
                     5 * fps                  // Maximum trace length
                     );

    int k = 0;

    double freq = cv::getTickFrequency();

    int64 allTime = 0;

    bool manualMode = false;
    int framesCounter = StartFrame + 1;
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
            writer.open(outFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), capture.get(cv::CAP_PROP_FPS), frame.size(), true);
        }

        int64 t1 = cv::getTickCount();


        bool findLargestObject = false;
        bool filterRects = true;
        std::vector<cv::Rect> faceRects;
        cascade.detectMultiScale(gray,
                                 faceRects,
                                 1.1,
                                 (filterRects || findLargestObject) ? 3 : 0,
                                 findLargestObject ? cv::CASCADE_FIND_BIGGEST_OBJECT : 0,
                                 cv::Size(gray.cols / 20, gray.rows / 20),
                                 cv::Size(gray.cols / 2, gray.rows / 2));
        std::vector<Point_t> centers;
        regions_t regions;
        for (auto rect : faceRects)
        {
            centers.push_back((rect.tl() + rect.br()) / 2);
            regions.push_back(rect);
        }

        tracker.Update(centers, regions, gray);

        int64 t2 = cv::getTickCount();

        allTime += t2 - t1;
        int currTime = cvRound(1000 * (t2 - t1) / freq);

        if (showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << tracker.tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : tracker.tracks)
        {
            if (track->IsRobust(8,                           // Minimal trajectory size
                                0.1f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, *track, colors);
            }
        }

        cv::imshow("Video", frame);

        int waitTime = manualMode ? 0 : std::max<int>(1, 1000 / fps - currTime);
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
        if (EndFrame && framesCounter > EndFrame)
        {
            break;
        }
    }

    std::cout << "work time = " << (allTime / freq) << std::endl;
    cv::waitKey(parser.get<int>("end_delay"));
}

// ----------------------------------------------------------------------
void PedestrianDetector(cv::CommandLineParser parser)
{
    std::string inFile = parser.get<std::string>(0);
    std::string outFile = parser.get<std::string>("out");

    cv::VideoWriter writer;

    std::vector<cv::Scalar> colors = { cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 127, 255), cv::Scalar(127, 0, 255), cv::Scalar(127, 0, 127) };
    cv::VideoCapture capture(inFile);
    if (!capture.isOpened())
    {
        return;
    }
    cv::namedWindow("Video");
    cv::Mat frame;
    cv::Mat gray;

    bool showLogs = parser.get<int>("show_logs") != 0;

    const int StartFrame = parser.get<int>("start_frame");
    const int EndFrame = parser.get<int>("end_frame");
    capture.set(cv::CAP_PROP_POS_FRAMES, StartFrame);

    const int fps = std::max(1, cvRound(capture.get(cv::CAP_PROP_FPS)));

    capture >> frame;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // If true then trajectories will be more smooth and accurate
    // But on high resolution videos with many objects may be to slow
    bool useLocalTracking = false;

#define USE_HOG 0

#if USE_HOG
    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
#else
    const int HUMAN_height = 108;
    const int HUMAN_width = 36;
    const int HUMAN_xdiv = 9;
    const int HUMAN_ydiv = 4;

    DetectionScanner scanner(HUMAN_height, HUMAN_width, HUMAN_xdiv, HUMAN_ydiv, 256, 0.8);

    std::string cascade1 = "../data/combined.txt.model";
    std::string cascade2 = "../data/combined.txt.model_";
    LoadCascade(cascade1, cascade2, scanner);
#endif

    CTracker tracker(useLocalTracking,
                     CTracker::DistRects,
                     CTracker::KalmanUnscented,
                     CTracker::FilterRect,
                     CTracker::TrackKCF,      // Use KCF tracker for collisions resolving
                     CTracker::MatchHungrian,
                     0.3f,                    // Delta time for Kalman filter
                     0.1f,                    // Accel noise magnitude for Kalman filter
                     gray.cols / 10.0f,       // Distance threshold between two frames
                     2 * fps,                 // Maximum allowed skipped frames
                     5 * fps                  // Maximum trace length
                     );

    int k = 0;

    double freq = cv::getTickFrequency();

    int64 allTime = 0;

    bool manualMode = false;
    int framesCounter = StartFrame + 1;
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
            writer.open(outFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), capture.get(cv::CAP_PROP_FPS), frame.size(), true);
        }

        int64 t1 = cv::getTickCount();

        std::vector<Point_t> centers;
        regions_t regions;

        std::vector<cv::Rect> foundRects;
        std::vector<cv::Rect> filteredRects;

#if USE_HOG
        hog.detectMultiScale(gray, foundRects, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 6, false);
#else
        IntImage<double> original;
        original.Load(gray);

        scanner.FastScan(original, foundRects, 2);
#endif

        nms(foundRects, filteredRects, 0.3f, 1);

        for (auto rect : filteredRects)
        {
            rect.x += cvRound(rect.width * 0.1f);
            rect.width = cvRound(rect.width * 0.8f);
            rect.y += cvRound(rect.height * 0.07f);
            rect.height = cvRound(rect.height * 0.8f);

            centers.push_back((rect.tl() + rect.br()) / 2);
            regions.push_back(rect);
        }

        tracker.Update(centers, regions, gray);

        int64 t2 = cv::getTickCount();

        allTime += t2 - t1;
        int currTime = cvRound(1000 * (t2 - t1) / freq);

        if (showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << tracker.tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : tracker.tracks)
        {
            if (track->IsRobust(fps / 2,                         // Minimal trajectory size
                                0.7f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, *track, colors);
            }
        }

        cv::imshow("Video", frame);

        int waitTime = manualMode ? 0 : std::max<int>(1, 1000 / fps - currTime);
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
        if (EndFrame && framesCounter > EndFrame)
        {
            break;
        }
    }

    std::cout << "work time = " << (allTime / freq) << std::endl;
    cv::waitKey(parser.get<int>("end_delay"));
}

// ----------------------------------------------------------------------
void HybridFaceDetector(cv::CommandLineParser parser)
{
    std::string inFile = parser.get<std::string>(0);
    std::string outFile = parser.get<std::string>("out");

    cv::VideoWriter writer;

    std::vector<cv::Scalar> colors = { cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 127, 255), cv::Scalar(127, 0, 255), cv::Scalar(127, 0, 127) };
    cv::VideoCapture capture(inFile);
    if (!capture.isOpened())
    {
        return;
    }
    cv::namedWindow("Video");
    cv::Mat frame;
    cv::Mat gray;

    bool showLogs = parser.get<int>("show_logs") != 0;

    const int StartFrame = parser.get<int>("start_frame");
    const int EndFrame = parser.get<int>("end_frame");
    capture.set(cv::CAP_PROP_POS_FRAMES, StartFrame);

    const int fps = std::max(1, cvRound(capture.get(cv::CAP_PROP_FPS)));

    capture >> frame;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // If true then trajectories will be more smooth and accurate
    // But on high resolution videos with many objects may be to slow
    bool useLocalTracking = false;

    cv::CascadeClassifier cascade;
    std::string fileName = "../data/haarcascade_frontalface_alt2.xml";
    cascade.load(fileName);
    if (cascade.empty())
    {
        std::cerr << "Cascade not opened!" << std::endl;
        return;
    }

    CDetector detector(BackgroundSubtract::ALG_MOG, useLocalTracking, gray);
    detector.SetMinObjectSize(cv::Size(gray.cols / 50, gray.rows / 50));

    CTracker tracker(useLocalTracking,
                     CTracker::DistCenters,   // For this distance type threshold must be from 0 to 1
                     CTracker::KalmanUnscented,
                     CTracker::FilterRect,
                     CTracker::TrackKCF,      // Use KCF tracker for collisions resolving
                     CTracker::MatchHungrian,
                     0.3f,                    // Delta time for Kalman filter
                     0.1f,                    // Accel noise magnitude for Kalman filter
                     gray.cols / 10,          // Distance threshold between two frames
                     2 * fps,                 // Maximum allowed skipped frames
                     5 * fps                  // Maximum trace length
                     );

    std::vector<cv::Rect> prevRects;

    int k = 0;

    double freq = cv::getTickFrequency();

    int64 allTime = 0;

    bool manualMode = false;
    int framesCounter = StartFrame + 1;
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
            writer.open(outFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), capture.get(cv::CAP_PROP_FPS), frame.size(), true);
        }

        int64 t1 = cv::getTickCount();


        bool findLargestObject = false;
        bool filterRects = true;
        std::vector<cv::Rect> faceRects;
        cascade.detectMultiScale(gray,
                                 faceRects,
                                 1.1,
                                 (filterRects || findLargestObject) ? 1 : 0,
                                 findLargestObject ? cv::CASCADE_FIND_BIGGEST_OBJECT : 0,
                                 cv::Size(gray.cols / 20, gray.rows / 20),
                                 cv::Size(gray.cols / 2, gray.rows / 2));

        detector.Detect(gray);
        const regions_t& regions1 = detector.GetDetects();

        std::vector<float> scores(faceRects.size(), 0.5f);
        for (const auto& reg : regions1)
        {
            faceRects.push_back(reg.m_rect);
            scores.push_back(0.4f);
        }
        faceRects.insert(faceRects.end(), prevRects.begin(), prevRects.end());
        scores.insert(scores.end(), prevRects.size(), 0.4f);
        std::vector<cv::Rect> allRects;
        nms2(faceRects, scores, allRects, 0.3f, 1, 0.7);

        std::vector<Point_t> centers;
        regions_t regions;
        for (auto rect : allRects)
        {
            centers.push_back((rect.tl() + rect.br()) / 2);
            regions.push_back(rect);
        }

        tracker.Update(centers, regions, gray);

        int64 t2 = cv::getTickCount();

        allTime += t2 - t1;
        int currTime = cvRound(1000 * (t2 - t1) / freq);

        if (showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << tracker.tracks.size() << ", time = " << currTime << std::endl;
        }

        prevRects.clear();
        for (const auto& track : tracker.tracks)
        {
            if (track->IsRobust(8,                           // Minimal trajectory size
                                0.1f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, *track, colors);
                prevRects.push_back(track->GetLastRect());
            }
        }

        detector.CalcMotionMap(frame);

        cv::imshow("Video", frame);

        int waitTime = manualMode ? 0 : std::max<int>(1, 1000 / fps - currTime);
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
        if (EndFrame && framesCounter > EndFrame)
        {
            break;
        }
    }

    std::cout << "work time = " << (allTime / freq) << std::endl;
    cv::waitKey(parser.get<int>("end_delay"));
}

// ----------------------------------------------------------------------
static void Help()
{
    printf("\nExamples of the Multitarget tracking algorithm\n"
           "Usage: \n"
           "          ./MultitargetTracker <path to movie file> [--example]=<number of example 0..3> [--start_frame]=<start a video from this position> [--end_frame]=<play a video to this position> [--end_delay]=<delay in milliseconds after video ending> [--out]=<name of result video file> [--show_logs]=<show logs> \n\n"
           "Press:\n"
           "\'m\' key for change mode: play|pause. When video is paused you can press any key for get next frame. \n\n"
           "Press Esc to exit from video \n\n"
           );
}

const char* keys =
{
    "{ @1             |../data/atrium.avi  | movie file | }"
    "{ e  example     |1                   | number of example 0 - MouseTracking, 1 - MotionDetector, 2 - FaceDetector, 3 - PedestrianDetector | }"
    "{ sf start_frame |0                   | Start a video from this position | }"
    "{ ef end_frame   |0                   | Play a video to this position (if 0 then played to the end of file) | }"
    "{ ed end_delay   |0                   | Delay in milliseconds after video ending | }"
    "{ o  out         |                    | Name of result video file | }"
    "{ sl show_logs   |1                   | Show Trackers logs | }"
};

// ----------------------------------------------------------------------
int main(int argc, char** argv)
{
    Help();

    cv::CommandLineParser parser(argc, argv, keys);

    int exampleNum = parser.get<int>("example");;

    switch (exampleNum)
    {
    case 0:
        MouseTracking(parser);
        break;

    case 1:
        MotionDetector(parser);
        break;

    case 2:
        FaceDetector(parser);
        break;

    case 3:
        PedestrianDetector(parser);
        break;

    case 4:
        HybridFaceDetector(parser);
        break;
    }


    cv::destroyAllWindows();
    return 0;
}
