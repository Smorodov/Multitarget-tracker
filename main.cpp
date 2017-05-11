#include "opencv2/opencv.hpp"
#include "BackgroundSubtract.h"
#include "Detector.h"

#include <opencv2/highgui/highgui_c.h>
#include "Ctracker.h"
#include <iostream>
#include <vector>

#include "pedestrians/c4-pedestrian-detector.h"

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
void nms(
        const std::vector<cv::Rect>& srcRects,
        std::vector<cv::Rect>& resRects,
        float thresh,
        int neighbors = 0
        )
{
    resRects.clear();

    const size_t size = srcRects.size();
    if (!size)
    {
        return;
    }

    // Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
    std::multimap<int, size_t> idxs;
    for (size_t i = 0; i < size; ++i)
    {
        idxs.insert(std::pair<int, size_t>(srcRects[i].br().y, i));
    }

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0)
    {
        // grab the last rectangle
        auto lastElem = --std::end(idxs);
        const cv::Rect& rect1 = srcRects[lastElem->second];

        int neigborsCount = 0;

        idxs.erase(lastElem);

        for (auto pos = std::begin(idxs); pos != std::end(idxs); )
        {
            // grab the current rectangle
            const cv::Rect& rect2 = srcRects[pos->second];

            float intArea = (rect1 & rect2).area();
            float unionArea = rect1.area() + rect2.area() - intArea;
            float overlap = intArea / unionArea;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh)
            {
                pos = idxs.erase(pos);
                ++neigborsCount;
            }
            else
            {
                ++pos;
            }
        }
        if (neigborsCount >= neighbors)
        {
            resRects.push_back(rect1);
        }
    }
}

// ----------------------------------------------------------------------
void MouseTracking(int argc, char** argv)
{
    // First argument ingnored

    std::string outFile;
    if (argc > 2)
    {
        outFile = argv[2];
    }

    cv::VideoWriter writer;

    int k = 0;
    cv::Scalar Colors[] = { cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 255, 255) };
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

    CTracker tracker(useLocalTracking, CTracker::CentersDist, CTracker::KalmanLinear, CTracker::FilterCenter, CTracker::TrackNone, CTracker::MatchHungrian, 0.2f, 0.5f, 100.0f, 25, 25);
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
            const auto& tracks = tracker.tracks[i];

            if (tracks->m_trace.size() > 1)
            {
                for (size_t j = 0; j < tracks->m_trace.size() - 1; j++)
                {
                    cv::line(frame, tracks->m_trace[j], tracks->m_trace[j + 1], Colors[i % 6], 2, CV_AA);
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
void MotionDetector(int argc, char** argv)
{
    std::string inFile("../data/atrium.avi");

    if (argc > 1)
    {
        inFile = argv[1];
    }

    std::string outFile;
    if (argc > 2)
    {
        outFile = argv[2];
    }

    cv::VideoWriter writer;

    cv::Scalar Colors[] = { cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 127, 255), cv::Scalar(127, 0, 255), cv::Scalar(127, 0, 127) };
    cv::VideoCapture capture(inFile);
    if (!capture.isOpened())
    {
        return;
    }
    cv::namedWindow("Video");
    cv::Mat frame;
    cv::Mat gray;

    const int StartFrame = 0;
    capture.set(cv::CAP_PROP_POS_FRAMES, StartFrame);

    const int fps = std::max(1, cvRound(capture.get(cv::CAP_PROP_FPS)));

    capture >> frame;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // If true then trajectories will be more smooth and accurate
    // But on high resolution videos with many objects may be to slow
    bool useLocalTracking = true;

    CDetector detector(BackgroundSubtract::ALG_CNT, useLocalTracking, gray);
    detector.SetMinObjectSize(cv::Size(gray.cols / 50, gray.rows / 50));
    //detector.SetMinObjectSize(cv::Size(2, 2));

    CTracker tracker(useLocalTracking,
                     CTracker::RectsDist,
                     CTracker::KalmanUnscented,
                     CTracker::FilterRect,
                     CTracker::TrackKCF,      // Use KCF tracker for collisions resolving
                     CTracker::MatchBipart,
                     0.3f,                    // Delta time for Kalman filter
                     0.1f,                    // Accel noise magnitude for Kalman filter
                     gray.cols / 20.0f,       // Distance threshold between two frames
                     fps,                     // Maximum allowed skipped frames
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

        const std::vector<Point_t>& centers = detector.Detect(gray);
        const regions_t& regions = detector.GetDetects();

        tracker.Update(centers, regions, gray);

        int64 t2 = cv::getTickCount();

        allTime += t2 - t1;
        int currTime = cvRound(1000 * (t2 - t1) / freq);

        std::cout << "Frame " << framesCounter << ": tracks = " << tracker.tracks.size() << ", time = " << currTime << std::endl;

        for (const auto& track : tracker.tracks)
        {
            if (track->IsRobust(fps / 2,                     // Minimal trajectory size
                                0.7f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.5f, 4.0f))      // Min and max ratio: width / height
                    )
            {
                cv::rectangle(frame, track->GetLastRect(), cv::Scalar(0, 255, 0), 1, CV_AA);

                cv::Scalar cl = Colors[track->m_trackID % 9];

                for (size_t j = 0; j < track->m_trace.size() - 1; ++j)
                {
                    cv::line(frame, track->m_trace[j], track->m_trace[j + 1], cl, 1, CV_AA);
                    if (!track->m_trace.HasRaw(j + 1))
                    {
                        cv::circle(frame, track->m_trace[j + 1], 4, cl, 1, CV_AA);
                    }
                }
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
        if (framesCounter > 215)
        {
            //break;
        }
    }

    std::cout << "work time = " << (allTime / freq) << std::endl;
    cv::waitKey(0);
}

// ----------------------------------------------------------------------
void FaceDetector(int argc, char** argv)
{
    std::string inFile("../data/smuglanka.mp4");

    if (argc > 1)
    {
        inFile = argv[1];
    }

    std::string outFile;
    if (argc > 2)
    {
        outFile = argv[2];
    }

    cv::VideoWriter writer;

    cv::Scalar Colors[] = { cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 127, 255), cv::Scalar(127, 0, 255), cv::Scalar(127, 0, 127) };
    cv::VideoCapture capture(inFile);
    if (!capture.isOpened())
    {
        return;
    }
    cv::namedWindow("Video");
    cv::Mat frame;
    cv::Mat gray;

    const int StartFrame = 0;
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
                     CTracker::RectsDist,
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

        std::cout << "Frame " << framesCounter << ": tracks = " << tracker.tracks.size() << ", time = " << currTime << std::endl;

        for (const auto& track : tracker.tracks)
        {
            if (track->IsRobust(8,                           // Minimal trajectory size
                                0.5f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                cv::rectangle(frame, track->GetLastRect(), cv::Scalar(0, 255, 0), 1, CV_AA);

                cv::Scalar cl = Colors[track->m_trackID % 9];

                for (size_t j = 0; j < track->m_trace.size() - 1; ++j)
                {
                    cv::line(frame, track->m_trace[j], track->m_trace[j + 1], cl, 1, CV_AA);
                    if (!track->m_trace.HasRaw(j + 1))
                    {
                        cv::circle(frame, track->m_trace[j + 1], 4, cl, 1, CV_AA);
                    }
                }
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
        if (framesCounter > 215)
        {
            //break;
        }
    }

    std::cout << "work time = " << (allTime / freq) << std::endl;
    cv::waitKey(0);
}

// ----------------------------------------------------------------------
void PedestrianDetector(int argc, char** argv)
{
    std::string inFile("TownCentreXVID.avi");

    if (argc > 1)
    {
        inFile = argv[1];
    }

    std::string outFile;
    if (argc > 2)
    {
        outFile = argv[2];
    }

    cv::VideoWriter writer;

    cv::Scalar Colors[] = { cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 127, 255), cv::Scalar(127, 0, 255), cv::Scalar(127, 0, 127) };
    cv::VideoCapture capture(inFile);
    if (!capture.isOpened())
    {
        return;
    }
    cv::namedWindow("Video");
    cv::Mat frame;
    cv::Mat gray;

    const int StartFrame = 0;
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
                     CTracker::RectsDist,
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

        std::cout << "Frame " << framesCounter << ": tracks = " << tracker.tracks.size() << ", time = " << currTime << std::endl;

        for (const auto& track : tracker.tracks)
        {
            if (track->IsRobust(fps / 2,                         // Minimal trajectory size
                                0.7f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                cv::rectangle(frame, track->GetLastRect(), cv::Scalar(0, 255, 0), 1, CV_AA);

                cv::Scalar cl = Colors[track->m_trackID % 9];

                for (size_t j = 0; j < track->m_trace.size() - 1; ++j)
                {
                    cv::line(frame, track->m_trace[j], track->m_trace[j + 1], cl, 1, CV_AA);
                    if (!track->m_trace.HasRaw(j + 1))
                    {
                        cv::circle(frame, track->m_trace[j + 1], 4, cl, 1, CV_AA);
                    }
                }
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
        if (framesCounter > 215)
        {
            //break;
        }
    }

    std::cout << "work time = " << (allTime / freq) << std::endl;
    cv::waitKey(0);
}

// ----------------------------------------------------------------------
int main(int argc, char** argv)
{
     int ExampleNum = 1;

     switch (ExampleNum)
     {
     case 0:
         MouseTracking(argc, argv);
         break;

     case 1:
         MotionDetector(argc, argv);
         break;

     case 2:
         FaceDetector(argc, argv);
         break;

     case 3:
         PedestrianDetector(argc, argv);
         break;
     }


	cv::destroyAllWindows();
	return 0;
}
