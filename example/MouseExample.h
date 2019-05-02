#pragma once

#include "Ctracker.h"

#include <iostream>
#include <vector>

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

    TrackerSettings settings;
    settings.m_useLocalTracking = useLocalTracking;
    settings.m_distType = tracking::DistCenters;
    settings.m_kalmanType = tracking::KalmanLinear;
    settings.m_filterGoal = tracking::FilterCenter;
    settings.m_lostTrackType = tracking::TrackNone;
    settings.m_matchType = tracking::MatchHungrian;
    settings.m_dt = 0.2f;
    settings.m_accelNoiseMag = 0.5f;
    settings.m_distThres = 100.0f;
    settings.m_maximumAllowedSkippedFrames = 25;
    settings.m_maxTraceLength = 25;

    CTracker tracker(settings);

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
            regions.push_back(CRegion(cv::Rect(cvRound(p.x), cvRound(p.y), 1, 1)));
        }


        for (size_t i = 0; i < pts.size(); i++)
        {
#if (CV_VERSION_MAJOR >= 4)
            cv::circle(frame, pts[i], 3, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
#else
			cv::circle(frame, pts[i], 3, cv::Scalar(0, 255, 0), 1, CV_AA);
#endif
        }

        tracker.Update(regions, cv::UMat(), 100);

		auto tracks = tracker.GetTracks();
        std::cout << tracks.size() << std::endl;

        for (size_t i = 0; i < tracks.size(); i++)
        {
            const auto& track = tracks[i];

            if (track.m_trace.size() > 1)
            {
                for (size_t j = 0; j < track.m_trace.size() - 1; j++)
                {
#if (CV_VERSION_MAJOR >= 4)
                    cv::line(frame, track.m_trace[j], track.m_trace[j + 1], colors[i % colors.size()], 2, cv::LINE_AA);
#else
					cv::line(frame, track.m_trace[j], track.m_trace[j + 1], colors[i % colors.size()], 2, CV_AA);
#endif
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
