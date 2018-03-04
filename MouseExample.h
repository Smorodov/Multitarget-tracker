#pragma once

//#include <opencv2/highgui/highgui_c.h>

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

    CTracker tracker(useLocalTracking,
                     tracking::DistCenters,
                     tracking::KalmanLinear,
                     tracking::FilterCenter,
                     tracking::TrackNone,
                     tracking::MatchHungrian,
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
            regions.push_back(CRegion(cv::Rect(cvRound(p.x), cvRound(p.y), 1, 1)));
        }


        for (size_t i = 0; i < pts.size(); i++)
        {
            cv::circle(frame, pts[i], 3, cv::Scalar(0, 255, 0), 1, CV_AA);
        }

        tracker.Update(regions, cv::UMat(), 100);

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
