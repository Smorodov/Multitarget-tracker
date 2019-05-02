#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>

#include "BaseDetector.h"
#include "Ctracker.h"

// ----------------------------------------------------------------------

///
/// \brief The FrameInfo class
///
struct FrameInfo
{
	cv::Mat m_frame;
    cv::UMat m_clFrame;
	cv::UMat m_gray;
	regions_t m_regions;
	std::vector<TrackingObject> m_tracks;
	int64 m_dt = 0;
	float m_fps = 0;

	int m_inDetector = 0; // 0 - not in Detector, 1 - detector start processing, 2 - objects was detected
	int m_inTracker = 0; // 0 - not in Tracker, 1 - objects was tracked
};

#include "Queue.h"

// ----------------------------------------------------------------------

///
/// \brief The AsyncDetector class
///
class AsyncDetector
{
public:
    AsyncDetector(const cv::CommandLineParser& parser);
    ~AsyncDetector();

    void Process();

private:
    bool m_showLogs = false;
    float m_fps = 0;

    std::string m_inFile;
    std::string m_outFile;
    int m_startFrame = 0;
    int m_endFrame = 0;
    int m_finishDelay = 0;
    std::vector<cv::Scalar> m_colors;

	FramesQueue m_framesQue;

	void DrawData(FrameInfo* frameInfo, int framesCounter, int currTime);

    void DrawTrack(cv::Mat frame, int resizeCoeff, const TrackingObject& track, bool drawTrajectory = true);

    static void CaptureThread(std::string fileName, int startFrame, float* fps, FramesQueue* framesQue, bool* stopFlag);
    static void DetectThread(const config_t& config, cv::UMat firstGray, FramesQueue* framesQue, bool* stopFlag);
	static void TrackingThread(const TrackerSettings& settings, FramesQueue* framesQue, bool* stopFlag);
};
