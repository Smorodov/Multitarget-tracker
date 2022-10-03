#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <chrono>
#include <atomic>
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
	regions_t m_regions;
	std::vector<TrackingObject> m_tracks;
	int64 m_dt = 0;
	float m_fps = 0;
	size_t m_frameInd = 0;

	static constexpr int StateNotProcessed = 0;
	static constexpr int StateInProcess = 1;
	static constexpr int StateCompleted = 2;
	static constexpr int StateSkipped = 3;

    std::atomic<int> m_inDetector; // 0 - not in Detector, 1 - detector started processing, 2 - objects was detected, 3 - detector skip this frame
    std::atomic<int> m_inTracker;  // 0 - not in Tracker, 1 - tracker started processing, 2 - objects was tracked

    FrameInfo(size_t frameInd)
        : m_frameInd(frameInd), m_inDetector(StateNotProcessed), m_inTracker(StateNotProcessed)
    {
    }
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
    ~AsyncDetector() = default;

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

	void DrawData(frame_ptr frameInfo, int framesCounter, int currTime);
    void DrawTrack(cv::Mat frame, const TrackingObject& track, bool drawTrajectory = true);

    static void CaptureThread(std::string fileName, int startFrame, float* fps, FramesQueue* framesQue, bool* stopFlag);
    static void DetectThread(const config_t& config, cv::Mat firstFrame, FramesQueue* framesQue, bool* stopFlag);
	static void TrackingThread(const TrackerSettings& settings, FramesQueue* framesQue, bool* stopFlag);
};
