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

// ----------------------------------------------------------------------

///
/// \brief The CarsCounting class
///
class CarsCounting
{
public:
    CarsCounting(const cv::CommandLineParser& parser);
    virtual ~CarsCounting();

    void Process();

protected:
    std::unique_ptr<BaseDetector> m_detector;
    std::unique_ptr<CTracker> m_tracker;

    bool m_showLogs = false;
    float m_fps = 0;
    bool m_useLocalTracking = false;

    virtual bool GrayProcessing() const;

    virtual bool InitTracker(cv::UMat frame);

    void Detection(cv::Mat frame, cv::UMat grayFrame, regions_t& regions);
    void Tracking(cv::Mat frame, cv::UMat grayFrame, const regions_t& regions);

    virtual void DrawData(cv::Mat frame, int framesCounter, int currTime);

    void DrawTrack(cv::Mat frame,
                   int resizeCoeff,
                   const CTrack& track,
                   bool drawTrajectory = true,
                   bool isStatic = false);

private:
    bool m_isTrackerInitialized = false;
    std::string m_inFile;
    std::string m_outFile;
    int m_startFrame = 0;
    int m_endFrame = 0;
    int m_finishDelay = 0;
    std::vector<cv::Scalar> m_colors;

    int m_minObjWidth = 10;
};
