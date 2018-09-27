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

    bool m_showLogs;
    float m_fps;
    bool m_useLocalTracking;

    int m_captureTimeOut;
    int m_trackingTimeOut;

    static void CaptureAndDetect(CarsCounting* thisPtr,
                                 bool* stopCapture,
                                 std::mutex* frameLock,
                                 std::condition_variable* frameCond,
                                 std::mutex* trackLock,
                                 std::condition_variable* trackCond);

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

    int m_minObjWidth = 10;
};
