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
/// \brief The Gate struct
///
struct Gate
{
    bool m_gateOpen = false;
    mutable std::condition_variable m_cond;
    mutable std::mutex m_mutex;

    void Lock()
    {
        m_mutex.lock();
        m_gateOpen = false;
    }
    void Unlock()
    {
        m_gateOpen = true;
        m_mutex.unlock();
    }

    void OpenGate()
    {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_gateOpen = true;
        }
        m_cond.notify_all();
    }

    void WaitAtGate()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cond.wait(lock, [this]{ return m_gateOpen; });
        m_gateOpen = false;
    }

    bool WaitAtGateFor(int timeOut)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        bool res = m_cond.wait_for(lock, std::chrono::milliseconds(timeOut), [this]{ return m_gateOpen; });
        m_gateOpen = false;
        return res;
    }
};

// ----------------------------------------------------------------------

///
/// \brief The VideoExample class
///
class VideoExample
{
public:
    VideoExample(const cv::CommandLineParser& parser);
    virtual ~VideoExample();

    void AsyncProcess();
    void SyncProcess();
    void FutureProcess();

protected:
    std::unique_ptr<BaseDetector> m_detector;
    std::unique_ptr<CTracker> m_tracker;

    bool m_showLogs = true;
    float m_fps = 25;

    int m_captureTimeOut = 60000;
    int m_trackingTimeOut = 60000;

    static void CaptureAndDetect(VideoExample* thisPtr, bool* stopCapture, Gate* frameLock, Gate* trackLock);

    virtual bool GrayProcessing() const;

    virtual bool InitTracker(cv::UMat frame) = 0;

    void Detection(cv::Mat frame, cv::UMat grayFrame, regions_t& regions);
    void Tracking(cv::Mat frame, cv::UMat grayFrame, const regions_t& regions);

    virtual void DrawData(cv::Mat frame, int framesCounter, int currTime) = 0;

    void DrawTrack(cv::Mat frame, int resizeCoeff, const TrackingObject& track, bool drawTrajectory = true);

private:
    bool m_isTrackerInitialized = false;
    std::string m_inFile;
    std::string m_outFile;
    int m_fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    int m_startFrame = 0;
    int m_endFrame = 0;
    int m_finishDelay = 0;
    std::vector<cv::Scalar> m_colors;

    struct FrameInfo
    {
        cv::Mat m_frame;
        cv::UMat m_gray;
        regions_t m_regions;
        int64 m_dt = 0;
    };
    FrameInfo m_frameInfo[2];

    int m_currFrame = 0;

    bool OpenCapture(cv::VideoCapture& capture);
    bool WriteFrame(cv::VideoWriter& writer, const cv::Mat& frame);
};
