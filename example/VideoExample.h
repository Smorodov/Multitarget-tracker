#pragma once

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <atomic>

#include "BaseDetector.h"
#include "Ctracker.h"
#include "FileLogger.h"
#include "cvatAnnotationsGenerator.h"

///
/// \brief The Frame struct
///
class Frame
{
public:
    Frame() = default;
    Frame(cv::Mat imgBGR)
    {
        m_mBGR = imgBGR;
    }

    ///
    bool empty() const
    {
        return m_mBGR.empty();
    }

    ///
    const cv::Mat& GetMatBGR()
    {
        return m_mBGR;
    }
    ///
    cv::Mat& GetMatBGRWrite()
    {
        m_umBGRGenerated = false;
        m_mGrayGenerated = false;
        m_umGrayGenerated = false;
        return m_mBGR;
    }
    ///
    const cv::Mat& GetMatGray()
    {
        if (m_mGray.empty() || !m_mGrayGenerated)
        {
            if (m_umGray.empty() || !m_umGrayGenerated)
                cv::cvtColor(m_mBGR, m_mGray, cv::COLOR_BGR2GRAY);
            else
                m_mGray = m_umGray.getMat(cv::ACCESS_READ);
            m_mGrayGenerated = true;
        }
        return m_mGray;
    }
    ///
    const cv::UMat& GetUMatBGR()
    {
        std::thread::id lastThreadID = std::this_thread::get_id();

        if (m_umBGR.empty() || !m_umBGRGenerated || lastThreadID != m_umBGRThreadID)
        {
            m_umBGR = m_mBGR.getUMat(cv::ACCESS_READ);
            m_umBGRGenerated = true;
            m_umBGRThreadID = lastThreadID;
        }
        return m_umBGR;
    }
    ///
    const cv::UMat& GetUMatGray()
    {
        std::thread::id lastThreadID = std::this_thread::get_id();

        if (m_umGray.empty() || !m_umGrayGenerated || lastThreadID != m_umGrayThreadID)
        {
            if (m_mGray.empty() || !m_mGrayGenerated)
            {
                if (m_umBGR.empty() || !m_umBGRGenerated || lastThreadID != m_umGrayThreadID)
                    cv::cvtColor(m_mBGR, m_umGray, cv::COLOR_BGR2GRAY);
                else
                    cv::cvtColor(m_umBGR, m_umGray, cv::COLOR_BGR2GRAY);
            }
            else
            {
                m_umGray = m_mGray.getUMat(cv::ACCESS_READ);
            }
            m_umGrayGenerated = true;
            m_umGrayThreadID = lastThreadID;
        }
        return m_umGray;
    }

private:
    cv::Mat m_mBGR;
    cv::Mat m_mGray;
    cv::UMat m_umBGR;
    cv::UMat m_umGray;
    bool m_umBGRGenerated = false;
    bool m_mGrayGenerated = false;
    bool m_umGrayGenerated = false;
    std::thread::id m_umBGRThreadID;
    std::thread::id m_umGrayThreadID;
};

///
/// \brief The FrameInfo struct
///
struct FrameInfo
{
    ///
    FrameInfo()
    {
        m_frames.reserve(m_batchSize);
        m_regions.reserve(m_batchSize);
        m_frameInds.reserve(m_batchSize);
    }
    ///
    FrameInfo(size_t batchSize)
        : m_batchSize(batchSize)
    {
        m_frames.reserve(m_batchSize);
        m_regions.reserve(m_batchSize);
        m_frameInds.reserve(m_batchSize);
    }

    ///
    void SetBatchSize(size_t batchSize)
    {
        m_batchSize = batchSize;
        m_frames.reserve(m_batchSize);
        m_regions.reserve(m_batchSize);
        m_frameInds.reserve(m_batchSize);
    }

    ///
    void CleanRegions()
    {
        if (m_regions.size() != m_batchSize)
            m_regions.resize(m_batchSize);
        for (auto& regions : m_regions)
        {
            regions.clear();
        }
    }

    ///
    void CleanTracks()
    {
        if (m_tracks.size() != m_batchSize)
            m_tracks.resize(m_batchSize);
        for (auto& tracks : m_tracks)
        {
            tracks.clear();
        }
    }

    std::vector<Frame> m_frames;
    std::vector<regions_t> m_regions;
    std::vector<std::vector<TrackingObject>> m_tracks;
    std::vector<int> m_frameInds;

    size_t m_batchSize = 1;

    int64 m_dt = 0;

    std::condition_variable m_cond;
    std::mutex m_mutex;
    std::atomic<bool> m_captured { false };
};

///
/// \brief The VideoExample class
///
class VideoExample
{
public:
    VideoExample(const cv::CommandLineParser& parser);
    VideoExample(const VideoExample&) = delete;
    VideoExample(VideoExample&&) = delete;
    VideoExample& operator=(const VideoExample&) = delete;
    VideoExample& operator=(VideoExample&&) = delete;

    virtual ~VideoExample() = default;

    void AsyncProcess();
    void SyncProcess();

protected:
    std::unique_ptr<BaseDetector> m_detector;
    std::unique_ptr<BaseTracker> m_tracker;

    bool m_showLogs = true;
    float m_fps = 25;
	cv::Size m_frameSize;
	int m_framesCount = 0;

	size_t m_batchSize = 1;

    int m_captureTimeOut = 60000;
    int m_trackingTimeOut = 60000;

    ResultsLog m_resultsLog;
	CVATAnnotationsGenerator m_cvatAnnotationsGenerator;

    static void CaptureAndDetect(VideoExample* thisPtr, std::atomic<bool>& stopCapture);

    virtual bool InitDetector(cv::UMat frame) = 0;
    virtual bool InitTracker(cv::UMat frame) = 0;

    void Detection(FrameInfo& frame);
    void Tracking(FrameInfo& frame);

    virtual void DrawData(cv::Mat frame, const std::vector<TrackingObject>& tracks, int framesCounter, int currTime) = 0;
    virtual void DrawTrack(cv::Mat frame, const TrackingObject& track, bool drawTrajectory, int framesCounter);

    TrackerSettings m_trackerSettings;
    bool m_trackerSettingsLoaded = false;

    std::vector<cv::Scalar> m_colors;

private:
	std::vector<TrackingObject> m_tracks;

    bool m_isTrackerInitialized = false;
    bool m_isDetectorInitialized = false;
    std::string m_inFile;
    std::string m_outFile;
    int m_fourcc = cv::VideoWriter::fourcc('h', '2', '6', '5'); //cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    int m_startFrame = 0;
    int m_endFrame = 0;
    int m_finishDelay = 0;

    FrameInfo m_frameInfo[2];

    bool OpenCapture(cv::VideoCapture& capture);
    bool WriteFrame(cv::VideoWriter& writer, const cv::Mat& frame);
};

///
void DrawFilledRect(cv::Mat& frame, const cv::Rect& rect, cv::Scalar cl, int alpha);
