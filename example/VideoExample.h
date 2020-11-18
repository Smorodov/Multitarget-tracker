#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <atomic>

#include "BaseDetector.h"
#include "Ctracker.h"

///
/// \brief The ResultsLog class
///
class ResultsLog
{
public:
	///
	ResultsLog(const std::string& fileName)
		: m_fileName(fileName)
	{
	}

	///
	~ResultsLog()
	{
		WriteAll(true);
	}

	///
	bool Open()
	{
		m_resCSV.close();
		if (m_fileName.size() > 5)
		{
			m_resCSV.open(m_fileName);
			return m_resCSV.is_open();
		}
		return false;
	}

	///
	bool AddTrack(int framesCounter, size_t trackID, const cv::Rect& brect, objtype_t type, float confidence)
	{
		if (m_resCSV.is_open())
		{
			auto frame = m_frames.find(framesCounter);
			if (frame == std::end(m_frames))
			{
				DetectsOnFrame tmpFrame;
				tmpFrame.m_detects.emplace_back(trackID, brect, type, confidence);
				m_frames.emplace(framesCounter, tmpFrame);
			}
			else
			{
				frame->second.m_detects.emplace_back(trackID, brect, type, confidence);
			}
			return true;
		}
		return false;
	}

	///
	void AddRobustTrack(size_t trackID)
	{
		m_robustIDs.insert(trackID);
	}

private:
	std::string m_fileName;
	std::ofstream m_resCSV;

	///
	struct Detection
	{
		cv::Rect m_rect;
		objtype_t m_type;
		float m_conf = 0.f;
		size_t m_trackID = 0;

		Detection(size_t trackID, const cv::Rect& brect, objtype_t type, float confidence)
		{
			m_type = type;
			m_rect = brect;
			m_conf = confidence;
			m_trackID = trackID;
		}
	};

	///
	struct DetectsOnFrame
	{
		std::vector<Detection> m_detects;
	};
	std::map<int, DetectsOnFrame> m_frames;
	std::set<size_t> m_robustIDs;

	///
	void WriteAll(bool byFrames)
	{
		if (byFrames)
		{
#if 1
			char delim = ',';
			for (const auto& frame : m_frames)
			{
				for (const auto& detect : frame.second.m_detects)
				{
					if (m_robustIDs.find(detect.m_trackID) != std::end(m_robustIDs))
					{
						m_resCSV << frame.first << delim << TypeConverter::Type2Str(detect.m_type) << delim << detect.m_rect.x << delim << detect.m_rect.y << delim <<
							detect.m_rect.width << delim << detect.m_rect.height << delim <<
							detect.m_conf << delim << std::endl;
					}
				}
			}
#else
			char delim = '	';
			for (const auto& frame : m_frames)
			{
				for (const auto& detect : frame.second.m_detects)
				{
					if (m_robustIDs.find(detect.m_trackID) != std::end(m_robustIDs))
					{
						m_resCSV << frame.first << delim << detect.m_type << delim << detect.m_rect.x << delim << detect.m_rect.y << delim <<
							(detect.m_rect.x + detect.m_rect.width) << delim << (detect.m_rect.y + detect.m_rect.height) << delim <<
							detect.m_conf << delim << detect.m_trackID << std::endl;
					}
				}
			}
#endif
		}
		else
		{
			char delim = ',';
			for (size_t id : m_robustIDs)
			{
				for (const auto& frame : m_frames)
				{
					for (const auto& detect : frame.second.m_detects)
					{
						if (detect.m_trackID == id)
						{
							m_resCSV << frame.first << delim << id << delim << detect.m_rect.x << delim << detect.m_rect.y << delim <<
								detect.m_rect.width << delim << detect.m_rect.height << delim <<
								detect.m_conf << ",-1,-1,-1," << std::endl;
							break;
						}
					}
				}
			}
		}
	}
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
    std::unique_ptr<CTracker> m_tracker;

	std::vector<TrackingObject> m_tracks;

    bool m_showLogs = true;
    float m_fps = 25;

    int m_captureTimeOut = 60000;
    int m_trackingTimeOut = 60000;

	ResultsLog m_resultsLog;

    static void CaptureAndDetect(VideoExample* thisPtr, std::atomic<bool>& stopCapture);

    virtual bool InitDetector(cv::UMat frame) = 0;
    virtual bool InitTracker(cv::UMat frame) = 0;

    void Detection(cv::Mat frame, regions_t& regions);
    void Tracking(cv::Mat frame, const regions_t& regions);

    virtual void DrawData(cv::Mat frame, int framesCounter, int currTime) = 0;

    void DrawTrack(cv::Mat frame, int resizeCoeff, const TrackingObject& track, bool drawTrajectory, int framesCounter);

private:
    bool m_isTrackerInitialized = false;
    bool m_isDetectorInitialized = false;
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
        regions_t m_regions;
        int64 m_dt = 0;

        std::condition_variable m_cond;
        std::mutex m_mutex;
        bool m_captured = false;
    };
    FrameInfo m_frameInfo[2];

    bool OpenCapture(cv::VideoCapture& capture);
    bool WriteFrame(cv::VideoWriter& writer, const cv::Mat& frame);
};
