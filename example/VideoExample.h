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
						m_resCSV << frame.first << delim << TypeConverter::Type2Str(detect.m_type) << delim << detect.m_rect.x << delim << detect.m_rect.y << delim <<
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
	bool m_captured = false;
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

    bool m_showLogs = true;
    float m_fps = 25;

	size_t m_batchSize = 1;

    int m_captureTimeOut = 60000;
    int m_trackingTimeOut = 60000;

    ResultsLog m_resultsLog;

    static void CaptureAndDetect(VideoExample* thisPtr, std::atomic<bool>& stopCapture);

    virtual bool InitDetector(cv::UMat frame) = 0;
    virtual bool InitTracker(cv::UMat frame) = 0;

    void Detection(FrameInfo& frame);
    void Tracking(FrameInfo& frame);

    virtual void DrawData(cv::Mat frame, const std::vector<TrackingObject>& tracks, int framesCounter, int currTime) = 0;

    void DrawTrack(cv::Mat frame, int resizeCoeff, const TrackingObject& track, bool drawTrajectory, int framesCounter);

	TrackerSettings m_trackerSettings;
	bool m_trackerSettingsLoaded = false;
	bool ParseTrackerSettings(const std::string& settingsFile);

private:
	std::vector<TrackingObject> m_tracks;

    bool m_isTrackerInitialized = false;
    bool m_isDetectorInitialized = false;
    std::string m_inFile;
    std::string m_outFile;
    int m_fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    int m_startFrame = 0;
    int m_endFrame = 0;
    int m_finishDelay = 0;
    std::vector<cv::Scalar> m_colors;

    FrameInfo m_frameInfo[2];

    bool OpenCapture(cv::VideoCapture& capture);
    bool WriteFrame(cv::VideoWriter& writer, const cv::Mat& frame);
};
