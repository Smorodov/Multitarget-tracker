#pragma once

#include <iostream>
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
/// \brief The CombinedDetector class
///
class CombinedDetector
{
public:
    CombinedDetector(const cv::CommandLineParser& parser);
    CombinedDetector(const CombinedDetector&) = delete;
    CombinedDetector(CombinedDetector&&) = delete;
    CombinedDetector& operator=(const CombinedDetector&) = delete;
    CombinedDetector& operator=(CombinedDetector&&) = delete;

    ~CombinedDetector() = default;

    void SyncProcess();

protected:
    std::unique_ptr<BaseDetector> m_detectorBGFG;
    std::unique_ptr<BaseTracker> m_trackerBGFG;

	std::unique_ptr<BaseDetector> m_detectorDNN;
	std::unique_ptr<BaseTracker> m_trackerDNN;

	std::vector<TrackingObject> m_tracksBGFG;
	std::vector<TrackingObject> m_tracksDNN;

	struct Bbox
	{
		cv::Rect m_rect;
		int m_lifeTime = 0;

		Bbox(const cv::Rect& rect, int lifeTime)
			: m_rect(rect), m_lifeTime(lifeTime)
		{
		}
	};
	std::vector<Bbox> m_oldBoxes;
	int m_maxLifeTime = 40;
	float m_bboxIoUThresh = 0.7f;
	bool AddBbox(const cv::Rect& rect);
	void CleanBboxes();

    bool m_showLogs = true;
    float m_fps = 25;
	bool m_flipV = false;

	bool InitDetector(cv::UMat frame);
    bool InitTracker(cv::UMat frame);

    void DetectAndTrack(cv::Mat frame);

    void DrawData(cv::Mat frame, int framesCounter, int currTime);

    void DrawTrack(cv::Mat frame, int resizeCoeff, const TrackingObject& track, bool drawTrajectory = true);

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

	int m_minStaticTime = 10;

    bool OpenCapture(cv::VideoCapture& capture);
    bool WriteFrame(cv::VideoWriter& writer, const cv::Mat& frame);
};
