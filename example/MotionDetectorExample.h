#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <map>

#include "VideoExample.h"

#ifdef USE_CLIP
#include <ruclip/ruclip.h>

///
/// \brief ParseClipBackend
/// Maps the [clip] backend name from the ini file to the ruCLIP backend.
///
inline ruclip::BackendKind ParseClipBackend(const std::string& name)
{
	if (name == "torch" || name == "libtorch")
		return ruclip::BackendKind::Torch;
	if (name == "dnn" || name == "opencv")
		return ruclip::BackendKind::Dnn;
	return ruclip::BackendKind::TensorRT;
}
#endif // USE_CLIP

///
/// \brief The MotionDetectorExample class
///
class MotionDetectorExample final : public VideoExample
{
public:
    MotionDetectorExample(const cv::CommandLineParser& parser)
        : VideoExample(parser)
    {
#ifdef USE_CLIP
		// ruCLIP is an external package (find_package(ruCLIP)); all parameters
		// come from the [clip] section of the settings ini file. The model
		// directory must contain config.json, bpe.model and the ONNX files
		// (clip_visual.onnx, clip_textual.onnx); the TensorRT engine is built
		// and cached next to the ONNX files on the first run.
		const TrackerSettings::ClipSettings& clipSettings = m_trackerSettings.m_clipSettings;
		if (clipSettings.m_modelDir.empty() || clipSettings.m_labels.empty())
		{
			m_logger->warn("RuCLIP is enabled but [clip] model_dir/labels are not set in the settings ini - classification is disabled");
		}
		else
		{
			m_clipLabels = clipSettings.m_labels;
			m_clipRejectLabel = clipSettings.m_rejectLabel.empty() ? std::string("другое") : clipSettings.m_rejectLabel;
			m_clipThreshold = clipSettings.m_threshold;
			m_clipRecheckInterval = std::max(1, clipSettings.m_recheckInterval);
			m_clipMaxBatch = std::max(1, clipSettings.m_maxBatch);
			m_clipMinObjectSize = cv::Size(std::max(1, clipSettings.m_minObjectWidth), std::max(1, clipSettings.m_minObjectHeight));
			try
			{
				ruclip::ModelOptions options;
				options.modelDir = clipSettings.m_modelDir;
				options.backend = ParseClipBackend(clipSettings.m_backend);
				options.deviceIndex = clipSettings.m_deviceIndex;
				options.maxBatch = m_clipMaxBatch;
				options.useFp16 = clipSettings.m_useFp16;
				m_clip = std::make_unique<ruclip::Model>(options);
				m_logger->info("RuCLIP: model = {}, backend = {}, fp16 = {}, labels = {}, reject = {}, threshold = {}, recheck = {}",
					clipSettings.m_modelDir, clipSettings.m_backend, clipSettings.m_useFp16,
					m_clipLabels.size(), m_clipRejectLabel, m_clipThreshold, m_clipRecheckInterval);
			}
			catch (const std::exception& e)
			{
				m_logger->error("RuCLIP init failed: {}", e.what());
			}
		}
#endif // USE_CLIP

		m_logger->info("MotionDetectorExample");
	}

protected:
    ///
    /// \brief InitDetector
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame) override
    {
		m_logger->info("MotionDetectorExample::InitDetector");

		m_minObjWidth = 2;

        config_t config;
		config.emplace("useRotatedRect", "0");

        tracking::Detectors detectorType = tracking::Detectors::Motion_MOG;

		switch (detectorType)
		{
		case tracking::Detectors::Motion_VIBE:
			config.emplace("samples", "20");
			config.emplace("pixelNeighbor", "1");
			config.emplace("distanceThreshold", "20");
			config.emplace("matchingThreshold", "3");
			config.emplace("updateFactor", "16");
			break;
		case tracking::Detectors::Motion_MOG:
            config.emplace("history", std::to_string(cvRound(5000 * m_fps)));
			config.emplace("nmixtures", "3");
			config.emplace("backgroundRatio", "0.7");
			config.emplace("noiseSigma", "0");
			break;
		case tracking::Detectors::Motion_GMG:
			config.emplace("initializationFrames", "50");
			config.emplace("decisionThreshold", "0.7");
			break;
		case tracking::Detectors::Motion_CNT:
			config.emplace("minPixelStability", "15");
			config.emplace("maxPixelStability", std::to_string(cvRound(20 * m_minStaticTime * m_fps)));
			config.emplace("useHistory", "1");
			config.emplace("isParallel", "1");
			break;
		case tracking::Detectors::Motion_MOG2:
			config.emplace("history", std::to_string(cvRound(20 * m_minStaticTime * m_fps)));
			config.emplace("varThreshold", "10");
			config.emplace("detectShadows", "1");
			break;
		}
        m_detector = BaseDetector::CreateDetector(detectorType, config, frame);

        if (m_detector.get())
        {
            m_detector->SetMinObjectSize(cv::Size(m_minObjWidth, m_minObjWidth));
            return true;
        }
        return false;
    }
    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame) override
    {
		m_logger->info("MotionDetectorExample::InitTracker");

		if (!m_trackerSettingsLoaded)
		{
            m_trackerSettings.SetDistance(tracking::DistCenters);
			m_trackerSettings.m_kalmanType = tracking::KalmanLinear;
			m_trackerSettings.m_filterGoal = tracking::FilterCenter;
            m_trackerSettings.m_lostTrackType = tracking::TrackNone; // Use visual objects tracker for collisions resolving. Used if m_filterGoal == tracking::FilterRect
			m_trackerSettings.m_matchType = tracking::MatchHungrian;
			m_trackerSettings.m_useAcceleration = false;             // Use constant acceleration motion model
            m_trackerSettings.m_dt = m_trackerSettings.m_useAcceleration ? 0.05f : 0.3f; // Delta time for Kalman filter
            m_trackerSettings.m_accelNoiseMag = 0.1f;                // Accel noise magnitude for Kalman filter
            m_trackerSettings.m_distThres = 0.95f;                   // Distance threshold between region and object on two frames
#if 1
            m_trackerSettings.m_minAreaRadiusPix = frame.rows / 5.f;
#else
			m_trackerSettings.m_minAreaRadiusPix = -1.f;
#endif
			m_trackerSettings.m_minAreaRadiusK = 0.8f;

            m_trackerSettings.m_useAbandonedDetection = false;
			if (m_trackerSettings.m_useAbandonedDetection)
			{
				m_trackerSettings.m_minStaticTime = m_minStaticTime;
				m_trackerSettings.m_maxStaticTime = 10;
				m_trackerSettings.m_maximumAllowedLostTime = m_trackerSettings.m_minStaticTime;      // Maximum allowed lost time
				m_trackerSettings.m_maxTraceLength = 2 * m_trackerSettings.m_maximumAllowedLostTime; // Maximum trace length
			}
			else
			{
				m_trackerSettings.m_maximumAllowedLostTime = 2.;       // Maximum allowed lost time
                m_trackerSettings.m_maxTraceLength = 2.;               // Maximum trace length
			}
		}

        m_tracker = BaseTracker::CreateTracker(m_trackerSettings, m_fps);
        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
	/// \param tracks
    /// \param framesCounter
    /// \param currTime
    ///
    void DrawData(cv::Mat frame, const std::vector<TrackingObject>& tracks, int framesCounter, int currTime) override
    {
		m_logger->info("Frame {0} ({1}): tracks = {2}, time = {3}", framesCounter, m_framesCount, tracks.size(), currTime);

#ifdef USE_CLIP
		// Drop cache entries of tracks that are no longer present.
		{
			std::unordered_set<size_t> currentIds;
			currentIds.reserve(tracks.size());
			for (const auto& track : tracks)
				currentIds.insert(track.m_ID.m_val);
			for (auto it = m_clipCache.begin(); it != m_clipCache.end(); )
			{
				if (currentIds.find(it->first) == currentIds.end())
					it = m_clipCache.erase(it);
				else
					++it;
			}
		}

		// Classify a track when it first appears and re-check it every
		// m_clipRecheckInterval frames instead of running ruCLIP on every
		// track of every frame.
		{
			std::vector<size_t> needClassify;
			needClassify.reserve(tracks.size());
			for (size_t i = 0; i < tracks.size(); ++i)
			{
				auto it = m_clipCache.find(tracks[i].m_ID.m_val);
				if (it == m_clipCache.end() || framesCounter - it->second.lastFrame >= m_clipRecheckInterval)
					needClassify.push_back(i);
			}
			if (!needClassify.empty())
				ClassifyTracks(frame, tracks, needClassify, framesCounter);
		}
#endif // USE_CLIP

        for (size_t i = 0; i < tracks.size(); ++i)
        {
			const auto& track = tracks[i];
#ifdef USE_CLIP
			std::string clipText;
			auto clipIt = m_clipCache.find(track.m_ID.m_val);
			if (clipIt != m_clipCache.end() && !clipIt->second.label.empty())
				clipText = clipIt->second.label + ", " + std::to_string(clipIt->second.conf);
#endif // USE_CLIP
            if (track.m_isStatic)
            {
#ifdef USE_CLIP
                DrawTrack(frame, track, false, framesCounter, clipText);
#else
				DrawTrack(frame, track, false, framesCounter);
#endif //USE_CLIP
				std::string label = "abandoned " + track.m_ID.ID2Str();
				int baseLine = 0;
				cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_TRIPLEX, 0.5, 1, &baseLine);

				cv::Rect brect = track.m_rrect.boundingRect();
				if (brect.x < 0)
				{
					brect.width = std::min(brect.width, frame.cols - 1);
					brect.x = 0;
				}
				else if (brect.x + brect.width >= frame.cols)
				{
					brect.x = std::max(0, frame.cols - brect.width - 1);
					brect.width = std::min(brect.width, frame.cols - 1);
				}
				if (brect.y - labelSize.height < 0)
				{
					brect.height = std::min(brect.height, frame.rows - 1);
					brect.y = labelSize.height;
				}
				else if (brect.y + brect.height >= frame.rows)
				{
					brect.y = std::max(0, frame.rows - brect.height - 1);
					brect.height = std::min(brect.height, frame.rows - 1);
				}
				DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 0, 255), 150);
				cv::putText(frame, label, brect.tl(), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
            else
            {
				auto velocity = sqrt(sqr(track.m_velocity[0]) + sqr(track.m_velocity[1]));
				if (track.IsRobust(4,             // Minimal trajectory size
					0.3f,                         // Minimal ratio raw_trajectory_points / trajectory_lenght
					cv::Size2f(0.2f, 5.0f),       // Min and max ratio: width / height
					2))
					//velocity > 30               // Velocity more than 30 pixels per second
				{
					//track_t mean = 0;
					//track_t stddev = 0;
					//TrackingObject::LSParams lsParams;
					//if (track.LeastSquares2(20, mean, stddev, lsParams) && mean > stddev)
					{
#ifdef USE_CLIP
						DrawTrack(frame, track, true, framesCounter, clipText);
#else
						DrawTrack(frame, track, true, framesCounter);
#endif //USE_CLIP
					}
				}
            }
        }
        m_detector->CalcMotionMap(frame);
    }

#ifdef USE_CLIP
	///
	/// \brief ClassifyTracks
	/// Runs ruCLIP on the given track indices (in chunks of m_clipMaxBatch) and
	/// updates m_clipCache. Tracks rejected by the "unknown" prompt or by the
	/// confidence threshold are cached with an empty label.
	///
	void ClassifyTracks(cv::Mat frame, const std::vector<TrackingObject>& tracks,
		const std::vector<size_t>& indices, int framesCounter)
	{
		if (!m_clip || m_clipLabels.empty() || indices.empty())
			return;

		std::vector<std::string> queryLabels = m_clipLabels;
		queryLabels.push_back(m_clipRejectLabel);

		std::vector<cv::Mat> images;
		std::vector<size_t> imageToTrack;
		images.reserve(indices.size());
		imageToTrack.reserve(indices.size());

		for (size_t idx : indices)
		{
			const cv::Rect rect = Clamp(tracks[idx].GetBoundingRect(), frame.size());
			if (rect.width < m_clipMinObjectSize.width || rect.height < m_clipMinObjectSize.height)
			{
				m_clipCache[tracks[idx].m_ID.m_val] = ClipCacheEntry{ std::string(), 0.f, framesCounter };
				continue;
			}
			images.emplace_back(frame, rect);
			imageToTrack.push_back(idx);
		}

		for (size_t start = 0; start < images.size(); start += static_cast<size_t>(m_clipMaxBatch))
		{
			const size_t end = std::min(images.size(), start + static_cast<size_t>(m_clipMaxBatch));
			const int startIdx = static_cast<int>(start);
			const int endIdx = static_cast<int>(end);
			std::vector<cv::Mat> chunk(images.begin() + startIdx, images.begin() + endIdx);
			const ruclip::TensorF32 probs = m_clip->ProbsPerImage(chunk, queryLabels);
			const int64_t labelsCount = probs.cols();
			for (size_t k = 0; k < chunk.size(); ++k)
			{
				int64_t bestInd = 0;
				float bestConf = probs.At(static_cast<int64_t>(k), 0);
				for (int64_t labelInd = 1; labelInd < labelsCount; ++labelInd)
				{
					const float conf = probs.At(static_cast<int64_t>(k), labelInd);
					if (bestConf < conf)
					{
						bestConf = conf;
						bestInd = labelInd;
					}
				}

				const size_t trackIdx = imageToTrack[start + k];
				const bool isReject = (bestInd == static_cast<int64_t>(m_clipLabels.size()));
				const bool rejected = isReject || (bestConf < m_clipThreshold);
				m_clipCache[tracks[trackIdx].m_ID.m_val] = ClipCacheEntry{
					rejected ? std::string() : m_clipLabels[static_cast<size_t>(bestInd)],
					rejected ? 0.f : bestConf,
					framesCounter };
			}
		}
	}
#endif // USE_CLIP

private:
    int m_minObjWidth = 8;
    int m_minStaticTime = 5;

#ifdef USE_CLIP
	std::unique_ptr<ruclip::Model> m_clip;
	std::vector<std::string> m_clipLabels;
	std::string m_clipRejectLabel = "другое";
	float m_clipThreshold = 0.5f;
	int m_clipRecheckInterval = 30;
	int m_clipMaxBatch = 32;
	cv::Size m_clipMinObjectSize{ 16, 16 };

	struct ClipCacheEntry
	{
		std::string label;
		float conf = 0.f;
		int lastFrame = -1000000;
	};
	std::unordered_map<size_t, ClipCacheEntry> m_clipCache;
#endif // USE_CLIP
};
