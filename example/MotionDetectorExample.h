#pragma once

#include <iostream>
#include <vector>
#include <map>

#include "VideoExample.h"

#ifdef USE_CLIP
#include "ruclip/ClipAPI.h"
#endif // USE_CLIP

///
/// \brief The MotionDetectorExample class
///
class MotionDetectorExample final : public VideoExample
{
public:
    MotionDetectorExample(const cv::CommandLineParser& parser)
        : VideoExample(parser), m_minObjWidth(10)
    {
#ifdef USE_CLIP
		std::string clipModel = "C:/work/clip/ruclip_/CLIP/data/ruclip-vit-large-patch14-336";
		std::string bpeModel = "C:/work/clip/ruclip_/CLIP/data/ruclip-vit-large-patch14-336/bpe.model";
		m_clip.Init(clipModel, bpeModel, 336, 0, { "pedestrian", "person", "suv", "pickup", "car", "truck", "bus" });
#endif // USE_CLIP
	}

protected:
    ///
    /// \brief InitDetector
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame) override
    {
        m_minObjWidth = frame.cols / 20;

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
            config.emplace("history", std::to_string(cvRound(50 * m_fps)));
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
		case tracking::Detectors::Motion_SuBSENSE:
			break;
		case tracking::Detectors::Motion_LOBSTER:
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
		if (!m_trackerSettingsLoaded)
		{
            m_trackerSettings.SetDistance(tracking::DistRects);
			m_trackerSettings.m_kalmanType = tracking::KalmanLinear;
			m_trackerSettings.m_filterGoal = tracking::FilterCenter;
            m_trackerSettings.m_lostTrackType = tracking::TrackNone; // Use visual objects tracker for collisions resolving. Used if m_filterGoal == tracking::FilterRect
			m_trackerSettings.m_matchType = tracking::MatchHungrian;
			m_trackerSettings.m_useAcceleration = false;             // Use constant acceleration motion model
            m_trackerSettings.m_dt = m_trackerSettings.m_useAcceleration ? 0.05f : 0.5f; // Delta time for Kalman filter
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
				m_trackerSettings.m_maximumAllowedSkippedFrames = cvRound(m_trackerSettings.m_minStaticTime * m_fps); // Maximum allowed skipped frames
				m_trackerSettings.m_maxTraceLength = 2 * m_trackerSettings.m_maximumAllowedSkippedFrames;        // Maximum trace length
			}
			else
			{
				m_trackerSettings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
                m_trackerSettings.m_maxTraceLength = cvRound(2 * m_fps);              // Maximum trace length
			}
		}

        m_tracker = BaseTracker::CreateTracker(m_trackerSettings);
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
        if (m_showLogs)
			std::cout << "Frame " << framesCounter << " (" << m_framesCount << "): tracks = " << tracks.size() << ", time = " << currTime << std::endl;

#ifdef USE_CLIP
		std::vector<CLIPResult> clipResult;
		std::vector<cv::Rect> clipRects;
		clipRects.reserve(tracks.size());
		for (const auto& track : tracks)
		{
			clipRects.emplace_back(track.GetBoundingRect());
		}
		m_clip.ProcessFrame(frame, clipRects, clipResult);
#endif // USE_CLIP

        for (size_t i = 0; i < tracks.size(); ++i)
        {
			const auto& track = tracks[i];
            if (track.m_isStatic)
            {
#ifdef USE_CLIP
                DrawTrack(frame, track, false, framesCounter, clipResult[i].m_label + ", " + std::to_string(clipResult[i].m_conf));
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
					cv::Size2f(0.2f, 5.0f)))    // Min and max ratio: width / height
					//velocity > 30                // Velocity more than 30 pixels per second
				{
					//track_t mean = 0;
					//track_t stddev = 0;
					//TrackingObject::LSParams lsParams;
					//if (track.LeastSquares2(20, mean, stddev, lsParams) && mean > stddev)
					{
#ifdef USE_CLIP
						DrawTrack(frame, track, true, framesCounter, clipResult[i].m_label + ", " + std::to_string(clipResult[i].m_conf));
#else
						DrawTrack(frame, track, true, framesCounter);
#endif //USE_CLIP
					}
				}
            }
        }
        m_detector->CalcMotionMap(frame);
    }

private:
    int m_minObjWidth = 8;
    int m_minStaticTime = 5;

#ifdef USE_CLIP
	ClassificationCLIP m_clip;
#endif // USE_CLIP
};
