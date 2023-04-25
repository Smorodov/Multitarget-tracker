#pragma once

#include <iostream>
#include <vector>
#include <map>

#include "VideoExample.h"

///
/// \brief The MotionDetectorExample class
///
class MotionDetectorExample final : public VideoExample
{
public:
    MotionDetectorExample(const cv::CommandLineParser& parser)
        : VideoExample(parser), m_minObjWidth(10)
    {
    }

protected:
    ///
    /// \brief InitDetector
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame) override
    {
        m_minObjWidth = 0;//frame.cols / 20;

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

        for (const auto& track : tracks)
        {
            if (track.m_isStatic)
            {
                DrawTrack(frame, track, false, framesCounter);

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
					cv::Size2f(0.2f, 5.0f)) &&    // Min and max ratio: width / height
					velocity > 30)                // Velocity more than 30 pixels per second
				{
					track_t mean = 0;
					track_t stddev = 0;
					TrackingObject::LSParams lsParams;
					if (track.LeastSquares2(20, mean, stddev, lsParams) && mean > stddev)
					{
						DrawTrack(frame, track, true, framesCounter);
					}
				}
            }
        }
        m_detector->CalcMotionMap(frame);
    }

private:
    int m_minObjWidth = 8;
    int m_minStaticTime = 5;
};

// ----------------------------------------------------------------------

///
/// \brief The FaceDetectorExample class
///
class FaceDetectorExample final : public VideoExample
{
public:
    FaceDetectorExample(const cv::CommandLineParser& parser)
        : VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitDetector
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame) override
    {
#ifdef _WIN32
        std::string pathToModel = "../../data/";
#else
        std::string pathToModel = "../data/";
#endif

        config_t config;
        config.emplace("cascadeFileName", pathToModel + "haarcascade_frontalface_alt2.xml");
        m_detector = BaseDetector::CreateDetector(tracking::Detectors::Face_HAAR, config, frame);
		return (m_detector.get() != nullptr);
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
			m_trackerSettings.SetDistance(tracking::DistJaccard);
			m_trackerSettings.m_kalmanType = tracking::KalmanUnscented;
			m_trackerSettings.m_filterGoal = tracking::FilterRect;
			m_trackerSettings.m_lostTrackType = tracking::TrackCSRT;      // Use visual objects tracker for collisions resolving. Used if m_filterGoal == tracking::FilterRect
			m_trackerSettings.m_matchType = tracking::MatchHungrian;
			m_trackerSettings.m_dt = 0.3f;                                // Delta time for Kalman filter
			m_trackerSettings.m_accelNoiseMag = 0.1f;                     // Accel noise magnitude for Kalman filter
			m_trackerSettings.m_distThres = 0.8f;                         // Distance threshold between region and object on two frames
			m_trackerSettings.m_minAreaRadiusPix = frame.rows / 20.f;
			m_trackerSettings.m_maximumAllowedSkippedFrames = cvRound(m_fps / 2);   // Maximum allowed skipped frames
			m_trackerSettings.m_maxTraceLength = cvRound(5 * m_fps);            // Maximum trace length
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

        for (const auto& track : tracks)
        {
            if (track.IsRobust(8,                           // Minimal trajectory size
                                0.4f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f)))      // Min and max ratio: width / height
                DrawTrack(frame, track, true, framesCounter);
        }
        m_detector->CalcMotionMap(frame);
    }
};

// ----------------------------------------------------------------------

///
/// \brief The PedestrianDetectorExample class
///
class PedestrianDetectorExample final : public VideoExample
{
public:
    PedestrianDetectorExample(const cv::CommandLineParser& parser)
        : VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitDetector
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame) override
    {
        tracking::Detectors detectorType = tracking::Detectors::Pedestrian_C4; // tracking::Detectors::Pedestrian_HOG;

#ifdef _WIN32
        std::string pathToModel = "../../data/";
#else
        std::string pathToModel = "../data/";
#endif

        config_t config;
        config.emplace("detectorType", (detectorType == tracking::Pedestrian_HOG) ? "HOG" : "C4");
        config.emplace("cascadeFileName1", pathToModel + "combined.txt.model");
        config.emplace("cascadeFileName2", pathToModel + "combined.txt.model_");
        m_detector = BaseDetector::CreateDetector(detectorType, config, frame);
		return (m_detector.get() != nullptr);
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
			m_trackerSettings.m_filterGoal = tracking::FilterRect;
			m_trackerSettings.m_lostTrackType = tracking::TrackCSRT;   // Use visual objects tracker for collisions resolving. Used if m_filterGoal == tracking::FilterRect
			m_trackerSettings.m_matchType = tracking::MatchHungrian;
			m_trackerSettings.m_dt = 0.3f;                             // Delta time for Kalman filter
			m_trackerSettings.m_accelNoiseMag = 0.1f;                  // Accel noise magnitude for Kalman filter
			m_trackerSettings.m_distThres = 0.8f;                      // Distance threshold between region and object on two frames
			m_trackerSettings.m_minAreaRadiusPix = frame.rows / 20.f;
			m_trackerSettings.m_maximumAllowedSkippedFrames = cvRound(m_fps);   // Maximum allowed skipped frames
			m_trackerSettings.m_maxTraceLength = cvRound(5 * m_fps);   // Maximum trace length
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

        for (const auto& track : tracks)
        {
			if (track.IsRobust(cvRound(m_fps / 2),          // Minimal trajectory size
                                0.4f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f)))      // Min and max ratio: width / height
                DrawTrack(frame, track, true, framesCounter);
        }
        m_detector->CalcMotionMap(frame);
    }
};

// ----------------------------------------------------------------------

///
/// \brief The OpenCVDNNExample class
///
class OpenCVDNNExample final : public VideoExample
{
public:
	OpenCVDNNExample(const cv::CommandLineParser& parser)
		: VideoExample(parser)
	{
	}

protected:
	///
	/// \brief InitDetector
	/// \param frame
	/// \return
	///
	bool InitDetector(cv::UMat frame) override
	{
		config_t config;
		if (!m_trackerSettingsLoaded)
		{
#ifdef _WIN32
			std::string pathToModel = "../../data/";
#else
			std::string pathToModel = "../data/";
#endif
			enum class NNModels
			{
				TinyYOLOv3 = 0,
				YOLOv3,
				YOLOv4,
				TinyYOLOv4,
				MobileNetSSD
			};
			NNModels usedModel = NNModels::MobileNetSSD;
			switch (usedModel)
			{
			case NNModels::TinyYOLOv3:
				config.emplace("modelConfiguration", pathToModel + "yolov3-tiny.cfg");
				config.emplace("modelBinary", pathToModel + "yolov3-tiny.weights");
				config.emplace("classNames", pathToModel + "coco.names");
				config.emplace("confidenceThreshold", "0.5");
				break;

			case NNModels::YOLOv3:
				config.emplace("modelConfiguration", pathToModel + "yolov3.cfg");
				config.emplace("modelBinary", pathToModel + "yolov3.weights");
				config.emplace("classNames", pathToModel + "coco.names");
				config.emplace("confidenceThreshold", "0.7");
				break;

			case NNModels::YOLOv4:
				config.emplace("modelConfiguration", pathToModel + "yolov4.cfg");
				config.emplace("modelBinary", pathToModel + "yolov4.weights");
				config.emplace("classNames", pathToModel + "coco.names");
				config.emplace("confidenceThreshold", "0.5");
				break;

			case NNModels::TinyYOLOv4:
				config.emplace("modelConfiguration", pathToModel + "yolov4-tiny.cfg");
				config.emplace("modelBinary", pathToModel + "yolov4-tiny.weights");
				config.emplace("classNames", pathToModel + "coco.names");
				config.emplace("confidenceThreshold", "0.5");
				break;

			case NNModels::MobileNetSSD:
				config.emplace("modelConfiguration", pathToModel + "MobileNetSSD_deploy.prototxt");
				config.emplace("modelBinary", pathToModel + "MobileNetSSD_deploy.caffemodel");
				config.emplace("classNames", pathToModel + "voc.names");
				config.emplace("confidenceThreshold", "0.5");
				break;
			}
			config.emplace("maxCropRatio", "-1");

			config.emplace("dnnTarget", "DNN_TARGET_CPU");
			config.emplace("dnnBackend", "DNN_BACKEND_DEFAULT");
		}
		else
		{
			config.emplace("modelConfiguration", m_trackerSettings.m_nnConfig);
			config.emplace("modelBinary", m_trackerSettings.m_nnWeights);
			config.emplace("confidenceThreshold", std::to_string(m_trackerSettings.m_confidenceThreshold));
			config.emplace("classNames", m_trackerSettings.m_classNames);
			config.emplace("maxCropRatio", std::to_string(m_trackerSettings.m_maxCropRatio));
			config.emplace("maxBatch", std::to_string(m_trackerSettings.m_maxBatch));
			config.emplace("gpuId", std::to_string(m_trackerSettings.m_gpuId));
			config.emplace("net_type", m_trackerSettings.m_netType);
			config.emplace("inference_precision", m_trackerSettings.m_inferencePrecision);
			config.emplace("video_memory", std::to_string(m_trackerSettings.m_maxVideoMemory));
			config.emplace("dnnTarget", m_trackerSettings.m_dnnTarget);
			config.emplace("dnnBackend", m_trackerSettings.m_dnnBackend);
			config.emplace("inWidth", std::to_string(m_trackerSettings.m_inputSize.width));
			config.emplace("inHeight", std::to_string(m_trackerSettings.m_inputSize.height));

			for (auto wname : m_trackerSettings.m_whiteList)
			{
				config.emplace("white_list", wname);
			}
		}
		m_detector = BaseDetector::CreateDetector(tracking::Detectors::DNN_OCV, config, frame);
		return (m_detector.get() != nullptr);
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
			m_trackerSettings.SetDistance(tracking::DistCenters);
			m_trackerSettings.m_kalmanType = tracking::KalmanLinear;
			m_trackerSettings.m_filterGoal = tracking::FilterRect;
			m_trackerSettings.m_lostTrackType = tracking::TrackCSRT;      // Use visual objects tracker for collisions resolving. Used if m_filterGoal == tracking::FilterRect
			m_trackerSettings.m_matchType = tracking::MatchHungrian;
			m_trackerSettings.m_useAcceleration = false;                   // Use constant acceleration motion model
			m_trackerSettings.m_dt = m_trackerSettings.m_useAcceleration ? 0.05f : 0.4f; // Delta time for Kalman filter
			m_trackerSettings.m_accelNoiseMag = 0.2f;                     // Accel noise magnitude for Kalman filter
			m_trackerSettings.m_distThres = 0.8f;                         // Distance threshold between region and object on two frames
#if 0
			m_trackerSettings.m_minAreaRadiusPix = frame.rows / 20.f;
#else
			m_trackerSettings.m_minAreaRadiusPix = -1.f;
#endif
			m_trackerSettings.m_minAreaRadiusK = 0.8f;
			m_trackerSettings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
			m_trackerSettings.m_maxTraceLength = cvRound(2 * m_fps);      // Maximum trace length
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

		for (const auto& track : tracks)
		{
			if (track.IsRobust(3,                           // Minimal trajectory size
				0.5f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
				cv::Size2f(0.1f, 8.0f)))      // Min and max ratio: width / height
			{
				DrawTrack(frame, track, false, framesCounter);


				std::stringstream label;
				label << TypeConverter::Type2Str(track.m_type) << std::setprecision(2) << ": " << track.m_confidence;

				int baseLine = 0;
				cv::Size labelSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_TRIPLEX, 0.5, 1, &baseLine);

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
				//DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(200, 200, 200), 150);
				//cv::putText(frame, label.str(), brect.tl(), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 0, 0));
			}
		}

		//m_detector->CalcMotionMap(frame);
	}
};

#ifdef BUILD_YOLO_LIB
// ----------------------------------------------------------------------

///
/// \brief The YoloDarknetExample class
///
class YoloDarknetExample final : public VideoExample
{
public:
	YoloDarknetExample(const cv::CommandLineParser& parser)
		: VideoExample(parser)
	{
	}

protected:
    ///
    /// \brief InitDetector
    /// \param frame
    /// \return
    ///
	bool InitDetector(cv::UMat frame) override
	{
		config_t config;

        if (!m_trackerSettingsLoaded)
        {
#ifdef _WIN32
            std::string pathToModel = "../../data/";
#else
            std::string pathToModel = "../data/";
#endif
            size_t maxBatch = 1;
            enum class YOLOModels
            {
                TinyYOLOv3 = 0,
                YOLOv3,
                YOLOv4,
                TinyYOLOv4,
                ScaledYOLOv4
            };
            YOLOModels usedModel = YOLOModels::YOLOv4;
            switch (usedModel)
            {
            case YOLOModels::TinyYOLOv3:
                config.emplace("modelConfiguration", pathToModel + "yolov3-tiny.cfg");
                config.emplace("modelBinary", pathToModel + "yolov3-tiny.weights");
                config.emplace("confidenceThreshold", "0.5");
                break;

            case YOLOModels::YOLOv3:
                config.emplace("modelConfiguration", pathToModel + "yolov3.cfg");
                config.emplace("modelBinary", pathToModel + "yolov3.weights");
                config.emplace("confidenceThreshold", "0.7");
                break;

            case YOLOModels::YOLOv4:
                config.emplace("modelConfiguration", pathToModel + "yolov4.cfg");
                config.emplace("modelBinary", pathToModel + "yolov4.weights");
                config.emplace("confidenceThreshold", "0.5");
                break;

            case YOLOModels::TinyYOLOv4:
                config.emplace("modelConfiguration", pathToModel + "yolov4-tiny.cfg");
                config.emplace("modelBinary", pathToModel + "yolov4-tiny.weights");
                config.emplace("confidenceThreshold", "0.5");
                maxBatch = 4;
                break;

            case YOLOModels::ScaledYOLOv4:
                config.emplace("modelConfiguration", pathToModel + "yolov4-csp.cfg");
                config.emplace("modelBinary", pathToModel + "yolov4-csp.weights");
                config.emplace("confidenceThreshold", "0.5");
                maxBatch = 2;
                break;
            }
            if (maxBatch < m_batchSize)
                maxBatch = m_batchSize;
            config.emplace("maxBatch", std::to_string(maxBatch));
            config.emplace("classNames", pathToModel + "coco.names");
            config.emplace("maxCropRatio", "-1");

            config.emplace("white_list", "person");
            config.emplace("white_list", "car");
            config.emplace("white_list", "bicycle");
            config.emplace("white_list", "motorbike");
            config.emplace("white_list", "bus");
            config.emplace("white_list", "truck");
        }
        else
        {
            config.emplace("modelConfiguration", m_trackerSettings.m_nnConfig);
            config.emplace("modelBinary", m_trackerSettings.m_nnWeights);
            config.emplace("confidenceThreshold", std::to_string(m_trackerSettings.m_confidenceThreshold));
            config.emplace("classNames", m_trackerSettings.m_classNames);
            config.emplace("maxCropRatio", std::to_string(m_trackerSettings.m_maxCropRatio));
            config.emplace("maxBatch", std::to_string(m_trackerSettings.m_maxBatch));
            config.emplace("gpuId", std::to_string(m_trackerSettings.m_gpuId));
            config.emplace("net_type", m_trackerSettings.m_netType);
            config.emplace("inference_precision", m_trackerSettings.m_inferencePrecision);
			config.emplace("video_memory", std::to_string(m_trackerSettings.m_maxVideoMemory));

			for (auto wname : m_trackerSettings.m_whiteList)
			{
				config.emplace("white_list", wname);
			}
        }

        m_detector = BaseDetector::CreateDetector(tracking::Detectors::Yolo_Darknet, config, frame);
		return (m_detector.get() != nullptr);
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
			bool useDeepSORT = false;
			if (useDeepSORT)
			{
#ifdef _WIN32
				std::string pathToModel = "../../data/";
#else
				std::string pathToModel = "../data/";
#endif

#if 1
				m_trackerSettings.m_embeddings.emplace_back(pathToModel + "open_model_zoo/person-reidentification-retail-0286/FP16-INT8/person-reidentification-retail-0286.xml",
                                                            pathToModel + "open_model_zoo/person-reidentification-retail-0286/FP16-INT8/person-reidentification-retail-0286.bin",
                                                            cv::Size(128, 256),
                                                            std::vector<objtype_t>{ TypeConverter::Str2Type("person") });
#endif

#if 0
				m_trackerSettings.m_embeddings.emplace_back(pathToModel + "open_model_zoo/vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.xml",
                                                            pathToModel + "open_model_zoo/vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.bin",
                                                            cv::Size(208, 208),
                                                            std::vector<objtype_t>{ TypeConverter::Str2Type("car"), TypeConverter::Str2Type("bus"), TypeConverter::Str2Type("truck"), TypeConverter::Str2Type("vehicle") });
#endif

				std::array<track_t, tracking::DistsCount> distType{
					0.f,   // DistCenters
					0.f,   // DistRects
					0.5f,  // DistJaccard
					0.f,   // DistHist
					0.5f   // DistFeatureCos
				};
				if (!m_trackerSettings.SetDistances(distType))
					std::cerr << "SetDistances failed! Absolutly summ must be equal 1" << std::endl;
			}
			else
			{
				m_trackerSettings.SetDistance(tracking::DistCenters);
			}

			m_trackerSettings.m_kalmanType = tracking::KalmanLinear;
			m_trackerSettings.m_filterGoal = tracking::FilterRect;
			m_trackerSettings.m_lostTrackType = useDeepSORT ? tracking::TrackNone : tracking::TrackCSRT; // Use visual objects tracker for collisions resolving. Used if m_filterGoal == tracking::FilterRect
			m_trackerSettings.m_matchType = tracking::MatchHungrian;
			m_trackerSettings.m_useAcceleration = false;                   // Use constant acceleration motion model
			m_trackerSettings.m_dt = m_trackerSettings.m_useAcceleration ? 0.05f : 0.4f; // Delta time for Kalman filter
			m_trackerSettings.m_accelNoiseMag = 0.2f;                     // Accel noise magnitude for Kalman filter
			m_trackerSettings.m_distThres = useDeepSORT ? 0.7f : 0.8f;    // Distance threshold between region and object on two frames
#if 0
			m_trackerSettings.m_minAreaRadiusPix = frame.rows / 20.f;
#else
			m_trackerSettings.m_minAreaRadiusPix = -1.f;
#endif
			m_trackerSettings.m_minAreaRadiusK = 0.8f;
			m_trackerSettings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
			m_trackerSettings.m_maxTraceLength = cvRound(2 * m_fps);      // Maximum trace length
		}
        m_trackerSettings.AddNearTypes(TypeConverter::Str2Type("car"), TypeConverter::Str2Type("bus"), true);
        m_trackerSettings.AddNearTypes(TypeConverter::Str2Type("car"), TypeConverter::Str2Type("truck"), true);
        m_trackerSettings.AddNearTypes(TypeConverter::Str2Type("bus"), TypeConverter::Str2Type("truck"), true);
        m_trackerSettings.AddNearTypes(TypeConverter::Str2Type("person"), TypeConverter::Str2Type("bicycle"), true);
        m_trackerSettings.AddNearTypes(TypeConverter::Str2Type("person"), TypeConverter::Str2Type("motorbike"), true);

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

		for (const auto& track : tracks)
		{
            if (track.IsRobust(3,                           // Minimal trajectory size
                0.5f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
				cv::Size2f(0.1f, 8.0f)))      // Min and max ratio: width / height
			{
				DrawTrack(frame, track, true, framesCounter);


				std::stringstream label;
#if 1
				label << TypeConverter::Type2Str(track.m_type) << std::setprecision(2) << ": " << track.m_confidence;
#else
				label << TypeConverter::Type2Str(track.m_type) << " " << std::setprecision(2) << track.m_velocity << ": " << track.m_confidence;
#endif
				int baseLine = 0;
				cv::Size labelSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_TRIPLEX, 0.5, 1, &baseLine);

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
				//DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(200, 200, 200), 150);
                //cv::putText(frame, label.str(), brect.tl(), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 0, 0));
			}
		}

        //m_detector->CalcMotionMap(frame);
	}
};

#endif

#ifdef BUILD_YOLO_TENSORRT
// ----------------------------------------------------------------------

///
/// \brief The YoloTensorRTExample class
///
class YoloTensorRTExample final : public VideoExample
{
public:
	YoloTensorRTExample(const cv::CommandLineParser& parser)
		: VideoExample(parser)
	{
	}

protected:
	///
	/// \brief InitDetector
	/// \param frame
	/// \return
	///
	bool InitDetector(cv::UMat frame) override
	{
		config_t config;
        if (!m_trackerSettingsLoaded)
        {
#ifdef _WIN32
            std::string pathToModel = "../../data/";
#else
            std::string pathToModel = "../data/";
#endif
            size_t maxBatch = 1;
            enum class YOLOModels
            {
                TinyYOLOv3 = 0,
                YOLOv3,
                YOLOv4,
                TinyYOLOv4,
                YOLOv5,
                YOLOv6,
                YOLOv7,
				YOLOv7Mask,
				YOLOv8
            };
            YOLOModels usedModel = YOLOModels::YOLOv5;
            switch (usedModel)
            {
            case YOLOModels::TinyYOLOv3:
                config.emplace("modelConfiguration", pathToModel + "yolov3-tiny.cfg");
                config.emplace("modelBinary", pathToModel + "yolov3-tiny.weights");
                config.emplace("confidenceThreshold", "0.5");
                config.emplace("inference_precision", "FP32");
                config.emplace("net_type", "YOLOV3");
                maxBatch = 4;
                config.emplace("maxCropRatio", "2");
                break;

            case YOLOModels::YOLOv3:
                config.emplace("modelConfiguration", pathToModel + "yolov3.cfg");
                config.emplace("modelBinary", pathToModel + "yolov3.weights");
                config.emplace("confidenceThreshold", "0.7");
                config.emplace("inference_precision", "FP32");
                config.emplace("net_type", "YOLOV3");
                maxBatch = 2;
                config.emplace("maxCropRatio", "-1");
                break;

            case YOLOModels::YOLOv4:
                config.emplace("modelConfiguration", pathToModel + "yolov4.cfg");
                config.emplace("modelBinary", pathToModel + "yolov4.weights");
                config.emplace("confidenceThreshold", "0.4");
                config.emplace("inference_precision", "FP32");
                config.emplace("net_type", "YOLOV4");
                maxBatch = 1;
                config.emplace("maxCropRatio", "-1");
                break;

            case YOLOModels::TinyYOLOv4:
                config.emplace("modelConfiguration", pathToModel + "yolov4-tiny.cfg");
                config.emplace("modelBinary", pathToModel + "yolov4-tiny.weights");
                config.emplace("confidenceThreshold", "0.5");
                config.emplace("inference_precision", "FP32");
                config.emplace("net_type", "YOLOV4_TINY");
                maxBatch = 1;
                config.emplace("maxCropRatio", "-1");
                break;

            case YOLOModels::YOLOv5:
                config.emplace("modelConfiguration", pathToModel + "yolov5s.cfg");
                config.emplace("modelBinary", pathToModel + "yolov5s.weights");
                config.emplace("confidenceThreshold", "0.5");
                config.emplace("inference_precision", "FP32");
                config.emplace("net_type", "YOLOV5");
                maxBatch = 1;
                config.emplace("maxCropRatio", "-1");
                break;

            case YOLOModels::YOLOv6:
                config.emplace("modelConfiguration", pathToModel + "yolov6s.onnx");
                config.emplace("modelBinary", pathToModel + "yolov6s.onnx");
                config.emplace("confidenceThreshold", "0.5");
                config.emplace("inference_precision", "FP32");
                config.emplace("net_type", "YOLOV6");
                maxBatch = 1;
                config.emplace("maxCropRatio", "-1");
                break;

            case YOLOModels::YOLOv7:
                config.emplace("modelConfiguration", pathToModel + "yolov7.onnx");
                config.emplace("modelBinary", pathToModel + "yolov7.onnx");
                config.emplace("confidenceThreshold", "0.2");
                config.emplace("inference_precision", "FP32");
                config.emplace("net_type", "YOLOV7");
                maxBatch = 1;
                config.emplace("maxCropRatio", "-1");
                break;

			case YOLOModels::YOLOv7Mask:
				config.emplace("modelConfiguration", pathToModel + "yolov7-mask.onnx");
				config.emplace("modelBinary", pathToModel + "yolov7-mask.onnx");
				config.emplace("confidenceThreshold", "0.2");
				config.emplace("inference_precision", "FP32");
				config.emplace("net_type", "YOLOV7Mask");
				maxBatch = 1;
				config.emplace("maxCropRatio", "-1");
				break;

			case YOLOModels::YOLOv8:
				config.emplace("modelConfiguration", pathToModel + "yolov8s.onnx");
				config.emplace("modelBinary", pathToModel + "yolov8s.onnx");
				config.emplace("confidenceThreshold", "0.2");
				config.emplace("inference_precision", "FP32");
				config.emplace("net_type", "YOLOV8");
				maxBatch = 1;
				config.emplace("maxCropRatio", "-1");
				break;
            }
            if (maxBatch < m_batchSize)
                maxBatch = m_batchSize;
            config.emplace("maxBatch", std::to_string(maxBatch));
            config.emplace("classNames", pathToModel + "coco.names");

			//config.emplace("white_list", "person");
			//config.emplace("white_list", "car");
			//config.emplace("white_list", "bicycle");
			//config.emplace("white_list", "motorbike");
			//config.emplace("white_list", "bus");
			//config.emplace("white_list", "truck");
        }
        else
        {
            config.emplace("modelConfiguration", m_trackerSettings.m_nnConfig);
            config.emplace("modelBinary", m_trackerSettings.m_nnWeights);
            config.emplace("confidenceThreshold", std::to_string(m_trackerSettings.m_confidenceThreshold));
            config.emplace("classNames", m_trackerSettings.m_classNames);
            config.emplace("maxCropRatio", std::to_string(m_trackerSettings.m_maxCropRatio));
            config.emplace("maxBatch", std::to_string(m_trackerSettings.m_maxBatch));
            config.emplace("gpuId", std::to_string(m_trackerSettings.m_gpuId));
            config.emplace("net_type", m_trackerSettings.m_netType);
            config.emplace("inference_precision", m_trackerSettings.m_inferencePrecision);
			config.emplace("video_memory", std::to_string(m_trackerSettings.m_maxVideoMemory));

			for (auto wname : m_trackerSettings.m_whiteList)
			{
				config.emplace("white_list", wname);
			}
        }

		m_detector = BaseDetector::CreateDetector(tracking::Detectors::Yolo_TensorRT, config, frame);
		return (m_detector.get() != nullptr);
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
			m_trackerSettings.SetDistance(tracking::DistCenters);
			m_trackerSettings.m_kalmanType = tracking::KalmanLinear;
			m_trackerSettings.m_filterGoal = tracking::FilterCenter;
			m_trackerSettings.m_lostTrackType = tracking::TrackKCF;      // Use visual objects tracker for collisions resolving. Used if m_filterGoal == tracking::FilterRect
			m_trackerSettings.m_matchType = tracking::MatchHungrian;
			m_trackerSettings.m_dt = 0.3f;                                // Delta time for Kalman filter
			m_trackerSettings.m_accelNoiseMag = 0.2f;                     // Accel noise magnitude for Kalman filter
			m_trackerSettings.m_distThres = 0.8f;                         // Distance threshold between region and object on two frames
			m_trackerSettings.m_minAreaRadiusPix = frame.rows / 20.f;
			m_trackerSettings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
			m_trackerSettings.m_maxTraceLength = cvRound(5 * m_fps);      // Maximum trace length
		}
        m_trackerSettings.AddNearTypes(TypeConverter::Str2Type("car"), TypeConverter::Str2Type("bus"), false);
        m_trackerSettings.AddNearTypes(TypeConverter::Str2Type("car"), TypeConverter::Str2Type("truck"), false);
        m_trackerSettings.AddNearTypes(TypeConverter::Str2Type("person"), TypeConverter::Str2Type("bicycle"), true);
        m_trackerSettings.AddNearTypes(TypeConverter::Str2Type("person"), TypeConverter::Str2Type("motorbike"), true);

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

		for (const auto& track : tracks)
		{
			if (track.IsRobust(2,                           // Minimal trajectory size
				0.5f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
				cv::Size2f(0.1f, 8.0f)))      // Min and max ratio: width / height
			{
				DrawTrack(frame, track, true, framesCounter);

				std::stringstream label;
				label << TypeConverter::Type2Str(track.m_type) << " " << std::setprecision(2) << track.m_velocity << ": " << track.m_confidence;
				int baseLine = 0;
				cv::Size labelSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_TRIPLEX, 0.5, 1, &baseLine);

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
				//DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(200, 200, 200), 150);
				//cv::putText(frame, label.str(), brect.tl(), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 0, 0));
			}
		}

		m_detector->CalcMotionMap(frame);
	}
};

#endif
