#pragma once

#include <iostream>
#include <vector>
#include <map>

#include "VideoExample.h"

///
/// \brief The MotionDetectorExample class
///
class MotionDetectorExample : public VideoExample
{
public:
    MotionDetectorExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser),
          m_minObjWidth(10)
    {
    }

protected:
    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
    {
        m_minObjWidth = frame.cols / 50;

        const int minStaticTime = 5;

        config_t config;
#if 1
        config["history"] = std::to_string(cvRound(10 * minStaticTime * m_fps));
        config["varThreshold"] = "16";
        config["detectShadows"] = "1";
        config["useRotatedRect"] = "0";
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Motion_MOG2, config, frame));
#else
        config["minPixelStability"] = "15";
        config["maxPixelStability"] = "900";
        config["useHistory"] = "1";
        config["isParallel"] = "1";
        config["useRotatedRect"] = "0";
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Motion_CNT, config, frame));
#endif
        m_detector->SetMinObjectSize(cv::Size(m_minObjWidth, m_minObjWidth));

        TrackerSettings settings;
        settings.m_distType = tracking::DistCenters;
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackKCF;       // Use visual objects tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.4f;                             // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.5f;                  // Accel noise magnitude for Kalman filter
        settings.m_distThres = frame.rows / 20.f;         // Distance threshold between region and object on two frames

        settings.m_useAbandonedDetection = true;
        if (settings.m_useAbandonedDetection)
        {
            settings.m_minStaticTime = minStaticTime;
            settings.m_maxStaticTime = 60;
            settings.m_maximumAllowedSkippedFrames = cvRound(settings.m_minStaticTime * m_fps); // Maximum allowed skipped frames
            settings.m_maxTraceLength = 2 * settings.m_maximumAllowedSkippedFrames;        // Maximum trace length
        }
        else
        {
            settings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
            settings.m_maxTraceLength = cvRound(4 * m_fps);              // Maximum trace length
        }

        m_tracker = std::make_unique<CTracker>(settings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
		auto tracks = m_tracker->GetTracks();

        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : tracks)
        {
            if (track.m_isStatic)
            {
                DrawTrack(frame, 1, track, true);
            }
            else
            {
                if (track.IsRobust(cvRound(m_fps / 4),          // Minimal trajectory size
                                    0.7f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                    cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                        )
                {
                    DrawTrack(frame, 1, track, true);
                }
            }
        }

        m_detector->CalcMotionMap(frame);
    }

private:
    int m_minObjWidth;
};

// ----------------------------------------------------------------------

///
/// \brief The FaceDetectorExample class
///
class FaceDetectorExample : public VideoExample
{
public:
    FaceDetectorExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
    {
#ifdef _WIN32
		std::string pathToModel = "../../data/";
#else
		std::string pathToModel = "../data/";
#endif

        config_t config;
        config["cascadeFileName"] = pathToModel + "haarcascade_frontalface_alt2.xml";
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Face_HAAR, config, frame));
        if (!m_detector.get())
        {
            return false;
        }
        m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));

        TrackerSettings settings;
        settings.m_distType = tracking::DistJaccard;
        settings.m_kalmanType = tracking::KalmanUnscented;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackCSRT;       // Use visual objects tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                             // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                  // Accel noise magnitude for Kalman filter
        settings.m_distThres = 0.8f;           // Distance threshold between region and object on two frames
        settings.m_maximumAllowedSkippedFrames = cvRound(m_fps / 2);   // Maximum allowed skipped frames
        settings.m_maxTraceLength = cvRound(5 * m_fps);            // Maximum trace length

        m_tracker = std::make_unique<CTracker>(settings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
		auto tracks = m_tracker->GetTracks();

        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : tracks)
        {
            if (track.IsRobust(8,                           // Minimal trajectory size
                                0.4f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, track);
            }
        }

        m_detector->CalcMotionMap(frame);
    }
};

// ----------------------------------------------------------------------

///
/// \brief The PedestrianDetectorExample class
///
class PedestrianDetectorExample : public VideoExample
{
public:
    PedestrianDetectorExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
    {
        tracking::Detectors detectorType = tracking::Detectors::Pedestrian_C4; // tracking::Detectors::Pedestrian_HOG;

#ifdef _WIN32
		std::string pathToModel = "../../data/";
#else
		std::string pathToModel = "../data/";
#endif

        config_t config;
        config["detectorType"] = (detectorType == tracking::Pedestrian_HOG) ? "HOG" : "C4";
        config["cascadeFileName1"] = pathToModel + "combined.txt.model";
        config["cascadeFileName2"] = pathToModel + "combined.txt.model_";
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(detectorType, config, frame));
        if (!m_detector.get())
        {
            return false;
        }
        m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));


        TrackerSettings settings;
        settings.m_distType = tracking::DistRects;
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackCSRT;       // Use visual objects tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                             // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                  // Accel noise magnitude for Kalman filter
        settings.m_distThres = frame.rows / 10.f;         // Distance threshold between region and object on two frames
        settings.m_maximumAllowedSkippedFrames = cvRound(m_fps);   // Maximum allowed skipped frames
        settings.m_maxTraceLength = cvRound(5 * m_fps);   // Maximum trace length

        m_tracker = std::make_unique<CTracker>(settings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
		auto tracks = m_tracker->GetTracks();

        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : tracks)
        {
			if (track.IsRobust(cvRound(m_fps / 2),          // Minimal trajectory size
                                0.4f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, track);
            }
        }

        m_detector->CalcMotionMap(frame);
    }
};

// ----------------------------------------------------------------------

///
/// \brief The SSDMobileNetExample class
///
class SSDMobileNetExample : public VideoExample
{
public:
    SSDMobileNetExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
    {
#ifdef _WIN32
		std::string pathToModel = "../../data/";
#else
		std::string pathToModel = "../data/";
#endif
        config_t config;
        config["modelConfiguration"] = pathToModel + "MobileNetSSD_deploy.prototxt";
        config["modelBinary"] = pathToModel + "MobileNetSSD_deploy.caffemodel";
        config["confidenceThreshold"] = "0.5";
        config["maxCropRatio"] = "3.0";
        config["dnnTarget"] = "DNN_TARGET_CPU";
        config["dnnBackend"] = "DNN_BACKEND_INFERENCE_ENGINE";

        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::SSD_MobileNet, config, frame));
        if (!m_detector.get())
        {
            return false;
        }
        m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));

        TrackerSettings settings;
        settings.m_distType = tracking::DistRects;
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackCSRT;       // Use visual objects tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                                // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                     // Accel noise magnitude for Kalman filter
        settings.m_distThres = frame.rows / 10.f;            // Distance threshold between region and object on two frames
        settings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
        settings.m_maxTraceLength = cvRound(5 * m_fps);      // Maximum trace length

        m_tracker = std::make_unique<CTracker>(settings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
		auto tracks = m_tracker->GetTracks();

        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : tracks)
        {
            if (track.IsRobust(5,                           // Minimal trajectory size
                                0.2f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, track);

                std::string label = track.m_type + ": " + std::to_string(track.m_confidence);
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                cv::Rect brect = track.m_rrect.boundingRect();
#if (CV_VERSION_MAJOR >= 4)
                cv::rectangle(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 255, 255), cv::FILLED);
#else
                cv::rectangle(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 255, 255), CV_FILLED);
#endif
                cv::putText(frame, label, brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }

        m_detector->CalcMotionMap(frame);
    }

    ///
    /// \brief GrayProcessing
    /// \return
    ///
    bool GrayProcessing() const
    {
        return false;
    }
};

// ----------------------------------------------------------------------

///
/// \brief The YoloExample class
///
class YoloExample : public VideoExample
{
public:
    YoloExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
    {
        config_t config;
        const int yoloTest = 0;

#ifdef _WIN32
		std::string pathToModel = "../../data/";
#else
		std::string pathToModel = "../data/";
#endif

        switch (yoloTest)
        {
        case 0:
            config["modelConfiguration"] = pathToModel + "tiny-yolo.cfg";
            config["modelBinary"] = pathToModel + "tiny-yolo.weights";
            break;

        case 1:
            config["modelConfiguration"] = pathToModel + "yolov3-tiny.cfg";
            config["modelBinary"] = pathToModel + "yolov3-tiny.weights";
            config["classNames"] = pathToModel + "coco.names";
            break;
        }

        config["confidenceThreshold"] = "0.1";
        config["maxCropRatio"] = "2.0";
        config["dnnTarget"] = "DNN_TARGET_CPU";
        config["dnnBackend"] = "DNN_BACKEND_INFERENCE_ENGINE";

        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Yolo_OCV, config, frame));
        if (!m_detector.get())
        {
            return false;
        }
        m_detector->SetMinObjectSize(cv::Size(frame.cols / 40, frame.rows / 40));

        TrackerSettings settings;
        settings.m_distType = tracking::DistRects;
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackCSRT;       // Use visual objects tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                                // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.2f;                     // Accel noise magnitude for Kalman filter
        settings.m_distThres = frame.rows / 10.f;            // Distance threshold between region and object on two frames
        settings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
        settings.m_maxTraceLength = cvRound(5 * m_fps);      // Maximum trace length

        m_tracker = std::make_unique<CTracker>(settings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
		auto tracks = m_tracker->GetTracks();

        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : tracks)
        {
            if (track.IsRobust(1,                           // Minimal trajectory size
                                0.1f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, track);

                std::string label = track.m_type + ": " + std::to_string(track.m_confidence);
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                cv::Rect brect = track.m_rrect.boundingRect();
#if (CV_VERSION_MAJOR >= 4)
                cv::rectangle(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 255, 255), cv::FILLED);
#else
                cv::rectangle(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 255, 255), CV_FILLED);
#endif
                cv::putText(frame, label, brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }

        m_detector->CalcMotionMap(frame);
    }

    ///
    /// \brief GrayProcessing
    /// \return
    ///
    bool GrayProcessing() const
    {
        return false;
    }
};

#ifdef BUILD_YOLO_LIB
// ----------------------------------------------------------------------

///
/// \brief The YoloDarknetExample class
///
class YoloDarknetExample : public VideoExample
{
public:
	YoloDarknetExample(const cv::CommandLineParser& parser)
		:
		VideoExample(parser)
	{
	}

protected:
    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
	bool InitTracker(cv::UMat frame)
	{
		config_t config;

#ifdef _WIN32
		std::string pathToModel = "../../data/";
#else
		std::string pathToModel = "../data/";
#endif

		config["modelConfiguration"] = pathToModel + "yolov3-tiny.cfg";
		config["modelBinary"] = pathToModel + "yolov3-tiny.weights";
		config["classNames"] = pathToModel + "coco.names";
		config["confidenceThreshold"] = "0.1";
		config["maxCropRatio"] = "2.0";

        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Yolo_Darknet, config, frame));
		if (!m_detector.get())
		{
			return false;
		}
		m_detector->SetMinObjectSize(cv::Size(frame.cols / 40, frame.rows / 40));

		TrackerSettings settings;
		settings.m_distType = tracking::DistRects;
		settings.m_kalmanType = tracking::KalmanLinear;
		settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackCSRT;       // Use visual objects tracker for collisions resolving
		settings.m_matchType = tracking::MatchHungrian;
		settings.m_dt = 0.3f;                                // Delta time for Kalman filter
		settings.m_accelNoiseMag = 0.2f;                     // Accel noise magnitude for Kalman filter
		settings.m_distThres = frame.rows / 10.f;            // Distance threshold between region and object on two frames
		settings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
		settings.m_maxTraceLength = cvRound(5 * m_fps);      // Maximum trace length

		m_tracker = std::make_unique<CTracker>(settings);

		return true;
	}

    ///
    /// \brief DrawData
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
	void DrawData(cv::Mat frame, int framesCounter, int currTime)
	{
		auto tracks = m_tracker->GetTracks();

		if (m_showLogs)
		{
			std::cout << "Frame " << framesCounter << ": tracks = " << tracks.size() << ", time = " << currTime << std::endl;
		}

		for (const auto& track : tracks)
		{
			if (track.IsRobust(1,                           // Minimal trajectory size
				0.1f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
				cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
				)
			{
				DrawTrack(frame, 1, track);

				std::string label = track.m_type + ": " + std::to_string(track.m_confidence);
				int baseLine = 0;
				cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                cv::Rect brect = track.m_rrect.boundingRect();
#if (CV_VERSION_MAJOR >= 4)
                cv::rectangle(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 255, 255), cv::FILLED);
#else
                cv::rectangle(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 255, 255), CV_FILLED);
#endif
                cv::putText(frame, label, brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
			}
		}

		m_detector->CalcMotionMap(frame);
	}

	///
	/// \brief GrayProcessing
	/// \return
	///
	bool GrayProcessing() const
	{
		return false;
	}
};

#endif
