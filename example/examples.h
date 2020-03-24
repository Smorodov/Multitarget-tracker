#pragma once

#include <iostream>
#include <vector>
#include <map>

#include "VideoExample.h"

///
/// \brief DrawFilledRect
///
void DrawFilledRect(cv::Mat& frame, const cv::Rect& rect, cv::Scalar cl, int alpha)
{
	if (alpha)
	{
		const int alpha_1 = 255 - alpha;
		const int nchans = frame.channels();
		int color[3] = { cv::saturate_cast<int>(cl[0]), cv::saturate_cast<int>(cl[1]), cv::saturate_cast<int>(cl[2]) };
		for (int y = rect.y; y < rect.y + rect.height; ++y)
		{
			uchar* ptr = frame.ptr(y) + nchans * rect.x;
			for (int x = rect.x; x < rect.x + rect.width; ++x)
			{
				for (int i = 0; i < nchans; ++i)
				{
					ptr[i] = cv::saturate_cast<uchar>((alpha_1 * ptr[i] + alpha * color[i]) / 255);
				}
				ptr += nchans;
			}
		}
	}
	else
	{
		cv::rectangle(frame, rect, cl, cv::FILLED);
	}
}

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
    /// \brief InitDetector
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame)
    {
        m_minObjWidth = frame.cols / 50;

        config_t config;
#if 1
        config.emplace("history", std::to_string(cvRound(10 * m_minStaticTime * m_fps)));
        config.emplace("varThreshold", "16");
        config.emplace("detectShadows", "1");
        config.emplace("useRotatedRect", "0");
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Motion_MOG2, config, frame));
#else
        config.emplace("minPixelStability", "15");
        config.emplace("maxPixelStability", "900");
        config.emplace("useHistory", "1");
        config.emplace("isParallel", "1");
        config.emplace("useRotatedRect", "0");
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Motion_CNT, config, frame));
#endif

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
    bool InitTracker(cv::UMat frame)
    {
        TrackerSettings settings;
		settings.SetDistance(tracking::DistCenters);
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterCenter;
        settings.m_lostTrackType = tracking::TrackKCF;       // Use visual objects tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.4f;                             // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.5f;                  // Accel noise magnitude for Kalman filter
        settings.m_distThres = 0.95f;                    // Distance threshold between region and object on two frames
        settings.m_minAreaRadius = frame.rows / 20.f;

        settings.m_useAbandonedDetection = false;
        if (settings.m_useAbandonedDetection)
        {
            settings.m_minStaticTime = m_minStaticTime;
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
    int m_minObjWidth = 8;
    int m_minStaticTime = 5;
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
    /// \brief InitDetector
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame)
    {
#ifdef _WIN32
        std::string pathToModel = "../../data/";
#else
        std::string pathToModel = "../data/";
#endif

        config_t config;
        config.emplace("cascadeFileName", pathToModel + "haarcascade_frontalface_alt2.xml");
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Face_HAAR, config, frame));
        if (m_detector.get())
        {
            m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));
            return true;
        }
        return false;
    }
    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
    {
        TrackerSettings settings;
		settings.SetDistance(tracking::DistJaccard);
        settings.m_kalmanType = tracking::KalmanUnscented;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackCSRT;      // Use visual objects tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                                // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                     // Accel noise magnitude for Kalman filter
        settings.m_distThres = 0.8f;                         // Distance threshold between region and object on two frames
        settings.m_minAreaRadius = frame.rows / 20.f;
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
    /// \brief InitDetector
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame)
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
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(detectorType, config, frame));
        if (m_detector.get())
        {
            m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));
            return true;
        }
        return false;
    }
    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
    {
        TrackerSettings settings;
		settings.SetDistance(tracking::DistRects);
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackCSRT;   // Use visual objects tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                             // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                  // Accel noise magnitude for Kalman filter
        settings.m_distThres = 0.8f;                      // Distance threshold between region and object on two frames
        settings.m_minAreaRadius = frame.rows / 20.f;
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
    /// \brief InitDetector(
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame)
    {
#ifdef _WIN32
        std::string pathToModel = "../../data/";
#else
        std::string pathToModel = "../data/";
#endif
        config_t config;
        config.emplace("modelConfiguration", pathToModel + "MobileNetSSD_deploy.prototxt");
        config.emplace("modelBinary", pathToModel + "MobileNetSSD_deploy.caffemodel");
        config.emplace("confidenceThreshold", "0.5");
        config.emplace("maxCropRatio", "3.0");
        config.emplace("dnnTarget", "DNN_TARGET_CPU");
        config.emplace("dnnBackend", "DNN_BACKEND_INFERENCE_ENGINE");

        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::SSD_MobileNet, config, frame));
        if (m_detector.get())
        {
            m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));
            return true;
        }
        return false;
    }
    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
    {
        TrackerSettings settings;
		settings.SetDistance(tracking::DistRects);
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackCSRT;      // Use visual objects tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                                // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                     // Accel noise magnitude for Kalman filter
        settings.m_distThres = 0.8f;                         // Distance threshold between region and object on two frames
        settings.m_minAreaRadius = frame.rows / 20.f;
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
				DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(200, 200, 200), 150);
                cv::putText(frame, label, brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }

        m_detector->CalcMotionMap(frame);
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
    /// \brief InitDetector(
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame)
    {
        config_t config;
        const int yoloTest = 1;

#ifdef _WIN32
        std::string pathToModel = "../../data/";
#else
        std::string pathToModel = "../data/";
#endif

        switch (yoloTest)
        {
        case 0:
            config.emplace("modelConfiguration", pathToModel + "tiny-yolo.cfg");
            config.emplace("modelBinary", pathToModel + "tiny-yolo.weights");
            break;

        case 1:
            config.emplace("modelConfiguration", pathToModel + "yolov3-tiny.cfg");
            config.emplace("modelBinary", pathToModel + "yolov3-tiny.weights");
            config.emplace("classNames", pathToModel + "coco.names");
            break;
        }

        config.emplace("confidenceThreshold", "0.1");
        config.emplace("maxCropRatio", "2.0");
        config.emplace("dnnTarget", "DNN_TARGET_CPU");
        config.emplace("dnnBackend", "DNN_BACKEND_OPENCV");

        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Yolo_OCV, config, frame));
        if (m_detector.get())
        {
            m_detector->SetMinObjectSize(cv::Size(frame.cols / 40, frame.rows / 40));
            return true;
        }
        return false;
    }
    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
    {
        TrackerSettings settings;
		settings.SetDistance(tracking::DistRects);
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackCSRT;      // Use visual objects tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                                // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.2f;                     // Accel noise magnitude for Kalman filter
        settings.m_distThres = 0.8f;                         // Distance threshold between region and object on two frames
        settings.m_minAreaRadius = frame.rows / 20.f;
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
				DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(200, 200, 200), 150);
                cv::putText(frame, label, brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }

        m_detector->CalcMotionMap(frame);
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
    /// \brief InitDetector
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame)
    {
        config_t config;

#ifdef _WIN32
        std::string pathToModel = "../../data/";
#else
        std::string pathToModel = "../data/";
#endif
#if 0
        config.emplace("modelConfiguration", pathToModel + "yolov3-tiny.cfg");
        config.emplace("modelBinary", pathToModel + "yolov3-tiny.weights");
		config.emplace("confidenceThreshold", "0.5");
#else
        config.emplace("modelConfiguration", pathToModel + "yolov3.cfg");
        config.emplace("modelBinary", pathToModel + "yolov3.weights");
		config.emplace("confidenceThreshold", "0.7");
#endif
        config.emplace("classNames", pathToModel + "coco.names");
        config.emplace("maxCropRatio", "-1");

        config.emplace("white_list", "person");
        config.emplace("white_list", "car");
        config.emplace("white_list", "bicycle");
        config.emplace("white_list", "motorbike");
        config.emplace("white_list", "bus");
        config.emplace("white_list", "truck");
        //config.emplace("white_list", "traffic light");
        //config.emplace("white_list", "stop sign");

        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Yolo_Darknet, config, frame));
        if (m_detector.get())
        {
            m_detector->SetMinObjectSize(cv::Size(frame.cols / 40, frame.rows / 40));
            return true;
        }
        return false;
    }

    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
	{
		TrackerSettings settings;
        settings.SetDistance(tracking::DistCenters);
		settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterCenter;
        settings.m_lostTrackType = tracking::TrackKCF;      // Use visual objects tracker for collisions resolving
		settings.m_matchType = tracking::MatchHungrian;
		settings.m_dt = 0.3f;                                // Delta time for Kalman filter
		settings.m_accelNoiseMag = 0.2f;                     // Accel noise magnitude for Kalman filter
        settings.m_distThres = 0.8f;                         // Distance threshold between region and object on two frames
        settings.m_minAreaRadius = frame.rows / 20.f;
		settings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
		settings.m_maxTraceLength = cvRound(5 * m_fps);      // Maximum trace length

		settings.AddNearTypes("car", "bus", false);
		settings.AddNearTypes("car", "truck", false);
		settings.AddNearTypes("person", "bicycle", true);
		settings.AddNearTypes("person", "motorbike", true);

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
            if (track.IsRobust(2,                           // Minimal trajectory size
                0.5f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
				cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
				)
			{
				DrawTrack(frame, 1, track);


				std::stringstream label;
				label << track.m_type << " " << std::setprecision(2) << track.m_velocity << ": " << track.m_confidence;
				int baseLine = 0;
				cv::Size labelSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

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
				DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(200, 200, 200), 150);
                cv::putText(frame, label.str(), brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
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
class YoloTensorRTExample : public VideoExample
{
public:
	YoloTensorRTExample(const cv::CommandLineParser& parser)
		:
		VideoExample(parser)
	{
	}

protected:
	///
	/// \brief InitDetector
	/// \param frame
	/// \return
	///
	bool InitDetector(cv::UMat frame)
	{
		config_t config;

#ifdef _WIN32
		std::string pathToModel = "../../data/";
#else
		std::string pathToModel = "../data/";
#endif
#if 0
		config.emplace("modelConfiguration", pathToModel + "yolov3-tiny.cfg");
		config.emplace("modelBinary", pathToModel + "yolov3-tiny.weights");
		config.emplace("confidenceThreshold", "0.5");
#else
		config.emplace("modelConfiguration", pathToModel + "yolov3.cfg");
		config.emplace("modelBinary", pathToModel + "yolov3.weights");
		config.emplace("confidenceThreshold", "0.7");
#endif
		config.emplace("classNames", pathToModel + "coco.names");
		config.emplace("maxCropRatio", "-1");

		config.emplace("white_list", "person");
		config.emplace("white_list", "car");
		config.emplace("white_list", "bicycle");
		config.emplace("white_list", "motorbike");
		config.emplace("white_list", "bus");
		config.emplace("white_list", "truck");
		//config.emplace("white_list", "traffic light");
		//config.emplace("white_list", "stop sign");

		m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Yolo_TensorRT, config, frame));
		if (m_detector.get())
		{
			m_detector->SetMinObjectSize(cv::Size(frame.cols / 40, frame.rows / 40));
			return true;
		}
		return false;
	}

	///
	/// \brief InitTracker
	/// \param frame
	/// \return
	///
	bool InitTracker(cv::UMat frame)
	{
		TrackerSettings settings;
		settings.SetDistance(tracking::DistCenters);
		settings.m_kalmanType = tracking::KalmanLinear;
		settings.m_filterGoal = tracking::FilterCenter;
		settings.m_lostTrackType = tracking::TrackKCF;      // Use visual objects tracker for collisions resolving
		settings.m_matchType = tracking::MatchHungrian;
		settings.m_dt = 0.3f;                                // Delta time for Kalman filter
		settings.m_accelNoiseMag = 0.2f;                     // Accel noise magnitude for Kalman filter
		settings.m_distThres = 0.8f;                         // Distance threshold between region and object on two frames
		settings.m_minAreaRadius = frame.rows / 20.f;
		settings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
		settings.m_maxTraceLength = cvRound(5 * m_fps);      // Maximum trace length

		settings.AddNearTypes("car", "bus", false);
		settings.AddNearTypes("car", "truck", false);
		settings.AddNearTypes("person", "bicycle", true);
		settings.AddNearTypes("person", "motorbike", true);

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
			if (track.IsRobust(2,                           // Minimal trajectory size
				0.5f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
				cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
				)
			{
				DrawTrack(frame, 1, track);


				std::stringstream label;
				label << track.m_type << " " << std::setprecision(2) << track.m_velocity << ": " << track.m_confidence;
				int baseLine = 0;
				cv::Size labelSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

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
				DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(200, 200, 200), 150);
				cv::putText(frame, label.str(), brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
			}
		}

		//m_detector->CalcMotionMap(frame);
	}
};

#endif
