#pragma once

#include <iostream>
#include <vector>
#include <map>

#include "VideoExample.h"


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
			m_trackerSettings.m_maximumAllowedLostTime = 2.; // Maximum allowed skipped frames
			m_trackerSettings.m_maxTraceLength = 2.;         // Maximum trace length
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

// ----------------------------------------------------------------------

#ifdef BUILD_ONNX_TENSORRT

///
/// \brief The ONNXTensorRTExample class
///
class ONNXTensorRTExample final : public VideoExample
{
public:
	ONNXTensorRTExample(const cv::CommandLineParser& parser)
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
                YOLOv8,
				YOLOV8_OBB,
                YOLOv8Mask,
				YOLOv9,
				YOLOv10,
				YOLOv11,
				YOLOv11_OBB,
				YOLOv11Mask,
				YOLOv12
            };
            YOLOModels usedModel = YOLOModels::YOLOv9;
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

			case YOLOModels::YOLOV8_OBB:
				config.emplace("modelConfiguration", pathToModel + "yolov8s-obb.onnx");
				config.emplace("modelBinary", pathToModel + "yolov8s-obb.onnx");
				config.emplace("confidenceThreshold", "0.2");
				config.emplace("inference_precision", "FP16");
				config.emplace("net_type", "YOLOV8_OBB");
				maxBatch = 1;
				config.emplace("maxCropRatio", "-1");
				break;

			case YOLOModels::YOLOv8Mask:
				config.emplace("modelConfiguration", pathToModel + "yolov8s-seg.onnx");
				config.emplace("modelBinary", pathToModel + "yolov8s-seg.onnx");
				config.emplace("confidenceThreshold", "0.2");
				config.emplace("inference_precision", "FP32");
				config.emplace("net_type", "YOLOV8Mask");
				maxBatch = 1;
				config.emplace("maxCropRatio", "-1");
				break;

			case YOLOModels::YOLOv9:
				config.emplace("modelConfiguration", pathToModel + "yolov9-c.onnx");
				config.emplace("modelBinary", pathToModel + "yolov9-c.onnx");
				config.emplace("confidenceThreshold", "0.2");
				config.emplace("inference_precision", "FP32");
				config.emplace("net_type", "YOLOV9");
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

            m_logger->info("YoloTensorRTExample:");
            m_logger->info("modelConfiguration: {}", m_trackerSettings.m_nnConfig);
            m_logger->info("modelBinary: {}", m_trackerSettings.m_nnWeights);
            m_logger->info("confidenceThreshold: {}", m_trackerSettings.m_confidenceThreshold);
            m_logger->info("classNames: {}", m_trackerSettings.m_classNames);
            m_logger->info("maxCropRatio: {}", m_trackerSettings.m_maxCropRatio);
            m_logger->info("maxBatch: {}", m_trackerSettings.m_maxBatch);
            m_logger->info("gpuId: {}", m_trackerSettings.m_gpuId);
            m_logger->info("net_type: {}", m_trackerSettings.m_netType);
            m_logger->info("inference_precision: {}", m_trackerSettings.m_inferencePrecision);
            m_logger->info("video_memory: {}", m_trackerSettings.m_maxVideoMemory);
            for (auto wname : m_trackerSettings.m_whiteList)
            {
				m_logger->info("white name: {}", wname);
            }
        }

		m_detector = BaseDetector::CreateDetector(tracking::Detectors::ONNX_TensorRT, config, frame);
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
			bool useDeepSORT = true;
			if (useDeepSORT)
			{
#ifdef _WIN32
				std::string pathToModel = "../../data/";
#else
				std::string pathToModel = "../data/";
#endif

				m_trackerSettings.m_embeddings.emplace_back(pathToModel + "reid/osnet_x0_25_msmt17.onnx", pathToModel + "reid/osnet_x0_25_msmt17.onnx",
					cv::Size(128, 256),
					std::vector<objtype_t>{ TypeConverter::Str2Type("person"), TypeConverter::Str2Type("car"), TypeConverter::Str2Type("bus"), TypeConverter::Str2Type("truck"), TypeConverter::Str2Type("vehicle") });

				std::array<track_t, tracking::DistsCount> distType{
					0.f,   // DistCenters
					0.f,   // DistRects
					0.5f,  // DistJaccard
					0.f,   // DistHist
					0.5f,  // DistFeatureCos
					0.f    // DistMahalanobis
				};
				if (!m_trackerSettings.SetDistances(distType))
					m_logger->error("SetDistances failed! Absolutly summ must be equal 1");
			}
			else
			{
				m_trackerSettings.SetDistance(tracking::DistCenters);
			}

			//m_trackerSettings.SetDistance(tracking::DistCenters);
			m_trackerSettings.m_kalmanType = tracking::KalmanLinear;
			m_trackerSettings.m_filterGoal = tracking::FilterCenter;
			m_trackerSettings.m_lostTrackType = tracking::TrackKCF;       // Use visual objects tracker for collisions resolving. Used if m_filterGoal == tracking::FilterRect
			m_trackerSettings.m_matchType = tracking::MatchHungrian;
			m_trackerSettings.m_dt = 0.3f;                                // Delta time for Kalman filter
			m_trackerSettings.m_accelNoiseMag = 0.2f;                     // Accel noise magnitude for Kalman filter
			m_trackerSettings.m_distThres = 0.8f;                         // Distance threshold between region and object on two frames
			m_trackerSettings.m_minAreaRadiusPix = frame.rows / 20.f;
			m_trackerSettings.m_maximumAllowedLostTime = 2.;              // Maximum allowed skipped frames
			m_trackerSettings.m_maxTraceLength = 5.;                      // Maximum trace length
		}
        //m_trackerSettings.AddNearTypes(TypeConverter::Str2Type("car"), TypeConverter::Str2Type("bus"), false);
        //m_trackerSettings.AddNearTypes(TypeConverter::Str2Type("car"), TypeConverter::Str2Type("truck"), false);
        //m_trackerSettings.AddNearTypes(TypeConverter::Str2Type("person"), TypeConverter::Str2Type("bicycle"), true);
        //m_trackerSettings.AddNearTypes(TypeConverter::Str2Type("person"), TypeConverter::Str2Type("motorbike"), true);

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

		static float averFps = 0;
		if (averFps == 0)
			averFps = 1000.f / currTime;
		else
			averFps = 0.9f * averFps + 0.1f * (1000.f / currTime);
		cv::putText(frame, std::to_string(cvRound(averFps)) + " fps", cv::Point(10, 40), cv::FONT_HERSHEY_TRIPLEX, (frame.cols > 1000) ? 1.5 : 1.0, cv::Scalar(255, 0, 255));

		for (const auto& track : tracks)
		{
            if (track.IsRobust(2,                           // Minimal trajectory size
                0.5f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                cv::Size2f(0.1f, 8.0f), 2))      // Min and max ratio: width / height
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

		//m_detector->CalcMotionMap(frame);
	}
};

#endif // BUILD_ONNX_TENSORRT
