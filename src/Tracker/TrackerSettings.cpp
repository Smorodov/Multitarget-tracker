#include <sstream>
#include "TrackerSettings.h"
#include <inih/INIReader.h>

///
/// \brief CarsCounting::ParseTrackerSettings
///
bool ParseTrackerSettings(const std::string& settingsFile, TrackerSettings& trackerSettings)
{
    bool res = false;

    std::cout << "ParseTrackerSettings: " << settingsFile << " ..." << std::endl;

    INIReader reader(settingsFile);

    if (reader.ParseError() >= 0)
    {
        std::cout << "ParseTrackerSettings - readed" << std::endl;

        trackerSettings = TrackerSettings();

        // Read tracking settings
        auto distType = reader.GetInteger("tracking", "distance_type", -1);
        if (distType >= 0 && distType < (int)tracking::DistsCount)
            trackerSettings.SetDistance((tracking::DistType)distType);

        auto kalmanType = reader.GetInteger("tracking", "kalman_type", -1);
        if (kalmanType >= 0 && kalmanType < (int)tracking::KalmanCount)
            trackerSettings.m_kalmanType = (tracking::KalmanType)kalmanType;

        auto filterGoal = reader.GetInteger("tracking", "filter_goal", -1);
        if (filterGoal >= 0 && filterGoal < (int)tracking::FiltersCount)
            trackerSettings.m_filterGoal = (tracking::FilterGoal)filterGoal;

        auto lostTrackType = reader.GetInteger("tracking", "lost_track_type", -1);
        if (lostTrackType >= 0 && lostTrackType < (int)tracking::SingleTracksCount)
            trackerSettings.m_lostTrackType = (tracking::LostTrackType)lostTrackType;

        auto matchType = reader.GetInteger("tracking", "match_type", -1);
        if (matchType >= 0 && matchType < (int)tracking::MatchCount)
            trackerSettings.m_matchType = (tracking::MatchType)matchType;

        trackerSettings.m_useAcceleration = reader.GetInteger("tracking", "use_aceleration", 0) != 0; // Use constant acceleration motion model
        trackerSettings.m_dt = static_cast<track_t>(reader.GetReal("tracking", "delta_time", 0.4));  // Delta time for Kalman filter
        trackerSettings.m_accelNoiseMag = static_cast<track_t>(reader.GetReal("tracking", "accel_noise", 0.2)); // Accel noise magnitude for Kalman filter
        trackerSettings.m_distThres = static_cast<track_t>(reader.GetReal("tracking", "dist_thresh", 0.8));     // Distance threshold between region and object on two frames
        trackerSettings.m_minAreaRadiusPix = static_cast<track_t>(reader.GetReal("tracking", "min_area_radius_pix", -1.));
        trackerSettings.m_minAreaRadiusK = static_cast<track_t>(reader.GetReal("tracking", "min_area_radius_k", 0.8));
        trackerSettings.m_maximumAllowedSkippedFrames = reader.GetInteger("tracking", "max_skip_frames", 50); // Maximum allowed skipped frames
        trackerSettings.m_maxTraceLength = reader.GetInteger("tracking", "max_trace_len", 50);                 // Maximum trace length
        trackerSettings.m_useAbandonedDetection = reader.GetInteger("tracking", "detect_abandoned", 0) != 0;
        trackerSettings.m_minStaticTime = reader.GetInteger("tracking", "min_static_time", 5);
        trackerSettings.m_maxStaticTime = reader.GetInteger("tracking", "max_static_time", 25);
        trackerSettings.m_maxSpeedForStatic = reader.GetInteger("tracking", "max_speed_for_static", 10);


        // Read detection settings
        trackerSettings.m_nnWeights = reader.GetString("detection", "nn_weights", "data/yolov4-tiny_best.weights");
        trackerSettings.m_nnConfig = reader.GetString("detection", "nn_config", "data/yolov4-tiny.cfg");
        trackerSettings.m_classNames = reader.GetString("detection", "class_names", "data/traffic.names");
        trackerSettings.m_confidenceThreshold = static_cast<track_t>(reader.GetReal("detection", "confidence_threshold", 0.5));
        trackerSettings.m_maxCropRatio = static_cast<track_t>(reader.GetReal("detection", "max_crop_ratio", -1));
        trackerSettings.m_maxBatch = reader.GetInteger("detection", "max_batch", 1);
        trackerSettings.m_gpuId = reader.GetInteger("detection", "gpu_id", 0);
        trackerSettings.m_netType = reader.GetString("detection", "net_type", "YOLOV4");
        trackerSettings.m_inferencePrecision = reader.GetString("detection", "inference_precision", "FP16");
        trackerSettings.m_detectorBackend = reader.GetInteger("detection", "detector_backend", (int)tracking::Detectors::DNN_OCV);
        trackerSettings.m_dnnTarget = reader.GetString("detection", "ocv_dnn_target", "DNN_TARGET_CPU");
        trackerSettings.m_dnnBackend = reader.GetString("detection", "ocv_dnn_backend", "DNN_BACKEND_OPENCV");
		trackerSettings.m_maxVideoMemory = reader.GetInteger("detection", "video_memory", 0);
        trackerSettings.m_inputSize.width = reader.GetInteger("detection", "input_width", 0);
        trackerSettings.m_inputSize.height = reader.GetInteger("detection", "input_height", 0);

		std::stringstream whiteList{ reader.GetString("detection", "white_list", "") };
		trackerSettings.m_whiteList.clear();
		std::string wname;
		while (std::getline(whiteList, wname, ';'))
		{
			trackerSettings.m_whiteList.push_back(wname);
		}

        res = true;
    }
    std::cout << "ParseTrackerSettings: " << res << std::endl;
    return res;
}
