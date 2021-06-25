#pragma once
#include <vector>
#include <array>
#include <numeric>

#include "defines.h"

///
/// \brief The TrackerSettings struct
///
struct TrackerSettings
{
    ///
    /// Tracker settings
    ///

    tracking::KalmanType m_kalmanType = tracking::KalmanLinear;
    tracking::FilterGoal m_filterGoal = tracking::FilterCenter;
    tracking::LostTrackType m_lostTrackType = tracking::TrackKCF; // Used if m_filterGoal == tracking::FilterRect
    tracking::MatchType m_matchType = tracking::MatchHungrian;

    std::array<track_t, tracking::DistsCount> m_distType;

    ///
    /// \brief m_dt
    /// Time step for Kalman
    ///
    track_t m_dt = 1.0f;

    ///
    /// \brief m_accelNoiseMag
    /// Noise magnitude for Kalman
    ///
    track_t m_accelNoiseMag = 0.1f;

    ///
    /// \brief m_useAcceleration
    /// Constant velocity or constant acceleration motion model
    ///
    bool m_useAcceleration = false;

    ///
    /// \brief m_distThres
    /// Distance threshold for Assignment problem: from 0 to 1
    ///
    track_t m_distThres = 0.8f;

    ///
    /// \brief m_minAreaRadius
    /// Minimal area radius in pixels for objects centers
    ///
    track_t m_minAreaRadiusPix = 20.f;

    ///
    /// \brief m_minAreaRadius
    /// Minimal area radius in ration for object size. Used if m_minAreaRadiusPix < 0
    ///
    track_t m_minAreaRadiusK = 0.5f;

    ///
    /// \brief m_maximumAllowedSkippedFrames
    /// If the object don't assignment more than this frames then it will be removed
    ///
    size_t m_maximumAllowedSkippedFrames = 25;

    ///
    /// \brief m_maxTraceLength
    /// The maximum trajectory length
    ///
    size_t m_maxTraceLength = 50;

    ///
    /// \brief m_useAbandonedDetection
    /// Detection abandoned objects
    ///
    bool m_useAbandonedDetection = false;

    ///
    /// \brief m_minStaticTime
    /// After this time (in seconds) the object is considered abandoned
    ///
    int m_minStaticTime = 5;
    ///
    /// \brief m_maxStaticTime
    /// After this time (in seconds) the abandoned object will be removed
    ///
    int m_maxStaticTime = 25;
    ///
    /// \brief m_maxSpeedForStatic
    /// Speed in pixels
    /// If speed of object is more that this value than object is non static
    ///
    int m_maxSpeedForStatic = 10;

    ///
    /// \brief m_nearTypes
    /// Object types that can be matched while tracking
    ///
    std::map<objtype_t, std::set<objtype_t>> m_nearTypes;


    ///
    /// Detector settings
    ///

    ///
    std::string m_nnWeights = "data/yolov4-tiny_best.weights";
    
    ///
    std::string m_nnConfig = "data/yolov4-tiny.cfg";
    
    ///
    std::string m_classNames = "data/traffic.names";

    ///
    float m_confidenceThreshold = 0.5f;

    ///
    float m_maxCropRatio = -1.f;

    ///
    int m_maxBatch = 1;

    ///
    int m_gpuId = 0;

    ///
    /// YOLOV2
    /// YOLOV3
    /// YOLOV2_TINY
    /// YOLOV3_TINY
    /// YOLOV4
    /// YOLOV4_TINY
    /// YOLOV5
    std::string m_netType = "YOLOV4_TINY";

    ///
    /// INT8
    /// FP16
    /// FP32
    std::string m_inferencePrecison = "FP16";

    // opencv_dnn = 12
    // darknet_cudnn = 10
    // tensorrt = 11
    int m_detectorBackend = 11;

    // DNN_TARGET_CPU
    // DNN_TARGET_OPENCL
    // DNN_TARGET_OPENCL_FP16
    // DNN_TARGET_MYRIAD
    // DNN_TARGET_CUDA
    // DNN_TARGET_CUDA_FP16
    std::string m_dnnTarget = "DNN_TARGET_CPU";

    // DNN_BACKEND_DEFAULT
    // DNN_BACKEND_HALIDE
    // DNN_BACKEND_INFERENCE_ENGINE
    // DNN_BACKEND_OPENCV
    // DNN_BACKEND_VKCOM
    // DNN_BACKEND_CUDA
    // DNN_BACKEND_INFERENCE_ENGINE_NGRAPH
    // DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019
    std::string m_dnnBackend = "DNN_BACKEND_OPENCV";


    ///
    struct EmbeddingParams
    {
        ///
        /// \brief m_embeddingCfgName
        /// Neural network config file for embeddings
        ///
        std::string m_embeddingCfgName;
        ///
        /// \brief m_embeddingWeightsName
        /// Neural network weights file for embeddings
        ///
        std::string m_embeddingWeightsName;

		///
		cv::Size m_inputLayer{128, 256};

		///
		std::vector<ObjectTypes> m_objectTypes;

		EmbeddingParams(const std::string& embeddingCfgName, const std::string& embeddingWeightsName,
			const cv::Size& inputLayer, const std::vector<ObjectTypes>& objectTypes)
			: m_embeddingCfgName(embeddingCfgName),
			m_embeddingWeightsName(embeddingWeightsName),
			m_inputLayer(inputLayer),
			m_objectTypes(objectTypes)
		{
			assert(!m_objectTypes.empty());
		}
    };
	///
	std::vector<EmbeddingParams> m_embeddings;

	///
	TrackerSettings()
	{
		m_distType[tracking::DistCenters] = 0.0f;
		m_distType[tracking::DistRects] = 0.0f;
		m_distType[tracking::DistJaccard] = 0.5f;
		m_distType[tracking::DistHist] = 0.5f;
		m_distType[tracking::DistFeatureCos] = 0.0f;

		assert(CheckDistance());
	}

	///
	bool CheckDistance() const
	{
		track_t sum = std::accumulate(m_distType.begin(), m_distType.end(), 0.0f);
		track_t maxOne = std::max(1.0f, std::fabs(sum));
		//std::cout << "CheckDistance: " << sum << " - " << (std::numeric_limits<track_t>::epsilon() * maxOne) << ", " << std::fabs(sum - 1.0f) << std::endl;
		return std::fabs(sum - 1.0f) <= std::numeric_limits<track_t>::epsilon() * maxOne;
	}

	///
	bool SetDistances(std::array<track_t, tracking::DistsCount> distType)
	{
		bool res = true;
		auto oldDists = m_distType;
		m_distType = distType;
		if (!CheckDistance())
		{
			m_distType = oldDists;
			res = false;
		}
		return res;
	}

	///
	bool SetDistance(tracking::DistType distType)
	{
		std::fill(m_distType.begin(), m_distType.end(), 0.0f);
		m_distType[distType] = 1.f;
		return true;
	}

	///
	void AddNearTypes(ObjectTypes type1, ObjectTypes type2, bool sym)
	{
		auto AddOne = [&](objtype_t type1, objtype_t type2)
		{
			auto it = m_nearTypes.find(type1);
			if (it == std::end(m_nearTypes))
				m_nearTypes[type1] = std::set<objtype_t>{ type2 };
			else
				it->second.insert(type2);
		};
		AddOne((objtype_t)type1, (objtype_t)type2);
		if (sym)
			AddOne((objtype_t)type2, (objtype_t)type1);
	}

	///
	bool CheckType(objtype_t type1, objtype_t type2) const
	{
		bool res = (type1 == bad_type) || (type2 == bad_type) || (type1 == type2);
		if (!res)
		{
			auto it = m_nearTypes.find(type1);
			if (it != std::end(m_nearTypes))
				res = it->second.find(type2) != std::end(it->second);
		}
		return res;
	}
};

///
bool ParseTrackerSettings(const std::string& settingsFile, TrackerSettings& trackerSettings);
