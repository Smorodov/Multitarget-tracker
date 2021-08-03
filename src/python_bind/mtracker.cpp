#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(std::map<std::string, std::string>)

#include <algorithm>

#include "../common/defines.h"
#include "../Tracker/Ctracker.h"
#include "../Detector/BaseDetector.h"
#include "../Detector/MotionDetector.h"
#include "ndarray_converter.h"

namespace py = pybind11;

///
class PyDetector : public BaseDetector
{
public:
	using BaseDetector::BaseDetector;

	bool Init(const config_t& config) override {
		PYBIND11_OVERLOAD_PURE(
			bool,          // Return type
			BaseDetector,  // Parent class
			Init,          // Name of function in C++ (must match Python name)
			config         // Argument(s)
		);
	}

    void Detect(cv::Mat frame) override {
		PYBIND11_OVERLOAD_PURE(
			void,          // Return type
			BaseDetector,  // Parent class
			Detect,        // Name of function in C++ (must match Python name)
			frame          // Argument(s)
		);
	}

	bool CanGrayProcessing() const override {
		PYBIND11_OVERLOAD_PURE(
			bool,          // Return type
			BaseDetector,  // Parent class
			CanGrayProcessing
		);
	}
};

///
class PyMotionDetector : public MotionDetector
{
public:
    using MotionDetector::MotionDetector;

    bool Init(const config_t& config) override {
        PYBIND11_OVERLOAD_PURE(
            bool,          // Return type
            BaseDetector,  // Parent class
            Init,          // Name of function in C++ (must match Python name)
            config         // Argument(s)
        );
    }

    void Detect(cv::Mat frame) override {
        PYBIND11_OVERLOAD_PURE(
            void,          // Return type
            BaseDetector,  // Parent class
            Detect,        // Name of function in C++ (must match Python name)
            frame          // Argument(s)
        );
    }

    bool CanGrayProcessing() const override {
        PYBIND11_OVERLOAD_PURE(
            bool,          // Return type
            BaseDetector,  // Parent class
            CanGrayProcessing
        );
    }
};

///
cv::Mat read_image(std::string image_name)
{
    return cv::imread(image_name, cv::IMREAD_COLOR);
}

///
void show_image(cv::Mat image)
{
    cv::imshow("image_from_Cpp", image);
    cv::waitKey(0);
}

///
cv::Mat passthru(cv::Mat image)
{
    return image;
}

///
cv::Mat cloneimg(cv::Mat image)
{
    return image.clone();
}

///
PYBIND11_MODULE(pymtracking, m)
{
    NDArrayConverter::init_numpy();

	m.doc() = R"pbdoc(
        mtracking library
        -----------------------
        .. currentmodule:: pymtracking
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    py::bind_map<std::map<std::string, std::string>>(m, "MapStringString");
    py::bind_map<std::map<std::string, double>>(m, "MapStringDouble");

    py::class_<TrackerSettings>(m, "TrackerSettings")
            .def(py::init<>())
            .def("CheckDistance", &TrackerSettings::CheckDistance)
            .def("SetDistances", &TrackerSettings::SetDistances)
            .def("SetDistance", &TrackerSettings::SetDistance)
            .def("AddNearTypes", &TrackerSettings::AddNearTypes)
            .def("CheckType", &TrackerSettings::CheckType)
            .def_readwrite("kalmanType", &TrackerSettings::m_kalmanType)
            .def_readwrite("filterGoal", &TrackerSettings::m_filterGoal)
            .def_readwrite("lostTrackType", &TrackerSettings::m_lostTrackType)
            .def_readwrite("matchType", &TrackerSettings::m_matchType)
            .def_readwrite("dt", &TrackerSettings::m_dt)
            .def_readwrite("accelNoiseMag", &TrackerSettings::m_accelNoiseMag)
            .def_readwrite("useAcceleration", &TrackerSettings::m_useAcceleration)
            .def_readwrite("distThres", &TrackerSettings::m_distThres)
            .def_readwrite("minAreaRadiusPix", &TrackerSettings::m_minAreaRadiusPix)
            .def_readwrite("minAreaRadiusK", &TrackerSettings::m_minAreaRadiusK)
            .def_readwrite("maximumAllowedSkippedFrames", &TrackerSettings::m_maximumAllowedSkippedFrames)
            .def_readwrite("maxTraceLength", &TrackerSettings::m_maxTraceLength)
            .def_readwrite("useAbandonedDetection", &TrackerSettings::m_useAbandonedDetection)
            .def_readwrite("minStaticTime", &TrackerSettings::m_minStaticTime)
            .def_readwrite("maxStaticTime", &TrackerSettings::m_maxStaticTime);

    py::class_<CRegion>(m, "CRegion")
            .def(py::init<>())
            .def("x", &CRegion::x)
            .def("y", &CRegion::y)
            .def("width", &CRegion::width)
            .def("height", &CRegion::height)
            .def_readwrite("type", &CRegion::m_type)
            .def_readwrite("confidence", &CRegion::m_confidence);

    py::class_<TrackingObject>(m, "TrackingObject")
            .def(py::init<>())
            .def("IsRobust", &TrackingObject::IsRobust)
            .def("x", &TrackingObject::x)
            .def("y", &TrackingObject::y)
            .def("width", &TrackingObject::width)
            .def("height", &TrackingObject::height)
            .def_readwrite("ID", &TrackingObject::m_ID)
            .def_readwrite("isStatic", &TrackingObject::m_isStatic)
            .def_readwrite("outOfTheFrame", &TrackingObject::m_outOfTheFrame)
            .def_readwrite("type", &TrackingObject::m_type)
            .def_readwrite("confidence", &TrackingObject::m_confidence)
            .def_readwrite("velocity", &TrackingObject::m_velocity);

    py::class_<CTracker> mtracker(m, "MTracker");
    mtracker.def(py::init<const TrackerSettings&>());
    mtracker.def("Update", &CTracker::Update);
    mtracker.def("CanGrayFrameToTrack", &CTracker::CanGrayFrameToTrack);
    mtracker.def("CanColorFrameToTrack", &CTracker::CanColorFrameToTrack);
    mtracker.def("GetTracksCount", &CTracker::GetTracksCount);
    mtracker.def("GetTracks", &CTracker::GetTracks);

    py::enum_<tracking::DistType>(mtracker, "DistType")
            .value("DistCenters", tracking::DistType::DistCenters)
            .value("DistRects", tracking::DistType::DistRects)
            .value("DistJaccard", tracking::DistType::DistJaccard)
            .export_values();

    py::enum_<tracking::FilterGoal>(mtracker, "FilterGoal")
            .value("FilterCenter", tracking::FilterGoal::FilterCenter)
            .value("FilterRect", tracking::FilterGoal::FilterRect)
            .export_values();

    py::enum_<tracking::KalmanType>(mtracker, "KalmanType")
            .value("KalmanLinear", tracking::KalmanType::KalmanLinear)
            .value("KalmanUnscented", tracking::KalmanType::KalmanUnscented)
            .value("KalmanAugmentedUnscented", tracking::KalmanType::KalmanAugmentedUnscented)
            .export_values();

    py::enum_<tracking::MatchType>(mtracker, "MatchType")
            .value("MatchHungrian", tracking::MatchType::MatchHungrian)
            .value("MatchBipart", tracking::MatchType::MatchBipart)
            .export_values();

    py::enum_<tracking::LostTrackType>(mtracker, "LostTrackType")
            .value("TrackNone", tracking::LostTrackType::TrackNone)
            .value("TrackKCF", tracking::LostTrackType::TrackKCF)
            .value("TrackMIL", tracking::LostTrackType::TrackMIL)
            .value("TrackMedianFlow", tracking::LostTrackType::TrackMedianFlow)
            .value("TrackGOTURN", tracking::LostTrackType::TrackGOTURN)
            .value("TrackMOSSE", tracking::LostTrackType::TrackMOSSE)
            .value("TrackCSRT", tracking::LostTrackType::TrackCSRT)
            .value("TrackDAT", tracking::LostTrackType::TrackDAT)
            .value("TrackSTAPLE", tracking::LostTrackType::TrackSTAPLE)
            .value("TrackLDES", tracking::LostTrackType::TrackLDES)
            .export_values();

	py::class_<BaseDetector, PyDetector>(m, "BaseDetector")
		.def(py::init<cv::Mat>())
		.def("Init", &BaseDetector::Init)
		.def("Detect", &BaseDetector::Detect)
		.def("ResetModel", &BaseDetector::ResetModel)
		.def("CanGrayProcessing", &BaseDetector::CanGrayProcessing)
		.def("SetMinObjectSize", &BaseDetector::SetMinObjectSize)
		.def("GetDetects", &BaseDetector::GetDetects)
		.def("CalcMotionMap", &BaseDetector::CalcMotionMap);

    py::class_<MotionDetector, PyMotionDetector> mdetector(m, "MotionDetector");
    mdetector.def(py::init<BackgroundSubtract::BGFG_ALGS, cv::Mat&>());
    mdetector.def("Init", &MotionDetector::Init);
    mdetector.def("Detect", &MotionDetector::Detect);
    mdetector.def("ResetModel", &MotionDetector::ResetModel);
    mdetector.def("CanGrayProcessing", &MotionDetector::CanGrayProcessing);
    mdetector.def("SetMinObjectSize", &MotionDetector::SetMinObjectSize);
    mdetector.def("GetDetects", &MotionDetector::GetDetects);
    mdetector.def("CalcMotionMap", &MotionDetector::CalcMotionMap);

    py::enum_<BackgroundSubtract::BGFG_ALGS>(mdetector, "BGFG_ALGS")
            .value("VIBE", BackgroundSubtract::BGFG_ALGS::ALG_VIBE)
            .value("MOG", BackgroundSubtract::BGFG_ALGS::ALG_MOG)
            .value("GMG", BackgroundSubtract::BGFG_ALGS::ALG_GMG)
            .value("CNT", BackgroundSubtract::BGFG_ALGS::ALG_CNT)
            .value("SuBSENSE", BackgroundSubtract::BGFG_ALGS::ALG_SuBSENSE)
            .value("LOBSTER", BackgroundSubtract::BGFG_ALGS::ALG_LOBSTER)
            .value("MOG2", BackgroundSubtract::BGFG_ALGS::ALG_MOG2)
            .export_values();

    m.def("read_image", &read_image, "A function that read an image",
          py::arg("image"));

    m.def("show_image", &show_image, "A function that show an image",
          py::arg("image"));

    m.def("passthru", &passthru, "Passthru function", py::arg("image"));
    m.def("clone", &cloneimg, "Clone function", py::arg("image"));

#define VERSION_INFO "1.0.1"
#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}
