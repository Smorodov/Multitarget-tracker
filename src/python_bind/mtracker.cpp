#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/stl_bind.h>

// pybind11 2.5 expects a global ssize_t which MSVC does not define.
#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include <pybind11/numpy.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../mtracking/defines.h"
#include "../Tracker/BaseTracker.h"
#include "../Detector/BaseDetector.h"
#include "../Detector/MotionDetector.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::multimap<std::string, std::string>)
PYBIND11_MAKE_OPAQUE(std::map<std::string, std::string>)
PYBIND11_MAKE_OPAQUE(std::map<std::string, double>)

namespace pybind11 { namespace detail
{
template <>
struct type_caster<cv::Mat>
{
public:
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

    // numpy.ndarray -> cv::Mat (C-contiguous arrays only)
    bool load(handle obj, bool)
    {
        array b = reinterpret_borrow<array>(obj);
        buffer_info info = b.request();

        if ((info.ndim != 2 && info.ndim != 3) ||
            info.shape[0] < 0 || info.shape[1] < 0 ||
            (info.ndim == 3 && info.shape[2] < 0))
        {
            throw std::logic_error("Unsupported ndarray shape: expected 2D or 3D array");
        }
        if (!(b.flags() & array::c_style))
        {
            throw std::logic_error("Only C-contiguous numpy arrays are supported");
        }

        const int nh = static_cast<int>(info.shape[0]);
        const int nw = static_cast<int>(info.shape[1]);
        const int nc = (info.ndim == 3) ? static_cast<int>(info.shape[2]) : 1;
        if (nc <= 0 || nc > 4)
        {
            throw std::logic_error("Unsupported number of channels");
        }

        int dtype = -1;
        if (info.format == format_descriptor<unsigned char>::format())
            dtype = CV_8UC(nc);
        else if (info.format == format_descriptor<int>::format())
            dtype = CV_32SC(nc);
        else if (info.format == format_descriptor<float>::format())
            dtype = CV_32FC(nc);
        else
            throw std::logic_error("Unsupported ndarray type: only uint8, int32 and float32 are supported");

        value = cv::Mat(nh, nw, dtype, info.ptr);
        return true;
    }

    // cv::Mat -> numpy.ndarray (copy, so the array owns its data)
    static handle cast(const cv::Mat& mat, return_value_policy, handle)
    {
        if (mat.dims != 2)
        {
            throw std::logic_error("Only 2D cv::Mat can be converted to numpy.ndarray");
        }

        std::string format;
        switch (mat.depth())
        {
        case CV_8U:
            format = format_descriptor<unsigned char>::format();
            break;
        case CV_32S:
            format = format_descriptor<int>::format();
            break;
        case CV_32F:
            format = format_descriptor<float>::format();
            break;
        default:
            throw std::logic_error("Unsupported cv::Mat depth: only uint8, int32 and float32 are supported");
        }

        const int nc = mat.channels();
        std::vector<py::ssize_t> shape;
        if (nc == 1)
            shape = { mat.rows, mat.cols };
        else
            shape = { mat.rows, mat.cols, nc };

        py::array result(py::dtype(format), shape);
        if (mat.rows > 0 && mat.cols > 0)
        {
            char* dst = static_cast<char*>(result.mutable_data());
            const size_t rowBytes = static_cast<size_t>(mat.cols) * mat.elemSize();
            for (int y = 0; y < mat.rows; ++y)
            {
                std::memcpy(dst + static_cast<size_t>(y) * rowBytes, mat.ptr(y), rowBytes);
            }
        }
        return result.release();
    }
};

template <typename T>
struct type_caster<cv::Size_<T>>
{
    PYBIND11_TYPE_CASTER(cv::Size_<T>, _("tuple_wh"));

    bool load(handle obj, bool)
    {
        if (!py::isinstance<py::tuple>(obj))
        {
            throw std::logic_error("Size(w, h) should be a tuple");
        }

        py::tuple pt = reinterpret_borrow<py::tuple>(obj);
        if (pt.size() != 2)
        {
            throw std::logic_error("Size(w, h) tuple should be size of 2");
        }

        value = cv::Size_<T>(static_cast<T>(pt[0].cast<T>()), static_cast<T>(pt[1].cast<T>()));
        return true;
    }

    static handle cast(const cv::Size_<T>& sz, return_value_policy, handle)
    {
        return py::make_tuple(sz.width, sz.height).release();
    }
};

template <>
struct type_caster<cv::Point>
{
    PYBIND11_TYPE_CASTER(cv::Point, _("tuple_xy"));

    bool load(handle obj, bool)
    {
        if (!py::isinstance<py::tuple>(obj))
        {
            throw std::logic_error("Point(x, y) should be a tuple");
        }

        py::tuple pt = reinterpret_borrow<py::tuple>(obj);
        if (pt.size() != 2)
        {
            throw std::logic_error("Point(x, y) tuple should be size of 2");
        }

        value = cv::Point(pt[0].cast<int>(), pt[1].cast<int>());
        return true;
    }

    static handle cast(const cv::Point& pt, return_value_policy, handle)
    {
        return py::make_tuple(pt.x, pt.y).release();
    }
};
}} // namespace pybind11::detail

namespace
{
config_t ToConfig(const KeyVal& kv)
{
    config_t config;
    for (const auto& item : kv.m_config)
        config.emplace(item.first, item.second);
    return config;
}

time_point_t ToTimePoint(double seconds)
{
    return time_point_t(std::chrono::duration_cast<time_point_t::duration>(
        std::chrono::duration<double>(seconds)));
}
} // namespace

///
cv::Mat read_image(std::string image_name)
{
    return cv::imread(image_name, cv::IMREAD_COLOR);
}

///
void show_image(cv::Mat image)
{
#ifndef SILENT_WORK
    cv::imshow("image_from_Cpp", image);
    cv::waitKey(0);
#endif
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
    m.doc() = "mtracking library";

    py::class_<KeyVal>(m, "KeyVal")
            .def(py::init<>())
            .def("Add", &KeyVal::Add);

    py::bind_map<std::multimap<std::string, std::string>>(m, "MapStringString");
    py::bind_map<std::map<std::string, double>>(m, "MapStringDouble");

    py::class_<TrackerSettings>(m, "TrackerSettings")
            .def(py::init<>())
            .def("CheckDistance", &TrackerSettings::CheckDistance)
            .def("SetDistances", [](TrackerSettings& settings, const std::vector<track_t>& distances)
            {
                if (distances.size() != static_cast<size_t>(tracking::DistsCount))
                    throw std::runtime_error("SetDistances expects " + std::to_string(tracking::DistsCount) + " values");

                std::array<track_t, tracking::DistsCount> values{};
                std::copy(distances.begin(), distances.end(), values.begin());
                return settings.SetDistances(values);
            })
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
            .def_readwrite("distThresLocalCartesian", &TrackerSettings::m_distThresLocalCartesian)
            .def_readwrite("minAreaRadiusPix", &TrackerSettings::m_minAreaRadiusPix)
            .def_readwrite("minAreaRadiusK", &TrackerSettings::m_minAreaRadiusK)
            .def_readwrite("maximumAllowedLostTime", &TrackerSettings::m_maximumAllowedLostTime)
            .def_readwrite("maxTraceLength", &TrackerSettings::m_maxTraceLength)
            .def_readwrite("useAbandonedDetection", &TrackerSettings::m_useAbandonedDetection)
            .def_readwrite("minStaticTime", &TrackerSettings::m_minStaticTime)
            .def_readwrite("maxStaticTime", &TrackerSettings::m_maxStaticTime);

    py::enum_<tracking::TrackerTemplate>(m, "TrackerTemplate")
            .value("UniversalTracker", tracking::TrackerTemplate::UniversalTracker)
            .value("ByteTrack", tracking::TrackerTemplate::ByteTrack)
            .export_values();

    py::class_<cv::Rect>(m, "CvRect")
            .def(py::init<>())
            .def_readwrite("x", &cv::Rect::x)
            .def_readwrite("y", &cv::Rect::y)
            .def_readwrite("width", &cv::Rect::width)
            .def_readwrite("height", &cv::Rect::height);

    py::class_<cv::RotatedRect>(m, "CvRRect")
            .def(py::init<>())
            .def("brect", &cv::RotatedRect::boundingRect);

    py::class_<CRegion>(m, "CRegion")
            .def(py::init<>())
            .def(py::init<const cv::Rect&>())
            .def_readwrite("brect", &CRegion::m_brect)
            .def_readwrite("type", &CRegion::m_type)
            .def_readwrite("confidence", &CRegion::m_confidence);

    py::class_<TrackingObject>(m, "TrackingObject")
            .def(py::init<>())
            .def("IsRobust", &TrackingObject::IsRobust,
                 py::arg("minTraceSize"), py::arg("minRawRatio"), py::arg("sizeRatio"),
                 py::arg("lastDetectsCount") = 0)
            .def("GetTrajectory", &TrackingObject::GetTrajectory)
            .def("GetBoundingRect", &TrackingObject::GetBoundingRect)
            .def_property_readonly("ID", [](const TrackingObject& object) { return object.m_ID.m_val; })
            .def_readwrite("rrect", &TrackingObject::m_rrect)
            .def_readwrite("isStatic", &TrackingObject::m_isStatic)
            .def_readwrite("staticTime", &TrackingObject::m_staticTime)
            .def_readwrite("outOfTheFrame", &TrackingObject::m_outOfTheFrame)
            .def_readwrite("type", &TrackingObject::m_type)
            .def_readwrite("confidence", &TrackingObject::m_confidence)
            .def_property_readonly("velocity", [](const TrackingObject& object)
            {
                return py::make_tuple(object.m_velocity[0], object.m_velocity[1]);
            })
            .def_readwrite("speedMps", &TrackingObject::m_speedMps);

    py::class_<BaseTracker> mtracker(m, "MTracker");
    mtracker.def(py::init(&BaseTracker::CreateTracker),
                 py::arg("settings"), py::arg("fps"));
    mtracker.def("Update", [](BaseTracker& tracker, const regions_t& regions, cv::Mat frame, double timestamp)
    {
        tracker.UpdateMat(regions, frame, ToTimePoint(timestamp));
    }, py::arg("regions"), py::arg("frame"), py::arg("timestamp"));
    mtracker.def("Update", [](BaseTracker& tracker, const regions_t& regions, cv::Mat frame)
    {
        tracker.UpdateMat(regions, frame, std::chrono::system_clock::now());
    }, py::arg("regions"), py::arg("frame"));
    mtracker.def("GetTracks", &BaseTracker::GetTracksCopy);

    py::enum_<tracking::DistType>(mtracker, "DistType")
            .value("DistCenters", tracking::DistType::DistCenters)
            .value("DistRects", tracking::DistType::DistRects)
            .value("DistJaccard", tracking::DistType::DistJaccard)
            .value("DistHist", tracking::DistType::DistHist)
            .value("DistFeatureCos", tracking::DistType::DistFeatureCos)
            .value("DistMahalanobis", tracking::DistType::DistMahalanobis)
            .value("DistLocalCartesian", tracking::DistType::DistLocalCartesian)
            .export_values();

    py::enum_<tracking::FilterGoal>(mtracker, "FilterGoal")
            .value("FilterCenter", tracking::FilterGoal::FilterCenter)
            .value("FilterRect", tracking::FilterGoal::FilterRect)
            .value("FilterRRect", tracking::FilterGoal::FilterRRect)
            .export_values();

    py::enum_<tracking::KalmanType>(mtracker, "KalmanType")
            .value("KalmanLinear", tracking::KalmanType::KalmanLinear)
            .value("KalmanUnscented", tracking::KalmanType::KalmanUnscented)
            .value("KalmanAugmentedUnscented", tracking::KalmanType::KalmanAugmentedUnscented)
            .export_values();

    py::enum_<tracking::MatchType>(mtracker, "MatchType")
            .value("MatchHungrian", tracking::MatchType::MatchHungrian)
            .value("MatchLAPJV", tracking::MatchType::MatchLAPJV)
            .export_values();

    py::enum_<tracking::LostTrackType>(mtracker, "LostTrackType")
            .value("TrackNone", tracking::LostTrackType::TrackNone)
            .value("TrackKCF", tracking::LostTrackType::TrackKCF)
            .value("TrackCSRT", tracking::LostTrackType::TrackCSRT)
            .value("TrackDaSiamRPN", tracking::LostTrackType::TrackDaSiamRPN)
            .value("TrackNano", tracking::LostTrackType::TrackNano)
            .value("TrackVit", tracking::LostTrackType::TrackVit)
            .export_values();

    py::class_<BaseDetector> baseDetector(m, "BaseDetector");
    baseDetector.def(py::init(&BaseDetector::CreateDetectorKV),
                     py::arg("detectorType"), py::arg("config"), py::arg("frame"));
    baseDetector.def("Init", [](BaseDetector& detector, const KeyVal& config)
    {
        return detector.Init(ToConfig(config));
    }, py::arg("config"));
    baseDetector.def("Detect", [](BaseDetector& detector, cv::Mat frame)
    {
        detector.DetectMat(frame);
    }, py::arg("frame"));
    baseDetector.def("ResetModel", [](BaseDetector& detector, cv::Mat frame, cv::Rect roi)
    {
        detector.ResetModel(frame.getUMat(cv::ACCESS_READ), roi);
    }, py::arg("frame"), py::arg("roi"));
    baseDetector.def("CanGrayProcessing", &BaseDetector::CanGrayProcessing);
    baseDetector.def("SetMinObjectSize", &BaseDetector::SetMinObjectSize);
    baseDetector.def("GetDetects", &BaseDetector::GetDetects);
    baseDetector.def("CalcMotionMap", [](BaseDetector& detector, cv::Mat frame, bool drawOnlyMasks)
    {
        detector.CalcMotionMap(frame, drawOnlyMasks);
    }, py::arg("frame"), py::arg("drawOnlyMasks") = true);

    py::class_<MotionDetector, BaseDetector> motionDetector(m, "MotionDetector");
    motionDetector.def(py::init<BackgroundSubtract::BGFG_ALGS, const cv::Mat&>(),
                       py::arg("algType"), py::arg("frame"));
    motionDetector.def("Init", [](MotionDetector& detector, const KeyVal& config)
    {
        return detector.Init(ToConfig(config));
    }, py::arg("config"));
    motionDetector.def("Detect", [](MotionDetector& detector, cv::Mat frame)
    {
        detector.DetectMat(frame);
    }, py::arg("frame"));
    motionDetector.def("SetMinObjectSize", &MotionDetector::SetMinObjectSize);
    motionDetector.def("GetDetects", &MotionDetector::GetDetects);
    motionDetector.def("CalcMotionMap", [](MotionDetector& detector, cv::Mat frame, bool drawOnlyMasks)
    {
        detector.CalcMotionMap(frame, drawOnlyMasks);
    }, py::arg("frame"), py::arg("drawOnlyMasks") = true);

    py::enum_<BackgroundSubtract::BGFG_ALGS>(motionDetector, "BGFG_ALGS")
            .value("VIBE", BackgroundSubtract::BGFG_ALGS::ALG_VIBE)
            .value("MOG", BackgroundSubtract::BGFG_ALGS::ALG_MOG)
            .value("GMG", BackgroundSubtract::BGFG_ALGS::ALG_GMG)
            .value("CNT", BackgroundSubtract::BGFG_ALGS::ALG_CNT)
            .value("MOG2", BackgroundSubtract::BGFG_ALGS::ALG_MOG2)
            .export_values();

    py::enum_<tracking::Detectors>(baseDetector, "Detectors")
            .value("VIBE", tracking::Detectors::Motion_VIBE)
            .value("MOG", tracking::Detectors::Motion_MOG)
            .value("GMG", tracking::Detectors::Motion_GMG)
            .value("CNT", tracking::Detectors::Motion_CNT)
            .value("MOG2", tracking::Detectors::Motion_MOG2)
            .value("Yolo_TensorRT", tracking::Detectors::ONNX_TensorRT)
            .value("DNN_OCV", tracking::Detectors::DNN_OCV)
            .export_values();

    m.def("read_image", &read_image, "Read an image from a file", py::arg("image"));
    m.def("show_image", &show_image, "Show an image", py::arg("image"));
    m.def("passthru", &passthru, "Return the same image", py::arg("image"));
    m.def("clone", &cloneimg, "Clone an image", py::arg("image"));

    m.attr("__version__") = "1.1.0";
}
