#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/stl_bind.h>

#include <algorithm>

#include "../common/defines.h"
#include "../Tracker/Ctracker.h"
#include "../Detector/BaseDetector.h"
#include "../Detector/MotionDetector.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::multimap<std::string, std::string>)
PYBIND11_MAKE_OPAQUE(std::map<std::string, std::string>)
PYBIND11_MAKE_OPAQUE(std::map<std::string, double>)

#include <Python.h>
#include <pybind11/numpy.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#if PY_VERSION_HEX >= 0x03000000
    #define PyInt_Check PyLong_Check
    #define PyInt_AsLong PyLong_AsLong
#endif

namespace pybind11 { namespace detail{
template<>
struct type_caster<cv::Mat>{
public:
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

    //! 1. cast numpy.ndarray to cv::Mat
    bool load(handle obj, bool)
    {
        array b = reinterpret_borrow<array>(obj);
        buffer_info info = b.request();

        //const int ndims = (int)info.ndim;
        int nh = 1;
        int nw = 1;
        int nc = 1;
        int ndims = static_cast<int>(info.ndim);
        if(ndims == 2){
            nh = static_cast<int>(info.shape[0]);
            nw = static_cast<int>(info.shape[1]);
        } else if(ndims == 3){
            nh = static_cast<int>(info.shape[0]);
            nw = static_cast<int>(info.shape[1]);
            nc = static_cast<int>(info.shape[2]);
        }else{
            char msg[64];
            std::sprintf(msg, "Unsupported dim %d, only support 2d, or 3-d", ndims);
            throw std::logic_error(msg);
            return false;
        }

        int dtype;
        if(info.format == format_descriptor<unsigned char>::format()){
            dtype = CV_8UC(nc);
        }else if (info.format == format_descriptor<int>::format()){
            dtype = CV_32SC(nc);
        }else if (info.format == format_descriptor<float>::format()){
            dtype = CV_32FC(nc);
        }else{
            throw std::logic_error("Unsupported type, only support uchar, int32, float");
            return false;
        }

        value = cv::Mat(nh, nw, dtype, info.ptr);
        return true;
    }

    //! 2. cast cv::Mat to numpy.ndarray
    static handle cast(const cv::Mat& mat, return_value_policy, handle /*defval*/)
    {
        std::string format = format_descriptor<unsigned char>::format();
        size_t elemsize = sizeof(unsigned char);
        int nw = mat.cols;
        int nh = mat.rows;
        int nc = mat.channels();
        int depth = mat.depth();
        int type = mat.type();
        int dim = (depth == type)? 2 : 3;

        if(depth == CV_8U){
            format = format_descriptor<unsigned char>::format();
            elemsize = sizeof(unsigned char);
        }else if(depth == CV_32S){
            format = format_descriptor<int>::format();
            elemsize = sizeof(int);
        }else if(depth == CV_32F){
            format = format_descriptor<float>::format();
            elemsize = sizeof(float);
        }else{
            throw std::logic_error("Unsupport type, only support uchar, int32, float");
        }

        std::vector<size_t> bufferdim;
        std::vector<size_t> strides;
        if (dim == 2) {
            bufferdim = {(size_t) nh, (size_t) nw};
            strides = {elemsize * (size_t) nw, elemsize};
        } else if (dim == 3) {
            bufferdim = {(size_t) nh, (size_t) nw, (size_t) nc};
            strides = {(size_t) elemsize * nw * nc, (size_t) elemsize * nc, (size_t) elemsize};
        }
        return array(buffer_info( mat.data,  elemsize,  format, dim, bufferdim, strides )).release();
    }
};

template<typename T>
struct type_caster<cv::Size_<T>>{

    PYBIND11_TYPE_CASTER(cv::Size_<T>, _("tuple_wh"));

    bool load(handle obj, bool)
    {
        if(!py::isinstance<py::tuple>(obj))
        {
            std::logic_error("Size(w, h) should be a tuple!");
            return false;
        }

        py::tuple pt = reinterpret_borrow<py::tuple>(obj);
        if (pt.size() != 2)
        {
            std::logic_error("Size(w, h) tuple should be size of 2");
            return false;
        }

        value = cv::Size(static_cast<int>(pt[0].cast<T>()), static_cast<int>(pt[1].cast<T>()));
        return true;
    }

    static handle cast(const cv::Size_<T>& sz, return_value_policy, handle)
    {
        return py::make_tuple(sz.width, sz.height).release();
    }
};

template<>
struct type_caster<cv::Point>{

    PYBIND11_TYPE_CASTER(cv::Point, _("tuple_xy"));

    bool load(handle obj, bool)
    {
        if (!py::isinstance<py::tuple>(obj))
        {
            std::logic_error("Point(x,y) should be a tuple!");
            return false;
        }

        py::tuple pt = reinterpret_borrow<py::tuple>(obj);
        if(pt.size()!=2)
        {
            std::logic_error("Point(x,y) tuple should be size of 2");
            return false;
        }

        value = cv::Point(pt[0].cast<int>(), pt[1].cast<int>());
        return true;
    }

    static handle cast(const cv::Point& pt, return_value_policy, handle)
    {
        return py::make_tuple(pt.x, pt.y).release();
    }
};


}}//! end namespace pybind11::detail


///
class PyDetector : public BaseDetector
{
public:
	using BaseDetector::BaseDetector;

    PyDetector()
    {
        std::cout << "PyDetector" << std::endl;
    }

	bool Init(const config_t& config) override
    {
        PYBIND11_OVERLOAD_PURE(bool, BaseDetector, Init, config);
    }

    void DetectMat(cv::Mat frame) override
    {
        PYBIND11_OVERLOAD(void, BaseDetector, DetectMat, frame);
    }

    bool CanGrayProcessing() const override
    {
        PYBIND11_OVERLOAD_PURE(bool, BaseDetector, CanGrayProcessing, );
    }
};

///
class PyMotionDetector : public MotionDetector
{
public:
    using MotionDetector::MotionDetector;

    bool Init(const config_t& config) override
    {
        PYBIND11_OVERLOAD(bool, MotionDetector, Init, config);
    }

    bool CanGrayProcessing() const override
    {
        PYBIND11_OVERLOAD(bool, MotionDetector, CanGrayProcessing, );
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

class PyBaseTracker : public BaseTracker
{
public:
    using BaseTracker::BaseTracker;

    void UpdateMat(const regions_t& regions, cv::Mat currFrame, float fps) override
    {
        PYBIND11_OVERLOAD_PURE(void, BaseTracker, Update, regions, currFrame, fps);
    }

    bool CanGrayFrameToTrack() const override
    {
        PYBIND11_OVERLOAD_PURE(bool, BaseTracker, CanGrayFrameToTrack, );
    }

    bool CanColorFrameToTrack() const override
    {
        PYBIND11_OVERLOAD_PURE(bool, BaseTracker, CanColorFrameToTrack, );
    }

    size_t GetTracksCount() const override
    {
        PYBIND11_OVERLOAD_PURE(size_t, BaseTracker, GetTracksCount, );
    }

    std::vector<TrackingObject> GetTracksCopy() const override
    {
        PYBIND11_OVERLOAD(std::vector<TrackingObject>, BaseTracker, GetTracksCopy);
    }

    void GetRemovedTracks(std::vector<track_id_t>& trackIDs) const override
    {
        PYBIND11_OVERLOAD_PURE(void, BaseTracker, GetRemovedTracks, trackIDs);
    }
};

///
PYBIND11_MODULE(pymtracking, m)
{
	m.doc() = R"pbdoc(
        mtracking library
        -----------------------
        .. currentmodule:: pymtracking
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    py::class_<KeyVal>(m, "KeyVal")
            .def(py::init<>())
            .def("Add", &KeyVal::Add);

    py::bind_map<std::multimap<std::string, std::string>>(m, "MapStringString");
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
            .def_readwrite("brect", &CRegion::m_brect)
            .def_readwrite("type", &CRegion::m_type)
            .def_readwrite("confidence", &CRegion::m_confidence);

    py::class_<TrackingObject>(m, "TrackingObject")
            .def(py::init<>())
            .def("IsRobust", &TrackingObject::IsRobust)
            .def("GetTrajectory", &TrackingObject::GetTrajectory)
            .def("GetBoundingRect", &TrackingObject::GetBoundingRect)
            .def_readwrite("rrect", &TrackingObject::m_rrect)
            .def_readwrite("ID", &TrackingObject::m_ID)
            .def_readwrite("isStatic", &TrackingObject::m_isStatic)
            .def_readwrite("outOfTheFrame", &TrackingObject::m_outOfTheFrame)
            .def_readwrite("type", &TrackingObject::m_type)
            .def_readwrite("confidence", &TrackingObject::m_confidence)
            .def_readwrite("velocity", &TrackingObject::m_velocity);

    py::class_<BaseTracker, PyBaseTracker> mtracker(m, "MTracker");
    mtracker.def(py::init(&BaseTracker::CreateTracker));
    mtracker.def("Update", &BaseTracker::UpdateMat);
    mtracker.def("CanGrayFrameToTrack", &BaseTracker::CanGrayFrameToTrack);
    mtracker.def("CanColorFrameToTrack", &BaseTracker::CanColorFrameToTrack);
    mtracker.def("GetTracksCount", &BaseTracker::GetTracksCount);
    mtracker.def("GetTracks", &BaseTracker::GetTracksCopy);

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

	py::class_<BaseDetector, PyDetector> base_detector(m, "BaseDetector");
    base_detector.def(py::init(&BaseDetector::CreateDetectorKV));
    base_detector.def("Init", &BaseDetector::Init);
    base_detector.def("Detect", &BaseDetector::DetectMat);
    base_detector.def("ResetModel", &BaseDetector::ResetModel);
    base_detector.def("CanGrayProcessing", &BaseDetector::CanGrayProcessing);
    base_detector.def("SetMinObjectSize", &BaseDetector::SetMinObjectSize);
    base_detector.def("GetDetects", &BaseDetector::GetDetects);
    base_detector.def("CalcMotionMap", &BaseDetector::CalcMotionMap);

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

    py::enum_<tracking::Detectors>(base_detector, "Detectors")
        .value("VIBE", tracking::Detectors::Motion_VIBE)
        .value("MOG", tracking::Detectors::Motion_MOG)
        .value("GMG", tracking::Detectors::Motion_GMG)
        .value("CNT", tracking::Detectors::Motion_CNT)
        .value("SuBSENSE", tracking::Detectors::Motion_SuBSENSE)
        .value("LOBSTER", tracking::Detectors::Motion_LOBSTER)
        .value("MOG2", tracking::Detectors::Motion_MOG2)
        .value("Face_HAAR", tracking::Detectors::Face_HAAR)
        .value("Pedestrian_HOG", tracking::Detectors::Pedestrian_HOG)
        .value("Pedestrian_C4", tracking::Detectors::Pedestrian_C4)
        .value("Yolo_Darknet", tracking::Detectors::Yolo_Darknet)
        .value("Yolo_TensorRT", tracking::Detectors::Yolo_TensorRT)
        .value("DNN_OCV", tracking::Detectors::DNN_OCV)
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
