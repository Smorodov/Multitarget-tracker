#include "CarsCounting.h"
#include <inih/INIReader.h>

///
/// \brief CarsCounting::CarsCounting
/// \param parser
///
CarsCounting::CarsCounting(const cv::CommandLineParser& parser)
    : VideoExample(parser)
{
#ifdef _WIN32
    std::string pathToModel = "../../data/";
#else
    std::string pathToModel = "../data/";
#endif

    m_drawHeatMap = parser.get<int>("heat_map") != 0;

    std::string settingsFile = parser.get<std::string>("settings");
    m_trackerSettingsLoaded = ParseTrackerSettings(settingsFile, m_trackerSettings);

    std::cout << "Inference loaded (" << m_trackerSettingsLoaded << ") from " << settingsFile << ": used " << m_trackerSettings.m_detectorBackend << " backend, weights: " << m_trackerSettings.m_nnWeights << ", config: " << m_trackerSettings.m_nnConfig << ", names: " << m_trackerSettings.m_classNames << std::endl;

    m_geoBindFile = parser.get<std::string>("geo_bind");
}

///
/// \brief CarsCounting::DrawTrack
/// \param frame
/// \param track
/// \param drawTrajectory
/// \param framesCounters
///
void CarsCounting::DrawTrack(cv::Mat frame, const TrackingObject& track, bool drawTrajectory, int framesCounter)
{
    cv::Rect brect = track.m_rrect.boundingRect();

    m_resultsLog.AddTrack(framesCounter, track.m_ID, brect, track.m_type, track.m_confidence);
    m_resultsLog.AddRobustTrack(track.m_ID);

    if (track.m_isStatic)
    {
#if (CV_VERSION_MAJOR >= 4)
        cv::rectangle(frame, brect, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
#else
        cv::rectangle(frame, brect, cv::Scalar(255, 0, 255), 2, CV_AA);
#endif
    }
    else
    {
#if (CV_VERSION_MAJOR >= 4)
        cv::rectangle(frame, brect, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
#else
        cv::rectangle(frame, brect, cv::Scalar(0, 255, 0), 1, CV_AA);
#endif

        if (!m_geoParams.Empty())
        {
            int traceSize = static_cast<int>(track.m_trace.size());
            int period = std::min(2 * cvRound(m_fps), traceSize);
            const auto& from = m_geoParams.Pix2Geo(track.m_trace[traceSize - period]);
            const auto& to = m_geoParams.Pix2Geo(track.m_trace[traceSize - 1]);
            auto dist = DistanceInMeters(from, to);

            std::stringstream label;
            if (period >= cvRound(m_fps) / 4)
            {
                auto velocity = (3.6f * dist * m_fps) / period;
                //std::cout << TypeConverter::Type2Str(track.m_type) << ": distance " << std::fixed << std::setw(2) << std::setprecision(2) << dist << " on time " << (period / m_fps) << " with velocity " << velocity << " km/h: " << track.m_confidence << std::endl;
                if (velocity < 1.f || std::isnan(velocity))
                    velocity = 0;
                //label << TypeConverter::Type2Str(track.m_type) << " " << std::fixed << std::setw(2) << std::setprecision(2) << velocity << " km/h";
                label << TypeConverter::Type2Str(track.m_type) << " " << cvRound(velocity) << " km/h";

                int baseLine = 0;
                double fontScale = (frame.cols < 2000) ? 0.5 : 1.;
                cv::Size labelSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_TRIPLEX, fontScale, 1, &baseLine);

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
                cv::rectangle(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(200, 200, 200), cv::FILLED);
                cv::putText(frame, label.str(), brect.tl(), cv::FONT_HERSHEY_TRIPLEX, fontScale, cv::Scalar(0, 0, 0));

                if (velocity > 3)
                    AddToHeatMap(brect);
            }
        }
    }

    if (drawTrajectory)
    {
        cv::Scalar cl = m_colors[track.m_ID.ID2Module(m_colors.size())];

        for (size_t j = 0; j < track.m_trace.size() - 1; ++j)
        {
            const TrajectoryPoint& pt1 = track.m_trace.at(j);
            const TrajectoryPoint& pt2 = track.m_trace.at(j + 1);
#if (CV_VERSION_MAJOR >= 4)
            cv::line(frame, pt1.m_prediction, pt2.m_prediction, cl, 1, cv::LINE_AA);
#else
            cv::line(frame, pt1.m_prediction, pt2.m_prediction, cl, 1, CV_AA);
#endif
            if (!pt2.m_hasRaw)
            {
#if (CV_VERSION_MAJOR >= 4)
                cv::circle(frame, pt2.m_prediction, 4, cl, 1, cv::LINE_AA);
#else
                cv::circle(frame, pt2.m_prediction, 4, cl, 1, CV_AA);
#endif
            }
        }
    }
}

///
/// \brief CarsCounting::InitDetector
/// \param frame
///
bool CarsCounting::InitDetector(cv::UMat frame)
{
    if (!m_trackerSettingsLoaded)
        return false;

    config_t config;

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

    m_detector = BaseDetector::CreateDetector((tracking::Detectors)m_trackerSettings.m_detectorBackend, config, frame);

    return m_detector.operator bool();
}

///
/// \brief CarsCounting::InitTracker
/// \param grayFrame
///
bool CarsCounting::InitTracker(cv::UMat frame)
{
    if (!m_trackerSettingsLoaded)
        return false;

    if (m_drawHeatMap)
    {
        if (frame.channels() == 3)
            m_keyFrame = frame.getMat(cv::ACCESS_READ).clone();
        else
            cv::cvtColor(frame, m_keyFrame, cv::COLOR_GRAY2BGR);
        m_heatMap = cv::Mat(m_keyFrame.size(), CV_32FC1, cv::Scalar::all(0));
    }

    const int minStaticTime = 5;

    TrackerSettings settings;
    settings.SetDistance(tracking::DistJaccard);
    settings.m_kalmanType = tracking::KalmanLinear;
    settings.m_filterGoal = tracking::FilterCenter;
    settings.m_lostTrackType = tracking::TrackCSRT; // Use KCF tracker for collisions resolving. Used if m_filterGoal == tracking::FilterRect
    settings.m_matchType = tracking::MatchHungrian;
    settings.m_dt = 0.3f;                           // Delta time for Kalman filter
    settings.m_accelNoiseMag = 0.2f;                // Accel noise magnitude for Kalman filter
    settings.m_distThres = 0.7f;                    // Distance threshold between region and object on two frames
    settings.m_minAreaRadiusPix = frame.rows / 20.f;
    settings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames

    settings.AddNearTypes(TypeConverter::Str2Type("car"), TypeConverter::Str2Type("bus"), false);
    settings.AddNearTypes(TypeConverter::Str2Type("car"), TypeConverter::Str2Type("truck"), false);
    settings.AddNearTypes(TypeConverter::Str2Type("person"), TypeConverter::Str2Type("bicycle"), true);
    settings.AddNearTypes(TypeConverter::Str2Type("person"), TypeConverter::Str2Type("motorbike"), true);

    settings.m_useAbandonedDetection = false;
    if (settings.m_useAbandonedDetection)
    {
        settings.m_minStaticTime = minStaticTime;
        settings.m_maxStaticTime = 60;
        settings.m_maximumAllowedSkippedFrames = cvRound(settings.m_minStaticTime * m_fps); // Maximum allowed skipped frames
        settings.m_maxTraceLength = 2 * settings.m_maximumAllowedSkippedFrames;        // Maximum trace length
    }
    else
    {
        settings.m_maximumAllowedSkippedFrames = cvRound(10 * m_fps); // Maximum allowed skipped frames
        settings.m_maxTraceLength = cvRound(4 * m_fps);              // Maximum trace length
    }

    m_tracker = BaseTracker::CreateTracker(settings);

    ReadGeobindings(frame.size());
    return true;
}

///
/// \brief CarsCounting::DrawData
/// \param frame
///
void CarsCounting::DrawData(cv::Mat frame, const std::vector<TrackingObject>& tracks, int framesCounter, int currTime)
{
    if (m_showLogs)
        std::cout << "Frame " << framesCounter << ": tracks = " << tracks.size() << ", time = " << currTime << std::endl;

#if 1 // Debug output
    if (!m_geoParams.Empty())
    {
        std::vector<cv::Point> points = m_geoParams.GetFramePoints();
        for (size_t i = 0; i < points.size(); ++i)
        {
            cv::line(frame, points[i % points.size()], points[(i + 1) % points.size()], cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }
#endif

    for (const auto& track : tracks)
    {
        if (track.m_isStatic)
        {
            DrawTrack(frame, track, true, framesCounter);
        }
        else
        {
            if (track.IsRobust(cvRound(m_fps / 4),          // Minimal trajectory size
                               0.8f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                               cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, track, true, framesCounter);

                CheckLinesIntersection(track, static_cast<float>(frame.cols), static_cast<float>(frame.rows));
            }
        }
    }
    //m_detector->CalcMotionMap(frame);

    if (!m_geoParams.Empty())
    {
        cv::Mat geoMap = m_geoParams.DrawTracksOnMap(tracks);
		if (!geoMap.empty())
		{
#ifndef SILENT_WORK
			cv::namedWindow("Geo map", cv::WINDOW_NORMAL);
			cv::imshow("Geo map", geoMap);
#endif
            if (true)
            {
                double k = 0.25;
                cv::Size mapPreview(cvRound(frame.cols * k), cvRound(((frame.cols * k) / geoMap.cols) * geoMap.rows));
                cv::resize(geoMap, frame(cv::Rect(frame.cols - mapPreview.width - 1, frame.rows - mapPreview.height - 1, mapPreview.width, mapPreview.height)), mapPreview, 0, 0, cv::INTER_CUBIC);
            }
		}
    }

    for (const auto& rl : m_lines)
    {
        rl.Draw(frame);
    }

    cv::Mat heatMap = DrawHeatMap();
#ifndef SILENT_WORK
    if (!heatMap.empty())
        cv::imshow("Heat map", heatMap);
#endif
}

///
/// \brief CarsCounting::AddLine
/// \param newLine
///
void CarsCounting::AddLine(const RoadLine& newLine)
{
    m_lines.push_back(newLine);
}

///
/// \brief CarsCounting::GetLine
/// \param lineUid
/// \return
///
bool CarsCounting::GetLine(unsigned int lineUid, RoadLine& line)
{
    for (const auto& rl : m_lines)
    {
        if (rl.m_uid == lineUid)
        {
            line = rl;
            return true;
        }
    }
    return false;
}

///
/// \brief CarsCounting::RemoveLine
/// \param lineUid
/// \return
///
bool CarsCounting::RemoveLine(unsigned int lineUid)
{
    for (auto it = std::begin(m_lines); it != std::end(m_lines);)
    {
        if (it->m_uid == lineUid)
            it = m_lines.erase(it);
        else
            ++it;
    }
    return false;
}

///
/// \brief CarsCounting::CheckLinesIntersection
/// \param track
///
void CarsCounting::CheckLinesIntersection(const TrackingObject& track, float xMax, float yMax)
{
    auto Pti2f = [&](cv::Point pt)
    {
        return cv::Point2f(pt.x / xMax, pt.y / yMax);
    };

    constexpr size_t minTrack = 5;
    if (track.m_trace.size() >= minTrack)
    {
        for (auto& rl : m_lines)
        {
            rl.IsIntersect(track.m_ID, Pti2f(track.m_trace[track.m_trace.size() - minTrack]), Pti2f(track.m_trace[track.m_trace.size() - 1]));
        }
    }
}

///
/// \brief CarsCounting::DrawHeatMap
///
cv::Mat CarsCounting::DrawHeatMap()
{
    cv::Mat res;
    if (!m_heatMap.empty())
    {
        cv::normalize(m_heatMap, m_normHeatMap, 255, 0, cv::NORM_MINMAX, CV_8UC1);
        cv::applyColorMap(m_normHeatMap, m_colorMap, cv::COLORMAP_HOT);
        cv::bitwise_or(m_keyFrame, m_colorMap, res);
    }
    return res;
}

///
/// \brief CarsCounting::AddToHeatMap
///
void CarsCounting::AddToHeatMap(const cv::Rect& rect)
{
    if (m_heatMap.empty())
        return;

    constexpr float w = 0.001f;
    for (int y = 0; y < rect.height; ++y)
    {
        float* heatPtr = m_heatMap.ptr<float>(rect.y + y) + rect.x;
        for (int x = 0; x < rect.width; ++x)
        {
            heatPtr[x] += w;
        }
    }
}

///
/// \brief CarsCounting::ReadGeobindings
///
bool CarsCounting::ReadGeobindings(cv::Size frameSize)
{
    bool res = true;
    INIReader reader(m_geoBindFile);
    
    int parseError = reader.ParseError();
    if (parseError < 0)
    {
        std::cerr << "GeoBindFile file " << m_geoBindFile << " does not exist!" << std::endl;
        res = false;
    }
    else if (parseError > 0)
    {
        std::cerr << "GeoBindFile file " << m_geoBindFile << " parse error in line: " << parseError << std::endl;
        res = false;
    }
    if (!res)
        return res;

    // Read frame-map bindings
    std::vector<cv::Point2d> geoPoints;
    std::vector<cv::Point> framePoints;
    for (size_t i = 0;; ++i)
    {
        cv::Point2d geoPoint;
        std::string lat = "lat" + std::to_string(i);
        std::string lon = "lon" + std::to_string(i);
        std::string px_x = "px_x" + std::to_string(i);
        std::string px_y = "px_y" + std::to_string(i);
        if (reader.HasValue("points", lat) && reader.HasValue("points", lon) && reader.HasValue("points", px_x) && reader.HasValue("points", px_y))
        {
            geoPoints.emplace_back(reader.GetReal("points", lat, 0), reader.GetReal("points", lon, 0));
            framePoints.emplace_back(cvRound(reader.GetReal("points", px_x, 0) * frameSize.width), cvRound(reader.GetReal("points", px_y, 0) * frameSize.height));
        }
        else
        {
            break;
        }
    }
    res = m_geoParams.SetKeyPoints(framePoints, geoPoints);

    // Read map image
    std::string mapFile = reader.GetString("map", "file", "");
    std::vector<cv::Point2d> mapGeoCorners;
    mapGeoCorners.emplace_back(reader.GetReal("map", "left_top_lat", 0), reader.GetReal("map", "left_top_lon", 0));
    mapGeoCorners.emplace_back(reader.GetReal("map", "right_top_lat", 0), reader.GetReal("map", "right_top_lon", 0));
    mapGeoCorners.emplace_back(reader.GetReal("map", "right_bottom_lat", 0), reader.GetReal("map", "right_bottom_lon", 0));
    mapGeoCorners.emplace_back(reader.GetReal("map", "left_bottom_lat", 0), reader.GetReal("map", "left_bottom_lon", 0));
    m_geoParams.SetMapParams(mapFile, mapGeoCorners);

    // Read lines
    std::cout <<"Read lines:" << std::endl;
    for (size_t i = 0;; ++i)
    {
        std::string line = "line" + std::to_string(i);
        std::string x0 = line + "_x0";
        std::string y0 = line + "_y0";
        std::string x1 = line + "_x1";
        std::string y1 = line + "_y1";
        if (reader.HasValue("lines", x0) && reader.HasValue("lines", y0) && reader.HasValue("lines", x1) && reader.HasValue("lines", y1))
        {
            cv::Point2f p0(static_cast<float>(reader.GetReal("lines", x0, 0)), static_cast<float>(reader.GetReal("lines", y0, 0)));
            cv::Point2f p1(static_cast<float>(reader.GetReal("lines", x1, 0)), static_cast<float>(reader.GetReal("lines", y1, 0)));
            std::cout << "Line" << i << ": " << p0 << " - " << p1 << std::endl;
            AddLine(RoadLine(p0, p1, static_cast<unsigned int>(i)));
        }
        else
        {
            break;
        }
    }

    return res;
}
