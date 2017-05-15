#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <array>
#include <chrono>

#include "defines.h"
#include "track.h"
#include "LocalTracker.h"

// ----------------------------------------------------------------------
#define SAVE_TRAJECTORIES 1

#if SAVE_TRAJECTORIES
#include <fstream>

class SaveTrajectories
{
public:
    SaveTrajectories()
    {

    }
    ~SaveTrajectories()
    {
    }

    bool Open(std::string fileName)
    {
        if (fileName.length() > 0)
        {
            m_file.open(fileName.c_str(), std::ios_base::out | std::ios_base::trunc);
        }
        return m_file.is_open();
    }

    bool NewTrack(const CTrack& track)
    {
        if (m_file.is_open())
        {
            std::string delim = ",";
            int type = 2;

            if (track.m_trace.size() > 25)
            {
                for (size_t j = 0; j < track.m_trace.size(); ++j)
                {
                    const TrajectoryPoint& pt = track.m_trace.at(j);

                    m_file << pt.m_frameInd << delim
                           << track.m_trackID << delim
                           << type << delim
                           << pt.m_prediction.x << delim
                           << pt.m_prediction.y << delim
                           << (static_cast<track_t>(pt.m_size.width) / static_cast<track_t>(pt.m_size.height))<< delim
                           << pt.m_time << delim
                           << j << std::endl;
                }
            }
            return true;
        }
        return false;
    }

private:
    std::ofstream m_file;
};
#endif

// --------------------------------------------------------------------------
class CTracker
{
public:
    enum DistType
    {
        CentersDist = 0,
        RectsDist = 1
    };
    enum FilterGoal
    {
        FilterCenter = 0,
        FilterRect = 1
    };
    enum KalmanType
    {
        KalmanLinear = 0,
        KalmanUnscented = 1
    };
	enum MatchType
	{
		MatchHungrian = 0,
		MatchBipart = 1
	};
    enum LostTrackType
    {
        TrackNone = 0,
        TrackKCF = 1
    };

    CTracker(bool useLocalTracking,
             DistType distType,
             KalmanType kalmanType,
             FilterGoal filterGoal,
             LostTrackType useExternalTrackerForLostObjects,
			 MatchType matchType,
             track_t dt_,
             track_t accelNoiseMag_,
             track_t dist_thres_ = 60,
             size_t maximum_allowed_skipped_frames_ = 10,
             size_t max_trace_length_ = 10,
             std::string trajectoryFileName = "");
	~CTracker(void);

    tracks_t tracks;
    void Update(const std::vector<Point_t>& detections, const regions_t& regions, cv::Mat grayFrame, int frameInd = 0);

#if SAVE_TRAJECTORIES
    void WriteAllTracks();
#endif

private:
    // Use local tracking for regions between two frames
    bool m_useLocalTracking;

    DistType m_distType;
    KalmanType m_kalmanType;
    FilterGoal m_filterGoal;
    LostTrackType m_useExternalTrackerForLostObjects;
	MatchType m_matchType;

	// Шаг времени опроса фильтра
	track_t dt;

	track_t accelNoiseMag;

	// Порог расстояния. Если точки находятся дуг от друга на расстоянии,
	// превышающем этот порог, то эта пара не рассматривается в задаче о назначениях.
	track_t dist_thres;
	// Максимальное количество кадров которое трек сохраняется не получая данных о измерений.
    size_t maximum_allowed_skipped_frames;
	// Максимальная длина следа
    size_t max_trace_length;

	size_t NextTrackID;

    LocalTracker m_localTracker;

    cv::Mat m_prevFrame;

#if SAVE_TRAJECTORIES
    SaveTrajectories m_saveTraj;
#endif
};
