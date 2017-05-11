#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <array>

#include "defines.h"
#include "track.h"
#include "LocalTracker.h"

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
             size_t max_trace_length_ = 10);
	~CTracker(void);

    tracks_t tracks;
    void Update(const std::vector<Point_t>& detections, const regions_t& regions, cv::Mat grayFrame);

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
};
