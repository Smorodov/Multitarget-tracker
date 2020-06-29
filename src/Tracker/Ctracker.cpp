#include "Ctracker.h"

///
/// \brief CTracker::CTracker
/// Tracker. Manage tracks. Create, remove, update.
/// \param settings
///
CTracker::CTracker(const TrackerSettings& settings)
    :
      m_settings(settings),
      m_nextTrackID(0)
{
    ShortPathCalculator* spcalc = nullptr;
    SPSettings spSettings = { settings.m_distThres, 12 };
    switch (m_settings.m_matchType)
    {
    case tracking::MatchHungrian:
        spcalc = new SPHungrian(spSettings);
        break;
    case tracking::MatchBipart:
        spcalc = new SPBipart(spSettings);
        break;
    }
    assert(spcalc != nullptr);
    m_SPCalculator = std::unique_ptr<ShortPathCalculator>(spcalc);
}

///
/// \brief CTracker::~CTracker
///
CTracker::~CTracker(void)
{
}

///
/// \brief CTracker::Update
/// \param regions
/// \param currFrame
/// \param fps
///
void CTracker::Update(
        const regions_t& regions,
        cv::UMat currFrame,
        float fps
        )
{
    UpdateTrackingState(regions, currFrame, fps);

    currFrame.copyTo(m_prevFrame);
}

///
/// \brief CTracker::UpdateTrackingState
/// \param regions
/// \param currFrame
/// \param fps
///
void CTracker::UpdateTrackingState(
        const regions_t& regions,
        cv::UMat currFrame,
        float fps
        )
{
    const size_t N = m_tracks.size();	// Tracking objects
    const size_t M = regions.size();	// Detections or regions

    assignments_t assignment(N, -1); // Assignments regions -> tracks

    std::vector<RegionEmbedding> regionEmbeddings;

    if (!m_tracks.empty())
    {
        // Distance matrix between all tracks to all regions
        distMatrix_t costMatrix(N * M);
        const track_t maxPossibleCost = static_cast<track_t>(currFrame.cols * currFrame.rows);
        track_t maxCost = 0;
        CreateDistaceMatrix(regions, regionEmbeddings, costMatrix, maxPossibleCost, maxCost, currFrame);

        // Solving assignment problem (shortest paths)
        m_SPCalculator->Solve(costMatrix, N, M, assignment, maxCost);

        // clean assignment from pairs with large distance
        for (size_t i = 0; i < assignment.size(); i++)
        {
            if (assignment[i] != -1)
            {
				if (costMatrix[i + assignment[i] * N] > m_settings.m_distThres)
                {
                    assignment[i] = -1;
                    m_tracks[i]->SkippedFrames()++;
                }
            }
            else
            {
                // If track have no assigned detect, then increment skipped frames counter.
                m_tracks[i]->SkippedFrames()++;
            }
        }

        // If track didn't get detects long time, remove it.
        for (size_t i = 0; i < m_tracks.size();)
        {
            if (m_tracks[i]->SkippedFrames() > m_settings.m_maximumAllowedSkippedFrames ||
				m_tracks[i]->IsOutOfTheFrame() ||
                    m_tracks[i]->IsStaticTimeout(cvRound(fps * (m_settings.m_maxStaticTime - m_settings.m_minStaticTime))))
            {
                m_tracks.erase(m_tracks.begin() + i);
                assignment.erase(assignment.begin() + i);
            }
			else
			{
				++i;
			}
        }
    }

    // Search for unassigned detects and start new tracks for them.
    for (size_t i = 0; i < regions.size(); ++i)
    {
        if (find(assignment.begin(), assignment.end(), i) == assignment.end())
        {
            if (regionEmbeddings.empty())
                m_tracks.push_back(std::make_unique<CTrack>(regions[i],
                                                            m_settings.m_kalmanType,
                                                            m_settings.m_dt,
                                                            m_settings.m_accelNoiseMag,
                                                            m_settings.m_useAcceleration,
                                                            m_nextTrackID++,
                                                            m_settings.m_filterGoal == tracking::FilterRect,
                                                            m_settings.m_lostTrackType));
            else
                m_tracks.push_back(std::make_unique<CTrack>(regions[i],
                                                            regionEmbeddings[i],
                                                            m_settings.m_kalmanType,
                                                            m_settings.m_dt,
                                                            m_settings.m_accelNoiseMag,
                                                            m_settings.m_useAcceleration,
                                                            m_nextTrackID++,
                                                            m_settings.m_filterGoal == tracking::FilterRect,
                                                            m_settings.m_lostTrackType));
        }
    }

    // Update Kalman Filters state
    const ptrdiff_t stop_i = static_cast<ptrdiff_t>(assignment.size());
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < stop_i; ++i)
    {
        // If track updated less than one time, than filter state is not correct.
        if (assignment[i] != -1) // If we have assigned detect, then update using its coordinates,
        {
            m_tracks[i]->SkippedFrames() = 0;
            if (regionEmbeddings.empty())
                m_tracks[i]->Update(regions[assignment[i]],
                        true, m_settings.m_maxTraceLength,
                        m_prevFrame, currFrame,
                        m_settings.m_useAbandonedDetection ? cvRound(m_settings.m_minStaticTime * fps) : 0);
            else
                m_tracks[i]->Update(regions[assignment[i]], regionEmbeddings[assignment[i]],
                        true, m_settings.m_maxTraceLength,
                        m_prevFrame, currFrame,
                        m_settings.m_useAbandonedDetection ? cvRound(m_settings.m_minStaticTime * fps) : 0);
        }
        else				     // if not continue using predictions
        {
            m_tracks[i]->Update(CRegion(), false, m_settings.m_maxTraceLength, m_prevFrame, currFrame, 0);
        }
    }
}

///
/// \brief CTracker::CreateDistaceMatrix
/// \param regions
/// \param costMatrix
/// \param maxPossibleCost
/// \param maxCost
///
void CTracker::CreateDistaceMatrix(const regions_t& regions, std::vector<RegionEmbedding>& regionEmbeddings, distMatrix_t& costMatrix, track_t maxPossibleCost, track_t& maxCost, cv::UMat currFrame)
{
    const size_t N = m_tracks.size();	// Tracking objects
    maxCost = 0;

	for (size_t i = 0; i < N; ++i)
	{
		const auto& track = m_tracks[i];

		// Calc predicted area for track
		cv::Size_<track_t> minRadius;
		if (m_settings.m_minAreaRadiusPix < 0)
		{
			minRadius.width = m_settings.m_minAreaRadiusK * track->LastRegion().m_rrect.size.width;
			minRadius.height = m_settings.m_minAreaRadiusK * track->LastRegion().m_rrect.size.height;
		}
		else
		{
			minRadius.width = m_settings.m_minAreaRadiusPix;
			minRadius.height = m_settings.m_minAreaRadiusPix;
		}
		cv::RotatedRect predictedArea = track->CalcPredictionEllipse(minRadius);

		// Calc distance between track and regions
		for (size_t j = 0; j < regions.size(); ++j)
		{
			const auto& reg = regions[j];

			auto dist = maxPossibleCost;
			if (m_settings.CheckType(m_tracks[i]->LastRegion().m_type, reg.m_type))
			{
				dist = 0;
				size_t ind = 0;
				if (m_settings.m_distType[ind] > 0.0f && ind == tracking::DistCenters)
				{
#if 1
                    track_t ellipseDist = track->IsInsideArea(reg.m_rrect.center, predictedArea);
                    if (ellipseDist > 1)
                        dist += m_settings.m_distType[ind];
                    else
                        dist += ellipseDist * m_settings.m_distType[ind];
#else
					dist += m_settings.m_distType[ind] * track->CalcDistCenter(reg);
#endif
				}
				++ind;

				if (m_settings.m_distType[ind] > 0.0f && ind == tracking::DistRects)
				{
#if 1
                    track_t ellipseDist = track->IsInsideArea(reg.m_rrect.center, predictedArea);
					if (ellipseDist < 1)
					{
						track_t dw = track->WidthDist(reg);
						track_t dh = track->HeightDist(reg);
						dist += m_settings.m_distType[ind] * (1 - (1 - ellipseDist) * (dw + dh) * 0.5f);
					}
					else
					{
						dist += m_settings.m_distType[ind];
					}
					//std::cout << "dist = " << dist << ", ed = " << ellipseDist << ", dw = " << dw << ", dh = " << dh << std::endl;
#else
					dist += m_settings.m_distType[ind] * track->CalcDistRect(reg);
#endif
				}
				++ind;

				if (m_settings.m_distType[ind] > 0.0f && ind == tracking::DistJaccard)
					dist += m_settings.m_distType[ind] * track->CalcDistJaccard(reg);
				++ind;

				if (m_settings.m_distType[ind] > 0.0f && ind == tracking::DistHist)
                {
                    if (regionEmbeddings.empty())
                        regionEmbeddings.resize(regions.size());
                    dist += m_settings.m_distType[ind] * track->CalcDistHist(reg, regionEmbeddings[j].m_hist, currFrame);
                }
				++ind;
				assert(ind == tracking::DistsCount);
			}

			costMatrix[i + j * N] = dist;
			if (dist > maxCost)
				maxCost = dist;
		}
	}
}
