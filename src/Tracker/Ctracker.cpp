#include "Ctracker.h"
#include "ShortPathCalculator.h"
#include "EmbeddingsCalculator.hpp"
#include "track.h"

///
/// \brief The CTracker class
///
class CTracker final : public BaseTracker
{
public:
    CTracker(const TrackerSettings& settings);
	CTracker(const CTracker&) = delete;
	CTracker(CTracker&&) = delete;
	CTracker& operator=(const CTracker&) = delete;
	CTracker& operator=(CTracker&&) = delete;
	
	~CTracker(void) = default;

    void Update(const regions_t& regions, cv::UMat currFrame, float fps) override;

    bool CanGrayFrameToTrack() const override;
	bool CanColorFrameToTrack() const override;
    size_t GetTracksCount() const override;
	void GetTracks(std::vector<TrackingObject>& tracks) const override;
    void GetRemovedTracks(std::vector<track_id_t>& trackIDs) const override;

private:
    TrackerSettings m_settings;

	tracks_t m_tracks;

    track_id_t m_nextTrackID;
    std::vector<track_id_t> m_removedObjects;

    cv::UMat m_prevFrame;

    std::unique_ptr<ShortPathCalculator> m_SPCalculator;
    std::map<objtype_t, std::shared_ptr<EmbeddingsCalculator>> m_embCalculators;

    void CreateDistaceMatrix(const regions_t& regions, const std::vector<RegionEmbedding>& regionEmbeddings, distMatrix_t& costMatrix, track_t maxPossibleCost, track_t& maxCost);
    void UpdateTrackingState(const regions_t& regions, cv::UMat currFrame, float fps);
	void CalcEmbeddins(std::vector<RegionEmbedding>& regionEmbeddings, const regions_t& regions, cv::UMat currFrame) const;
};
// ----------------------------------------------------------------------

///
/// \brief CTracker::CTracker
/// Manage tracks: create, remove, update.
/// \param settings
///
CTracker::CTracker(const TrackerSettings& settings)
    : m_settings(settings)
{
    m_SPCalculator.reset();
    SPSettings spSettings = { settings.m_distThres, 12 };
    switch (m_settings.m_matchType)
    {
    case tracking::MatchHungrian:
        m_SPCalculator = std::make_unique<SPHungrian>(spSettings);
        break;
    case tracking::MatchBipart:
        m_SPCalculator = std::make_unique<SPBipart>(spSettings);
        break;
    }
    assert(m_SPCalculator);

	for (const auto& embParam : settings.m_embeddings)
	{
		std::shared_ptr<EmbeddingsCalculator> embCalc = std::make_shared<EmbeddingsCalculator>();
		if (!embCalc->Initialize(embParam.m_embeddingCfgName, embParam.m_embeddingWeightsName, embParam.m_inputLayer))
		{
			std::cerr << "EmbeddingsCalculator initialization error: " << embParam.m_embeddingCfgName << ", " << embParam.m_embeddingWeightsName << std::endl;
		}
		else
		{
			for (auto objType : embParam.m_objectTypes)
			{
				m_embCalculators.try_emplace((objtype_t)objType, embCalc);
			}
		}
	}
}

///
    /// \brief CanGrayFrameToTrack
    /// \return
    ///
bool CTracker::CanGrayFrameToTrack() const
{
    bool needColor = (m_settings.m_lostTrackType == tracking::LostTrackType::TrackGOTURN) ||
        (m_settings.m_lostTrackType == tracking::LostTrackType::TrackDAT) ||
        (m_settings.m_lostTrackType == tracking::LostTrackType::TrackSTAPLE) ||
        (m_settings.m_lostTrackType == tracking::LostTrackType::TrackLDES);
    return !needColor;
}

///
/// \brief CanColorFrameToTrack
/// \return
///
bool CTracker::CanColorFrameToTrack() const
{
    return true;
}

///
/// \brief GetTracksCount
/// \return
///
size_t CTracker::GetTracksCount() const
{
    return m_tracks.size();
}

///
/// \brief GetTracks
/// \return
///
void CTracker::GetTracks(std::vector<TrackingObject>& tracks) const
{
    tracks.clear();

    if (m_tracks.size() > tracks.capacity())
        tracks.reserve(m_tracks.size());
    for (const auto& track : m_tracks)
    {
        tracks.emplace_back(track->ConstructObject());
    }
}

///
/// \brief GetRemovedTracks
/// \return
///
void CTracker::GetRemovedTracks(std::vector<track_id_t>& trackIDs) const
{
    trackIDs.assign(std::begin(m_removedObjects), std::end(m_removedObjects));
}

///
/// \brief CTracker::Update
/// \param regions
/// \param currFrame
/// \param fps
///
void CTracker::Update(const regions_t& regions, cv::UMat currFrame, float fps)
{
    m_removedObjects.clear();

    UpdateTrackingState(regions, currFrame, fps);

    currFrame.copyTo(m_prevFrame);
}

#define DRAW_DBG_ASSIGNMENT 0

///
/// \brief CTracker::UpdateTrackingState
/// \param regions
/// \param currFrame
/// \param fps
///
void CTracker::UpdateTrackingState(const regions_t& regions,
                                   cv::UMat currFrame,
                                   float fps)
{
    const size_t N = m_tracks.size();	// Tracking objects
    const size_t M = regions.size();	// Detections or regions

    assignments_t assignment(N, -1); // Assignments regions -> tracks

    std::vector<RegionEmbedding> regionEmbeddings;
    CalcEmbeddins(regionEmbeddings, regions, currFrame);

#if DRAW_DBG_ASSIGNMENT
    cv::Mat dbgAssignment = currFrame.getMat(cv::ACCESS_READ).clone();
    {
        cv::Mat foreground(dbgAssignment.size(), CV_8UC1, cv::Scalar(0, 0, 100));
        for (const auto& track : m_tracks)
        {
#if (CV_VERSION_MAJOR < 4)
            cv::ellipse(foreground, track->GetLastRect(), cv::Scalar(255, 255, 255), CV_FILLED);
#else
            cv::ellipse(foreground, track->GetLastRect(), cv::Scalar(255, 255, 255), cv::FILLED);
#endif
        }

        const int chans = dbgAssignment.channels();
        const int height = dbgAssignment.rows;
#pragma omp parallel for
        for (int y = 0; y < height; ++y)
        {
            uchar* imgPtr = dbgAssignment.ptr(y);
            const uchar* frgrndPtr = foreground.ptr(y);
            for (int x = 0; x < dbgAssignment.cols; ++x)
            {
                for (int ci = chans - 1; ci < chans; ++ci)
                {
                    imgPtr[ci] = cv::saturate_cast<uchar>(imgPtr[ci] + frgrndPtr[0]);
                }
                imgPtr += chans;
                ++frgrndPtr;
            }
        }

        for (const auto& reg : regions)
        {
            cv::rectangle(dbgAssignment, reg.m_brect, cv::Scalar(255, 0, 0), 1);
        }
    }
#endif

    if (!m_tracks.empty())
    {
        // Distance matrix between all tracks to all regions
        distMatrix_t costMatrix(N * M);
        const track_t maxPossibleCost = static_cast<track_t>(currFrame.cols * currFrame.rows);
        track_t maxCost = 0;
        CreateDistaceMatrix(regions, regionEmbeddings, costMatrix, maxPossibleCost, maxCost);

        // Solving assignment problem (shortest paths)
        m_SPCalculator->Solve(costMatrix, N, M, assignment, maxCost);

        // clean assignment from pairs with large distance
        for (size_t i = 0; i < assignment.size(); i++)
        {
#if DRAW_DBG_ASSIGNMENT
            std::stringstream ss;
            if (assignment[i] != -1)
            {
                ss << std::fixed << std::setprecision(2) << costMatrix[i + assignment[i] * N];

				if (costMatrix[i + assignment[i] * N] > m_settings.m_distThres)
                {
                    ss << ">" << m_settings.m_distThres;
                    cv::line(dbgAssignment, m_tracks[i]->GetLastRect().center, regions[assignment[i]].m_rrect.center, cv::Scalar(0, 0, 255), 1);
                    cv::rectangle(dbgAssignment, m_tracks[i]->LastRegion().m_brect, cv::Scalar(0, 0, 255), 1);
                }
                else
                {
                    ss << "<" << m_settings.m_distThres;
                    cv::line(dbgAssignment, m_tracks[i]->GetLastRect().center, regions[assignment[i]].m_rrect.center, cv::Scalar(0, 255, 0), 1);
                    cv::rectangle(dbgAssignment, m_tracks[i]->LastRegion().m_brect, cv::Scalar(0, 255, 0), 1);
                }

                for (size_t ri = 0; ri < regions.size(); ++ri)
                {
                    if (ri != assignment[i] && costMatrix[i + ri * N] < 1)
                    {
                        std::stringstream liness;
                        liness << std::fixed << std::setprecision(2) << costMatrix[i + ri * N];
                        auto p1 = m_tracks[i]->GetLastRect().center;
                        auto p2 = regions[ri].m_rrect.center;
                        cv::line(dbgAssignment, p1, p2, cv::Scalar(255, 0, 255), 1);
                        cv::putText(dbgAssignment, liness.str(), cv::Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2), cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(0, 0, 0), 1, 8);
                    }
                }
            }
            else
            {
                // If track have no assigned detect, then increment skipped frames counter.
                cv::rectangle(dbgAssignment, m_tracks[i]->LastRegion().m_brect, cv::Scalar(255, 0, 255), 1);
                for (size_t ri = 0; ri < regions.size(); ++ri)
                {
                    if (costMatrix[i + ri * N] < 1)
                    {
                        std::stringstream liness;
                        liness << std::fixed << std::setprecision(2) << costMatrix[i + ri * N];
                        auto p1 = m_tracks[i]->GetLastRect().center;
                        auto p2 = regions[ri].m_rrect.center;
                        cv::line(dbgAssignment, p1, p2, cv::Scalar(255, 0, 255), 1);
                        cv::putText(dbgAssignment, liness.str(), cv::Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2), cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(0, 0, 0), 1, 8);
                    }
                }
            }
            if (ss.str().length() > 0)
            {
                auto brect = m_tracks[i]->LastRegion().m_brect;
                cv::putText(dbgAssignment, ss.str(), cv::Point(brect.x, brect.y), cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(0, 0, 0), 1, 8);
            }
#endif

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
                m_removedObjects.push_back(m_tracks[i]->GetID());
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
                                                            m_nextTrackID,
                                                            m_settings.m_filterGoal == tracking::FilterRect,
                                                            m_settings.m_lostTrackType));
            else
                m_tracks.push_back(std::make_unique<CTrack>(regions[i],
                                                            regionEmbeddings[i],
                                                            m_settings.m_kalmanType,
                                                            m_settings.m_dt,
                                                            m_settings.m_accelNoiseMag,
                                                            m_settings.m_useAcceleration,
                                                            m_nextTrackID,
                                                            m_settings.m_filterGoal == tracking::FilterRect,
                                                            m_settings.m_lostTrackType));
            m_nextTrackID = m_nextTrackID.NextID();
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
            //std::cout << "Update track " << i << " for " << assignment[i] << " region, regionEmbeddings.size = " << regionEmbeddings.size() << std::endl;
            if (regionEmbeddings.empty())
                m_tracks[i]->Update(regions[assignment[i]],
                        true, m_settings.m_maxTraceLength,
                        m_prevFrame, currFrame,
                        m_settings.m_useAbandonedDetection ? cvRound(m_settings.m_minStaticTime * fps) : 0, m_settings.m_maxSpeedForStatic);
            else
                m_tracks[i]->Update(regions[assignment[i]], regionEmbeddings[assignment[i]],
                        true, m_settings.m_maxTraceLength,
                        m_prevFrame, currFrame,
                        m_settings.m_useAbandonedDetection ? cvRound(m_settings.m_minStaticTime * fps) : 0, m_settings.m_maxSpeedForStatic);
        }
        else				     // if not continue using predictions
        {
            m_tracks[i]->Update(CRegion(), false, m_settings.m_maxTraceLength, m_prevFrame, currFrame, 0, m_settings.m_maxSpeedForStatic);
        }
    }

#if DRAW_DBG_ASSIGNMENT
    cv::imshow("dbgAssignment", dbgAssignment);
    //cv::waitKey(1);
#endif

}

///
/// \brief CTracker::CreateDistaceMatrix
/// \param regions
/// \param costMatrix
/// \param maxPossibleCost
/// \param maxCost
///
void CTracker::CreateDistaceMatrix(const regions_t& regions,
                                   const std::vector<RegionEmbedding>& regionEmbeddings,
                                   distMatrix_t& costMatrix,
                                   track_t maxPossibleCost,
                                   track_t& maxCost)
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
				// Euclidean distance between centers
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

				// Euclidean distance between bounding rectangles
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

				// Intersection over Union, IoU
				if (m_settings.m_distType[ind] > 0.0f && ind == tracking::DistJaccard)
					dist += m_settings.m_distType[ind] * track->CalcDistJaccard(reg);
				++ind;

				// Bhatacharia distance between histograms
				if (m_settings.m_distType[ind] > 0.0f && ind == tracking::DistHist)
                {
                    dist += m_settings.m_distType[ind] * track->CalcDistHist(regionEmbeddings[j]);
                }
				++ind;

				// Cosine distance between embeddings
                if (m_settings.m_distType[ind] > 0.0f && ind == tracking::DistFeatureCos)
                {
                    if (reg.m_type == track->LastRegion().m_type)
                    {
                        auto resCos = track->CalcCosine(regionEmbeddings[j]);
                        if (resCos)
                        {
                            dist += m_settings.m_distType[ind] * resCos.value();
                            //std::cout << "CalcCosine: " << TypeConverter::Type2Str(track->LastRegion().m_type) << ", reg = " << reg.m_brect << ", track = " << track->LastRegion().m_brect << ": res = " << resCos.value() << ", dist = " << dist << std::endl;
                        }
                        else
                        {
                            dist /= m_settings.m_distType[ind];
                            //std::cout << "CalcCosine: " << TypeConverter::Type2Str(track->LastRegion().m_type) << ", reg = " << reg.m_brect << ", track = " << track->LastRegion().m_brect << ": res = 1, weight = " << m_settings.m_distType[ind] << ", dist = " << dist << std::endl;
                        }
                    }
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

///
/// \brief CTracker::CalcEmbeddins
/// \param regionEmbeddings
/// \param regions
/// \param currFrame
///
void CTracker::CalcEmbeddins(std::vector<RegionEmbedding>& regionEmbeddings, const regions_t& regions, cv::UMat currFrame) const
{
    if (!regions.empty())
    {
        regionEmbeddings.resize(regions.size());
        // Bhatacharia distance between histograms
        if (m_settings.m_distType[tracking::DistHist] > 0.0f)
        {
            for (size_t j = 0; j < regions.size(); ++j)
            {
                    int bins = 64;
                    std::vector<int> histSize;
                    std::vector<float> ranges;
                    std::vector<int> channels;

                    for (int i = 0, stop = currFrame.channels(); i < stop; ++i)
                    {
                        histSize.push_back(bins);
                        ranges.push_back(0);
                        ranges.push_back(255);
                        channels.push_back(i);
                    }

                    std::vector<cv::UMat> regROI = { currFrame(regions[j].m_brect) };
                    cv::calcHist(regROI, channels, cv::Mat(), regionEmbeddings[j].m_hist, histSize, ranges, false);
                    cv::normalize(regionEmbeddings[j].m_hist, regionEmbeddings[j].m_hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
            }
        }

        // Cosine distance between embeddings
        if (m_settings.m_distType[tracking::DistFeatureCos] > 0.0f)
        {
            for (size_t j = 0; j < regions.size(); ++j)
            {
                if (regionEmbeddings[j].m_embedding.empty())
                {
                    //std::cout << "Search embCalc for " << TypeConverter::Type2Str(regions[j].m_type) << ": ";
                    auto embCalc = m_embCalculators.find(regions[j].m_type);
                    if (embCalc != std::end(m_embCalculators))
                    {
                        embCalc->second->Calc(currFrame, regions[j].m_brect, regionEmbeddings[j].m_embedding);
                        regionEmbeddings[j].m_embDot = regionEmbeddings[j].m_embedding.dot(regionEmbeddings[j].m_embedding);

                        //std::cout << "Founded! m_embedding = " << regionEmbeddings[j].m_embedding.size() << ", m_embDot = " << regionEmbeddings[j].m_embDot << std::endl;
                    }
                    else
                    {
                        //std::cout << "Not found" << std::endl;
                    }
                }
            }
        }
    }
}

///
/// BaseTracker::CreateTracker
///
std::unique_ptr<BaseTracker> BaseTracker::CreateTracker(const TrackerSettings& settings)
{
    return std::make_unique<CTracker>(settings);
}
