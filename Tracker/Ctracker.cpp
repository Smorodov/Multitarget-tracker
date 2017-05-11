#include "Ctracker.h"
#include "HungarianAlg.h"

#include <GTL/GTL.h>
#include "mygraph.h"
#include "mwbmatching.h"
#include "tokenise.h"

// ---------------------------------------------------------------------------
// Tracker. Manage tracks. Create, remove, update.
// ---------------------------------------------------------------------------
CTracker::CTracker(
        bool useLocalTracking,
        DistType distType,
        KalmanType kalmanType,
        FilterGoal filterGoal,
        LostTrackType useExternalTrackerForLostObjects,
		MatchType matchType,
        track_t dt_,
        track_t accelNoiseMag_,
        track_t dist_thres_,
        size_t maximum_allowed_skipped_frames_,
        size_t max_trace_length_
        )
    :
      m_useLocalTracking(useLocalTracking),
      m_distType(distType),
      m_kalmanType(kalmanType),
      m_filterGoal(filterGoal),
      m_useExternalTrackerForLostObjects(useExternalTrackerForLostObjects),
	  m_matchType(matchType),
      dt(dt_),
      accelNoiseMag(accelNoiseMag_),
      dist_thres(dist_thres_),
      maximum_allowed_skipped_frames(maximum_allowed_skipped_frames_),
      max_trace_length(max_trace_length_),
      NextTrackID(0)
{
}

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
CTracker::~CTracker(void)
{
}

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
void CTracker::Update(
        const std::vector<Point_t>& detections,
        const regions_t& regions,
        cv::Mat grayFrame
        )
{
    TKalmanFilter::KalmanType kalmanType = (m_kalmanType == KalmanLinear) ? TKalmanFilter::TypeLinear : TKalmanFilter::TypeUnscented;

    assert(detections.size() == regions.size());

    if (m_prevFrame.size() == grayFrame.size())
    {
        if (m_useLocalTracking)
        {
            m_localTracker.Update(tracks, m_prevFrame, grayFrame);
        }
    }

    // -----------------------------------
    // If there is no tracks yet, then every cv::Point begins its own track.
    // -----------------------------------
    if (tracks.size() == 0)
    {
        // If no tracks yet
        for (size_t i = 0; i < detections.size(); ++i)
        {
            tracks.push_back(std::make_unique<CTrack>(detections[i], regions[i], kalmanType, dt, accelNoiseMag, NextTrackID++, m_filterGoal == FilterRect, m_useExternalTrackerForLostObjects == TrackKCF));
        }
    }

    size_t N = tracks.size();		// треки
    size_t M = detections.size();	// детекты

    assignments_t assignment(N, -1); // назначения

    if (!tracks.empty())
    {
        // Матрица расстояний от N-ного трека до M-ного детекта.
        distMatrix_t Cost(N * M);

        // -----------------------------------
        // Треки уже есть, составим матрицу расстояний
        // -----------------------------------
		track_t maxCost = 0;
		switch (m_distType)
        {
        case CentersDist:
            for (size_t i = 0; i < tracks.size(); i++)
            {
                for (size_t j = 0; j < detections.size(); j++)
                {
					auto dist = tracks[i]->CalcDist(detections[j]);
					Cost[i + j * N] = dist;
					if (dist > maxCost)
					{
						maxCost = dist;
					}
                }
            }
            break;

        case RectsDist:
            for (size_t i = 0; i < tracks.size(); i++)
            {
                for (size_t j = 0; j < detections.size(); j++)
                {
					auto dist = tracks[i]->CalcDist(regions[j].m_rect);
					Cost[i + j * N] = dist;
					if (dist > maxCost)
					{
						maxCost = dist;
					}
                }
            }
            break;
        }
        // -----------------------------------
        // Solving assignment problem (tracks and predictions of Kalman filter)
        // -----------------------------------
		if (m_matchType == MatchHungrian)
		{
			AssignmentProblemSolver APS;
			APS.Solve(Cost, N, M, assignment, AssignmentProblemSolver::optimal);
		}
		else
		{
			MyGraph G;
			G.make_directed();

			std::vector<node> nodes(N + M);

			for (size_t i = 0; i < nodes.size(); ++i)
			{
				nodes[i] = G.new_node();
			}

			edge_map<int> weights(G, 100);
			for (size_t i = 0; i < tracks.size(); i++)
			{
				bool hasZeroEdge = false;

				for (size_t j = 0; j < detections.size(); j++)
				{
					track_t currCost = Cost[i + j * N];

					edge e = G.new_edge(nodes[i], nodes[N + j]);

					if (currCost < dist_thres)
					{
						int weight = maxCost - currCost + 1;
						G.set_edge_weight(e, weight);
						weights[e] = weight;
					}
					else
					{
						if (!hasZeroEdge)
						{
							G.set_edge_weight(e, 0);
							weights[e] = 0;
						}
						hasZeroEdge = true;
					}
				}
			}

			edges_t L = MAX_WEIGHT_BIPARTITE_MATCHING(G, weights);
			for (edges_t::iterator it = L.begin(); it != L.end(); ++it)
			{
				node a = it->source();
				node b = it->target();
				assignment[b.id()] = a.id() - N;
			}
		}

		// -----------------------------------
		// clean assignment from pairs with large distance
		// -----------------------------------
		for (size_t i = 0; i < assignment.size(); i++)
		{
			if (assignment[i] != -1)
			{
				if (Cost[i + assignment[i] * N] > dist_thres)
				{
					assignment[i] = -1;
                    tracks[i]->m_skippedFrames++;
				}
			}
			else
			{
				// If track have no assigned detect, then increment skipped frames counter.
                tracks[i]->m_skippedFrames++;
			}
		}

		// -----------------------------------
        // If track didn't get detects long time, remove it.
        // -----------------------------------
        for (int i = 0; i < static_cast<int>(tracks.size()); i++)
        {
            if (tracks[i]->m_skippedFrames > maximum_allowed_skipped_frames)
            {
                tracks.erase(tracks.begin() + i);
                assignment.erase(assignment.begin() + i);
                i--;
            }
        }
    }

    // -----------------------------------
    // Search for unassigned detects and start new tracks for them.
    // -----------------------------------
    for (size_t i = 0; i < detections.size(); ++i)
    {
        if (find(assignment.begin(), assignment.end(), i) == assignment.end())
        {
            tracks.push_back(std::make_unique<CTrack>(detections[i], regions[i], kalmanType, dt, accelNoiseMag, NextTrackID++, m_filterGoal == FilterRect, m_useExternalTrackerForLostObjects == TrackKCF));
        }
    }

    // Update Kalman Filters state

    for (size_t i = 0; i < assignment.size(); i++)
    {
        // If track updated less than one time, than filter state is not correct.

        if (assignment[i] != -1) // If we have assigned detect, then update using its coordinates,
        {
            tracks[i]->m_skippedFrames = 0;
            tracks[i]->Update(detections[assignment[i]], regions[assignment[i]], true, max_trace_length, m_prevFrame, grayFrame);
        }
        else				     // if not continue using predictions
        {
            tracks[i]->Update(Point_t(), CRegion(), false, max_trace_length, m_prevFrame, grayFrame);
        }
    }

    grayFrame.copyTo(m_prevFrame);
}
