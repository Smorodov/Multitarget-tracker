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
        cv::Mat gray_frame
        )
{
    assert(detections.size() == regions.size());

    if (m_useLocalTracking)
    {
        localTracker.Update(tracks, gray_frame);
    }

    // -----------------------------------
    // If there is no tracks yet, then every cv::Point begins its own track.
    // -----------------------------------
    if (tracks.size() == 0)
    {
        // If no tracks yet
        for (size_t i = 0; i < detections.size(); ++i)
        {
            tracks.push_back(std::make_unique<CTrack>(detections[i], regions[i], dt, accelNoiseMag, NextTrackID++, m_kalmanType == FilterRect));
        }
    }

    size_t N = tracks.size();		// треки
    size_t M = detections.size();	// детекты

    assignments_t assignment; // назначения

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
#if 0
		AssignmentProblemSolver APS;
        APS.Solve(Cost, N, M, assignment, AssignmentProblemSolver::optimal);
#else
		assignment.resize(N, -1);

		MyGraph G;
		G.make_directed();

		std::vector<node> nodes(N + M);

		for (size_t i = 0; i < nodes.size(); ++i)
		{
			nodes[i] = G.new_node();
		}

		for (size_t i = 0; i < tracks.size(); i++)
		{
			int maxDist = 0;
			bool haveEdge = false;
			int maxj = 0;
			for (size_t j = 0; j < detections.size(); j++)
			{
				track_t currCost = Cost[i + j * N];

				if (currCost < dist_thres)
				{
					int dist = maxCost - currCost  + 1;

					edge e = G.new_edge(nodes[i], nodes[N + j]);
					G.set_edge_weight(e, dist);

					haveEdge = true;

					if (dist > maxDist)
					{

					}
				}
			}
			if (!haveEdge)
			{

			}
		}

		edge_map<int> weights(G, 100);
		for (graph::edge_iterator eit = G.edges_begin(), eend = G.edges_end(); eit != eend; ++eit)
		{
			weights[*eit] = G.get_edge_weight(*eit);
		}

		list<edge> L = MAX_WEIGHT_BIPARTITE_MATCHING(G, weights);

		list <edge> edges;
		for (graph::edge_iterator eit = G.edges_begin(), eend = G.edges_end(); eit != eend; ++eit)
		{
			edges.push_back(*eit);
		}

		list<edge>::iterator lit = edges.begin();
		list<edge>::iterator lend = edges.end();
		while (lit != lend)
		{
			G.hide_edge(*lit);
			lit++;
		}

		for (list<edge>::iterator it = L.begin(); it != L.end(); ++it)
		{
			edge e = *it;
			G.restore_edge(e);

			node a = e.source();
			node b = e.target();

			assignment[b.id()] = a.id() - N;
		}
#endif


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
					tracks[i]->skipped_frames++;
				}
			}
			else
			{
				// If track have no assigned detect, then increment skipped frames counter.
				tracks[i]->skipped_frames++;
			}
		}

		// -----------------------------------
        // If track didn't get detects long time, remove it.
        // -----------------------------------
        for (int i = 0; i < static_cast<int>(tracks.size()); i++)
        {
            if (tracks[i]->skipped_frames > maximum_allowed_skipped_frames)
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
            tracks.push_back(std::make_unique<CTrack>(detections[i], regions[i], dt, accelNoiseMag, NextTrackID++, m_kalmanType == FilterRect));
        }
    }

    // Update Kalman Filters state

    for (size_t i = 0; i<assignment.size(); i++)
    {
        // If track updated less than one time, than filter state is not correct.

        if (assignment[i] != -1) // If we have assigned detect, then update using its coordinates,
        {
            tracks[i]->skipped_frames = 0;
            tracks[i]->Update(detections[assignment[i]], regions[assignment[i]], true, max_trace_length);
        }
        else				     // if not continue using predictions
        {
            tracks[i]->Update(Point_t(), CRegion(), false, max_trace_length);
        }
    }
}
