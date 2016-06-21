#include "Ctracker.h"

size_t CTrack::NextTrackID = 0;
// ---------------------------------------------------------------------------
// Track constructor.
// The track begins from initial cv::Point (pt)
// ---------------------------------------------------------------------------
CTrack::CTrack(Point_t pt, track_t dt, track_t Accel_noise_mag)
{
	track_id = NextTrackID;

	NextTrackID++;
	// Every track have its own Kalman filter,
	// it user for next cv::Point position prediction.
	KF = new TKalmanFilter(pt, dt, Accel_noise_mag);
	// Here stored cv::Points coordinates, used for next position prediction.
	prediction = pt;
	skipped_frames = 0;
}
// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
CTrack::~CTrack()
{
	// Free resources.
	delete KF;
}

// ---------------------------------------------------------------------------
// Tracker. Manage tracks. Create, remove, update.
// ---------------------------------------------------------------------------
CTracker::CTracker(track_t _dt, track_t _Accel_noise_mag, track_t _dist_thres, int _maximum_allowed_skipped_frames, int _max_trace_length)
{
	dt = _dt;
	Accel_noise_mag = _Accel_noise_mag;
	dist_thres = _dist_thres;
	maximum_allowed_skipped_frames = _maximum_allowed_skipped_frames;
	max_trace_length = _max_trace_length;
}
// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
void CTracker::Update(std::vector<Point_t>& detections)
{
	// -----------------------------------
	// If there is no tracks yet, then every cv::Point begins its own track.
	// -----------------------------------
	if (tracks.size() == 0)
	{
		// If no tracks yet
		for (int i = 0; i < detections.size(); i++)
		{
			tracks.push_back(std::make_unique<CTrack>(detections[i], dt, Accel_noise_mag));
		}
	}

	// -----------------------------------
	// «десь треки уже есть в любом случае
	// -----------------------------------
	size_t N = tracks.size();		// треки
	size_t M = detections.size();	// детекты

	// ћатрица рассто€ний от N-ного трека до M-ного детекта.
	std::vector< std::vector<track_t> > Cost(N, std::vector<track_t>(M));
	std::vector<int> assignment; // назначени€

	// -----------------------------------
	// “реки уже есть, составим матрицу рассто€ний
	// -----------------------------------
	for (int i = 0; i < tracks.size(); i++)
	{
		// Point_t prediction=tracks[i]->prediction;
		// std::cout << prediction << std::endl;
		for (int j = 0; j < detections.size(); j++)
		{
			Point_t diff = (tracks[i]->prediction - detections[j]);
			track_t dist = sqrtf(diff.x*diff.x + diff.y*diff.y);
			Cost[i][j] = dist;
		}
	}
	// -----------------------------------
	// Solving assignment problem (tracks and predictions of Kalman filter)
	// -----------------------------------
	AssignmentProblemSolver APS;
	APS.Solve(Cost, assignment, AssignmentProblemSolver::optimal);

	// -----------------------------------
	// clean assignment from pairs with large distance
	// -----------------------------------
	// Not assigned tracks
	std::vector<int> not_assigned_tracks;

	for (int i = 0; i<assignment.size(); i++)
	{
		if (assignment[i] != -1)
		{
			if (Cost[i][assignment[i]]>dist_thres)
			{
				assignment[i] = -1;
				// Mark unassigned tracks, and increment skipped frames counter,
				// when skipped frames counter will be larger than threshold, track will be deleted.
				not_assigned_tracks.push_back(i);
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
	for (int i = 0; i < tracks.size(); i++)
	{
		if (tracks[i]->skipped_frames > maximum_allowed_skipped_frames)
		{
			tracks.erase(tracks.begin() + i);
			assignment.erase(assignment.begin() + i);
			i--;
		}
	}
	// -----------------------------------
	// Search for unassigned detects
	// -----------------------------------
	std::vector<int> not_assigned_detections;
	std::vector<int>::iterator it;
	for (int i = 0; i < detections.size(); i++)
	{
		it = find(assignment.begin(), assignment.end(), i);
		if (it == assignment.end())
		{
			not_assigned_detections.push_back(i);
		}
	}

	// -----------------------------------
	// and start new tracks for them.
	// -----------------------------------
	if (not_assigned_detections.size() != 0)
	{
		for (int i = 0; i < not_assigned_detections.size(); i++)
		{
			tracks.push_back(std::make_unique<CTrack>(detections[not_assigned_detections[i]], dt, Accel_noise_mag));
		}
	}

	// Update Kalman Filters state

	for (int i = 0; i<assignment.size(); i++)
	{
		// If track updated less than one time, than filter state is not correct.

		tracks[i]->KF->GetPrediction();

		if (assignment[i] != -1) // If we have assigned detect, then update using its coordinates,
		{
			tracks[i]->skipped_frames = 0;
			tracks[i]->prediction = tracks[i]->KF->Update(detections[assignment[i]], 1);
		}
		else				  // if not continue using predictions
		{
			tracks[i]->prediction = tracks[i]->KF->Update(Point_t(0, 0), 0);
		}

		if (tracks[i]->trace.size()>max_trace_length)
		{
			tracks[i]->trace.erase(tracks[i]->trace.begin(), tracks[i]->trace.end() - max_trace_length);
		}

		tracks[i]->trace.push_back(tracks[i]->prediction);
		tracks[i]->KF->LastResult = tracks[i]->prediction;
	}

}
// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
CTracker::~CTracker(void)
{
}
