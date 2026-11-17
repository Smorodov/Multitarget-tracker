#pragma once

#include <fstream>
#include <iomanip>
#include <map>
#include <vector>
#include "trajectory.h"

///
/// \brief The MOTResultsWriter class
///
/// Accumulates per-frame tracks and saves them in the MOTChallenge
/// results format: one row per track box
///     frame,id,x,y,w,h,conf,-1,-1,-1
/// 1-based frame numbers, x/y = top-left corner of the bounding box.
/// The same format is accepted by evaluation toolkits for the
/// MOTChallenge (MOT17/20) and VisDrone-MOT datasets, so the tracker can
/// be scored on third-party benchmarks with standard tools (roadmap M3).
///
/// The pipeline feeds 1-based frame indices (VideoExample starts
/// framesCounter at start_frame + 1), which matches the MOT ground truth
/// numbering as is.
///
class MOTResultsWriter
{
public:
	///
	explicit MOTResultsWriter(const std::string& resFileName)
		: m_resFileName(resFileName)
	{
	}

	///
	bool NewDetects(int frameInd, const std::vector<TrackingObject>& tracks, size_t detectorInd)
	{
		if (m_resFileName.empty())
			return false;

		std::vector<TrackingObject>& frameTracks = m_detects[frameInd];
		if (detectorInd == 0)
		{
			frameTracks.insert(frameTracks.end(), tracks.begin(), tracks.end());
		}
		else
		{
			// Multiple detectors in one pipeline: keep their IDs apart
			// (the same convention as CVATAnnotationsGenerator).
			for (auto track : tracks)
			{
				track.m_ID.m_val += detectorInd * DetectorIDRange;
				frameTracks.push_back(track);
			}
		}

		return true;
	}

	///
	bool Save()
	{
		if (m_resFileName.empty())
			return false;

		std::ofstream resFile(m_resFileName);
		if (!resFile.is_open())
			return false;

		resFile << std::fixed << std::setprecision(4);
		for (const auto& [frameInd, tracks] : m_detects)
		{
			for (const auto& track : tracks)
			{
				const cv::Rect rect = track.m_rrect.boundingRect();
				const float conf = track.m_confidence >= 0.f ? track.m_confidence : 1.f;
				resFile << frameInd << ','
					<< track.m_ID.m_val << ','
					<< rect.x << ',' << rect.y << ','
					<< rect.width << ',' << rect.height << ','
					<< conf << ",-1,-1,-1" << std::endl;
			}
		}

		return resFile.good();
	}

private:
	std::string m_resFileName;

	std::map<int, std::vector<TrackingObject>> m_detects;

	static constexpr track_id_t::value_type DetectorIDRange = 1000000000;
};
