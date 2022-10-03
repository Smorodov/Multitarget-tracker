#pragma once

#include <fstream>
#include <unordered_set>
#include "defines.h"

///
class CVATAnnotationsGenerator
{
public:
	///
	CVATAnnotationsGenerator(const std::string& annFileName)
		: m_annFileName(annFileName)
	{
	}

	///
	bool NewDetects(int frameInd, const std::vector<TrackingObject>& tracks, size_t detectorInd)
	{
		if (m_annFileName.empty())
			return false;

		auto it = m_detects.find(frameInd);
		if (it == m_detects.end())
		{
			if (detectorInd == 0)
			{
				m_detects.emplace(frameInd, tracks);
			}
			else
			{
				std::vector<TrackingObject> tmpTracks = tracks;
				for (auto& track : tmpTracks)
				{
					track.m_ID.m_val += detectorInd * DetectorIDRange;
				}
				m_detects.emplace(frameInd, tmpTracks);
			}

			//it = m_detects.find(frameInd);
			//std::cout << "New detects 1: Frame " << frameInd << ", detector ind " << detectorInd << std::endl;
			//for (const auto& track : it->second)
			//{
			//	std::cout << "track  " << track.m_ID.ID2Str() << ", type = " << track.m_type << ", rect = " << track.m_rrect.boundingRect() << std::endl;
			//}
		}
		else
		{
			if (detectorInd == 0)
			{
				it->second.insert(it->second.end(), tracks.begin(), tracks.end());
			}
			else
			{
				std::vector<TrackingObject> tmpTracks = tracks;
				for (auto& track : tmpTracks)
				{
					track.m_ID.m_val += detectorInd * DetectorIDRange;
				}
				it->second.insert(it->second.end(), tmpTracks.begin(), tmpTracks.end());
			}

			//std::cout << "New detects 2: Frame " << frameInd << ", detector ind " << detectorInd << std::endl;
			//for (const auto& track : it->second)
			//{
			//	std::cout << "track  " << track.m_ID.ID2Str() << ", type = " << track.m_type << ", rect = " << track.m_rrect.boundingRect() << std::endl;
			//}
		}

		return true;
	}

	///
	bool Save(const std::string& videoFileName, int framesCount, cv::Size frameSize)
	{
		//PrintDetects();

		bool res = !m_annFileName.empty();
		if (!res)
			return res;

		std::ofstream annFile(m_annFileName);
		res = annFile.is_open();
		if (!res)
			return res;

		WriteMeta(annFile, videoFileName, framesCount, frameSize);

		auto WritePoly = [&](int frameInd, const cv::RotatedRect& rrect)
		{
			cv::Point2f pts[4];
			rrect.points(pts);
			annFile << "    <polygon frame=\"" << frameInd << "\" outside=\"0\" occluded=\"0\" keyframe=\"1\" points=\""
				<< pts[0].x << "," << pts[0].y << ";"
				<< pts[1].x << "," << pts[1].y << ";"
				<< pts[2].x << "," << pts[2].y << ";"
				<< pts[3].x << "," << pts[3].y << "\" z_order=\"0\">\n";
			annFile << "    </polygon>\n";
		};

		std::unordered_set<track_id_t::value_type> writedTracks;

		for (auto itStartFrame = std::begin(m_detects); itStartFrame != std::end(m_detects); ++itStartFrame)
		{
			for (const auto& track : itStartFrame->second)
			{
				if (writedTracks.find(track.m_ID.m_val) != std::end(writedTracks))
					continue;
				writedTracks.emplace(track.m_ID.m_val);

				annFile << "  <track id=\"" << track.m_ID.m_val << "\" label=\"" << TypeConverter::Type2Str(track.m_type) << "\" source=\"manual\">\n";
				WritePoly(itStartFrame->first, track.m_rrect);

				//std::cout << "track  " << track.m_ID.ID2Str() << ", type = " << track.m_type << ", rect = " << track.m_rrect.boundingRect() << std::endl;

				auto itNextFrame = itStartFrame;
				for (++itNextFrame; itNextFrame != std::end(m_detects); ++itNextFrame)
				{
					for (const auto& subTrack : itNextFrame->second)
					{
						if (track.m_ID.m_val != subTrack.m_ID.m_val)
							continue;

						WritePoly(itNextFrame->first, subTrack.m_rrect);

						//std::cout << "subTrack  " << subTrack.m_ID.ID2Str() << ", type = " << subTrack.m_type << ", rect = " << subTrack.m_rrect.boundingRect() << "\n";
						break;
					}
				}

				annFile << "  </track>\n";
			}
		}

		FinalMeta(annFile);

		return res;
	}

private:
	std::string m_annFileName;

	std::map<int, std::vector<TrackingObject>> m_detects;

	static constexpr track_id_t::value_type DetectorIDRange = 1000000000;

	///
	void PrintDetects()
	{
		std::cout << "Print detects:\n";
		for (auto it = m_detects.begin(); it != m_detects.end(); ++it)
		{
			std::cout << "Frame " << it->first << ": \n";
			for (const auto track : it->second)
			{
				std::cout << "track  " << track.m_ID.ID2Str() << ", type = " << track.m_type << ", rect = " << track.m_rrect.boundingRect() << "\n";
			}
		}
		std::cout.flush();
	}

	///
	template<typename TimePoint>
	std::string Time2Str(TimePoint now)
	{
		// get number of milliseconds for the current second
		// (remainder after division into seconds)
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

		// convert to std::time_t in order to convert to std::tm (broken time)
		auto timer = std::chrono::system_clock::to_time_t(now);

		// convert to broken time
#ifdef WIN32
		std::tm bt;
		localtime_s(&bt, &timer);
#else
		std::tm bt = *std::localtime(&timer);
#endif

		std::ostringstream oss;
		oss << std::put_time(&bt, "%Y-%m-%d %H:%M:%S");
		oss << '.' << std::setfill('0') << std::setw(3) << ms.count() << "+00:00";

		return oss.str();
	}

	///
	void WriteMeta(std::ofstream& annFile, const std::string& videoFileName, int framesCount, cv::Size frameSize)
	{
		std::string currTime = Time2Str(std::chrono::system_clock::now());

		annFile << "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n";
		annFile << "<annotations>\n";
		annFile << "  <version>1.1</version>\n";
		annFile << "  <meta>\n";
		annFile << "    <task>\n";
		annFile << "      <id>777</id>\n";
		annFile << "      <name>" << videoFileName << "</name>\n";
		annFile << "      <size>" << framesCount << "</size>\n";
		annFile << "      <mode>interpolation</mode>\n";
		annFile << "      <overlap>5</overlap>\n";
		annFile << "      <bugtracker></bugtracker>\n";
		annFile << "      <created>" << currTime << "</created>\n";
		annFile << "      <updated>" << currTime << "</updated>\n";
		annFile << "      <subset>default</subset>\n";
		annFile << "      <start_frame>" << 0 << "</start_frame>\n";
		annFile << "      <stop_frame>" << (framesCount - 1) << "</stop_frame>\n";
		annFile << "      <frame_filter></frame_filter>\n";
		annFile << "      <segments>\n";
		annFile << "        <segment>\n";
		annFile << "          <id>777</id>\n";
		annFile << "          <start>" << 0 << "</start>\n";
		annFile << "          <stop>" << (framesCount - 1) << "</stop>\n";
		annFile << "          <url>http://127.0.0.1:8080/?id=777</url>\n";
		annFile << "        </segment>\n";
		annFile << "      </segments>\n";
		annFile << "      <owner>\n";
		annFile << "        <username>user</username>\n";
		annFile << "        <email>user@de-id.ca</email>\n";
		annFile << "      </owner>\n";
		annFile << "      <assignee></assignee>\n";
		annFile << "      <labels>\n";
		annFile << "        <label>\n";
		annFile << "          <name>face</name>\n";
		annFile << "          <color>#906080</color>\n";
		annFile << "          <attributes>\n";
		annFile << "          </attributes>\n";
		annFile << "        </label>\n";
		annFile << "        <label>\n";
		annFile << "          <name>licence_plate</name>\n";
		annFile << "          <color>#d055ce</color>\n";
		annFile << "          <attributes>\n";
		annFile << "          </attributes>\n";
		annFile << "        </label>\n";
		annFile << "      </labels>\n";
		annFile << "      <original_size>\n";
		annFile << "        <width>" << frameSize.width << "</width>\n";
		annFile << "        <height>" << frameSize.height << "</height>\n";
		annFile << "      </original_size>\n";
		annFile << "    </task>\n";
		annFile << "    <dumped>" << currTime << "</dumped>\n";
		annFile << "    <source>" << videoFileName << "</source>\n";
		annFile << "  </meta>\n";
	}

	///
	void FinalMeta(std::ofstream& annFile)
	{
		annFile << "</annotations>\n";
	}
};
