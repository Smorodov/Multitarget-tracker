#pragma once
#include <fstream>
#include <map>
#include <unordered_set>

#include <opencv2/opencv.hpp>

#include "object_types.h"

///
/// \brief The ResultsLog class
///
class ResultsLog
{
public:
	///
	ResultsLog(const std::string& fileName, int writeEachNFrame)
		: m_fileName(fileName), m_writeEachNFrame(writeEachNFrame)
	{
	}

	///
	~ResultsLog()
	{
		WriteAll(true);
	}

	///
	bool Open()
	{
		m_resCSV.close();
		if (m_fileName.size() > 5)
		{
			m_resCSV.open(m_fileName);
			return m_resCSV.is_open();
		}
		return false;
	}

	///
	bool AddTrack(int framesCounter, track_id_t trackID, const cv::Rect& brect, objtype_t type, float confidence)
	{
		if (m_resCSV.is_open())
		{
			auto frame = m_frames.find(framesCounter);
			if (frame == std::end(m_frames))
			{
				DetectsOnFrame tmpFrame;
				tmpFrame.m_detects.emplace_back(trackID, brect, type, confidence);
				m_frames.emplace(framesCounter, tmpFrame);
			}
			else
			{
				frame->second.m_detects.emplace_back(trackID, brect, type, confidence);
			}
			return true;
		}
		return false;
	}

	///
	void AddRobustTrack(track_id_t trackID)
	{
		m_robustIDs.insert(trackID);
	}

    ///
    void Flush()
    {
        WriteAll(true);
        m_frames.clear();
    }

private:
	std::string m_fileName;
	std::ofstream m_resCSV;

	///
	struct Detection
	{
		cv::Rect m_rect;
		objtype_t m_type;
		float m_conf = 0.f;
		track_id_t m_trackID = 0;

		Detection(track_id_t trackID, const cv::Rect& brect, objtype_t type, float confidence)
		{
			m_type = type;
			m_rect = brect;
			m_conf = confidence;
			m_trackID = trackID;
		}
	};

	///
	struct DetectsOnFrame
	{
		std::vector<Detection> m_detects;
	};
	std::map<int, DetectsOnFrame> m_frames;
	std::unordered_set<track_id_t> m_robustIDs;
    int m_writeEachNFrame = 1;

	///
	void WriteAll(bool byFrames)
	{
		if (byFrames)
		{
#if 1
			char delim = ',';
			for (const auto& frame : m_frames)
			{
                if (frame.first % m_writeEachNFrame == 0)
                {
                    for (const auto& detect : frame.second.m_detects)
                    {
                        if (m_robustIDs.find(detect.m_trackID) != std::end(m_robustIDs))
                        {
                            m_resCSV << frame.first << delim << TypeConverter::Type2Str(detect.m_type) << delim << detect.m_rect.x << delim << detect.m_rect.y << delim <<
                                detect.m_rect.width << delim << detect.m_rect.height << delim <<
                                detect.m_conf << delim << std::endl;
                        }
                    }
                }
            }
#else
			char delim = '	';
			for (const auto& frame : m_frames)
			{
				for (const auto& detect : frame.second.m_detects)
				{
					if (m_robustIDs.find(detect.m_trackID) != std::end(m_robustIDs))
					{
						m_resCSV << frame.first << delim << TypeConverter::Type2Str(detect.m_type) << delim << detect.m_rect.x << delim << detect.m_rect.y << delim <<
							(detect.m_rect.x + detect.m_rect.width) << delim << (detect.m_rect.y + detect.m_rect.height) << delim <<
							detect.m_conf << delim << detect.m_trackID << std::endl;
					}
				}
			}
#endif
		}
		else
		{
			char delim = ',';
			for (auto id : m_robustIDs)
            {
                for (const auto& frame : m_frames)
                {
                    if (frame.first % m_writeEachNFrame == 0)
                    {
                        for (const auto& detect : frame.second.m_detects)
                        {
                            if (detect.m_trackID == id)
                            {
                                m_resCSV << frame.first << delim << id.ID2Str() << delim << detect.m_rect.x << delim << detect.m_rect.y << delim <<
                                    detect.m_rect.width << delim << detect.m_rect.height << delim <<
                                    detect.m_conf << ",-1,-1,-1," << std::endl;
                                break;
                            }
                        }
                    }
                }
            }
		}
	}
};
