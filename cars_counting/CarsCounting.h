#pragma once

#include "BaseDetector.h"

#include "Ctracker.h"
#include <iostream>
#include <vector>
#include <map>

// ----------------------------------------------------------------------

///
/// \brief The RoadLine struct
///
class RoadLine
{
public:
    ///
    /// \brief RoadLine
    ///
    RoadLine()
    {
    }
    RoadLine(const cv::Point2f& pt1, const cv::Point2f& pt2, unsigned int uid)
        :
          m_pt1(pt1), m_pt2(pt2), m_uid(uid)
    {
    }

    cv::Point2f m_pt1;
    cv::Point2f m_pt2;

    unsigned int m_uid = 0;

    int m_intersect1 = 0;
    int m_intersect2 = 0;

    ///
    /// \brief operator ==
    /// \param line
    /// \return
    ///
    bool operator==(const RoadLine &line) const
    {
        return line.m_uid == m_uid;
    }

    ///
    /// \brief Draw
    /// \param frame
    ///
    void Draw(cv::Mat frame) const
    {
        auto Ptf2i = [&](cv::Point2f pt) -> cv::Point
        {
            return cv::Point(cvRound(frame.cols * pt.x), cvRound(frame.rows * pt.y));
        };

        cv::line(frame, Ptf2i(m_pt1), Ptf2i(m_pt2), cv::Scalar(0, 255, 255), 1, cv::LINE_8, 0);

        std::string label = "Line " + std::to_string(m_uid) + ": " + std::to_string(m_intersect1) + "/" + std::to_string(m_intersect2);
        //int baseLine = 0;
        //cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::putText(frame, label, Ptf2i(0.5f * (m_pt1 + m_pt2)), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    ///
    /// \brief IsIntersect
    /// \param pt1
    /// \param pt2
    /// \return
    ///
    int IsIntersect(cv::Point2f pt1, cv::Point2f pt2)
    {
        bool isIntersect = CheckIntersection(pt1, pt2);
        int direction = 0;

        if (isIntersect)
        {
            cv::Point2f pt;
            if ((m_pt1.x <= m_pt2.x) && (m_pt1.y > m_pt1.y))
            {
                pt.x = (m_pt1.x + m_pt2.x) / 2.f - 0.01f;
                pt.y = (m_pt1.y + m_pt1.y) / 2.f - 0.01f;
            }
            else
            {
                if ((m_pt1.x <= m_pt2.x) && (m_pt1.y <= m_pt1.y))
                {
                    pt.x = (m_pt1.x + m_pt2.x) / 2.f + 0.01f;
                    pt.y = (m_pt1.y + m_pt1.y) / 2.f - 0.01f;
                }
                else
                {
                    if ((m_pt1.x > m_pt2.x) && (m_pt1.y > m_pt1.y))
                    {
                        pt.x = (m_pt1.x + m_pt2.x) / 2.f - 0.01f;
                        pt.y = (m_pt1.y + m_pt1.y) / 2.f + 0.01f;
                    }
                    else
                    {
                        if ((m_pt1.x > m_pt2.x) && (m_pt1.y <= m_pt1.y))
                        {
                            pt.x = (m_pt1.x + m_pt2.x) / 2.f + 0.01f;
                            pt.y = (m_pt1.y + m_pt1.y) / 2.f + 0.01f;
                        }
                    }
                }
            }
            if (CheckIntersection(pt1, pt))
            {
                direction = 1;
                ++m_intersect1;
            }
            else
            {
                direction = 2;
                ++m_intersect2;
            }
        }

        return direction;
    }

private:

    ///
    /// \brief CheckIntersection
    /// \param pt1
    /// \param pt2
    /// \return
    ///
    bool CheckIntersection(cv::Point2f pt1, cv::Point2f pt2) const
    {
        const float eps = 0.00001f; // Epsilon for equal comparing

        // First line equation
        float a1 = 0;
        float b1 = 0;
        bool trivial1 = false; // Is first line is perpendicular with OX

        if (fabs(m_pt1.x - m_pt2.x) < eps)
        {
            trivial1 = true;
        }
        else
        {
            a1 = (m_pt2.y - m_pt1.y) / (m_pt2.x - m_pt1.x);
            b1 = (m_pt2.x * m_pt1.y - m_pt1.x * m_pt2.y) / (m_pt2.x - m_pt1.x);
        }

        // Second line equation
        float a2 = 0;
        float b2 = 0;
        bool trivial2 = false; // Is second line is perpendicular with OX

        if (fabs(pt1.x - pt2.x) < eps)
        {
            trivial2 = true;
        }
        else
        {
            a2 = (pt2.y - pt1.y) / (pt2.x - pt1.x);
            b2 = (pt2.x * pt1.y - pt1.x * pt2.y) / (pt2.x - pt1.x);
        }

        //Определение координат пересечения прямых
        cv::Point2f intersectPt;

        bool isIntersect = true;
        if (trivial1)
        {
            if (trivial2)
            {
                isIntersect = (fabs(m_pt1.x - pt1.x) < eps);
            }
            else
            {
                intersectPt.x = m_pt1.x;
            }
            intersectPt.y = a2 * intersectPt.x + b2;
        }
        else
        {
            if (trivial2)
            {
                intersectPt.x = pt1.x;
            }
            else
            {
                if (fabs(a2 - a1) > eps)
                {
                    intersectPt.x = (b1 - b2) / (a2 - a1);
                }
                else
                {
                    isIntersect = false;
                }
            }
            intersectPt.y = a1 * intersectPt.x + b1;
        }

        if (isIntersect)
        {
            auto InRange = [](float val, float minVal, float  maxVal) -> bool
            {
                return (val >= minVal) && (val <= maxVal);
            };

            isIntersect = InRange(intersectPt.x, std::min(m_pt1.x, m_pt2.x), std::max(m_pt1.x, m_pt2.x) + eps) &&
                    InRange(intersectPt.x, std::min(pt1.x, pt2.x), std::max(pt1.x, pt2.x) + eps) &&
                    InRange(intersectPt.y, std::min(m_pt1.y, m_pt2.y), std::max(m_pt1.y, m_pt2.y) + eps) &&
                    InRange(intersectPt.y, std::min(pt1.y, pt2.y), std::max(pt1.y, pt2.y) + eps);
        }

        return isIntersect;
    }
};
// ----------------------------------------------------------------------

///
/// \brief The CarsCounting class
///
class CarsCounting
{
public:
    CarsCounting(const cv::CommandLineParser& parser);
    virtual ~CarsCounting();

    void Process();

    // Lines API
    void AddLine(const RoadLine& newLine);
    bool GetLine(unsigned int lineUid, RoadLine& line);
    bool RemoveLine(unsigned int lineUid);

protected:
    std::unique_ptr<BaseDetector> m_detector;
    std::unique_ptr<CTracker> m_tracker;

    bool m_showLogs = false;
    float m_fps = 0;
    bool m_useLocalTracking = false;

    virtual bool GrayProcessing() const;

    virtual bool InitTracker(cv::UMat frame);

    virtual void DrawData(cv::Mat frame, int framesCounter, int currTime);

    void DrawTrack(cv::Mat frame,
                   int resizeCoeff,
                   const TrackingObject& track,
                   bool drawTrajectory = true);

private:

    bool m_isTrackerInitialized = false;
    std::string m_inFile;
    std::string m_outFile;
    int m_startFrame = 0;
    int m_endFrame = 0;
    int m_finishDelay = 0;
    std::vector<cv::Scalar> m_colors;

    int m_minObjWidth = 10;

    // Road lines
    std::deque<RoadLine> m_lines;
    void CheckLinesIntersection(const TrackingObject& track, float xMax, float yMax, std::set<size_t>& currIntersections);
    std::set<size_t> m_lastIntersections;
};
