#pragma once

#include <memory>
#include <map>
#include <string>
#include "defines.h"

///
/// \brief The BaseDetector class
///
class BaseDetector
{
public:
    ///
    /// \brief BaseDetector
    /// \param collectPoints
    /// \param frame
    ///
    BaseDetector(bool collectPoints, cv::UMat& frame)
        : m_collectPoints(collectPoints)
    {
        m_minObjectSize.width = std::max(5, frame.cols / 100);
        m_minObjectSize.height = m_minObjectSize.width;
    }
    ///
    /// \brief ~BaseDetector
    ///
    virtual ~BaseDetector(void)
    {
    }

    typedef std::map<std::string, std::string> config_t;
    ///
    /// \brief Init
    /// \param config
    ///
    virtual bool Init(const config_t& config) = 0;

    ///
    /// \brief Detect
    /// \param frame
    ///
    virtual void Detect(cv::UMat& frame) = 0;

    ///
    /// \brief SetMinObjectSize
    /// \param minObjectSize
    ///
    void SetMinObjectSize(cv::Size minObjectSize)
    {
        m_minObjectSize = minObjectSize;
    }

    ///
    /// \brief GetDetects
    /// \return
    ///
    const regions_t& GetDetects() const
    {
        return m_regions;
    }

    ///
    /// \brief CollectPoints
    /// \param region
    ///
    virtual void CollectPoints(CRegion& region)
    {
        const int yStep = 5;
        const int xStep = 5;

        for (int y = region.m_rect.y, yStop = region.m_rect.y + region.m_rect.height; y < yStop; y += yStep)
        {
            for (int x = region.m_rect.x, xStop = region.m_rect.x + region.m_rect.width; x < xStop; x += xStep)
            {
                if (region.m_rect.contains(cv::Point(x, y)))
                {
                    region.m_points.push_back(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
                }
            }
        }

        if (region.m_points.empty())
        {
            region.m_points.push_back(cv::Point2f(region.m_rect.x + 0.5f * region.m_rect.width, region.m_rect.y + 0.5f * region.m_rect.height));
        }
    }

    ///
    /// \brief CalcMotionMap
    /// \param frame
    ///
    virtual void CalcMotionMap(cv::Mat frame)
    {
        if (m_motionMap.size() != frame.size())
        {
            m_motionMap = cv::Mat(frame.size(), CV_32FC1, cv::Scalar(0, 0, 0));
        }

        cv::Mat foreground(m_motionMap.size(), CV_8UC1, cv::Scalar(0, 0, 0));
        for (const auto& region : m_regions)
        {
            cv::ellipse(foreground,
                        cv::RotatedRect((region.m_rect.tl() + region.m_rect.br()) / 2, region.m_rect.size(), 0),
                                        cv::Scalar(255, 255, 255), CV_FILLED);
        }

        cv::Mat normFor;
        cv::normalize(foreground, normFor, 255, 0, cv::NORM_MINMAX, m_motionMap.type());

        double alpha = 0.95;
        cv::addWeighted(m_motionMap, alpha, normFor, 1 - alpha, 0, m_motionMap);

        const int chans = frame.channels();

        for (int y = 0; y < frame.rows; ++y)
        {
            uchar* imgPtr = frame.ptr(y);
            float* moPtr = reinterpret_cast<float*>(m_motionMap.ptr(y));
            for (int x = 0; x < frame.cols; ++x)
            {
                for (int ci = chans - 1; ci < chans; ++ci)
                {
                    imgPtr[ci] = cv::saturate_cast<uchar>(imgPtr[ci] + moPtr[0]);
                }
                imgPtr += chans;
                ++moPtr;
            }
        }
    }

protected:
    regions_t m_regions;

    cv::Size m_minObjectSize;

    bool m_collectPoints;

    cv::Mat m_motionMap;
};


///
/// \brief CreateDetector
/// \param detectorType
/// \param collectPoints
/// \param gray
/// \return
///
BaseDetector* CreateDetector(tracking::Detectors detectorType, const BaseDetector::config_t& config, bool collectPoints, cv::UMat& gray);
