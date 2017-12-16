#pragma once

#include <memory>
#include "defines.h"

///
/// \brief The BaseDetector class
///
class BaseDetector
{
public:
    BaseDetector(bool collectPoints, cv::UMat& gray)
        : m_collectPoints(collectPoints)
    {
        m_minObjectSize.width = std::max(5, gray.cols / 100);
        m_minObjectSize.height = m_minObjectSize.width;
    }
    virtual ~BaseDetector(void)
    {
    }

    virtual void Detect(cv::UMat& gray) = 0;

    void SetMinObjectSize(cv::Size minObjectSize)
    {
        m_minObjectSize = minObjectSize;
    }

    const regions_t& GetDetects() const
    {
        return m_regions;
    }

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
                                        cv::Scalar(255, 255, 255), -1);
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
BaseDetector* CreateDetector(tracking::Detectors detectorType, bool collectPoints, cv::UMat& gray);
