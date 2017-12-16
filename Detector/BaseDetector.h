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

    virtual void CalcMotionMap(cv::Mat frame) = 0;

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
