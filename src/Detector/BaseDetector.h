#pragma once

#include <memory>
#include "defines.h"

///
/// \brief The BaseDetector class
///
class BaseDetector
{
public:
    ///
    /// \brief BaseDetector
    /// \param frame
    ///
    BaseDetector(cv::UMat& frame)
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
	/// \brief CanGrayProcessing
	///
	virtual bool CanGrayProcessing() const = 0;

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
#if (CV_VERSION_MAJOR < 4)
            cv::ellipse(foreground, region.m_rrect, cv::Scalar(255, 255, 255), CV_FILLED);
#else
            cv::ellipse(foreground, region.m_rrect, cv::Scalar(255, 255, 255), cv::FILLED);
#endif
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

    cv::Mat m_motionMap;
};


///
/// \brief CreateDetector
/// \param detectorType
/// \param gray
/// \return
///
BaseDetector* CreateDetector(tracking::Detectors detectorType, const config_t& config, cv::UMat& gray);
