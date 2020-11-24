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
    BaseDetector(const cv::UMat& frame)
    {
        m_minObjectSize.width = std::max(5, frame.cols / 100);
        m_minObjectSize.height = m_minObjectSize.width;
    }
    ///
    /// \brief ~BaseDetector
    ///
    virtual ~BaseDetector(void) = default;

    ///
    /// \brief Init
    /// \param config
    ///
    virtual bool Init(const config_t& config) = 0;

    ///
    /// \brief Detect
    /// \param frame
    ///
    virtual void Detect(const cv::UMat& frame) = 0;

	///
	/// \brief ResetModel
	/// \param img
	/// \param roiRect
	///
	virtual void ResetModel(const cv::UMat& /*img*/, const cv::Rect& /*roiRect*/)
	{
	}

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
    virtual void CalcMotionMap(cv::Mat& frame)
    {
        if (m_motionMap.size() != frame.size())
            m_motionMap = cv::Mat(frame.size(), CV_32FC1, cv::Scalar(0, 0, 0));

        cv::Mat foreground(m_motionMap.size(), CV_8UC1, cv::Scalar(0, 0, 0));
        for (const auto& region : m_regions)
        {
#if (CV_VERSION_MAJOR < 4)
            cv::ellipse(foreground, region.m_rrect, cv::Scalar(255, 255, 255), CV_FILLED);
#else
            cv::ellipse(foreground, region.m_rrect, cv::Scalar(255, 255, 255), cv::FILLED);
#endif
        }

        cv::normalize(foreground, m_normFor, 255, 0, cv::NORM_MINMAX, m_motionMap.type());

        double alpha = 0.95;
        cv::addWeighted(m_motionMap, alpha, m_normFor, 1 - alpha, 0, m_motionMap);

        const int chans = frame.channels();
		const int height = frame.rows;
#pragma omp parallel for
        for (int y = 0; y < height; ++y)
        {
            uchar* imgPtr = frame.ptr(y);
            const float* moPtr = reinterpret_cast<float*>(m_motionMap.ptr(y));
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

	// Motion map for visualization current detections
    cv::Mat m_motionMap;
	cv::Mat m_normFor;

	std::set<objtype_t> m_classesWhiteList;

    std::vector<cv::Rect> GetCrops(float maxCropRatio, cv::Size netSize, cv::Size imgSize) const
    {
        std::vector<cv::Rect> crops;

        const float whRatio = static_cast<float>(netSize.width) / static_cast<float>(netSize.height);
        int cropHeight = cvRound(maxCropRatio * netSize.height);
        int cropWidth = cvRound(maxCropRatio * netSize.width);

        if (imgSize.width / (float)imgSize.height > whRatio)
        {
            if (cropHeight >= imgSize.height)
                cropHeight = imgSize.height;
            cropWidth = cvRound(cropHeight * whRatio);
        }
        else
        {
            if (cropWidth >= imgSize.width)
                cropWidth = imgSize.width;
            cropHeight = cvRound(cropWidth / whRatio);
        }

        //std::cout << "Frame size " << imgSize << ", crop size = " << cv::Size(cropWidth, cropHeight) << ", ratio = " << maxCropRatio << std::endl;

        const int stepX = 3 * cropWidth / 4;
        const int stepY = 3 * cropHeight / 4;
        for (int y = 0; y < imgSize.height; y += stepY)
        {
            bool needBreakY = false;
            if (y + cropHeight >= imgSize.height)
            {
                y = imgSize.height - cropHeight;
                needBreakY = true;
            }
            for (int x = 0; x < imgSize.width; x += stepX)
            {
                bool needBreakX = false;
                if (x + cropWidth >= imgSize.width)
                {
                    x = imgSize.width - cropWidth;
                    needBreakX = true;
                }
                crops.emplace_back(x, y, cropWidth, cropHeight);
                if (needBreakX)
                    break;
            }
            if (needBreakY)
                break;
        }
        return crops;
    }

	///
	bool FillTypesMap(const std::vector<std::string>& classNames)
	{
		bool res = true;

		m_typesMap.resize(classNames.size(), bad_type);
		for (size_t i = 0; i < classNames.size(); ++i)
		{
			objtype_t type = TypeConverter::Str2Type(classNames[i]);
			m_typesMap[i] = type;
			res = (type != bad_type);
		}
		return res;
	}
	
	///
	objtype_t T2T(size_t typeInd) const
	{
		if (typeInd < m_typesMap.size())
			return m_typesMap[typeInd];
		else
			return bad_type;
	}

private:
    std::vector<objtype_t> m_typesMap;
};


///
/// \brief CreateDetector
/// \param detectorType
/// \param gray
/// \return
///
BaseDetector* CreateDetector(tracking::Detectors detectorType, const config_t& config, cv::UMat& gray);
