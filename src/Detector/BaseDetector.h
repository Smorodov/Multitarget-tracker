#pragma once

#include <memory>
#include "defines.h"

///
/// \brief The KeyVal struct
///
struct KeyVal
{
    KeyVal() = default;
    void Add(const std::string& key, const std::string& val)
    {
        m_config.emplace_back(key, val);
    }

    std::vector<std::pair<std::string, std::string>> m_config;
};

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
    BaseDetector()
    {
        m_minObjectSize.width = 5;
        m_minObjectSize.height = m_minObjectSize.width;
    }
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
    /// \brief BaseDetector
    /// \param frame
    ///
    BaseDetector(const cv::Mat& frame)
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
    virtual void DetectMat(cv::Mat frame)
    {
        cv::UMat um = frame.getUMat(cv::ACCESS_READ);
        return Detect(um);
    }

    ///
    /// \brief Detect
    /// \param frames
    /// \param regions
    ///
    virtual void Detect(const std::vector<cv::UMat>& frames, std::vector<regions_t>& regions)
    {
        for (size_t i = 0; i < frames.size(); ++i)
        {
            Detect(frames[i]);
            auto res = GetDetects();
            regions[i].assign(std::begin(res), std::end(res));
        }
    }

    ///
    /// \brief ResetModel
    /// \param img
    /// \param roiRect
    ///
    virtual void ResetModel(const cv::UMat& /*img*/, const cv::Rect& /*roiRect*/)
    {
    }

    ///
    /// \brief ResetIgnoreMask
    ///
    virtual void ResetIgnoreMask()
    {
        if (!m_ignoreMask.empty())
            m_ignoreMask = 255;
    }

	///
	/// \brief UpdateIgnoreMask
	/// \param img
	/// \param roiRect
	///
	virtual void UpdateIgnoreMask(const cv::UMat& img, cv::Rect roiRect)
	{
        if (m_ignoreMask.empty())
            m_ignoreMask = cv::Mat(img.size(), CV_8UC1, cv::Scalar(255));

        auto Clamp = [](int& v, int& size, int hi)
        {
            if (v < 0)
            {
                size += v;
                v = 0;
            }
            else if (v + size > hi - 1)
            {
                size = hi - 1 - v;
            }
        };
        Clamp(roiRect.x, roiRect.width, m_ignoreMask.cols);
        Clamp(roiRect.y, roiRect.height, m_ignoreMask.rows);
        m_ignoreMask(roiRect) = 0;
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
        if (!m_ignoreMask.empty())
            cv::bitwise_and(foreground, m_ignoreMask, foreground);
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
#if 0
        if (!m_ignoreMask.empty())
            cv::imshow("ignoreMask", m_ignoreMask);
#endif
    }

    ///
    static std::unique_ptr<BaseDetector> CreateDetector(tracking::Detectors detectorType, const config_t& config, const cv::UMat& gray);
    static std::unique_ptr<BaseDetector> CreateDetectorKV(tracking::Detectors detectorType, const KeyVal& config, const cv::Mat& gray);


protected:
    regions_t m_regions;

    cv::Size m_minObjectSize{2, 2};

    cv::Mat m_ignoreMask;

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
			res &= (type != bad_type);
		}
		return res;
	}
	
	///
	objtype_t T2T(size_t typeInd) const
	{
		objtype_t res = (typeInd < m_typesMap.size()) ? m_typesMap[typeInd] : bad_type;
		return res;
	}

private:
    std::vector<objtype_t> m_typesMap;
};
