#pragma once

#include "BaseDetector.h"
#include "BackgroundSubtract.h"

///
/// \brief The MotionDetector class
///
class MotionDetector : public BaseDetector
{
public:
    MotionDetector(BackgroundSubtract::BGFG_ALGS algType, const cv::UMat& gray);
    MotionDetector(BackgroundSubtract::BGFG_ALGS algType, const cv::Mat& gray);
    ~MotionDetector(void) = default;

    bool Init(const config_t& config) override;

    void Detect(const cv::UMat& gray) override;

    bool CanGrayProcessing() const override
    {
        return true;
    }

    void CalcMotionMap(cv::Mat& frame) override;

    void ResetModel(const cv::UMat& img, const cv::Rect& roiRect) override;

private:
    void DetectContour();

    std::unique_ptr<BackgroundSubtract> m_backgroundSubst;

    cv::UMat m_fg;

    BackgroundSubtract::BGFG_ALGS m_algType = BackgroundSubtract::BGFG_ALGS::ALG_MOG2;
    bool m_useRotatedRect = false;
};
