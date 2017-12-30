#pragma once

#include "BaseDetector.h"
#include "BackgroundSubtract.h"

///
/// \brief The MotionDetector class
///
class MotionDetector : public BaseDetector
{
public:
    MotionDetector(BackgroundSubtract::BGFG_ALGS algType, bool collectPoints, cv::UMat& gray);
    ~MotionDetector(void);

    bool Init(const config_t& config);

    void Detect(cv::UMat& gray);

	void CalcMotionMap(cv::Mat frame);

private:
    void DetectContour();

    std::unique_ptr<BackgroundSubtract> m_backgroundSubst;

    cv::UMat m_fg;
};
