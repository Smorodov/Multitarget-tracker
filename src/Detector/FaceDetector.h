#pragma once

#include "BaseDetector.h"

///
/// \brief The FaceDetector class
///
class FaceDetector : public BaseDetector
{
public:
    FaceDetector(bool collectPoints, cv::UMat& gray);
    ~FaceDetector(void);

    bool Init(const config_t& config);

    void Detect(cv::UMat& gray);

private:
    cv::CascadeClassifier m_cascade;
};
