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

    bool Init(std::string cascadeFileName = "../data/haarcascade_frontalface_alt2.xml");

    void Detect(cv::UMat& gray);

private:
    cv::CascadeClassifier m_cascade;
};
