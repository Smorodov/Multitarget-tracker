#pragma once

#include "BaseDetector.h"

///
/// \brief The FaceDetector class
///
class FaceDetector : public BaseDetector
{
public:
    FaceDetector(const cv::UMat& gray);
    ~FaceDetector(void) = default;

    bool Init(const config_t& config);

    void Detect(const cv::UMat& gray);

	bool CanGrayProcessing() const
	{
		return true;
	}

private:
    cv::CascadeClassifier m_cascade;
};
