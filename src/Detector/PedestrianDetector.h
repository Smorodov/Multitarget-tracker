#pragma once

#include "BaseDetector.h"
#include "Detector/pedestrians/c4-pedestrian-detector.h"

///
/// \brief The PedestrianDetector class
///
class PedestrianDetector : public BaseDetector
{
public:
    enum DetectorTypes
    {
        HOG,
        C4
    };

    PedestrianDetector(bool collectPoints, cv::UMat& gray);
    ~PedestrianDetector(void);

    bool Init(const config_t& config);

    void Detect(cv::UMat& gray);

private:
    DetectorTypes m_detectorType;

    ///
    /// \brief m_hog
    /// HOG detector
    ///
    cv::HOGDescriptor m_hog;

    ///
    /// \brief m_scannerC4
    /// C4 detector
    ///
    DetectionScanner m_scannerC4;
    static const int HUMAN_height = 108;
    static const int HUMAN_width = 36;
    static const int HUMAN_xdiv = 9;
    static const int HUMAN_ydiv = 4;
};
