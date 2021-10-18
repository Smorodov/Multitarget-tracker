#pragma once

#include "BaseDetector.h"
#include "pedestrians/c4-pedestrian-detector.h"

///
/// \brief The PedestrianDetector class
///
class PedestrianDetector final : public BaseDetector
{
public:
    enum DetectorTypes
    {
        HOG,
        C4
    };

    PedestrianDetector(const cv::UMat& gray);
    PedestrianDetector(const cv::Mat& gray);
    ~PedestrianDetector(void) = default;

    bool Init(const config_t& config) override;

    void Detect(const cv::UMat& gray) override;

    bool CanGrayProcessing() const override
    {
        return true;
    }

private:
    DetectorTypes m_detectorType = HOG;

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
