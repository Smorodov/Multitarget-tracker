#pragma once

///
/// \brief The VOTTracker class
///
class VOTTracker
{
public:
    VOTTracker() = default;
    virtual ~VOTTracker() = default;

    virtual void Initialize(const cv::Mat &im, cv::Rect region) = 0;
    virtual cv::RotatedRect Update(const cv::Mat &im, float& confidence) = 0;
    virtual void Train(const cv::Mat &im, bool first) = 0;
};
