#pragma once

///
/// \brief The VOTTracker class
///
class VOTTracker
{
public:
    VOTTracker()
    {

    }
    virtual ~VOTTracker()
    {

    }

    virtual void Initialize(const cv::Mat &im, cv::Rect region) = 0;
    virtual cv::Rect Update(const cv::Mat &im, float& confidence) = 0;
    virtual void Train(const cv::Mat &im, bool first) = 0;
};
