#ifndef __VIBE_HPP__
#define __VIBE_HPP__

#include <opencv2/core/core.hpp>
#include <memory>

namespace vibe
{
    constexpr int RANDOM_BUFFER_SIZE = 65535;


class VIBE
{
public:
    VIBE(int channels, int samples, int pixel_neighbor, int distance_threshold, int matching_threshold, int update_factor);
    ~VIBE() = default;

    void update(const cv::Mat& img);
    cv::Mat& getMask();

	void ResetModel(const cv::Mat& img, const cv::Rect& roiRect);

private:
    size_t m_samples = 20;
    size_t m_channels = 1;
    int m_pixelNeighbor = 1;
    int m_distanceThreshold = 20;
    int m_matchingThreshold = 3;
    int m_updateFactor = 16;

    cv::Size m_size;
	typedef std::vector<uchar> model_t;
    model_t m_model;

    cv::Mat m_mask;

    unsigned int m_rng[RANDOM_BUFFER_SIZE];
    int m_rngIdx = 0;

    cv::Vec<size_t, 2> getRndNeighbor(int i, int j);
	void init(const cv::Mat& img);
};
}

#endif /*__VIBE_HPP__*/
