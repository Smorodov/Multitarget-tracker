#ifndef __VIBE_HPP__
#define __VIBE_HPP__

#include <opencv2/core/core.hpp>
#include <memory>

#define RANDOM_BUFFER_SIZE (65535)

namespace vibe
{

class VIBE
{
public:
    VIBE(int channels, int samples, int pixel_neighbor, int distance_threshold, int matching_threshold, int update_factor);
    ~VIBE();

    void update(const cv::Mat& img);
    cv::Mat& getMask();

	int GetChannels() const
	{
		return m_channels;
	}

	void ResetModel(const cv::Mat& img, const cv::Rect& roiRect);

private:
    int m_samples = 20;
    int m_channels = 1;
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

    cv::Vec2i getRndNeighbor(int i, int j);
	void init(const cv::Mat& img);
};

}

#endif /*__VIBE_HPP__*/
