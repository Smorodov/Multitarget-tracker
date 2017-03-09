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
    VIBE(int channels = 1, int samples = 20, int pixel_neighbor = 1, int distance_threshold = 20, int matching_threshold = 3, int update_factor = 16);
    ~VIBE();

    void update(const cv::Mat& img);
    cv::Mat& getMask();

	int GetChannels() const
	{
		return channels_;
	}

private:
    int samples_;
    int channels_;
    int pixel_neighbor_;
    int distance_threshold_;
    int matching_threshold_;
    int update_factor_;

    cv::Size size_;
    unsigned char *model_;

    cv::Mat mask_;

    unsigned int rng_[RANDOM_BUFFER_SIZE];
    int rng_idx_;

    cv::Vec2i getRndNeighbor(int i, int j);
	void init(const cv::Mat& img);
};

}

#endif /*__VIBE_HPP__*/
