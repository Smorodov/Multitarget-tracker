#include "vibe.hpp"
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <random>

namespace vibe
{
	namespace
	{
	// Thread-local generator: VIBE::update() runs inside an OpenMP parallel
	// loop, so a shared m_rng/m_rngIdx (as before) was a data race.
	int RandomIndex(int count)
	{
		if (count <= 1)
			return 0;

		static thread_local std::mt19937 rng{ std::random_device{}() };
		return static_cast<int>(rng() % static_cast<unsigned int>(count));
	}

	bool RandomHit(int count)
	{
		return RandomIndex(count) == 0;
	}
	}

	///
	VIBE::VIBE(int channels, int samples, int pixel_neighbor, int distance_threshold, int matching_threshold, int update_factor) :
		m_samples(static_cast<size_t>(std::max(1, samples))),
		m_channels(static_cast<size_t>(std::max(1, channels))),
		m_pixelNeighbor(std::max(0, pixel_neighbor)),
		m_distanceThreshold(std::max(1, distance_threshold)),
		m_matchingThreshold(std::max(0, matching_threshold)),
		m_updateFactor(std::max(1, update_factor))
	{
	}

	///
	cv::Vec2i VIBE::getRndNeighbor(int i, int j)
	{
		const int area = m_pixelNeighbor * 2 + 1;
		const int neighbor_count = area * area;
		if (neighbor_count <= 0)
			return cv::Vec2i(i, j);

		const int rnd = RandomIndex(neighbor_count);
		const int start_i = i - m_pixelNeighbor;
		const int start_j = j - m_pixelNeighbor;
		const int position_i = rnd / area;
		const int position_j = rnd % area;
		const int cur_i = std::max(std::min(start_i + position_i, m_size.height - 1), 0);
		const int cur_j = std::max(std::min(start_j + position_j, m_size.width - 1), 0);
		return cv::Vec2i(cur_i, cur_j);
	}

	///
	void VIBE::init(const cv::Mat &img)
	{
		m_size = img.size();

		const size_t imWidth = static_cast<size_t>(m_size.width);
		const size_t imHeight = static_cast<size_t>(m_size.height);
		const size_t chanSampl = static_cast<size_t>(m_channels * m_samples);

		m_model.resize(m_channels * static_cast<size_t>(m_samples) * imWidth * imHeight, 0);

		m_mask = cv::Mat(m_size, CV_8UC1, cv::Scalar::all(0));

		const uchar* image = img.data;
		for (size_t i = 0; i < imHeight; ++i)
		{
			for (size_t j = 0; j < imWidth; j++)
			{
                for (size_t c = 0; c < m_channels; ++c)
				{
					m_model[chanSampl * (imWidth * i + j) + c] = image[m_channels * imWidth * i + m_channels * j + c];
				}
                for (size_t s = 1; s < m_samples; ++s)
				{
					cv::Vec<size_t, 2> rnd_pos = getRndNeighbor(static_cast<int>(i), static_cast<int>(j));
					size_t img_idx = m_channels * imWidth * rnd_pos[0] + m_channels * rnd_pos[1];
					size_t model_idx = chanSampl * (imWidth * i + j) + m_channels * s;
                    for (size_t c = 0; c < m_channels; ++c)
					{
						m_model[model_idx + c] = image[img_idx + c];
					}
				}
			}
		}
	}

	///
	void VIBE::update(const cv::Mat& img)
	{
		if (m_size != img.size())
		{
			init(img);
			return;
		}

		int rowsCount = img.rows;
#pragma omp parallel for
		for (int i = 0; i < rowsCount; i++)
		{
			const uchar* img_ptr = img.ptr(i);
			uchar* mask_ptr = m_mask.ptr(i);

			for (int j = 0; j < img.cols; j++)
			{
				bool flag = false;
				int matching_counter = 0;
				model_t::value_type* model_ptr = &m_model[m_channels * m_samples * m_size.width * i + m_channels * m_samples * j];
                for (size_t s = 0; s < m_samples; ++s)
				{
                    size_t channels_counter = 0;
                    for (size_t c = 0; c < m_channels; ++c)
					{
						if (std::abs((int)model_ptr[c] - img_ptr[c]) < m_distanceThreshold)
							++channels_counter;
					}
					if (channels_counter == m_channels)
					{
						if (++matching_counter > m_matchingThreshold)
						{
							flag = true;
							break;
						}
					}
					model_ptr += m_channels;
				}

				if (flag)
				{
					mask_ptr[0] = 0;
					if (RandomHit(m_updateFactor))
					{
						const int sample = RandomIndex(static_cast<int>(m_samples));
						size_t model_idx = m_channels * m_samples * m_size.width * i + m_channels * m_samples * j + m_channels * static_cast<size_t>(sample);
                        for (size_t c = 0; c < m_channels; ++c)
						{
							m_model[model_idx + c] = img_ptr[c];
						}

						const cv::Vec2i rnd_pos = getRndNeighbor(i, j);
						const int rndSample = RandomIndex(static_cast<int>(m_samples));
						model_idx = m_channels * m_samples * m_size.width * rnd_pos[0] + m_channels * m_samples * rnd_pos[1] + m_channels * static_cast<size_t>(rndSample);
                        for (size_t c = 0; c < m_channels; ++c)
						{
							m_model[model_idx + c] = img_ptr[c];
						}
					}
				}
				else
				{
					mask_ptr[0] = 255;
				}
				img_ptr += m_channels;
				++mask_ptr;
			}
		}
	}

	///
	cv::Mat& VIBE::getMask()
	{
		return m_mask;
	}

	///
	void VIBE::ResetModel(const cv::Mat& img, const cv::Rect& roiRect)
	{
		const int top = std::max(0, roiRect.y);
		const int bottom = std::min(img.rows, roiRect.y + roiRect.height);
		const int left = std::max(0, roiRect.x);
		const int right = std::min(img.cols, roiRect.x + roiRect.width);
		for (int i = top; i < bottom; i++)
		{
			const uchar* img_ptr = img.ptr(i) + m_channels * left;
			uchar* mask_ptr = m_mask.ptr(i) + left;

			for (int j = left; j < right; j++)
			{
				if (*mask_ptr)
				{
					int matching_counter = 0;
					model_t::value_type* model_ptr = &m_model[m_channels * m_samples * m_size.width * i + m_channels * m_samples * j];
                    for (size_t s = 0; s < m_samples; ++s)
					{
                        size_t channels_counter = 0;
                        for (size_t c = 0; c < m_channels; ++c)
						{
							if (std::abs((int)model_ptr[c] - img_ptr[c]) >= m_distanceThreshold)
							{
								model_ptr[c] = img_ptr[c];
								++channels_counter;
							}
						}
						if (channels_counter == m_channels)
						{
							if (++matching_counter > m_matchingThreshold)
								break;
						}
						model_ptr += m_channels;
					}
				}

				img_ptr += m_channels;
				++mask_ptr;
			}
		}
	}
}
