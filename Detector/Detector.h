#pragma once
#include "BackgroundSubtract.h"
#include <iostream>
#include <vector>
#include "defines.h"

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
class CDetector
{
private:
	void DetectContour();

	std::unique_ptr<BackgroundSubtract> m_bs;
    std::vector<CRegion> m_regions;
	std::vector<Point_t> m_centers;
	cv::Mat m_fg;

	cv::Size m_minObjectSize;

public:
	CDetector(cv::Mat& gray);
	const std::vector<Point_t>& Detect(cv::Mat& gray);
	~CDetector(void);

	void SetMinObjectSize(cv::Size minObjectSize);

    const regions_t& GetDetects() const;
};
