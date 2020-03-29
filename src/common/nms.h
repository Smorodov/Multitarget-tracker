#pragma once
#include <opencv2/opencv.hpp>
#include <assert.h>

/**
 * @brief nms
 * Non maximum suppression
 * @param srcRects
 * @param resRects
 * @param thresh
 * @param neighbors
 */
inline void nms(
        const std::vector<cv::Rect>& srcRects,
        std::vector<cv::Rect>& resRects,
        float thresh,
        int neighbors = 0
        )
{
    resRects.clear();

    const size_t size = srcRects.size();
    if (!size)
    {
        return;
    }

    // Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
    std::multimap<int, size_t> idxs;
    for (size_t i = 0; i < size; ++i)
    {
        idxs.insert(std::pair<int, size_t>(srcRects[i].br().y, i));
    }

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0)
    {
        // grab the last rectangle
        auto lastElem = --std::end(idxs);
        const cv::Rect& rect1 = srcRects[lastElem->second];

        int neigborsCount = 0;

        idxs.erase(lastElem);

        for (auto pos = std::begin(idxs); pos != std::end(idxs); )
        {
            // grab the current rectangle
            const cv::Rect& rect2 = srcRects[pos->second];

			float intArea = static_cast<float>((rect1 & rect2).area());
			float unionArea = static_cast<float>(rect1.area() + rect2.area() - intArea);
            float overlap = intArea / unionArea;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh)
            {
                pos = idxs.erase(pos);
                ++neigborsCount;
            }
            else
            {
                ++pos;
            }
        }
        if (neigborsCount >= neighbors)
        {
            resRects.push_back(rect1);
        }
    }
}

/**
 * @brief nms2
 * Non maximum suppression with detection scores
 * @param srcRects
 * @param scores
 * @param resRects
 * @param thresh
 * @param neighbors
 */
inline void nms2(
        const std::vector<cv::Rect>& srcRects,
        const std::vector<float>& scores,
        std::vector<cv::Rect>& resRects,
        float thresh,
        int neighbors = 0,
        float minScoresSum = 0.f
        )
{
    resRects.clear();

    const size_t size = srcRects.size();
    if (!size)
    {
        return;
    }

    assert(srcRects.size() == scores.size());

    // Sort the bounding boxes by the detection score
    std::multimap<float, size_t> idxs;
    for (size_t i = 0; i < size; ++i)
    {
        idxs.insert(std::pair<float, size_t>(scores[i], i));
    }

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0)
    {
        // grab the last rectangle
        auto lastElem = --std::end(idxs);
        const cv::Rect& rect1 = srcRects[lastElem->second];

        int neigborsCount = 0;
        float scoresSum = lastElem->first;

        idxs.erase(lastElem);

        for (auto pos = std::begin(idxs); pos != std::end(idxs); )
        {
            // grab the current rectangle
            const cv::Rect& rect2 = srcRects[pos->second];

			float intArea = static_cast<float>((rect1 & rect2).area());
			float unionArea = static_cast<float>(rect1.area() + rect2.area() - intArea);
            float overlap = intArea / unionArea;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh)
            {
                scoresSum += pos->first;
                pos = idxs.erase(pos);
                ++neigborsCount;
            }
            else
            {
                ++pos;
            }
        }
        if (neigborsCount >= neighbors &&
                scoresSum >= minScoresSum)
        {
            resRects.push_back(rect1);
        }
    }
}


/**
 * @brief nms3
 * Non maximum suppression with detection scores
 * @param srcRects
 * @param resRects
 * @param thresh
 * @param neighbors
 */
template<typename OBJ, typename GET_RECT_FUNC, typename GET_SCORE_FUNC, typename GET_TYPE_FUNC>
inline void nms3(
        const std::vector<OBJ>& srcRects,
        std::vector<OBJ>& resRects,
        float thresh,
        GET_RECT_FUNC GetRect,
        GET_SCORE_FUNC GetScore,
        GET_TYPE_FUNC GetType,
        int neighbors = 0,
        float minScoresSum = 0.f
        )
{
    resRects.clear();

    const size_t size = srcRects.size();
    if (!size)
    {
        return;
    }

    // Sort the bounding boxes by the detection score
    std::multimap<float, size_t> idxs;
    for (size_t i = 0; i < size; ++i)
    {
        idxs.insert(std::pair<float, size_t>(GetScore(srcRects[i]), i));
    }

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0)
    {
        // grab the last rectangle
        auto lastElem = --std::end(idxs);
        size_t lastPos = lastElem->second;
        const cv::Rect& rect1 = GetRect(srcRects[lastPos]);
        auto type1 = GetType(srcRects[lastPos]);

        int neigborsCount = 0;
        float scoresSum = lastElem->first;

        idxs.erase(lastElem);

        for (auto pos = std::begin(idxs); pos != std::end(idxs); )
        {
            // grab the current rectangle
            auto type2 = GetType(srcRects[pos->second]);
            if (type1 == type2)
            {
                const cv::Rect& rect2 = GetRect(srcRects[pos->second]);

                float intArea = static_cast<float>((rect1 & rect2).area());
                float unionArea = static_cast<float>(rect1.area() + rect2.area() - intArea);
                float overlap = intArea / unionArea;

                // if there is sufficient overlap, suppress the current bounding box
                if (overlap > thresh)
                {
                    scoresSum += pos->first;
                    pos = idxs.erase(pos);
                    ++neigborsCount;
                }
                else
                {
                    ++pos;
                }
            }
            else
            {
                ++pos;
            }
        }
        if (neigborsCount >= neighbors &&
                scoresSum >= minScoresSum)
        {
            resRects.push_back(srcRects[lastPos]);
        }
    }
}
