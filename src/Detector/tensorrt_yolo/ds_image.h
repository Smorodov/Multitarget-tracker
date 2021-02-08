/**
MIT License

Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*
*/
#ifndef __IMAGE_H__
#define __IMAGE_H__

#include "trt_utils.h"

struct BBoxInfo;

class DsImage
{
public:
    DsImage();
    DsImage(const std::string& path, const std::string &s_net_type_, const int& inputH, const int& inputW);
    DsImage(const cv::Mat& mat_image_, const std::string &s_net_type_, const int& inputH, const int& inputW);
    int getImageHeight() const { return m_Height; }
    int getImageWidth() const { return m_Width; }
    cv::Mat getLetterBoxedImage() const { return m_LetterboxImage; }
    cv::Mat getOriginalImage() const { return m_OrigImage; }
    std::string getImageName() const { return m_ImageName; }
    void addBBox(BBoxInfo box, const std::string& labelName);
    void showImage() const;
    void saveImageJPEG(const std::string& dirPath) const;
    std::string exportJson() const;
	void letterbox(const int& inputH, const int& inputW);
private:
    int m_Height;
    int m_Width;
    int m_XOffset;
    int m_YOffset;
    float m_ScalingFactor;
    std::string m_ImagePath;
    cv::RNG m_RNG;
    std::string m_ImageName;
    std::vector<BBoxInfo> m_Bboxes;

    // unaltered original Image
    cv::Mat m_OrigImage;
    // letterboxed Image given to the network as input
    cv::Mat m_LetterboxImage;
    // final image marked with the bounding boxes
    cv::Mat m_MarkedImage;
};

#endif
