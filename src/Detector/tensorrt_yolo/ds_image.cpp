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
#include "ds_image.h"

#ifdef HAVE_FILESYSTEM
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

DsImage::DsImage(const cv::Mat& mat_image_, tensor_rt::ModelType net_type, const int& inputH, const int& inputW)
{
	m_OrigImage = mat_image_;
	m_Height = m_OrigImage.rows;
	m_Width = m_OrigImage.cols;
	if (!m_OrigImage.data || m_OrigImage.cols <= 0 || m_OrigImage.rows <= 0)
	{
		std::cout << "empty image !"<< std::endl;
		assert(0);
	}
	if (m_OrigImage.channels() != 3)
	{
		std::cout << "Non RGB images are not supported "<< std::endl;
		assert(0);
	}
    if (tensor_rt::ModelType::YOLOV5 == net_type || tensor_rt::ModelType::YOLOV6 == net_type ||
		tensor_rt::ModelType::YOLOV7 == net_type || tensor_rt::ModelType::YOLOV7Mask == net_type ||
		tensor_rt::ModelType::YOLOV8 == net_type)
	{
		// resize the DsImage with scale
		float r = std::min(static_cast<float>(inputH) / static_cast<float>(m_Height), static_cast<float>(inputW) / static_cast<float>(m_Width));
        int resizeH = (std::round(m_Height*r));
        int resizeW = (std::round(m_Width*r));

		// Additional checks for images with non even dims
		if ((inputW - resizeW) % 2) resizeW--;
		if ((inputH - resizeH) % 2) resizeH--;
		assert((inputW - resizeW) % 2 == 0);
		assert((inputH - resizeH) % 2 == 0);

		m_XOffset = (inputW - resizeW) / 2;
		m_YOffset = (inputH - resizeH) / 2;

		assert(2 * m_XOffset + resizeW == inputW);
		assert(2 * m_YOffset + resizeH == inputH);

		// resizing
        cv::resize(m_OrigImage, m_LetterboxImage, cv::Size(resizeW, resizeH), 0, 0, cv::INTER_LINEAR);
		// letterboxing
        cv::copyMakeBorder(m_LetterboxImage, m_LetterboxImage, m_YOffset, m_YOffset, m_XOffset, m_XOffset, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
	}
	else
	{
        cv::resize(m_OrigImage, m_LetterboxImage, cv::Size(inputW, inputH), 0, 0, cv::INTER_LINEAR);
	}
}

DsImage::DsImage(const std::string& path, tensor_rt::ModelType net_type, const int& inputH, const int& inputW)
{
    m_ImageName = fs::path(path).stem().string();
	m_OrigImage = cv::imread(path, cv::IMREAD_UNCHANGED);
	m_Height = m_OrigImage.rows;
	m_Width = m_OrigImage.cols;
	if (!m_OrigImage.data || m_OrigImage.cols <= 0 || m_OrigImage.rows <= 0)
	{
		std::cout << "Unable to open image : " << path << std::endl;
		assert(0);
	}

	if (m_OrigImage.channels() != 3)
	{
		std::cout << "Non RGB images are not supported : " << path << std::endl;
		assert(0);
	}

    if (tensor_rt::ModelType::YOLOV5 == net_type || tensor_rt::ModelType::YOLOV6 == net_type ||
		tensor_rt::ModelType::YOLOV7 == net_type || tensor_rt::ModelType::YOLOV7Mask == net_type ||
		tensor_rt::ModelType::YOLOV8 == net_type)
	{
		// resize the DsImage with scale
		float dim = std::max(m_Height, m_Width);
		int resizeH = ((m_Height / dim) * inputH);
		int resizeW = ((m_Width / dim) * inputW);
		m_ScalingFactor = static_cast<float>(resizeH) / static_cast<float>(m_Height);

		// Additional checks for images with non even dims
        if ((inputW - resizeW) % 2)
            resizeW--;
        if ((inputH - resizeH) % 2)
            resizeH--;
		assert((inputW - resizeW) % 2 == 0);
		assert((inputH - resizeH) % 2 == 0);

		m_XOffset = (inputW - resizeW) / 2;
		m_YOffset = (inputH - resizeH) / 2;

		assert(2 * m_XOffset + resizeW == inputW);
		assert(2 * m_YOffset + resizeH == inputH);

		// resizing
        cv::resize(m_OrigImage, m_LetterboxImage, cv::Size(resizeW, resizeH), 0, 0, cv::INTER_LINEAR);
		// letterboxing
        cv::copyMakeBorder(m_LetterboxImage, m_LetterboxImage, m_YOffset, m_YOffset, m_XOffset, m_XOffset, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
	}
	else
	{
        cv::resize(m_OrigImage, m_LetterboxImage, cv::Size(inputW, inputH), 0, 0, cv::INTER_LINEAR);
	}
}

void DsImage::letterbox(const int& inputH, const int& inputW)
{
	//m_OrigImage.copyTo(m_MarkedImage);
	m_Height = m_OrigImage.rows;
	m_Width = m_OrigImage.cols;

	// resize the DsImage with scale
	float dim = std::max(m_Height, m_Width);
	int resizeH = ((m_Height / dim) * inputH);
	int resizeW = ((m_Width / dim) * inputW);
	m_ScalingFactor = static_cast<float>(resizeH) / static_cast<float>(m_Height);

	// Additional checks for images with non even dims
	if ((inputW - resizeW) % 2) resizeW--;
	if ((inputH - resizeH) % 2) resizeH--;
	assert((inputW - resizeW) % 2 == 0);
	assert((inputH - resizeH) % 2 == 0);

	m_XOffset = (inputW - resizeW) / 2;
	m_YOffset = (inputH - resizeH) / 2;

	assert(2 * m_XOffset + resizeW == inputW);
	assert(2 * m_YOffset + resizeH == inputH);

	// resizing
	cv::resize(m_OrigImage, m_LetterboxImage, cv::Size(resizeW, resizeH), 0, 0, cv::INTER_LINEAR);
	// letterboxing
    cv::copyMakeBorder(m_LetterboxImage, m_LetterboxImage, m_YOffset, m_YOffset, m_XOffset, m_XOffset, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
	//	cv::imwrite("letter.jpg", m_LetterboxImage);
}

void DsImage::addBBox(BBoxInfo box, const std::string& labelName)
{
    m_Bboxes.push_back(box);
    const int x = cvRound(box.box.x1);
    const int y = cvRound(box.box.y1);
    const int w = cvRound(box.box.x2 - box.box.x1);
    const int h = cvRound(box.box.y2 - box.box.y1);
    const cv::Scalar color(m_RNG.uniform(0, 255), m_RNG.uniform(0, 255), m_RNG.uniform(0, 255));

    cv::rectangle(m_MarkedImage, cv::Rect(x, y, w, h), color, 1);
    const cv::Size tsize = cv::getTextSize(labelName, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1, nullptr);
    cv::rectangle(m_MarkedImage, cv::Rect(x, y, tsize.width + 3, tsize.height + 4), color, -1);
    cv::putText(m_MarkedImage, labelName.c_str(), cv::Point(x, y + tsize.height), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), 1);
}

void DsImage::showImage() const
{
#ifndef SILENT_WORK
    //cv::namedWindow(m_ImageName);
    //cv::imshow(m_ImageName.c_str(), m_MarkedImage);
    //cv::waitKey(0);
#endif
}

void DsImage::saveImageJPEG(const std::string& dirPath) const
{
    cv::imwrite(dirPath + m_ImageName + ".jpeg", m_MarkedImage);
}

std::string DsImage::exportJson() const
{
    if (m_Bboxes.empty()) return "";
    std::stringstream json;
    json.precision(2);
    json << std::fixed;
    for (uint32_t i = 0; i < m_Bboxes.size(); ++i)
    {
        json << "\n{\n";
        json << "  \"image_id\"         : " << std::stoi(m_ImageName) << ",\n";
        json << "  \"category_id\"      : " << m_Bboxes.at(i).classId << ",\n";
        json << "  \"bbox\"             : ";
        json << "[" << m_Bboxes.at(i).box.x1 << ", " << m_Bboxes.at(i).box.y1 << ", ";
        json << m_Bboxes.at(i).box.x2 - m_Bboxes.at(i).box.x1 << ", "
             << m_Bboxes.at(i).box.y2 - m_Bboxes.at(i).box.y1 << "],\n";
        json << "  \"score\"            : " << m_Bboxes.at(i).prob << "\n";
        if (i != m_Bboxes.size() - 1)
            json << "},";
        else
            json << "}";
    }
    return json.str();
}
