#pragma once

#include "YoloONNX.hpp"

///
/// \brief The YOLOv26_bb_onnx class
///
class YOLOv26_bb_onnx : public YoloONNX
{
public:
	YOLOv26_bb_onnx(std::vector<std::string>& inputTensorNames, std::vector<std::string>& outputTensorNames)
	{
		inputTensorNames.push_back("images");
		outputTensorNames.push_back("output0");
	}

protected:
	///
	/// \brief GetResult
	/// \param output
	/// \return
	///
	std::vector<tensor_rt::Result> GetResult(size_t imgIdx, int /*keep_topk*/, const std::vector<float*>& outputs, cv::Size frameSize)
	{
		std::vector<tensor_rt::Result> resBoxes;

		//0: name: images, size: 1x3x640x640
		//1: name: output0, size: 1x300x6

		const float fw = static_cast<float>(frameSize.width) / static_cast<float>(m_resizedROI.width);
		const float fh = static_cast<float>(frameSize.height) / static_cast<float>(m_resizedROI.height);

		auto output = outputs[0];

		size_t lenInd = 1;
		size_t len = static_cast<size_t>(m_outpuDims[0].d[lenInd]);
		auto volume = len * m_outpuDims[0].d[2];
		output += volume * imgIdx;
		//std::cout << "len = " << len << ", confThreshold = " << m_params.m_confThreshold << ", volume = " << volume << std::endl;

		for (size_t i = 0; i < len; ++i)
		{
			auto ind = i * m_outpuDims[0].d[2];

			float classConf = output[ind + 4];
			int64_t classId = output[ind + 5];

			if (classConf >= m_params.m_confThreshold)
			{
				float x = fw * (output[ind + 0] - m_resizedROI.x);
				float y = fh * (output[ind + 1] - m_resizedROI.y);
				float width = fw * (output[ind + 2] - output[ind + 0]);
				float height = fh * (output[ind + 3] - output[ind + 1]);

				//std::cout << "ind = " << ind << ", output[0] = " << output[ind + 0] << ", output[1] = " << output[ind + 1] << ", output[2] = " << output[ind + 2] << ", output[3] = " << output[ind + 3] << std::endl;
				//std::cout << "ind = " << ind << ", classConf = " << classConf << ", classId = " << classId << ", x = " << x << ", y = " << y << ", width = " << width << ", height = " << height << std::endl;

				resBoxes.emplace_back(classId, classConf, cv::Rect(cvRound(x), cvRound(y), cvRound(width), cvRound(height)));
			}
		}

		return resBoxes;
	}
};
