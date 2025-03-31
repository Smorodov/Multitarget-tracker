#pragma once

#include "YoloONNX.hpp"

///
/// \brief The DFINE_bb_onnx class
///
class DFINE_bb_onnx : public YoloONNX
{
public:
	DFINE_bb_onnx(std::vector<std::string>& inputTensorNames, std::vector<std::string>& outputTensorNames)
	{
		inputTensorNames.push_back("images");
		inputTensorNames.push_back("orig_target_sizes");
		outputTensorNames.push_back("labels");
		outputTensorNames.push_back("boxes");
		outputTensorNames.push_back("scores");
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

		//0: name: images, size : 32x3x640x640
		//1: name: orig_target_sizes, size : 1x2
		//2: name: labels, size : 32x300
		//3: name: boxes, size : 32x300x4
		//4: name: scores, size : 32x300

		const float fw = static_cast<float>(frameSize.width) / static_cast<float>(m_resizedROI.width);
		const float fh = static_cast<float>(frameSize.height) / static_cast<float>(m_resizedROI.height);

        //std::cout << "m_resizedROI: " << m_resizedROI << ", frameSize: " << frameSize << ", fw_h: " << cv::Size2f(fw, fh) << ", m_inputDims: " << cv::Point3i(m_inputDims.d[1], m_inputDims.d[2], m_inputDims.d[3]) << std::endl;

		auto labels = (const int64_t*)outputs[0];
		auto boxes = outputs[1];
		auto scores = outputs[2];

		for (size_t i = 0; i < m_outpuDims[0].d[1]; ++i)
		{
            float classConf = scores[i];
			int64_t classId = labels[i];

			if (classConf >= m_params.confThreshold)
			{
				auto ind = i * m_outpuDims[1].d[2];
                float x = fw * (boxes[ind + 0] - m_resizedROI.x);
                float y = fh * (boxes[ind + 1] - m_resizedROI.y);
                float width = fw * (boxes[ind + 2] - boxes[ind + 0]);
                float height = fh * (boxes[ind + 3] - boxes[ind + 1]);

				resBoxes.emplace_back(classId, classConf, cv::Rect(cvRound(x), cvRound(y), cvRound(width), cvRound(height)));
			}
		}

		return resBoxes;
	}
};
