#pragma once

#include "YoloONNX.hpp"

#include "nms.h"

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
        outputTensorNames.push_back("scores");
        outputTensorNames.push_back("labels");
		outputTensorNames.push_back("boxes");
    }

protected:
	///
	/// \brief GetResult
	/// \param output
	/// \return
	///
	std::vector<tensor_rt::Result> GetResult(size_t imgIdx, int /*keep_topk*/, const std::vector<float*>& outputs, cv::Size frameSize)
	{
		std::vector<tensor_rt::Result> tmpBoxes;

		//0: name: images, size : 1x3x640x640
		//1: name: orig_target_sizes, size : 1x2
		//2: name: labels, size : 1x300
		//3: name: boxes, size : 1x300x4
		//4: name: scores, size : 1x300

		const float fw = static_cast<float>(frameSize.width) / static_cast<float>(m_resizedROI.width);
		const float fh = static_cast<float>(frameSize.height) / static_cast<float>(m_resizedROI.height);

        //std::cout << "m_resizedROI: " << m_resizedROI << ", frameSize: " << frameSize << ", fw_h: " << cv::Size2f(fw, fh) << ", m_inputDims: " << cv::Point3i(m_inputDims.d[1], m_inputDims.d[2], m_inputDims.d[3]) << std::endl;

        auto labels = (const int64_t*)outputs[1];
        auto boxes = outputs[2];
        auto scores = outputs[0];

#if 0
        std::cout << "scores mem:\n";
        for (size_t ii = 0; ii < 15; ++ii)
        {
            std::cout << ii << ": ";
            for (size_t jj = 0; jj < 20; ++jj)
            {
                std::cout << scores[ii * 20 + jj] << " ";
            }
            std::cout << ";" << std::endl;
        }
        std::cout << std::endl;

        std::cout << "labels mem:\n";
        for (size_t ii = 0; ii < 15; ++ii)
        {
            std::cout << ii << ": ";
            for (size_t jj = 0; jj < 20; ++jj)
            {
                std::cout << labels[ii * 20 + jj] << " ";
            }
            std::cout << ";" << std::endl;
        }
        std::cout << std::endl;

        std::cout << "boxes mem:\n";
        for (size_t ii = 0; ii < 15; ++ii)
        {
            std::cout << ii << ": ";
            for (size_t jj = 0; jj < 20; ++jj)
            {
                std::cout << boxes[ii * 20 + jj] << " ";
            }
            std::cout << ";" << std::endl;
        }
        std::cout << std::endl;

        std::cout << "m_outpuDims[0].d[1] = " << m_outpuDims[0].d[1] << std::endl;
#endif

		for (size_t i = 0; i < static_cast<int>(m_outpuDims[0].d[1]); ++i)
		{
            float classConf = scores[i];
			int64_t classId = labels[i];

            //if (classId > 0)
            //    --classId;

			if (classConf >= m_params.m_confThreshold)
			{
				auto ind = i * m_outpuDims[1].d[2];
                float x = fw * (boxes[ind + 0] - m_resizedROI.x);
                float y = fh * (boxes[ind + 1] - m_resizedROI.y);
                float width = fw * (boxes[ind + 2] - boxes[ind + 0]);
                float height = fh * (boxes[ind + 3] - boxes[ind + 1]);

                //std::cout << "ind = " << ind << ", boxes[0] = " << boxes[ind + 0] << ", boxes[1] = " << boxes[ind + 1] << ", boxes[2] = " << boxes[ind + 2] << ", boxes[3] = " << boxes[ind + 3] << std::endl;
                //std::cout << "ind = " << ind << ", x = " << x << ", y = " << y << ", width = " << width << ", height = " << height << std::endl;

				tmpBoxes.emplace_back(classId, classConf, cv::Rect(cvRound(x), cvRound(y), cvRound(width), cvRound(height)));
			}
		}

        std::vector<tensor_rt::Result> resBoxes;
        resBoxes.reserve(tmpBoxes.size());

        nms3<tensor_rt::Result>(tmpBoxes, resBoxes, static_cast<track_t>(0.3),
            [](const tensor_rt::Result& reg) { return reg.m_brect; },
            [](const tensor_rt::Result& reg) { return reg.m_prob; },
            [](const tensor_rt::Result& reg) { return reg.m_id; },
            0, static_cast<track_t>(0));

		return resBoxes;
	}
};
