#pragma once

#include "YoloONNX.hpp"

///
/// \brief The RFDETR_bb_onnx class
///
class RFDETR_bb_onnx : public YoloONNX
{
public:
	RFDETR_bb_onnx(std::vector<std::string>& inputTensorNames, std::vector<std::string>& outputTensorNames)
	{
		inputTensorNames.push_back("input");
		outputTensorNames.push_back("dets");
		outputTensorNames.push_back("labels");
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

		//0: name: input, size : 1x3x560x560
		//1: name: dets, size : 1x300x4
		//2: name: labels, size : 1x300x91

		const float fw = static_cast<float>(frameSize.width) / static_cast<float>(m_resizedROI.width);
		const float fh = static_cast<float>(frameSize.height) / static_cast<float>(m_resizedROI.height);

        //std::cout << "m_resizedROI: " << m_resizedROI << ", frameSize: " << frameSize << ", fw_h: " << cv::Size2f(fw, fh) << ", m_inputDims: " << cv::Point3i(m_inputDims.d[1], m_inputDims.d[2], m_inputDims.d[3]) << std::endl;

		auto dets = outputs[0];
		auto labels = outputs[1];

		size_t ncInd = 2;
		size_t lenInd = 1;

        size_t nc = m_outpuDims[1].d[ncInd];
		size_t len = static_cast<size_t>(m_outpuDims[0].d[lenInd]) / m_params.m_explicitBatchSize;
		auto volume0 = len * m_outpuDims[0].d[ncInd]; // Volume(m_outpuDims[0]);
		dets += volume0 * imgIdx;
		auto volume1 = len * m_outpuDims[1].d[ncInd]; // Volume(m_outpuDims[0]);
		labels += volume1 * imgIdx;


        //std::cout << "len = " << len << ", nc = " << nc << ", m_params.confThreshold = " << m_params.confThreshold << ", volume0 = " << volume0 << ", volume1 = " << volume1 << std::endl;

		//for (size_t i = 0; i < len; ++i)
		//{
		//	std::cout << "labels: ";
		//	for (size_t j = 0; j < m_outpuDims[1].d[ncInd]; ++j)
		//	{
		//		std::cout << labels[j] << " | ";
		//	}
		//	std::cout << std::endl;
		//
		//	std::cout << "dets: ";
		//	for (size_t j = 0; j < m_outpuDims[0].d[ncInd]; ++j)
		//	{
		//		std::cout << dets[j] << " | ";
		//	}
		//	std::cout << std::endl;
		//}


		auto L2Conf = [](float v)
		{
			return 1.f / (1.f + std::exp(-v));
		};

		for (size_t i = 0; i < len; ++i)
		{
            float classConf = L2Conf(labels[0]);
            size_t classId = 0;
            for (size_t cli = 1; cli < nc; ++cli)
			{
				auto conf = L2Conf(labels[cli]);
				if (classConf < conf)
				{
					classConf = conf;
					classId = cli;
				}
			}
            if (classId > 0)
                --classId;

			if (classConf >= m_params.m_confThreshold)
			{
                float x = fw * (m_inputDims[0].d[2] * (dets[0] - dets[2] / 2.f) - m_resizedROI.x);
                float y = fh * (m_inputDims[0].d[3] * (dets[1] - dets[3] / 2.f) - m_resizedROI.y);
                float width = fw * m_inputDims[0].d[2] * dets[2];
                float height = fh * m_inputDims[0].d[3] * dets[3];

                //if (i == 0)
                //{
                //    std::cout << i << ": classConf = " << classConf << ", classId = " << classId << " (" << labels[classId] << "), rect = " << cv::Rect(cvRound(x), cvRound(y), cvRound(width), cvRound(height)) << std::endl;
                //    std::cout << "dets = " << dets[0] << ", " << dets[1] << ", " << dets[2] << ", " << dets[3] << std::endl;
                //}
				resBoxes.emplace_back(classId, classConf, cv::Rect(cvRound(x), cvRound(y), cvRound(width), cvRound(height)));
			}

			dets += m_outpuDims[0].d[ncInd];
			labels += m_outpuDims[1].d[ncInd];
		}

		return resBoxes;
	}
};
