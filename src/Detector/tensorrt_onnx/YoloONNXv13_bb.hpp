#pragma once

#include "YoloONNX.hpp"

///
/// \brief The YOLOv13_bb_onnx class
///
class YOLOv13_bb_onnx : public YoloONNX
{
public:
	YOLOv13_bb_onnx(std::vector<std::string>& inputTensorNames, std::vector<std::string>& outputTensorNames)
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
		//1: name: output0, size: 1x84x8400

		const float fw = static_cast<float>(frameSize.width) / static_cast<float>(m_resizedROI.width);
		const float fh = static_cast<float>(frameSize.height) / static_cast<float>(m_resizedROI.height);

		auto output = outputs[0];

		size_t ncInd = 1;
		size_t lenInd = 2;
		int nc = static_cast<int>(m_outpuDims[0].d[ncInd] - 4);
		int dimensions = nc + 4;
		size_t len = static_cast<size_t>(m_outpuDims[0].d[lenInd]);// / m_params.explicitBatchSize;
		//auto Volume = [](const nvinfer1::Dims& d)
		//{
		//    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int>());
		//};
		auto volume = len * m_outpuDims[0].d[ncInd]; // Volume(m_outpuDims[0]);
		output += volume * imgIdx;
		//std::cout << "len = " << len << ", nc = " << nc << ", m_params.confThreshold = " << m_params.confThreshold << ", volume = " << volume << std::endl;

		cv::Mat rawMemory(1, dimensions * static_cast<int>(len), CV_32FC1, output);
		rawMemory = rawMemory.reshape(1, dimensions);
		cv::transpose(rawMemory, rawMemory);
		output = (float*)rawMemory.data;

		//std::cout << "output[0] mem:\n";
		//for (size_t ii = 0; ii < 100; ++ii)
		//{
		//    std::cout << ii << ": ";
		//    for (size_t jj = 0; jj < 20; ++jj)
		//    {
		//        std::cout << output[ii * 20 + jj] << " ";
		//    }
		//    std::cout << ";" << std::endl;
		//}
		//std::cout << ";" << std::endl;

		std::vector<int> classIds;
		std::vector<float> confidences;
		std::vector<cv::Rect> rectBoxes;
		classIds.reserve(len);
		confidences.reserve(len);
		rectBoxes.reserve(len);

		for (size_t i = 0; i < len; ++i)
		{
			// Box
			size_t k = i * (nc + 4);

			int classId = -1;
			float objectConf = 0.f;
			for (int j = 0; j < nc; ++j)
			{
				const float classConf = output[k + 4 + j];
				if (classConf > objectConf)
				{
					classId = j;
					objectConf = classConf;
				}
			}

			//if (i == 0)
			//	std::cout << i << ": object_conf = " << object_conf << ", class_conf = " << class_conf << ", classId = " << classId << ", rect = " << cv::Rect(cvRound(x), cvRound(y), cvRound(width), cvRound(height)) << std::endl;

			if (objectConf >= m_params.m_confThreshold)
			{
				classIds.push_back(classId);
				confidences.push_back(objectConf);

				// (center x, center y, width, height) to (x, y, w, h)
				float x = fw * (output[k] - output[k + 2] / 2 - m_resizedROI.x);
				float y = fh * (output[k + 1] - output[k + 3] / 2 - m_resizedROI.y);
				float width = fw * output[k + 2];
				float height = fh * output[k + 3];
				rectBoxes.emplace_back(cvRound(x), cvRound(y), cvRound(width), cvRound(height));
			}
		}

		// Non-maximum suppression to eliminate redudant overlapping boxes
		std::vector<int> indices;
		cv::dnn::NMSBoxes(rectBoxes, confidences, m_params.m_confThreshold, m_params.m_nmsThreshold, indices);
		resBoxes.reserve(indices.size());

		for (size_t bi = 0; bi < indices.size(); ++bi)
		{
			resBoxes.emplace_back(classIds[indices[bi]], confidences[indices[bi]], rectBoxes[indices[bi]]);
		}

		return resBoxes;
	}
};
