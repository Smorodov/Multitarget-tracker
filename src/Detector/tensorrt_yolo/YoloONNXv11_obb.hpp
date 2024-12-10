#pragma once

#include "YoloONNX.hpp"

///
/// \brief The YOLOv11_obb_onnx class
///
class YOLOv11_obb_onnx : public YoloONNX
{
protected:
	///
	/// \brief GetResult
	/// \param output
	/// \return
	///
	std::vector<tensor_rt::Result> GetResult(size_t imgIdx, int /*keep_topk*/, const std::vector<float*>& outputs, cv::Size frameSize)
	{
		std::vector<tensor_rt::Result> resBoxes;

		//0: name: images, size: 1x3x1024x1024
		//1: name: output0, size: 1x20x21504
		//20: 15 DOTA classes + x + y + w + h + a
		constexpr int shapeDataSize = 5;

		const float fw = static_cast<float>(frameSize.width) / static_cast<float>(m_resizedROI.width);
		const float fh = static_cast<float>(frameSize.height) / static_cast<float>(m_resizedROI.height);

		auto output = outputs[0];

		size_t ncInd = 1;
		size_t lenInd = 2;
		int nc = m_outpuDims[0].d[ncInd] - shapeDataSize;
		int dimensions = nc + shapeDataSize;
		size_t len = static_cast<size_t>(m_outpuDims[0].d[lenInd]) / m_params.explicitBatchSize;
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
		std::vector<cv::RotatedRect> rectBoxes;
		classIds.reserve(len);
		confidences.reserve(len);
		rectBoxes.reserve(len);

		for (size_t i = 0; i < len; ++i)
		{
			// Box
			size_t k = i * (nc + shapeDataSize);

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
			//{
			//	for (int jj = 0; jj < 20; ++jj)
			//	{
			//		std::cout << output[jj] << " ";
			//	}
			//	std::cout << std::endl;
			//}

			if (objectConf >= m_params.confThreshold)
			{
				classIds.push_back(classId);
				confidences.push_back(objectConf);

				// (center x, center y, width, height)
				float cx = fw * (output[k] - m_resizedROI.x);
				float cy = fh * (output[k + 1] - m_resizedROI.y);
				float width = fw * output[k + 2];
				float height = fh * output[k + 3];
				float angle = 180.f * output[k + nc + shapeDataSize - 1] / M_PI;
				rectBoxes.emplace_back(cv::Point2f(cx, cy), cv::Size2f(width, height), angle);

				//if (rectBoxes.size() == 1)
				//	std::cout << i << ": object_conf = " << objectConf << ", classId = " << classId << ", rect = " << rectBoxes.back().boundingRect() << ", angle = " << angle << std::endl;
			}
		}

		// Non-maximum suppression to eliminate redudant overlapping boxes
		//std::vector<int> indices;
		//cv::dnn::NMSBoxes(rectBoxes, confidences, m_params.confThreshold, m_params.nmsThreshold, indices);
		//resBoxes.reserve(indices.size());

		resBoxes.reserve(rectBoxes.size());
		for (size_t bi = 0; bi < rectBoxes.size(); ++bi)
		{
			resBoxes.emplace_back(classIds[bi], confidences[bi], rectBoxes[bi]);
		}

		return resBoxes;
	}
};
