#pragma once

#include "YoloONNX.hpp"

///
/// \brief The YOLOv7_bb_onnx class
///
class YOLOv7_bb_onnx : public YoloONNX
{
protected:
	///
/// \brief GetResult
/// \param output
/// \return
///
	std::vector<tensor_rt::Result> YoloONNX::GetResult(size_t imgIdx, int /*keep_topk*/, const std::vector<float*>& outputs, cv::Size frameSize)
	{
		std::vector<tensor_rt::Result> resBoxes;

		if (outputs.size() == 4)
		{
			auto dets = reinterpret_cast<int*>(outputs[0]);
			auto boxes = outputs[1];
			auto scores = outputs[2];
			auto classes = reinterpret_cast<int*>(outputs[3]);

			int objectsCount = m_outpuDims[1].d[1];

			const float fw = static_cast<float>(frameSize.width) / static_cast<float>(m_inputDims.d[3]);
			const float fh = static_cast<float>(frameSize.height) / static_cast<float>(m_inputDims.d[2]);

			//std::cout << "Dets[" << imgIdx << "] = " << dets[imgIdx] << ", objectsCount = " << objectsCount << std::endl;

			const size_t step1 = imgIdx * objectsCount;
			const size_t step2 = 4 * imgIdx * objectsCount;
			for (size_t i = 0; i < static_cast<size_t>(dets[imgIdx]); ++i)
			{
				// Box
				const size_t k = i * 4;
				float class_conf = scores[i + step1];
				int classId = classes[i + step1];
				if (class_conf >= m_params.confThreshold)
				{
					float x = fw * boxes[k + 0 + step2];
					float y = fh * boxes[k + 1 + step2];
					float width = fw * boxes[k + 2 + step2] - x;
					float height = fh * boxes[k + 3 + step2] - y;

					//if (i == 0)
					//{
					//    std::cout << i << ": class_conf = " << class_conf << ", classId = " << classId << " (" << classes[i + step1] << "), rect = " << cv::Rect(cvRound(x), cvRound(y), cvRound(width), cvRound(height)) << std::endl;
					//    std::cout << "boxes = " << boxes[k + 0 + step2] << ", " << boxes[k + 1 + step2] << ", " << boxes[k + 2 + step2] << ", " << boxes[k + 3 + step2] << std::endl;
					//}
					resBoxes.emplace_back(classId, class_conf, cv::Rect(cvRound(x), cvRound(y), cvRound(width), cvRound(height)));
				}
			}
		}
		else if (outputs.size() == 1)
		{
			const float fw = static_cast<float>(frameSize.width) / static_cast<float>(m_inputDims.d[3]);
			const float fh = static_cast<float>(frameSize.height) / static_cast<float>(m_inputDims.d[2]);

			auto output = outputs[0];

			size_t ncInd = 2;
			size_t lenInd = 1;
			if (m_outpuDims[0].nbDims == 2)
			{
				ncInd = 1;
				lenInd = 0;
			}
			int nc = m_outpuDims[0].d[ncInd] - 5;
			size_t len = static_cast<size_t>(m_outpuDims[0].d[lenInd]) / m_params.explicitBatchSize;
			//auto Volume = [](const nvinfer1::Dims& d)
			//{
			//    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int>());
			//};
			auto volume = len * m_outpuDims[0].d[ncInd]; // Volume(m_outpuDims[0]);
			output += volume * imgIdx;
			//std::cout << "len = " << len << ", nc = " << nc << ", m_params.confThreshold = " << m_params.confThreshold << ", volume = " << volume << std::endl;

			if (m_outpuDims[0].nbDims == 2) // With nms
			{
				std::vector<int> classIds;
				std::vector<float> confidences;
				std::vector<cv::Rect> rectBoxes;
				classIds.reserve(len);
				confidences.reserve(len);
				rectBoxes.reserve(len);

				for (size_t i = 0; i < len; ++i)
				{
					// Box
					size_t k = i * 7;
					float class_conf = output[k + 6];
					int classId = cvRound(output[k + 5]);
					if (class_conf >= m_params.confThreshold)
					{
						float x = fw * output[k + 1];
						float y = fh * output[k + 2];
						float width = fw * (output[k + 3] - output[k + 1]);
						float height = fh * (output[k + 4] - output[k + 2]);

						//if (i == 0)
						//	std::cout << i << ": class_conf = " << class_conf << ", classId = " << classId << ", rect = " << cv::Rect(cvRound(x), cvRound(y), cvRound(width), cvRound(height)) << std::endl;

						classIds.push_back(classId);
						confidences.push_back(class_conf);
						rectBoxes.emplace_back(cvRound(x), cvRound(y), cvRound(width), cvRound(height));

						//bboxes.emplace_back(classId, class_conf, cv::Rect(cvRound(x), cvRound(y), cvRound(width), cvRound(height)));
					}
				}

				// Non-maximum suppression to eliminate redudant overlapping boxes
				std::vector<int> indices;
				cv::dnn::NMSBoxes(rectBoxes, confidences, m_params.confThreshold, m_params.nmsThreshold, indices);
				resBoxes.reserve(indices.size());

				for (size_t bi = 0; bi < indices.size(); ++bi)
				{
					resBoxes.emplace_back(classIds[indices[bi]], confidences[indices[bi]], rectBoxes[indices[bi]]);
				}
			}
			else // Without nms
			{
				std::vector<int> classIds;
				std::vector<float> confidences;
				std::vector<cv::Rect> rectBoxes;
				classIds.reserve(len);
				confidences.reserve(len);
				rectBoxes.reserve(len);

				for (size_t i = 0; i < len; ++i)
				{
					// Box
					size_t k = i * (nc + 5);
					float object_conf = output[k + 4];

					//if (i == 0)
					//{
					//	std::cout << "mem" << i << ": ";
					//	for (size_t ii = 0; ii < nc + 5; ++ii)
					//	{
					//		std::cout << output[k + ii] << " ";
					//	}
					//	std::cout << std::endl;
					//}

					if (object_conf >= m_params.confThreshold)
					{
						// (center x, center y, width, height) to (x, y, w, h)
						float x = fw * (output[k] - output[k + 2] / 2);
						float y = fh * (output[k + 1] - output[k + 3] / 2);
						float width = fw * output[k + 2];
						float height = fh * output[k + 3];

						// Classes
						float class_conf = output[k + 5];
						int classId = 0;

						for (int j = 1; j < nc; ++j)
						{
							if (class_conf < output[k + 5 + j])
							{
								classId = j;
								class_conf = output[k + 5 + j];
							}
						}

						class_conf *= object_conf;

						//if (i == 0)
						//	std::cout << i << ": object_conf = " << object_conf << ", class_conf = " << class_conf << ", classId = " << classId << ", rect = " << cv::Rect(cvRound(x), cvRound(y), cvRound(width), cvRound(height)) << std::endl;

						classIds.push_back(classId);
						confidences.push_back(class_conf);
						rectBoxes.emplace_back(cvRound(x), cvRound(y), cvRound(width), cvRound(height));
					}
				}

				// Non-maximum suppression to eliminate redudant overlapping boxes
				std::vector<int> indices;
				cv::dnn::NMSBoxes(rectBoxes, confidences, m_params.confThreshold, m_params.nmsThreshold, indices);
				resBoxes.reserve(indices.size());

				for (size_t bi = 0; bi < indices.size(); ++bi)
				{
					resBoxes.emplace_back(classIds[indices[bi]], confidences[indices[bi]], rectBoxes[indices[bi]]);
				}
			}
		}

		return resBoxes;
	}
};
