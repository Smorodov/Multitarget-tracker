#pragma once

#include "YoloONNX.hpp"
#include "../../common/defines.h"

///
/// \brief The YOLOv7_instance_onnx class
///
class YOLOv7_instance_onnx : public YoloONNX
{
public:
	YOLOv7_instance_onnx(std::vector<std::string>& inputTensorNames, std::vector<std::string>& outputTensorNames)
	{
		inputTensorNames.push_back("images");
		outputTensorNames.push_back("output");
		outputTensorNames.push_back("516");
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

		const float fw = static_cast<float>(frameSize.width) / static_cast<float>(m_resizedROI.width);
		const float fh = static_cast<float>(frameSize.height) / static_cast<float>(m_resizedROI.height);

		size_t outInd = (outputs.size() == 0) ? 0 : 1;
		size_t segInd = (outputs.size() == 0) ? 1 : 0;

		auto output = outputs[0];

		//0: name: images, size : 1x3x640x640
		//1 : name : 516, size : 1x32x160x160
		//2 : name : onnx::Slice_542, size : 1x3x80x80x117
		//3 : name : onnx::Slice_710, size : 1x3x40x40x117
		//4 : name : onnx::Slice_878, size : 1x3x20x20x117
		//5 : name : output, size : 1x25200x117
		// 25200 = 3x80x80 + 3x40x40 + 3x20x20
		// 117 = x, y, w, h, c, 80 classes, 32 seg ancors
		// 80 * 8 = 640, 40 * 16 = 640, 20 * 32 = 640

		size_t ncInd = 2;
		size_t lenInd = 1;
		if (m_outpuDims[outInd].nbDims == 2)
		{
			ncInd = 1;
			lenInd = 0;
		}
		int nc = static_cast<int>(m_outpuDims[outInd].d[ncInd] - 5 - 32);
		size_t len = static_cast<size_t>(m_outpuDims[outInd].d[lenInd]);// / m_params.explicitBatchSize;
		//auto Volume = [](const nvinfer1::Dims& d)
		//{
		//    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int>());
		//};
		auto volume = len * m_outpuDims[outInd].d[ncInd]; // Volume(m_outpuDims[0]);
		output += volume * imgIdx;
		//std::cout << "len = " << len << ", nc = " << nc << ", m_params.confThreshold = " << m_params.confThreshold << ", volume = " << volume << std::endl;

#if 1
		int segWidth = 160;
		int segHeight = 160;
		int segChannels = 32;

		if (outputs.size() > 1)
		{
			//std::cout << "516 nbDims: " << m_outpuDims[segInd].nbDims << ", ";
			//for (size_t i = 0; i < m_outpuDims[segInd].nbDims; ++i)
			//{
			//    std::cout << m_outpuDims[segInd].d[i];
			//    if (i + 1 != m_outpuDims[segInd].nbDims)
			//        std::cout << "x";
			//}
			//std::cout << std::endl;
//
			//std::cout << "output nbDims: " << m_outpuDims[outInd].nbDims << ", ";
			//for (size_t i = 0; i < m_outpuDims[outInd].nbDims; ++i)
			//{
			//    std::cout << m_outpuDims[outInd].d[i];
			//    if (i + 1 != m_outpuDims[outInd].nbDims)
			//        std::cout << "x";
			//}
			//std::cout << std::endl;

			segChannels = static_cast<int>(m_outpuDims[segInd].d[1]);
			segWidth = static_cast<int>(m_outpuDims[segInd].d[2]);
			segHeight = static_cast<int>(m_outpuDims[segInd].d[3]);
		}
		cv::Mat maskProposals;
		std::vector<std::vector<float>> picked_proposals;
		int net_width = nc + 5 + segChannels;
#endif

		std::vector<int> classIds;
		std::vector<float> confidences;
		std::vector<cv::Rect> rectBoxes;
		classIds.reserve(len);
		confidences.reserve(len);
		rectBoxes.reserve(len);

		for (size_t i = 0; i < len; ++i)
		{
			// Box
			size_t k = i * (nc + 5 + 32);
			float object_conf = output[k + 4];

			if (i == 0)
			{
				//std::cout << "without nms: mem" << i << ": ";
				//for (size_t ii = 0; ii < nc; ++ii)
				//{
				//    std::cout << output[k + ii] << " ";
				//}
				//std::cout << ";" << std::endl;
				//for (size_t ii = nc; ii < nc + 4; ++ii)
				//{
				//    std::cout << output[k + ii] << " ";
				//}
				//std::cout << ";" << std::endl;
				//for (size_t ii = nc + 4; ii < nc + 4 + 32; ++ii)
				//{
				//    std::cout << output[k + ii] << " ";
				//}
				//std::cout << ";" << std::endl;
			}

			if (object_conf >= m_params.confThreshold)
			{
				// (center x, center y, width, height) to (x, y, w, h)
				float x = output[k] - output[k + 2] / 2;
				float y = output[k + 1] - output[k + 3] / 2;
				float width = output[k + 2];
				float height = output[k + 3];

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
				//    std::cout << i << ": object_conf = " << object_conf << ", class_conf = " << class_conf << ", classId = " << classId << ", rect = " << cv::Rect(cvRound(x), cvRound(y), cvRound(width), cvRound(height)) << std::endl;

				classIds.push_back(classId);
				confidences.push_back(class_conf);
				rectBoxes.emplace_back(cvRound(x), cvRound(y), cvRound(width), cvRound(height));

				std::vector<float> temp_proto(output + k + 5 + nc, output + k + net_width);
				picked_proposals.push_back(temp_proto);

				if (rectBoxes.size() >= CV_CN_MAX - 1)
					break;
			}
		}

		// Non-maximum suppression to eliminate redudant overlapping boxes
		std::vector<int> indices;
		cv::dnn::NMSBoxes(rectBoxes, confidences, m_params.confThreshold, m_params.nmsThreshold, indices);
		resBoxes.reserve(indices.size());

		for (size_t bi = 0; bi < indices.size(); ++bi)
		{
			resBoxes.emplace_back(classIds[indices[bi]], confidences[indices[bi]], Clamp(rectBoxes[indices[bi]], frameSize));
			maskProposals.push_back(cv::Mat(picked_proposals[indices[bi]]).t());
		}

		if (!maskProposals.empty())
		{
			// Mask processing
			const float* pdata = outputs[1];
			std::vector<float> maskFloat(pdata, pdata + segChannels * segWidth * segHeight);

			int INPUT_W = static_cast<int>(m_inputDims[0].d[3]);
			int INPUT_H = static_cast<int>(m_inputDims[0].d[2]);
			static constexpr float MASK_THRESHOLD = 0.5;

			cv::Mat mask_protos = cv::Mat(maskFloat);
			cv::Mat protos = mask_protos.reshape(0, { segChannels, segWidth * segHeight });

			cv::Mat matmulRes = (maskProposals * protos).t();//n*32 32*25600
			cv::Mat masks = matmulRes.reshape(static_cast<int>(resBoxes.size()), { segWidth, segHeight });
			std::vector<cv::Mat> maskChannels;
			split(masks, maskChannels);
			for (size_t i = 0; i < resBoxes.size(); ++i)
			{
				cv::Mat dest;
				cv::Mat mask;
				//sigmoid
				cv::exp(-maskChannels[i], dest);
				dest = 1.0 / (1.0 + dest);//160*160

				int padw = 0;
				int padh = 0;
				cv::Rect roi(int((float)padw / INPUT_W * segWidth), int((float)padh / INPUT_H * segHeight), int(segWidth - padw / 2), int(segHeight - padh / 2));
				dest = dest(roi);

				cv::resize(dest, mask, cv::Size(INPUT_W, INPUT_H), cv::INTER_NEAREST);

				resBoxes[i].m_boxMask = mask(resBoxes[i].m_brect) > MASK_THRESHOLD;

#if 0
				static int globalObjInd = 0;
				SaveMat(resBoxes[i].m_boxMask, std::to_string(globalObjInd++), ".png", "tmp", true);
#endif

				std::vector<std::vector<cv::Point>> contours;
				std::vector<cv::Vec4i> hierarchy;
#if (CV_VERSION_MAJOR < 4)
				cv::findContours(resBoxes[i].m_boxMask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point());
#else
				cv::findContours(resBoxes[i].m_boxMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point());
#endif
				for (const auto& contour : contours)
				{
					cv::Rect br = cv::boundingRect(contour);

					if (br.width >= 4 &&
						br.height >= 4)
					{
						int dx = resBoxes[i].m_brect.x;
						int dy = resBoxes[i].m_brect.y;

						cv::RotatedRect rr = (contour.size() < 5) ? cv::minAreaRect(contour) : cv::fitEllipse(contour);
						rr.center.x = (rr.center.x + dx - m_resizedROI.x) * fw;
						rr.center.y = (rr.center.y + dy - m_resizedROI.y) * fw;
						rr.size.width *= fw;
						rr.size.height *= fh;

						br.x = cvRound((dx + br.x - m_resizedROI.x) * fw);
						br.y = cvRound((dy + br.y - m_resizedROI.y) * fh);
						br.width = cvRound(br.width * fw);
						br.height = cvRound(br.height * fh);

						resBoxes[i].m_brect = br;
						resBoxes[i].m_rrect = rr;

						//std::cout << "resBoxes[" << i << "] br: " << br << ", rr: (" << rr.size << " from " << rr.center << ", " << rr.angle << ")" << std::endl;

						break;
					}
				}
			}
		}
		return resBoxes;
	}
};
