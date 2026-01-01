#pragma once

#include "YoloONNX.hpp"

///
/// \brief The DFINE_is_onnx class
///
class DFINE_is_onnx : public YoloONNX
{
public:
	DFINE_is_onnx(std::vector<std::string>& inputTensorNames, std::vector<std::string>& outputTensorNames)
	{
		inputTensorNames.push_back("input");
		outputTensorNames.push_back("logits");
		outputTensorNames.push_back("boxes");
		outputTensorNames.push_back("mask_probs");
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

		//0: name: input, size: 1x3x640x640
		//1: name: logits, size: 1x300x80
		//2: name: boxes, size: 1x300x4
		//3: name: mask_probs, size: 1x300x160x160


		//0: name: input, size: 1x3x432x432
		//1: name: dets, size: 1x200x4
		//2: name: labels, size: 1x200x91
		//3: name: 4245, size: 1x200x108x108

		const float fw = static_cast<float>(frameSize.width) / static_cast<float>(m_resizedROI.width);
		const float fh = static_cast<float>(frameSize.height) / static_cast<float>(m_resizedROI.height);

		cv::Size inputSize(m_inputDims[0].d[3], m_inputDims[0].d[2]);
		cv::Size2f inputSizef(static_cast<float>(inputSize.width), static_cast<float>(inputSize.height));

        //std::cout << "m_resizedROI: " << m_resizedROI << ", frameSize: " << frameSize << ", fw_h: " << cv::Size2f(fw, fh) << ", m_inputDims: " << cv::Point3i(m_inputDims.d[1], m_inputDims.d[2], m_inputDims.d[3]) << std::endl;

		int labelsInd = 0;
		int detsInd = 1;
		int segInd = 2;

		auto dets = outputs[detsInd];
		auto labels = outputs[labelsInd];

		auto masks = outputs[segInd];

		size_t ncInd = 2;
		size_t lenInd = 1;


        size_t nc = m_outpuDims[labelsInd].d[ncInd];
		size_t len = static_cast<size_t>(m_outpuDims[detsInd].d[lenInd]) / m_params.m_explicitBatchSize;
		auto volume0 = len * m_outpuDims[detsInd].d[ncInd]; // Volume(m_outpuDims[0]);
		dets += volume0 * imgIdx;
		auto volume1 = len * m_outpuDims[labelsInd].d[ncInd]; // Volume(m_outpuDims[0]);
		labels += volume1 * imgIdx;

		int segChannels = static_cast<int>(m_outpuDims[segInd].d[1]);
		int segWidth = static_cast<int>(m_outpuDims[segInd].d[2]);
		int segHeight = static_cast<int>(m_outpuDims[segInd].d[3]);
		masks += imgIdx * segChannels * segWidth * segHeight;

		cv::Mat binaryMask8U(segHeight, segWidth, CV_8UC1);

        //std::cout << "len = " << len << ", nc = " << nc << ", m_params.confThreshold = " << m_params.m_confThreshold << ", volume0 = " << volume0 << ", volume1 = " << volume1 << std::endl;

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

			if (classConf >= m_params.m_confThreshold)
			{
				float d0 = dets[0];
				float d1 = dets[1];
				float d2 = dets[2];
				float d3 = dets[3];

				float x = fw * (inputSizef.width * (d0 - d2 / 2.f) - m_resizedROI.x);
				float y = fh * (inputSizef.height * (d1 - d3 / 2.f) - m_resizedROI.y);
				float width = fw * inputSizef.width * d2;
				float height = fh * inputSizef.height * d3;

				//if (i == 0)
				//{
				//    std::cout << i << ": classConf = " << classConf << ", classId = " << classId << " (" << labels[classId] << "), rect = " << cv::Rect2f(x, y, width, height) << std::endl;
				//    std::cout << "dets = " << d0 << ", " << d1 << ", " << d2 << ", " << d3 << std::endl;
				//}
				resBoxes.emplace_back(classId, classConf, cv::Rect(cvRound(x), cvRound(y), cvRound(width), cvRound(height)));

				double maskThreshold = 0.1;
				for (int row = 0; row < segHeight; ++row)
				{
					const float* maskPtr = masks + row * segWidth;
					uchar* binMaskPtr = binaryMask8U.ptr(row);

					for (int col = 0; col < segWidth; ++col)
					{
						binMaskPtr[col] = (maskPtr[col] > maskThreshold) ? 255 : 0;
					}
				}

				tensor_rt::Result& resObj = resBoxes.back();

				cv::Rect smallRect;
				smallRect.x = cvRound(segHeight * (d0 - d2 / 2.f));
				smallRect.y = cvRound(segHeight * (d1 - d3 / 2.f));
				smallRect.width = cvRound(segHeight * d2);
				smallRect.height = cvRound(segHeight * d3);
				smallRect = Clamp(smallRect, cv::Size(segWidth, segHeight));

				if (smallRect.area() > 0)
				{
					cv::resize(binaryMask8U(smallRect), resObj.m_boxMask, resObj.m_brect.size(), 0, 0, cv::INTER_NEAREST);

#if 0
					static int globalObjInd = 0;
					SaveMat(mask, std::to_string(globalObjInd) + "_mask", ".png", "tmp", true);
					SaveMat(binaryMask, std::to_string(globalObjInd) + "_bin_mask", ".png", "tmp", true);
					SaveMat(binaryMask8U, std::to_string(globalObjInd) + "_bin_mask_8u", ".png", "tmp", true);
					SaveMat(resObj.m_boxMask, std::to_string(globalObjInd++) + "_obj_mask", ".png", "tmp", true);
					std::cout << "inputSize: " << inputSize << ", localRect: " << localRect << std::endl;
#endif

					std::vector<std::vector<cv::Point>> contours;
					cv::findContours(resObj.m_boxMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point());

					for (const auto& contour : contours)
					{
						cv::Rect br = cv::boundingRect(contour);

						if (br.width >= 4 &&
							br.height >= 4)
						{
							int dx = resObj.m_brect.x;
							int dy = resObj.m_brect.y;

							cv::RotatedRect rr = (contour.size() < 5) ? cv::minAreaRect(contour) : cv::fitEllipse(contour);
							rr.center.x = rr.center.x * fw + dx;
							rr.center.y = rr.center.y * fw + dy;
							rr.size.width *= fw;
							rr.size.height *= fh;

							br.x = cvRound(dx + br.x * fw);
							br.y = cvRound(dy + br.y * fh);
							br.width = cvRound(br.width * fw);
							br.height = cvRound(br.height * fh);

							resObj.m_brect = br;
							//resObj.m_rrect = rr;

							//std::cout << "resBoxes[" << i << "] br: " << br << ", rr: (" << rr.size << " from " << rr.center << ", " << rr.angle << ")" << std::endl;

							break;
						}
					}
				}
				else
				{
					resObj.m_boxMask = cv::Mat(resObj.m_brect.size(), CV_8UC1, cv::Scalar(255));
				}
			}

			dets += m_outpuDims[detsInd].d[ncInd];
			labels += m_outpuDims[labelsInd].d[ncInd];
			masks += segWidth * segHeight;
		}

		return resBoxes;
	}
};
