#pragma once

#include "YoloONNX.hpp"
#include "../../mtracking/defines.h"

///
/// \brief The YOLOv26_instance_onnx class
///
class YOLOv26_instance_onnx : public YoloONNX
{
public:
	YOLOv26_instance_onnx(std::vector<std::string>& inputTensorNames, std::vector<std::string>& outputTensorNames)
	{
		inputTensorNames.push_back("images");
		outputTensorNames.push_back("output0");
		outputTensorNames.push_back("output1");
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

		size_t outInd = 0;
		size_t segInd = 1;

		auto output = outputs[outInd];

		//0: name: images, size: 1x3x640x640
		//1: name: output0, size: 1x300x38
		//2: name: output1, size: 1x32x160x160

		size_t dimInd = 2;
		size_t lenInd = 1;
		int dimensions = static_cast<int>(m_outpuDims[outInd].d[dimInd]);
		size_t len = static_cast<size_t>(m_outpuDims[outInd].d[lenInd]);
		auto volume = len * dimensions;
		output += volume * imgIdx;
		//std::cout << "len = " << len << ", nc = " << nc << ", m_params.confThreshold = " << m_params.confThreshold << ", volume = " << volume << std::endl;

		int segWidth = 160;
		int segHeight = 160;
		int segChannels = 32;

		if (outputs.size() > 1)
		{
			segChannels = static_cast<int>(m_outpuDims[segInd].d[1]);
			segWidth = static_cast<int>(m_outpuDims[segInd].d[2]);
			segHeight = static_cast<int>(m_outpuDims[segInd].d[3]);
		}
		cv::Mat maskProposals;
		int netWidth = 6 + segChannels;

		for (size_t i = 0; i < len; ++i)
		{
			// Box
			size_t k = i * dimensions;
			
			float objectConf = output[k + 4];
			int classId = output[k + 5];

			if (objectConf >= m_params.m_confThreshold)
			{
				// (center x, center y, width, height) to (x, y, w, h)
                float x = output[k];
                float y = output[k + 1];
                float width = output[k + 2] - output[k];
                float height = output[k + 3] - output[k + 1];

				if (width > 4 && height > 4)
				{
					resBoxes.emplace_back(classId, objectConf, cv::Rect(cvRound(x), cvRound(y), cvRound(width), cvRound(height)));

					std::vector<float> tempProto(output + k + 6, output + k + netWidth);
					maskProposals.push_back(cv::Mat(tempProto).t());
				}
			}
		}

		//std::cout << "maskProposals.size = " << maskProposals.size() << std::endl;
		if (!maskProposals.empty())
		{
			// Mask processing
			const float* pdata = outputs[segInd];
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

				//std::cout << "m_brect = " << resBoxes[i].m_brect << ", dest = " << dest.size() << ", mask = " << mask.size() << std::endl;
				
				resBoxes[i].m_boxMask = mask(resBoxes[i].m_brect) > MASK_THRESHOLD;

				//std::cout << "m_boxMask = " << resBoxes[i].m_boxMask.size() << ", m_brect = " << resBoxes[i].m_brect << ", dest = " << dest.size() << ", mask = " << mask.size() << std::endl;

#if 0
				static int globalObjInd = 0;
				SaveMat(resBoxes[i].m_boxMask, std::to_string(globalObjInd++), ".png", "tmp", true);
#endif

#if 1
				std::vector<std::vector<cv::Point>> contours;
#if ((CV_VERSION_MAJOR > 4) || ((CV_VERSION_MAJOR == 4) && (CV_VERSION_MINOR > 9)))
				cv::findContoursLinkRuns(resBoxes[i].m_boxMask, contours);
#else
				cv::findContours(resBoxes[i].m_boxMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point());
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

						cv::resize(resBoxes[i].m_boxMask, resBoxes[i].m_boxMask, resBoxes[i].m_brect.size(), 0, 0, cv::INTER_NEAREST);

						//std::cout << "resBoxes[" << i << "] br: " << br << ", rr: (" << rr.size << " from " << rr.center << ", " << rr.angle << ")" << std::endl;

						break;
					}
				}
#endif
			}
		}
		return resBoxes;
	}
};
