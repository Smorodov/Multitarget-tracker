
#include "yolov4.h"

YoloV4::YoloV4(const uint32_t batch_size_,
	const NetworkInfo &network_info_,
	const InferParams &infer_params_) :
    Yolo(batch_size_, network_info_, infer_params_) {}

std::vector<BBoxInfo> YoloV4::decodeTensor(const int imageIdx, const int imageH, const int imageW, const TensorInfo& tensor)
{
	float scalingFactor
		= std::min(static_cast<float>(m_InputW) / imageW, static_cast<float>(m_InputH) / imageH);
	float xOffset = (m_InputW - scalingFactor * imageW) / 2;
	float yOffset = (m_InputH - scalingFactor * imageH) / 2;

	const float* detections = &tensor.hostBuffer[imageIdx * tensor.volume];

	std::vector<BBoxInfo> binfo;
	for (uint32_t y = 0; y < tensor.gridSize; ++y)
	{
		for (uint32_t x = 0; x < tensor.gridSize; ++x)
		{
			for (uint32_t b = 0; b < tensor.numBBoxes; ++b)
			{
				const float pw = tensor.anchors[tensor.masks[b] * 2];
				const float ph = tensor.anchors[tensor.masks[b] * 2 + 1];

				const int numGridCells = tensor.gridSize * tensor.gridSize;
				const int bbindex = y * tensor.gridSize + x;
				const float bx
					= x + detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 0)];

				const float by
					= y + detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 1)];
				const float bw
					= pw * detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 2)];
				const float bh
					= ph * detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 3)];

				const float objectness
					= detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 4)];

				float maxProb = 0.0f;
				int maxIndex = -1;

				for (uint32_t i = 0; i < tensor.numClasses; ++i)
				{
					float prob
						= (detections[bbindex
							+ numGridCells * (b * (5 + tensor.numClasses) + (5 + i))]);

					if (prob > maxProb)
					{
						maxProb = prob;
						maxIndex = i;
					}
				}
				maxProb = objectness * maxProb;

				if (maxProb > m_ProbThresh)
				{
					addBBoxProposal(bx, by, bw, bh, tensor.stride, scalingFactor, xOffset, yOffset,
						maxIndex, maxProb, imageW, imageH, binfo);
				}
			}
		}
	}
	return binfo;
}
