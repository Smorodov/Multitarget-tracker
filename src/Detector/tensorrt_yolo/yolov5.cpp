
#include "yolov5.h"


YoloV5::YoloV5(
	const NetworkInfo &network_info_,
	const InferParams &infer_params_) :
	Yolo( network_info_, infer_params_) {}

std::vector<BBoxInfo> YoloV5::decodeTensor(const int imageIdx, const int imageH, const int imageW, const TensorInfo& tensor)
{
	float	scale_h = 1.f;
	float	scale_w = 1.f;
	int	xOffset = 0;
	int yOffset = 0;
	calcuate_letterbox_message(m_InputH, m_InputW, imageH, imageW, scale_h, scale_w, xOffset, yOffset);
	const float* detections = &tensor.hostBuffer[imageIdx * tensor.volume];

	std::vector<BBoxInfo> binfo;
	for (uint32_t y = 0; y < tensor.grid_h; ++y)
	{
		for (uint32_t x = 0; x < tensor.grid_w; ++x)
		{
			for (uint32_t b = 0; b < tensor.numBBoxes; ++b)
			{
				const float pw = tensor.anchors[tensor.masks[b] * 2];
				const float ph = tensor.anchors[tensor.masks[b] * 2 + 1];

				const int numGridCells = tensor.grid_h * tensor.grid_w;
				const int bbindex = y * tensor.grid_w+ x;
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
					add_bbox_proposal(bx, by, bw, bh, tensor.stride_h, tensor.stride_w, scale_h, scale_w,xOffset, yOffset, maxIndex, maxProb, imageW, imageH, binfo);
				}
			}
		}
	}
	return binfo;
}
