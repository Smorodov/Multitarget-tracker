#ifndef CLASS_YOLOV5_H_
#define CLASS_YOLOV5_H_
#include "yolo.h"
class YoloV5 :public Yolo
{
public:
	YoloV5(const uint32_t batch_size_,
		const NetworkInfo &network_info_,
		const InferParams &infer_params_);

	BBox convert_bbox_res(const float& bx, const float& by, const float& bw, const float& bh,
		const uint32_t& stride_h_, const uint32_t& stride_w_, const uint32_t& netW, const uint32_t& netH)
	{
		BBox b;
		// Restore coordinates to network input resolution
		float x = bx * stride_w_;
		float y = by * stride_h_;

		b.x1 = x - bw / 2;
		b.x2 = x + bw / 2;

		b.y1 = y - bh / 2;
		b.y2 = y + bh / 2;

		b.x1 = clamp(b.x1, 0, netW);
		b.x2 = clamp(b.x2, 0, netW);
		b.y1 = clamp(b.y1, 0, netH);
		b.y2 = clamp(b.y2, 0, netH);

		return b;
	}

	inline void add_bbox_proposal(const float bx, const float by, const float bw, const float bh,
		const uint32_t stride_h_, const uint32_t stride_w_, const int maxIndex, const float maxProb,
		const uint32_t 	image_w, const uint32_t image_h,
		std::vector<BBoxInfo>& binfo)
	{
		BBoxInfo bbi;
		bbi.box = convert_bbox_res(bx, by, bw, bh, stride_h_,stride_w_, m_InputW, m_InputH);
		if ((bbi.box.x1 > bbi.box.x2) || (bbi.box.y1 > bbi.box.y2))
		{
			return;
		}
		convertBBoxImgRes(0, m_InputW, m_InputH, image_w, image_h, bbi.box);
		bbi.label = maxIndex;
		bbi.prob = maxProb;
		bbi.classId = getClassId(maxIndex);
		binfo.push_back(bbi);
	};
private:
	std::vector<BBoxInfo> decodeTensor(const int imageIdx,
		const int imageH,
		const int imageW,
		const TensorInfo& tensor) override;
};

#endif