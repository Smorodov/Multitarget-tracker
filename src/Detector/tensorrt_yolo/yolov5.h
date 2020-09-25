#ifndef CLASS_YOLOV5_H_
#define CLASS_YOLOV5_H_
#include "yolo.h"
class YoloV5 :public Yolo
{
public:
	YoloV5(
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

		b.x1 = clamp(b.x1, 0, static_cast<float>(netW));
		b.x2 = clamp(b.x2, 0, static_cast<float>(netW));
		b.y1 = clamp(b.y1, 0, static_cast<float>(netH));
		b.y2 = clamp(b.y2, 0, static_cast<float>(netH));

		return b;
	}

	
private:
	std::vector<BBoxInfo> decodeTensor(const int imageIdx,
		const int imageH,
		const int imageW,
		const TensorInfo& tensor) override;
};

#endif
