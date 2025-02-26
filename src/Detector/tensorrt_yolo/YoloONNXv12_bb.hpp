#pragma once

#include "YoloONNX.hpp"

///
/// \brief The YOLOv12_bb_onnx class
///
class YOLOv12_bb_onnx : public YOLOv11_bb_onnx
{
public:
	YOLOv12_bb_onnx(std::vector<std::string>& inputTensorNames, std::vector<std::string>& outputTensorNames)
		: YOLOv11_bb_onnx(inputTensorNames, outputTensorNames)
	{
	}

};
