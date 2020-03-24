#ifndef CLASS_YOLO_DETECTOR_HPP_
#define CLASS_YOLO_DETECTOR_HPP_

#include <opencv2/opencv.hpp>
#include "ds_image.h"
#include "trt_utils.h"
#include "yolo.h"
#include "yolov2.h"
#include "yolov3.h"

#include <experimental/filesystem>
#include <fstream>
#include <string>
#include <chrono>
#include <stdio.h>  /* defines FILENAME_MAX */

#include "class_detector.h"
//struct Result
//{
//	int		 id = -1;
//	float	 prob = 0.f;
//	cv::Rect rect;
//};
//
//enum ModelType
//{
//	YOLOV2 = 0,
//	YOLOV3,
//	YOLOV2_TINY,
//	YOLOV3_TINY
//};
//
//enum Precision
//{
//	INT8 = 0,
//	FP16,
//	FP32
//};

//struct Config
//{
//	std::string file_model_cfg						= "configs/yolov3.cfg";
//
//	std::string file_model_weights					= "configs/yolov3.weights";
//
//	float detect_thresh								= 0.9;
//
//	ModelType	net_type							= YOLOV3;
//
//	Precision	inference_precison					= INT8;
//
//	std::string calibration_image_list_file_txt     = "configs/calibration_images.txt";
//};

class YoloDectector
{
public:
	YoloDectector()
	{

	}
	~YoloDectector()
	{

	}

	void init(const tensor_rt::Config &config)
	{
		_config = config;

		this->set_gpu_id(_config.gpu_id);

		this->parse_config();

		this->build_net();
	}

	void detect(const cv::Mat		&mat_image,
				std::vector<tensor_rt::Result> &vec_result)
	{
		std::vector<DsImage> vec_ds_images;
		vec_result.clear();
		vec_ds_images.emplace_back(mat_image, _p_net->getInputH(), _p_net->getInputW());
		cv::Mat trtInput = blobFromDsImages(vec_ds_images, _p_net->getInputH(),_p_net->getInputW());
		_p_net->doInference(trtInput.data, vec_ds_images.size());
		for (uint32_t i = 0; i < vec_ds_images.size(); ++i)
		{
			auto curImage = vec_ds_images.at(i);
			auto binfo = _p_net->decodeDetections(i, curImage.getImageHeight(), curImage.getImageWidth());
			auto remaining = nmsAllClasses(_p_net->getNMSThresh(), binfo, _p_net->getNumClasses());
			for (const auto &b : remaining)
			{
				tensor_rt::Result res;
				res.id = b.label;
				res.prob = b.prob;
				const int x = b.box.x1;
				const int y = b.box.y1;
				const int w = b.box.x2 - b.box.x1;
				const int h = b.box.y2 - b.box.y1;
				res.rect = cv::Rect(x, y, w, h);
				vec_result.push_back(res);
			}
		}
	}

private:

	void set_gpu_id(const int id = 0)
	{
		cudaError_t status = cudaSetDevice(id);
		if (status != cudaSuccess)
		{
			std::cout << "gpu id :" + std::to_string(id) + " not exist !" << std::endl;
			assert(0);
		}
	}

	void parse_config()
	{
		_yolo_info.networkType = _vec_net_type[_config.net_type];
		_yolo_info.configFilePath = _config.file_model_cfg;
		_yolo_info.wtsFilePath = _config.file_model_weights;
		_yolo_info.precision = _vec_precision[_config.inference_precison];
		_yolo_info.deviceType = "kGPU";
		int npos = _yolo_info.wtsFilePath.find(".weights");
		assert(npos != std::string::npos
			&& "wts file file not recognised. File needs to be of '.weights' format");
		std::string dataPath = _yolo_info.wtsFilePath.substr(0, npos);
		_yolo_info.calibrationTablePath = dataPath + "-calibration.table";
		_yolo_info.enginePath = dataPath + "-" + _yolo_info.precision + ".engine";
		_yolo_info.inputBlobName = "data";

		_infer_param.printPerfInfo = false;
		_infer_param.printPredictionInfo = false;
		_infer_param.calibImages = _config.calibration_image_list_file_txt;
		_infer_param.calibImagesPath = "";
		_infer_param.probThresh = _config.detect_thresh;
		_infer_param.nmsThresh = 0.5;
	}

	void build_net()
	{
		if ((_config.net_type == tensor_rt::YOLOV2) || (_config.net_type == tensor_rt::YOLOV2_TINY))
		{
			_p_net = std::unique_ptr<Yolo>{ new YoloV2(1, _yolo_info, _infer_param) };
		}
		else if ((_config.net_type == tensor_rt::YOLOV3) || (_config.net_type == tensor_rt::YOLOV3_TINY))
		{
			_p_net = std::unique_ptr<Yolo>{ new YoloV3(1, _yolo_info, _infer_param) };
		}
		else
		{
			assert(false && "Unrecognised network_type. Network Type has to be one among the following : yolov2, yolov2-tiny, yolov3 and yolov3-tiny");
		}
	}

private:
	tensor_rt::Config _config;
	NetworkInfo _yolo_info;
	InferParams _infer_param;

	std::vector<std::string> _vec_net_type{ "yolov2","yolov3","yolov2-tiny","yolov3-tiny" };
	std::vector<std::string> _vec_precision{ "kINT8","kHALF","kFLOAT" };
	std::unique_ptr<Yolo> _p_net = nullptr;
};


#endif
