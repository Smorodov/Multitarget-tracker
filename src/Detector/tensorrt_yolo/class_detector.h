#ifndef CLASS_DETECTOR_H_
#define CLASS_DETECTOR_H_

#include "API.h"
#include <iostream>
#include <opencv2/opencv.hpp>

namespace tensor_rt
{
	struct Result
	{
		int		 id = -1;
		float	 prob = 0.f;
		cv::Rect rect;
	};

	enum ModelType
	{
		YOLOV2 = 0,
		YOLOV3,
		YOLOV2_TINY,
		YOLOV3_TINY
	};

	enum Precision
	{
		INT8 = 0,
		FP16,
		FP32
	};

	struct Config
	{
		std::string file_model_cfg = "configs/yolov3.cfg";

		std::string file_model_weights = "configs/yolov3.weights";

		float detect_thresh = 0.9;

		ModelType	net_type = YOLOV3;

		Precision	inference_precison = FP32;

		int	gpu_id = 0;

		std::string calibration_image_list_file_txt = "configs/calibration_images.txt";
	};

	class API Detector
	{
	public:
		explicit Detector();

		~Detector();

		void init(const Config &config);

		void detect(const cv::Mat &mat_image, std::vector<Result> &vec_result);

	private:

		Detector(const Detector &);
		const Detector &operator =(const Detector &);
		class Impl;
		Impl *_impl;
	};
}
#endif // !CLASS_QH_DETECTOR_H_