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

		Result(int id_, float prob_, cv::Rect r)
			: id(id_), prob(prob_), rect(r)
		{
		}
	};
	
	typedef std::vector<Result> BatchResult;

	enum ModelType
	{
        YOLOV2 = 0,
        YOLOV3,
        YOLOV2_TINY,
        YOLOV3_TINY,
        YOLOV4,
        YOLOV4_TINY,
        YOLOV5
	};

	enum Precision
	{
		INT8 = 0,
		FP16,
		FP32
	};

	struct Config
	{
		std::string file_model_cfg = "yolov4.cfg";

		std::string file_model_weights = "yolov4.weights";

		float detect_thresh = 0.9f;

		ModelType	net_type = YOLOV3;

		Precision	inference_precison = FP32;

		int	gpu_id = 0;

		uint32_t batch_size = 1;

		std::string calibration_image_list_file_txt = "configs/calibration_images.txt";
	};

	class API Detector
	{
	public:
		explicit Detector();

		~Detector();

		void init(const Config &config);

		void detect(const std::vector<cv::Mat> &mat_image, std::vector<BatchResult> &vec_batch_result);

		cv::Size get_input_size() const;

	private:

		Detector(const Detector &);
		const Detector &operator =(const Detector &);
		class Impl;
		Impl *_impl = nullptr;
	};
}
#endif // !CLASS_QH_DETECTOR_H_
