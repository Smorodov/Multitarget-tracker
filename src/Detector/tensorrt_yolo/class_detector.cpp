#include "class_detector.h"
#include "class_yolo_detector.hpp"

namespace tensor_rt
{

	class Detector::Impl
	{
	public:
		Impl() {}

		~Impl() {}

		YoloDectector _detector;
	};

	Detector::Detector()
	{
		_impl = new Impl();
	}

	Detector::~Detector()
	{
		if (_impl)
		{
			delete _impl;
			_impl = nullptr;
		}
	}

	void Detector::init(const Config &config)
	{
		_impl->_detector.init(config);
	}

	void Detector::detect(const cv::Mat &mat_image, std::vector<Result> &vec_result)
	{
		_impl->_detector.detect(mat_image, vec_result);
	}
}
