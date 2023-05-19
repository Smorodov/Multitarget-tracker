#ifndef CLASS_DETECTOR_H_
#define CLASS_DETECTOR_H_

#include "API.h"
#include <iostream>
#include <opencv2/opencv.hpp>

namespace tensor_rt
{
    ///
    /// \brief The Result struct
    ///
	struct Result
	{
        cv::RotatedRect m_rrect;
        cv::Rect m_brect;
        int m_id = -1;
        float m_prob = 0.f;
        cv::Mat m_boxMask;

		Result(int id, float prob, const cv::Rect& brect)
			: m_brect(brect), m_id(id), m_prob(prob)
		{
            m_rrect = cv::RotatedRect(m_brect.tl(), cv::Point2f(static_cast<float>(m_brect.x + m_brect.width), static_cast<float>(m_brect.y)), m_brect.br());
            if (m_rrect.size.width < 1)
                m_rrect.size.width = 1;
            if (m_rrect.size.height < 1)
                m_rrect.size.height = 1;
		}

        Result(int id, float prob, const cv::RotatedRect& rrect)
            : m_rrect(rrect), m_id(id), m_prob(prob)
        {
            m_brect = m_rrect.boundingRect();
        }
	};
	
	using BatchResult = std::vector<Result>;

    ///
    /// \brief The ModelType enum
    ///
    enum ModelType
	{
        YOLOV3,
        YOLOV3_TINY,
        YOLOV4,
        YOLOV4_TINY,
        YOLOV5,
        YOLOV6,
        YOLOV7,
        YOLOV7Mask,
        YOLOV8
	};

    ///
    /// \brief The Precision enum
    ///
	enum Precision
	{
		INT8 = 0,
		FP16,
		FP32
	};

    ///
    /// \brief The Config struct
    ///
    struct Config
    {
        std::string file_model_cfg = "yolov4.cfg";
        std::string file_model_weights = "yolov4.weights";
        float detect_thresh = 0.5f;
        ModelType net_type = YOLOV4;
        Precision inference_precision = FP32;
        int	gpu_id = 0;
        size_t video_memory = 0;
        uint32_t batch_size = 1;
        std::string calibration_image_list_file_txt = "configs/calibration_images.txt";
    };

    ///
    /// \brief The Detector class
    ///
	class API Detector
	{
	public:
        explicit Detector() noexcept;
        ~Detector();

        bool Init(const Config& config);

        void Detect(const std::vector<cv::Mat>& mat_image, std::vector<BatchResult>& vec_batch_result);

        cv::Size GetInputSize() const;

		class Impl;

	private:
		Detector(const Detector &);
		const Detector &operator =(const Detector &);

        Impl* m_impl = nullptr;
	};
}
#endif // !CLASS_QH_DETECTOR_H_
