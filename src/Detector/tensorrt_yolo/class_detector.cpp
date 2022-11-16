#include "class_detector.h"
#include "class_yolo_detector.hpp"
#include "YoloONNX.hpp"

namespace tensor_rt
{
    ///
    /// \brief The Detector::Impl class
    ///
    class Detector::Impl
	{
	public:
		Impl() = default;
        virtual ~Impl() = default;

        virtual bool Init(const Config& config) = 0;
        virtual void Detect(const std::vector<cv::Mat>& mat_image, std::vector<BatchResult>& vec_batch_result) = 0;
        virtual cv::Size GetInputSize() const = 0;
    };

    ///
    /// \brief The YoloDectectorImpl class
    ///
    class YoloDectectorImpl final : public Detector::Impl
    {
    public:
        virtual bool Init(const Config& config) override
        {
            m_detector.init(config);
            return true;
        }
        virtual void Detect(const std::vector<cv::Mat>& mat_image, std::vector<BatchResult>& vec_batch_result) override
        {
            m_detector.detect(mat_image, vec_batch_result);
        }
        virtual cv::Size GetInputSize() const override
        {
            return m_detector.get_input_size();
        }

    private:
        YoloDectector m_detector;
    };

    ///
    /// \brief The YoloDectectorImpl class
    ///
    class YoloONNXImpl final : public Detector::Impl
    {
    public:
        bool Init(const Config& config) override
        {
            // The onnx file to load
            m_params.onnxFileName = config.file_model_cfg; //"yolov6s.onnx"

            // Input tensor name of ONNX file & engine file
            if (config.net_type == ModelType::YOLOV6)
                m_params.inputTensorNames.push_back("image_arrays");
            else if (config.net_type == ModelType::YOLOV7)
                m_params.inputTensorNames.push_back("images");

            // Old batch configuration, it is zero if explicitBatch flag is true for the tensorrt engine
            // May be deprecated in the future
            m_params.batchSize = config.batch_size;
            // Threshold values
            m_params.confThreshold = config.detect_thresh;
            m_params.nmsThreshold = 0.5;

            m_params.videoMemory = config.video_memory;

            // Batch size, you can modify to other batch size values if needed
            m_params.explicitBatchSize = config.batch_size;

            m_params.m_precision = config.inference_precision;

            // Output tensors when BatchedNMSPlugin is available
            if (config.net_type == ModelType::YOLOV6)
                m_params.outputTensorNames.push_back("outputs");
            else if (config.net_type == ModelType::YOLOV7)
                m_params.outputTensorNames.push_back("output");

            std::string precisionStr;
            std::map<tensor_rt::Precision, std::string> dictprecision;
            dictprecision[tensor_rt::INT8] =  "kINT8";
            dictprecision[tensor_rt::FP16] = "kHALF";
            dictprecision[tensor_rt::FP32] = "kFLOAT";
            auto precision = dictprecision.find(m_params.m_precision);
            if (precision != dictprecision.end())
                precisionStr = precision->second;
            m_params.engingFileName = config.file_model_cfg + "-" + precisionStr + "-batch" + std::to_string(config.batch_size) + ".engine";

            return m_detector.Init(m_params);
        }

        void Detect(const std::vector<cv::Mat>& mat_image, std::vector<BatchResult>& vec_batch_result) override
        {
            vec_batch_result.clear();
            if (vec_batch_result.capacity() < mat_image.size())
                vec_batch_result.reserve(mat_image.size());

            for (const cv::Mat& frame : mat_image)
            {
                std::vector<tensor_rt::Result> bboxes;
                m_detector.Detect(frame, bboxes);
                vec_batch_result.emplace_back(bboxes);
            }
        }

        cv::Size GetInputSize() const override
        {
            return m_detector.GetInputSize();
        }

    private:
        YoloONNX m_detector;
        SampleYoloParams m_params;
    };

	///
	/// \brief Detector::Detector
	///
	Detector::Detector() noexcept
	{
	}

    ///
    /// \brief Detector::~Detector
    ///
    Detector::~Detector()
    {
        if (m_impl)
            delete m_impl;
    }

    ///
    /// \brief Detector::Init
    /// \param config
    ///
    bool Detector::Init(const Config& config)
	{
        if (m_impl)
            delete m_impl;

        if (config.net_type == ModelType::YOLOV6 || config.net_type == ModelType::YOLOV7)
            m_impl = new YoloONNXImpl();
        else
            m_impl = new YoloDectectorImpl();

        bool res = m_impl->Init(config);
        assert(res);
        return res;
	}

    ///
    /// \brief Detector::Detect
    /// \param mat_image
    /// \param vec_batch_result
    ///
    void Detector::Detect(const std::vector<cv::Mat>& mat_image, std::vector<BatchResult>& vec_batch_result)
	{
        m_impl->Detect(mat_image, vec_batch_result);
	}

    ///
    /// \brief Detector::GetInputSize
    /// \return
    ///
    cv::Size Detector::GetInputSize() const
	{
        return m_impl->GetInputSize();
	}
}
