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
            // The engine file to generate or to load
            // The engine file does not exist:
            //     This program will try to load onnx file and convert onnx into engine
            // The engine file exists:
            //     This program will load the engine file directly
            m_params.engingFileName = config.file_model_cfg + ".engine"; //"yolov6s.engine"

            // The onnx file to load
            m_params.onnxFileName = config.file_model_cfg; //"yolov6s.onnx"

            // Input tensor name of ONNX file & engine file
            m_params.inputTensorNames.push_back("image_arrays");

            // Old batch configuration, it is zero if explicitBatch flag is true for the tensorrt engine
            // May be deprecated in the future
            m_params.batchSize = config.batch_size;
            // Threshold values
            m_params.confThreshold = config.detect_thresh;
            m_params.nmsThreshold = 0.5;

            // Batch size, you can modify to other batch size values if needed
            m_params.explicitBatchSize = config.batch_size;
            m_params.width = 640;
            m_params.height = 640;

            m_params.int8 = (config.inference_precison == INT8);
            m_params.fp16 = (config.inference_precison == FP16);

            m_params.inputShape = std::vector<int>{ m_params.explicitBatchSize, 3, m_params.width, m_params.height };

            // Output shapes when BatchedNMSPlugin is available
            m_params.outputShapes.push_back(std::vector<int>{m_params.explicitBatchSize, 1});
            m_params.outputShapes.push_back(std::vector<int>{m_params.explicitBatchSize, m_params.keepTopK, 4});
            m_params.outputShapes.push_back(std::vector<int>{m_params.explicitBatchSize, m_params.keepTopK});
            m_params.outputShapes.push_back(std::vector<int>{m_params.explicitBatchSize, m_params.keepTopK});

            // Output tensors when BatchedNMSPlugin is available
            m_params.outputTensorNames.push_back("outputs");

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
            return cv::Size(m_params.width, m_params.height);
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

        if (config.net_type == ModelType::YOLOV6)
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
