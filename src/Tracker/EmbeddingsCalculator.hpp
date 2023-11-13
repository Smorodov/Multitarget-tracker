#pragma once

///
/// \brief The EmbeddingsCalculator class
///
class EmbeddingsCalculator
{
public:
    EmbeddingsCalculator() = default;
    virtual ~EmbeddingsCalculator() = default;

	///
	bool Initialize(const std::string& cfgName, const std::string& weightsName, const cv::Size& inputLayer)
	{
#ifdef USE_OCV_EMBEDDINGS
        m_inputLayer = inputLayer;

#if 1
		m_net = cv::dnn::readNet(weightsName);
#else
		m_net = cv::dnn::readNetFromTorch(weightsName);
#endif

		std::cout << "Re-id model " << weightsName << " loaded: " << (!m_net.empty()) << std::endl;

		if (!m_net.empty())
		{
#if (((CV_VERSION_MAJOR == 4) && (CV_VERSION_MINOR >= 2)) || (CV_VERSION_MAJOR > 4))
			std::map<cv::dnn::Target, std::string> dictTargets;
			dictTargets[cv::dnn::DNN_TARGET_CPU] = "DNN_TARGET_CPU";
			dictTargets[cv::dnn::DNN_TARGET_OPENCL] = "DNN_TARGET_OPENCL";
			dictTargets[cv::dnn::DNN_TARGET_OPENCL_FP16] = "DNN_TARGET_OPENCL_FP16";
			dictTargets[cv::dnn::DNN_TARGET_MYRIAD] = "DNN_TARGET_MYRIAD";
			dictTargets[cv::dnn::DNN_TARGET_CUDA] = "DNN_TARGET_CUDA";
			dictTargets[cv::dnn::DNN_TARGET_CUDA_FP16] = "DNN_TARGET_CUDA_FP16";

			std::map<int, std::string> dictBackends;
			dictBackends[cv::dnn::DNN_BACKEND_DEFAULT] = "DNN_BACKEND_DEFAULT";
			dictBackends[cv::dnn::DNN_BACKEND_HALIDE] = "DNN_BACKEND_HALIDE";
			dictBackends[cv::dnn::DNN_BACKEND_INFERENCE_ENGINE] = "DNN_BACKEND_INFERENCE_ENGINE";
			dictBackends[cv::dnn::DNN_BACKEND_OPENCV] = "DNN_BACKEND_OPENCV";
			dictBackends[cv::dnn::DNN_BACKEND_VKCOM] = "DNN_BACKEND_VKCOM";
			dictBackends[cv::dnn::DNN_BACKEND_CUDA] = "DNN_BACKEND_CUDA";
			dictBackends[1000000] = "DNN_BACKEND_INFERENCE_ENGINE_NGRAPH";
			dictBackends[1000000 + 1] = "DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019";

			std::cout << "Avaible pairs for Target - backend:" << std::endl;
			std::vector<std::pair<cv::dnn::Backend, cv::dnn::Target>> pairs = cv::dnn::getAvailableBackends();
			for (auto p : pairs)
			{
				std::cout << dictBackends[p.first] << " (" << p.first << ") - " << dictTargets[p.second] << " (" << p.second << ")" << std::endl;

				if (p.first == cv::dnn::DNN_BACKEND_CUDA)
				{
					//m_net.setPreferableTarget(p.second);
					//m_net.setPreferableBackend(p.first);
					//std::cout << "Set!" << std::endl;
				}
			}
#endif

			auto outNames = m_net.getUnconnectedOutLayersNames();
			auto outLayers = m_net.getUnconnectedOutLayers();
			auto outLayerType = m_net.getLayer(outLayers[0])->type;

			std::vector<cv::dnn::MatShape> outputs;
			std::vector<cv::dnn::MatShape> internals;
			m_net.getLayerShapes(cv::dnn::MatShape(), 0, outputs, internals);
			std::cout << "REID: getLayerShapes: outputs (" << outputs.size() << ") = " << (outputs.size() > 0 ? outputs[0].size() : 0) << ", internals (" << internals.size() << ") = " << (internals.size() > 0 ? internals[0].size() : 0) << std::endl;
			if (outputs.size() && outputs[0].size() > 3)
				std::cout << "outputs = [" << outputs[0][0] << ", " << outputs[0][1] << ", " << outputs[0][2] << ", " << outputs[0][3] << "], internals = [" << internals[0][0] << ", " << internals[0][1] << ", " << internals[0][2] << ", " << internals[0][3] << "]" << std::endl;
		}
		return !m_net.empty();
#else
        std::cerr << "EmbeddingsCalculator was disabled in CMAKE! Check SetDistances params." << std::endl;
        return false;
#endif
    }
	
	///
    bool IsInitialized() const
    {
#ifdef USE_OCV_EMBEDDINGS
        return !m_net.empty();
#else
        return false;
#endif
	}

	///
	void Calc(const cv::UMat& img, cv::Rect rect, cv::Mat& embedding)
    {
#ifdef USE_OCV_EMBEDDINGS
		auto Clamp = [](int& v, int& size, int hi) -> int
		{
			int res = 0;
			if (v < 0)
			{
				res = v;
				v = 0;
				return res;
			}
			else if (v + size > hi - 1)
			{
				res = v;
				v = hi - 1 - size;
				if (v < 0)
				{
					size += v;
					v = 0;
				}
				res -= v;
				return res;
			}
			return res;
		};
		Clamp(rect.x, rect.width, img.cols);
		Clamp(rect.y, rect.height, img.rows);

		cv::Mat obj;
		cv::resize(img(rect), obj, m_inputLayer, 0., 0., cv::INTER_CUBIC);
		cv::Mat blob = cv::dnn::blobFromImage(obj, 1.0 / 255.0, cv::Size(), cv::Scalar(), false, false, CV_32F);
		
		m_net.setInput(blob);
		//std::cout << "embedding: " << embedding.size() << ", chans = " << embedding.channels() << std::endl;
		cv::normalize(m_net.forward(), embedding);
#else
        std::cerr << "EmbeddingsCalculator was disabled in CMAKE! Check SetDistances params." << std::endl;
#endif
	}

private:
#ifdef USE_OCV_EMBEDDINGS
    cv::dnn::Net m_net;
    cv::Size m_inputLayer{ 128, 256 };
#endif
};
