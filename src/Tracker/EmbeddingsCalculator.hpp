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
		m_net = cv::dnn::readNet(weightsName, cfgName);
#else
		m_net = cv::dnn::readNetFromTensorflow(weightsName, cfgName);
#endif
		if (!m_net.empty())
		{
			m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
			m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
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

		cv::UMat obj;
		cv::resize(img(rect), obj, m_inputLayer, 0., 0., cv::INTER_LANCZOS4);
		cv::Mat blob = cv::dnn::blobFromImage(obj, 1.0, cv::Size(), cv::Scalar(), false, false);
		
		m_net.setInput(blob);
		embedding = m_net.forward();
		//std::cout << "embedding: " << embedding.size() << ", chans = " << embedding.channels() << std::endl;
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
