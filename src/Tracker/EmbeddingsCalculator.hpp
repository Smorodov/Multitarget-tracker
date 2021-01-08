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
	bool Initialize(const std::string& cfgName, const std::string& weightsName)
	{
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
	}
	
	///
	bool IsInitialized() const
	{
		return !m_net.empty();
	}

	///
	void Calc(const cv::UMat& img, const cv::Rect& rect, cv::Mat& embedding)
	{
		cv::UMat obj;
		//cv::resize(img(rect), obj, cv::Size(64, 128), 0., 0., cv::INTER_LINEAR); // mars-small128_1.pb
		cv::resize(img(rect), obj, cv::Size(208, 208), 0., 0., cv::INTER_LINEAR);  // vehicle-reid-0001
		cv::resize(img(rect), obj, cv::Size(128, 256), 0., 0., cv::INTER_LINEAR);  // person-reidentification-retail-0277
		cv::Mat blob = cv::dnn::blobFromImage(obj, 1.0, cv::Size(), cv::Scalar(), false, false);
		
		m_net.setInput(blob);
		embedding = m_net.forward();
		std::cout << "embedding: " << embedding.size() << ", chans = " << embedding.channels() << std::endl;
	}

private:
	cv::dnn::Net m_net;
};
