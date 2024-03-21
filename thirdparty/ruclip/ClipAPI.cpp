#include "ClipAPI.h"

#include "TorchHeader.h"
#include "RuCLIP.h"
#include "RuCLIPProcessor.h"

#include "../../src/common/defines.h"

///
class ClassificationCLIP::ClassificationCLIPImpl
{
public:
	ClassificationCLIPImpl() = default;
	~ClassificationCLIPImpl() = default;
	
	///
	bool Init(const std::string& pathToClip, const std::string& pathToBPE, int inputImgSize, int indGPU, const std::vector<std::string>& labels)
	{
		bool res = true;

		m_pathToClip = pathToClip;
		m_pathToBPE = pathToBPE;
		m_inputImgSize = inputImgSize;
		m_indGPU = indGPU;
		m_labels = labels;

		//torch::manual_seed(24);

		std::cout << "Set Torch device (" << m_indGPU << "): " << ((m_indGPU < 0) ? "CPU" : "GPU") << std::endl;
		if (m_indGPU >= 0 && torch::cuda::is_available())
		{
			std::cout << "CUDA is available! Running on GPU." << std::endl;
			m_device = torch::Device(torch::kCUDA, m_indGPU);
		}
		else
		{
			std::cout << "CUDA is not available! Running on CPU." << std::endl;
		}

		std::cout << "Load clip from: " << pathToClip << std::endl;
		m_clip = FromPretrained(pathToClip);
		m_clip->to(m_device);

		std::cout << "Load processor from: " << pathToBPE << std::endl;
		std::vector<double> normMean{ 0.48145466, 0.4578275, 0.40821073 };
		std::vector<double> normStd{ 0.26862954, 0.26130258, 0.27577711 };
		m_processor = std::make_unique<RuCLIPProcessor>(m_pathToBPE, m_inputImgSize, 77, normMean, normStd);

		m_processor->CacheText(m_labels);

		return res;
	}

	///
	bool ProcessFrame(const cv::Mat& frame, const std::vector<cv::Rect>& rois, std::vector<CLIPResult>& result)
	{
		bool res = false;

		if (rois.empty())
			return res;

		result.resize(rois.size());

		std::map<size_t, size_t> img2roi;

		std::cout << "Resizing..." << std::endl;
		std::vector<cv::Mat> images;
		images.reserve(rois.size());
		for (size_t i = 0; i < rois.size(); ++i)
		{
			cv::Rect r = Clamp(rois[i], frame.size());
			if (r.width > m_inputImgSize / 10 && r.height > m_inputImgSize / 10)
			{
				img2roi[images.size()] = i;
				cv::Mat tmp;
				cv::resize(cv::Mat(frame, r), tmp, cv::Size(m_inputImgSize, m_inputImgSize), cv::INTER_CUBIC);
				images.emplace_back(tmp);
			}
		}
		if (images.empty())
		{
			std::cout << "CLIP::ProcessFrame: empty images" << std::endl;
			return res;
		}

		std::cout << "Running on " << images.size() << "..." << std::endl;
		auto dummy_input = m_processor->operator()(images);
		try
		{
			torch::Tensor logits_per_image = m_clip->forward(dummy_input.first.to(m_device), dummy_input.second.to(m_device));
			torch::Tensor logits_per_text = logits_per_image.t();
			auto probs = logits_per_image.softmax(/*dim = */-1).detach().cpu();
			//std::cout << "probs per image: " << probs << std::endl;

			const float* tensorData = reinterpret_cast<const float*>(probs.data_ptr());
			for (size_t imgInd = 0; imgInd < images.size(); ++imgInd)
			{
				float bestConf = 0.;
				size_t bestInd = 0;
				for (size_t labelInd = 0; labelInd < m_labels.size(); ++labelInd)
				{
					if (bestConf < tensorData[labelInd])
					{
						bestConf = tensorData[labelInd];
						bestInd = labelInd;
					}
				}
				result[img2roi[imgInd]] = CLIPResult(m_labels[bestInd], bestConf);
				std::cout << "Object: " << m_labels[bestInd] << " - " << bestConf << std::endl;
				tensorData += m_labels.size();
			}
			res = true;
		}
		catch (std::exception& e)
		{
			res = false;
			std::cout << "ClassificationCLIP::ProcessFrame: " << e.what() << std::endl;
		}

		return res;
	}


private:
	void* m_pImpl = nullptr;

	std::string m_pathToClip = "";
	std::string m_pathToBPE = "";
	int m_inputImgSize = 336;
	int m_indGPU = -1; // -1 - use CPU

	torch::Device m_device{ torch::kCPU };
	CLIP m_clip = nullptr;
	std::unique_ptr<RuCLIPProcessor> m_processor;

	std::vector<std::string> m_labels{ "human", "pedestrian", "car", "vehicle", "truck", "bus" };
};

///
ClassificationCLIP::ClassificationCLIP() noexcept
{
}

///
ClassificationCLIP::~ClassificationCLIP()
{
	if (m_pImpl)
		delete m_pImpl;
}

///
bool ClassificationCLIP::Init(const std::string& pathToClip, const std::string& pathToBPE, int inputImgSize, int indGPU, const std::vector<std::string>& labels)
{
	if (m_pImpl)
		delete m_pImpl;

	m_pImpl = new ClassificationCLIPImpl();

	bool res = m_pImpl->Init(pathToClip, pathToBPE, inputImgSize, indGPU, labels);
	assert(res);
	return res;
}

///
bool ClassificationCLIP::ProcessFrame(const cv::Mat& frame, const std::vector<cv::Rect>& rois, std::vector<CLIPResult>& result)
{
	return m_pImpl->ProcessFrame(frame, rois, result);
}
