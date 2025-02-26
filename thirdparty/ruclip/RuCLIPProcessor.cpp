#include "RuCLIPProcessor.h"

///
inline torch::Tensor CVMatToTorchTensor(const cv::Mat img, const bool perm = true)
{
	auto tensor_image = torch::from_blob(img.data, { img.rows, img.cols, img.channels() }, at::kByte);
	if (perm)
		tensor_image = tensor_image.permute({ 2,0,1 });
	tensor_image.unsqueeze_(0);
	tensor_image = tensor_image.toType(c10::kFloat).div(255);
	return tensor_image;		//tensor_image.clone();
}

///
inline cv::Mat TorchTensorToCVMat(const torch::Tensor tensor_image, const bool perm = true)
{
	auto t = tensor_image.detach().squeeze().cpu();
	if (perm)
		t = t.permute({ 1, 2, 0 });
	t = t.mul(255).clamp(0, 255).to(torch::kU8);
	cv::Mat result_img;
	cv::Mat(static_cast<int>(t.size(0)), static_cast<int>(t.size(1)), CV_MAKETYPE(CV_8U, t.sizes().size() >= 3 ? static_cast<int>(t.size(2)) : 1), t.data_ptr()).copyTo(result_img);
	return result_img;
}

///
RuCLIPProcessor :: RuCLIPProcessor(
	const std::string& tokenizer_path,
	const int image_size /*= 224*/,
	const int text_seq_length /*= 77*/,
	const std::vector<double> norm_mean /*= { 0.48145466, 0.4578275, 0.40821073 }*/,
	const std::vector<double> norm_std /*= { 0.26862954, 0.26130258, 0.27577711 }*/
) : ImageSize(image_size), TextSeqLength(text_seq_length), NormMean(norm_mean), NormStd(norm_std)
{
	vkcom::Status status;
	Tokenizer = std::make_unique<vkcom::BaseEncoder>(tokenizer_path, -1, &status);
}

///!!!Локали-юникоды
torch::Tensor RuCLIPProcessor :: EncodeText(const/*std::vector<*/std::string &text)
{
	std::vector<std::vector<int32_t>> ret_ids;
	vkcom::Status status;
	////for (auto &it : text)
	////	it = lowercase(it);
	//text = lowercase(text);
	//output_type = vkcom::OutputType::ID, bos = false, eos = false, reverse = false, dropout_prob = 0.0
	std::vector <std::string> texts;
	texts.push_back(text);
	status = Tokenizer->encode_as_ids(texts, &ret_ids);
	if (status.code != 0)
		throw std::runtime_error("RuCLIPProcessor::EncodeText error : " + status.message);
	auto it = ret_ids[0];
	//for (auto &it : ret_ids)
	//{
	if (it.size() > TextSeqLength - 2)
		it.resize(TextSeqLength - 2);
	it.insert(it.begin(), bos_id);			//vector сдвинет при вставке
	it.push_back(eos_id);
	//}
	return PrepareTokens(it);
}

torch::Tensor RuCLIPProcessor::EncodeImage(const cv::Mat& img)
{
	torch::Tensor img_tensor = CVMatToTorchTensor(img, true);
	img_tensor = torch::data::transforms::Normalize<>(NormMean, NormStd)(img_tensor);
	return img_tensor;
}

torch::Tensor RuCLIPProcessor :: PrepareTokens(/*std::vector<*/std::vector<int32_t> tokens)		//Передаю по значению чтобы внутри иметь дело с копией
{
	torch::Tensor result;
	if (tokens.size() > TextSeqLength)
	{
		int32_t back = tokens.back();
		tokens.resize(TextSeqLength);
		tokens.back() = back;
	}
	int empty_positions = TextSeqLength - static_cast<int>(tokens.size());
	if (empty_positions > 0)
		result = torch::cat({ torch::tensor(tokens, torch::kLong), torch::zeros(empty_positions, torch::kLong) });  //position tokens after text
	return result;
}

///
void RuCLIPProcessor::CacheText(const std::vector <std::string>& texts)
{
	m_textsTensors.clear();
	for (auto& it : texts)
	{
		std::string s = it;
		torch::Tensor text_tensor = EncodeText(s);
		m_textsTensors.push_back(text_tensor);
	}
}

///
std::pair<torch::Tensor, torch::Tensor> RuCLIPProcessor::operator()(const std::vector <std::string> &texts, const std::vector <cv::Mat> &images)
{
	std::vector <torch::Tensor> texts_tensors;
	for (auto& it : texts)
	{
		std::string s = it;
		torch::Tensor text_tensor = EncodeText(s);
		texts_tensors.push_back(text_tensor);
	}

	std::vector <torch::Tensor> images_tensors;
	for (auto &it : images)
	{
		torch::Tensor img_tensor = CVMatToTorchTensor(it, true);
		img_tensor = torch::data::transforms::Normalize<>(NormMean, NormStd)(img_tensor);
		//img_tensor.clone();
		images_tensors.push_back(img_tensor);
	}
	return std::make_pair(!texts_tensors.empty()?/*torch::pad_sequence*/torch::stack(texts_tensors):torch::Tensor(), torch::pad_sequence(images_tensors).squeeze(0));
}

///
std::pair<torch::Tensor, torch::Tensor> RuCLIPProcessor::operator()(const std::vector <cv::Mat>& images)
{
	std::vector <torch::Tensor> images_tensors;
	for (auto& it : images)
	{
		torch::Tensor img_tensor = CVMatToTorchTensor(it, true);
		img_tensor = torch::data::transforms::Normalize<>(NormMean, NormStd)(img_tensor);
		//img_tensor.clone();
		images_tensors.push_back(img_tensor);
	}
	return std::make_pair(torch::stack(m_textsTensors), torch::pad_sequence(images_tensors).squeeze(0));
}
