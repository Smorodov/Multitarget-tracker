#include "RuCLIPProcessor.h"

RuCLIPProcessor :: RuCLIPProcessor(
	const std::filesystem::path &tokenizer_path,
	const int image_size /*= 224*/,
	const int text_seq_length /*= 77*/,
	const std::vector<double> norm_mean /*= { 0.48145466, 0.4578275, 0.40821073 }*/,
	const std::vector<double> norm_std /*= { 0.26862954, 0.26130258, 0.27577711 }*/
) : ImageSize(image_size), TextSeqLength(text_seq_length), NormMean(norm_mean), NormStd(norm_std)
{
	vkcom::Status status;
	Tokenizer = new vkcom::BaseEncoder(tokenizer_path.string(), -1, &status);
}

///!!!Локали-юникоды
torch::Tensor RuCLIPProcessor :: EncodeText(/*std::vector<*/std::string &text)
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
std::pair <torch::Tensor, torch::Tensor> RuCLIPProcessor :: operator () (const std::vector <std::string> &texts, const std::vector <cv::Mat> &images)
{
	std::vector <torch::Tensor> texts_tensors,
		images_tensors;
	for (auto &it : texts)
	{
		std::string s = it;
		torch::Tensor text_tensor = EncodeText(s);
		texts_tensors.push_back(text_tensor);
	}
	for (auto &it : images)
	{
		torch::Tensor img_tensor = CVMatToTorchTensor(it, true);
		img_tensor = torch::data::transforms::Normalize<>(NormMean, NormStd)(img_tensor);
		//img_tensor.clone();
		images_tensors.push_back(img_tensor);
	}
	return std::make_pair(/*torch::pad_sequence*/torch::stack(texts_tensors), torch::pad_sequence(images_tensors).squeeze(0));
}