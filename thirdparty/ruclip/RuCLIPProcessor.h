#pragma once

#include "json.hpp"
#include "bpe.h"
#include "TorchHeader.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <filesystem>
#include <fstream>

inline torch::Tensor CVMatToTorchTensor(const cv::Mat img, const bool perm = true)
{
	auto tensor_image = torch::from_blob(img.data, { img.rows, img.cols, img.channels() }, at::kByte);
	if (perm)
		tensor_image = tensor_image.permute({ 2,0,1 });
	tensor_image.unsqueeze_(0);
	tensor_image = tensor_image.toType(c10::kFloat).div(255);
	return tensor_image;		//tensor_image.clone();
}

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

//template <typename T>
//std::basic_string<T> lowercase(const std::basic_string<T>& s)
//{
//	std::basic_string<T> s2 = s;
//	std::transform(s2.begin(), s2.end(), s2.begin(),
//		[](const T v) { return static_cast<T>(std::tolower(v)); });
//	return s2;
//}
//
//template <typename T>
//std::basic_string<T> uppercase(const std::basic_string<T>& s)
//{
//	std::basic_string<T> s2 = s;
//	std::transform(s2.begin(), s2.end(), s2.begin(),
//		[](const T v) { return static_cast<T>(std::toupper(v)); });
//	return s2;
//}

///
class RuCLIPProcessor {
protected:
	uint32_t eos_id = 3,
		bos_id = 2,
		unk_id = 1,
		pad_id = 0;
	const int ImageSize{ 224 },
		TextSeqLength{ 77 };
	std::vector<double> NormMean,
		NormStd;
	vkcom::BaseEncoder * Tokenizer;
public:
	RuCLIPProcessor(
		const std::filesystem::path &tokenizer_path,
		const int image_size = 224,
		const int text_seq_length = 77,
		const std::vector<double> norm_mean = { 0.48145466, 0.4578275, 0.40821073 },
		const std::vector<double> norm_std = { 0.26862954, 0.26130258, 0.27577711 }
	);

	///!!!Локали-юникоды
	torch::Tensor EncodeText(/*std::vector<*/std::string &text);
	torch::Tensor PrepareTokens(/*std::vector<*/std::vector<int32_t> tokens);		//Передаю по значению чтобы внутри иметь дело с копией
	std::pair <torch::Tensor, torch::Tensor> operator () (const std::vector <std::string> &texts, const std::vector <cv::Mat> &images);

	static RuCLIPProcessor FromPretrained(const std::filesystem::path &folder)
	{
		std::filesystem::path tokenizer_path = folder / "bpe.model";
		using json = nlohmann::json;
		std::cout << tokenizer_path << std::endl;
		std::ifstream f(folder / "config.json");
		json config = json::parse(f);

		return RuCLIPProcessor(
			tokenizer_path,
			int(config["image_resolution"]),
			int(config["context_length"]),
			{ 0.48145466, 0.4578275, 0.40821073 },	//config.get("mean"),
			{ 0.26862954, 0.26130258, 0.27577711 }	//config.get("std")
		);
	}
};