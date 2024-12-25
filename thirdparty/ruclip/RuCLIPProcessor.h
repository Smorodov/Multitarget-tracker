#pragma once

#include "json.hpp"
#include "youtokentome/bpe.h"
#include "TorchHeader.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <filesystem>
#include <fstream>

///
class RuCLIPProcessor
{
public:
	RuCLIPProcessor(const std::string& tokenizer_path,
		const int image_size = 224,
		const int text_seq_length = 77,
		const std::vector<double> norm_mean = { 0.48145466, 0.4578275, 0.40821073 },
		const std::vector<double> norm_std = { 0.26862954, 0.26130258, 0.27577711 });

	///!!!Локали-юникоды
	torch::Tensor EncodeText(/*std::vector<*/std::string &text);
	torch::Tensor PrepareTokens(/*std::vector<*/std::vector<int32_t> tokens);		//Передаю по значению чтобы внутри иметь дело с копией
	std::pair <torch::Tensor, torch::Tensor> operator()(const std::vector <std::string>& texts, const std::vector <cv::Mat>& images);
	std::pair <torch::Tensor, torch::Tensor> operator()(const std::vector <cv::Mat>& images);

	void CacheText(const std::vector <std::string>& texts);

	static RuCLIPProcessor FromPretrained(const std::filesystem::path &folder)
	{
		std::filesystem::path tokenizer_path = folder / "bpe.model";
		using json = nlohmann::json;
		std::cout << tokenizer_path << std::endl;
		std::ifstream f(folder / "config.json");
		json config = json::parse(f);

		return RuCLIPProcessor(
			tokenizer_path.string(),
			int(config["image_resolution"]),
			int(config["context_length"]),
			{ 0.48145466, 0.4578275, 0.40821073 },	//config.get("mean"),
			{ 0.26862954, 0.26130258, 0.27577711 }	//config.get("std")
		);
	}

private:
	uint32_t eos_id = 3;
	uint32_t bos_id = 2;
	uint32_t unk_id = 1;
	uint32_t pad_id = 0;
	const int ImageSize{ 224 };
	const int TextSeqLength{ 77 };
	std::vector<double> NormMean;
	std::vector<double> NormStd;
	vkcom::BaseEncoder* Tokenizer = nullptr;

	std::vector<torch::Tensor> m_textsTensors;
};

//relevancy for batch size == 1 at this moment,   float lv = result.index({0,0}).item<float>();
///
///std::vector<torch::Tensor> canon_texts_tensors;
///canon_texts_tensors.push_back(ClipProcessor->EncodeText(std::string("объект")));
///canon_texts_tensors.push_back(ClipProcessor->EncodeText(std::string("вещи")));
///canon_texts_tensors.push_back(ClipProcessor->EncodeText(std::string("текстура")));
///int negatives_len =  (int)canon_texts_tensors.size();
///auto canon_features = Clip->EncodeText(torch::stack(canon_texts_tensors).to(Device)).to(torch::kCPU); ///[3, 768]
///canon_features = canon_features / canon_features.norm(2/*L2*/, -1, true);
///auto input = ClipProcessor->EncodeText(std::string("малый барабан"));
///auto text_features = Clip->EncodeText(input.unsqueeze(0).to(Device)).to(torch::kCPU);		///[1, 768]
///text_features = text_features / text_features.norm(2/*L2*/, -1, true);
///torch::Tensor image_features = PyramidClipEmbedding.GetPixelValue(i,j,0.5f,img_id,pyramid_embedder_properties,cv::Size(data.W, data.H)).to(torch::kCPU);
///image_features = image_features / image_features.norm(2/*L2*/, -1, true);
///torch::Tensor rel = Relevancy(image_features, text_features, canon_features);
///float lv = rel.index({0,0}).item<float>();
inline torch::Tensor Relevancy(torch::Tensor embeds, torch::Tensor positives, torch::Tensor negatives)
{
	auto embeds2 = torch::cat({ positives, negatives });
	auto logits = /*scale * */torch::mm(embeds, embeds2.t());              // [batch_size x phrases]
	auto positive_vals = logits.index({ "...", torch::indexing::Slice(0, 1) }); // [batch_size x 1]
	auto negative_vals = logits.index({ "...", torch::indexing::Slice(1, torch::indexing::None) });		// [batch_size x negative_phrase_n]
	auto repeated_pos = positive_vals.repeat({ 1, negatives.sizes()[0] }); // [batch_size x negative_phrase_n]
	auto sims = torch::stack({ repeated_pos, negative_vals }, -1);         // [batch_size x negative_phrase_n x 2]
	auto smx = torch::softmax(10 * sims, -1);                              // [batch_size x negative_phrase_n x 2]
	auto best_id = smx.index({ "...", 0 }).argmin(1);                      // [batch_size x 2]
	auto result = torch::gather(smx, 1, best_id.index({
		"...", torch::indexing::None, torch::indexing::None 
		}).expand(
			{ best_id.sizes()[0], negatives.sizes()[0], 2
			})
	).index(
		{ torch::indexing::Slice(), 0, torch::indexing::Slice()
		});// [batch_size x 2]
	return result;
}
