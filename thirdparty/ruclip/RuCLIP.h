#pragma once

#include "json.hpp"
#include <filesystem>
#include <fstream>

#include "TorchHeader.h"

///to handle fp16
class RCLayerNormImpl : public torch::nn::LayerNormImpl {
protected:
public:
	RCLayerNormImpl(std::vector<int64_t> normalized_shape) : LayerNormImpl(normalized_shape) {}
	virtual ~RCLayerNormImpl() {}

	torch::Tensor forward(const torch::Tensor &x) ///!override
	{ 
		auto orig_type = x.dtype();
		auto result = torch::nn::LayerNormImpl::forward(x.to(torch::kFloat32));
		return result.to(orig_type);
	}
};
TORCH_MODULE(RCLayerNorm);


class QuickGELUImpl : public torch::nn::Module {
protected:
public:
	QuickGELUImpl() : torch::nn::Module() {}
	virtual ~QuickGELUImpl() {}

	torch::Tensor forward(const torch::Tensor &x)
	{
		return x * torch::sigmoid(1.702f * x);
	}
};
TORCH_MODULE(QuickGELU);


class ResidualAttentionBlockImpl : public torch::nn::Module {
protected:
	torch::nn::MultiheadAttention Attn{ nullptr };
	RCLayerNorm Ln1{ nullptr };
	torch::nn::Sequential Mlp{ nullptr };
	RCLayerNorm Ln2{ nullptr };
	torch::Tensor AttnMask;
public:
	ResidualAttentionBlockImpl(const std::string& module_name, const int d_model, const int n_head, const torch::Tensor& attn_mask);
	virtual ~ResidualAttentionBlockImpl() {}
	torch::Tensor Attention(const torch::Tensor &x);
	torch::Tensor forward(const torch::Tensor &x);
	torch::nn::MultiheadAttention GetAttn() { return Attn; }
	torch::nn::Sequential GetMlp() { return Mlp; }
};
TORCH_MODULE(ResidualAttentionBlock);


class TransformerImpl : public torch::nn::Module {
protected:
	int Width,
		Layers,
		Heads;
	//torch::Tensor AttnMask;
	torch::nn::Sequential Resblocks;
public:
	TransformerImpl(const std::string &module_name, const int width, const int layers, const int heads, const torch::Tensor &attn_mask = torch::Tensor());
	virtual ~TransformerImpl() {}
	torch::Tensor forward(const torch::Tensor &x);
	void InitializeParameters();
};
TORCH_MODULE(Transformer);


class VisionTransformerImpl : public torch::nn::Module {
protected:
	int InputResolution,
		OutputDim;
	torch::nn::Conv2d Conv1{ nullptr };
	torch::Tensor ClassEmbedding,
		PositionalEmbedding,
		Proj;
	RCLayerNorm LnPre{ nullptr },
		LnPost{ nullptr };
	Transformer VTTransformer{ nullptr };
public:
	VisionTransformerImpl(
		const std::string& module_name,
		const int input_resolution,
		const int patch_size,
		const int width,
		const int layers,
		const int heads,
		const int output_dim
	);
	virtual ~VisionTransformerImpl() {}
	torch::Tensor forward(const torch::Tensor& x);
};
TORCH_MODULE(VisionTransformer);


class CLIPImpl : public torch::nn::Module {
protected:
	int EosId,
		//EmbedDim,
		//ImageResolution,
		//VisionLayers,
		//VisionWidth,
		//VisionPatchSize,
		ContextLength,
		VocabSize,
		TransformerWidth,
		/*TransformerHeads,*/
		TransformerLayers;
	VisionTransformer Visual{ nullptr };
	Transformer NVTransformer{ nullptr };
	torch::nn::Embedding TokenEmbedding{ nullptr };
	torch::Tensor PositionalEmbedding;
	RCLayerNorm LnFinal{ nullptr };
	torch::Tensor TextProjection,
		LogitScale;
public:
	CLIPImpl(
		const std::string &module_name,
		const int embed_dim,
		const int image_resolution,
		const int vision_layers,
		const int vision_width,
		const int vision_patch_size,
		const int context_length,
		const int vocab_size,
		const int transformer_width,
		const int transformer_heads,
		const int transformer_layers,
		const int eos_id = 3
	);
	virtual ~CLIPImpl() {}

	void InitializeParameters();
	torch::Tensor BuildAttentionMask();

	//auto dtype()
	//{
	//	return Visual.conv1.weight.dtype();
	//}
	
	///pixel_values : torch::Tensor	Processed images from RuCLIPProcessor class, out: image_latents : torch::Tensor Image embeddings
	torch::Tensor EncodeImage(torch::Tensor pixel_values);

	///input_ids : torch::Tensor Tokenized texts from RuCLIPProcessor class, out: text_latents : torch::Tensor Text embeddings
	torch::Tensor EncodeText(torch::Tensor input_ids);

	torch::Tensor forward(torch::Tensor input_ids, torch::Tensor pixel_values);

	torch::Tensor GetLogitScale() { return LogitScale; }
};
TORCH_MODULE(CLIP);

inline CLIP FromPretrained(const std::filesystem::path &folder)
{
	using json = nlohmann::json;
	std::filesystem::path path = folder / "config.json";
	std::cout << path << std::endl;
	std::ifstream f(path);
	json config = json::parse(f);

	// Создание модели
	auto clip = CLIP(
		"ruclip",
		int(config["embed_dim"]),
		int(config["image_resolution"]),
		int(config["vision_layers"]),
		int(config["vision_width"]),
		int(config["vision_patch_size"]),
		int(config["context_length"]),
		int(config["vocab_size"]),
		int(config["transformer_width"]),
		int(config["transformer_heads"]),
		int(config["transformer_layers"])
	);

	for (auto &k : clip->named_parameters())
		std::cout << k.key() << std::endl;
	//std::cout << "Model params count: " << Trainable::ParamsCount(clip) << std::endl;

	// Загрузка состояния модели из файла
	try {
		torch::load(clip, (folder / "jit_model.zip").string());
	}
	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
	}

	//		"mean" : [0.48145466, 0.4578275, 0.40821073] ,
	//		"std" : [0.26862954, 0.26130258, 0.27577711]

	return clip;
}
