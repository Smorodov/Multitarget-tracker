#include "RuCLIP.h"

ResidualAttentionBlockImpl :: ResidualAttentionBlockImpl(const std::string &module_name, const int d_model, const int n_head, const torch::Tensor &attn_mask)
	: torch::nn::Module(module_name)
{
	Attn = torch::nn::MultiheadAttention(d_model, n_head);
	Ln1 = RCLayerNorm(std::vector<int64_t>() = { (int64_t)d_model });
	Mlp = torch::nn::Sequential({
		{"c_fc", torch::nn::Linear(d_model, d_model * 4)},
		{"gelu", QuickGELU()},
		{"c_proj", torch::nn::Linear(d_model * 4, d_model)}
		});
	Ln2 = RCLayerNorm(std::vector<int64_t>() = { (int64_t)d_model });
	AttnMask = attn_mask;

	register_module("attn", Attn);
	register_module("ln_1", Ln1);
	register_module("mlp", Mlp);
	register_module("ln_2", Ln2);
	//register_buffer("attn_mask", AttnMask);
}

torch::Tensor ResidualAttentionBlockImpl :: Attention(const torch::Tensor &x)
{
	if (AttnMask.defined() && (AttnMask.numel() != 0))
		AttnMask = AttnMask.to(x.dtype()).to(x.device());
	/*return Attn(x, x, x, weights = False, attn_mask = self.attn_mask)[0];*/
	//std::tuple<Tensor, Tensor> forward(const Tensor & query, const Tensor & key, const Tensor & value, const Tensor & key_padding_mask = {}, bool need_weights = true, const Tensor & attn_mask = {}, bool average_attn_weights = true)
	return std::get<0>(Attn->forward(x, x, x, {}, false, AttnMask));
}

torch::Tensor ResidualAttentionBlockImpl :: forward(const torch::Tensor &x)
{
	auto result = x + Attention(Ln1(x));
	result = result + Mlp->forward(Ln2(result));
	return result;
}



TransformerImpl :: TransformerImpl(const std::string &module_name, const int width, const int layers, const int heads, const torch::Tensor &attn_mask /*= torch::Tensor()*/)
	: torch::nn::Module(module_name), Width(width), Layers(layers), Heads(heads)/*, AttnMask(attn_mask)*/
{
	for (int i = 0; i < layers; i++)
		Resblocks->push_back(ResidualAttentionBlock(module_name + "_" + std::to_string(i), width, heads, attn_mask));

	register_module("resblocks", Resblocks);		//???
	//for (int i = 0; i < Resblocks->size(); i++)
	//	register_module(module_name + "_res_attn_block_" + std::to_string(i), Resblocks[i]);
}

torch::Tensor TransformerImpl :: forward(const torch::Tensor& x)
{
	//!!!Сделать проверку и преобразование if (x.type() != )
	return Resblocks->forward(x);
}

void TransformerImpl :: InitializeParameters()
{
	float proj_std = powf(Width, -0.5f) * pow(2 * Layers, -0.5f);
	float attn_std = powf(Width, -0.5f);
	float fc_std = powf(2 * Width, -0.5f);

	for (int i = 0; i < Resblocks->size(); i++)
	{
		auto block = Resblocks[i]->as<ResidualAttentionBlock>();
		torch::nn::init::normal_(block->GetAttn()->in_proj_weight, 0., attn_std);
		torch::nn::init::normal_(block->GetAttn()->out_proj->weight, 0., proj_std);
		auto mlp = block->GetMlp();
		for (int j = 0; j < mlp->size(); j++)
		{
			if (mlp[j]->name() == "c_fc")
				torch::nn::init::normal_(mlp[j]->as<torch::nn::Linear>()->weight, 0., fc_std);
			if (mlp[j]->name() == "c_proj")
				torch::nn::init::normal_(mlp[j]->as<torch::nn::Linear>()->weight, 0., proj_std);
		}
	}
}



VisionTransformerImpl :: VisionTransformerImpl(
	const std::string &module_name,
	const int input_resolution,
	const int patch_size,
	const int width,
	const int layers,
	const int heads,
	const int output_dim
) : torch::nn::Module(module_name), InputResolution(input_resolution), OutputDim(output_dim)
{
	Conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, width, patch_size).stride(patch_size).bias(false));
	float scale = powf(width, -0.5);
	ClassEmbedding = scale * torch::randn(width);
	PositionalEmbedding = scale * torch::randn({ (int)pow(input_resolution / patch_size/*деление нацело*/, 2) + 1, width });
	LnPre = RCLayerNorm(std::vector<int64_t>() = { (int64_t)width });
	VTTransformer = Transformer("visual", width, layers, heads);
	LnPost = RCLayerNorm(std::vector<int64_t>() = { (int64_t)width });
	Proj = scale * torch::randn({ width, output_dim });

	register_buffer("class_embedding", ClassEmbedding);
	register_buffer("positional_embedding", PositionalEmbedding);
	register_buffer("proj", Proj);
	register_module("conv1", Conv1);
	register_module("ln_pre", LnPre);
	register_module("ln_post", LnPost);
	register_module("transformer", VTTransformer);
}

torch::Tensor VisionTransformerImpl :: forward(const torch::Tensor &x)
{
	//!!!Сделать проверку и преобразование if (x.type() != )
	auto res = Conv1(x);																					//shape = [*, width, grid, grid]
	res = res.reshape({ res.sizes()[0], res.sizes()[1], -1 });		//shape = [*, width, grid **2]
	res = res.permute({ 0, 2, 1 });																//shape = [*, grid **2, width]
	res = torch::cat({
			ClassEmbedding.to(res.dtype()) + torch::zeros({res.sizes()[0], 1, res.sizes().back()}, res.dtype()).to(x.device()),
			res
		}, 1);			//shape = [*, grid **2 + 1, width]
	res = res + PositionalEmbedding.to(res.dtype());
	res = LnPre(res);
	res = res.permute({ 1, 0, 2 });  // NLD->LND
	res = VTTransformer(res);
	res = res.permute({ 1, 0, 2 });  // LND->NLD
	res = LnPost(res.index({ torch::indexing::Slice(), 0, torch::indexing::Slice() }));
	if (Proj.defined() && Proj.numel() != 0)
		res = torch::mm(res, Proj);
	return res;
}



CLIPImpl :: CLIPImpl(
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
	const int eos_id /*= 3*/
) : torch::nn::Module(module_name), EosId(eos_id), ContextLength(context_length), VocabSize(vocab_size), TransformerWidth(transformer_width), TransformerLayers(transformer_layers)
{
	int vision_heads = vision_width / 64;
	Visual = VisionTransformer("visual", image_resolution, vision_patch_size, vision_width, vision_layers, vision_heads, embed_dim);
	NVTransformer = Transformer("transformer", transformer_width, transformer_layers, transformer_heads, BuildAttentionMask());

	TokenEmbedding = torch::nn::Embedding(vocab_size, transformer_width);
	PositionalEmbedding = torch::empty({ context_length, transformer_width });	//!!!type, device
	
	std::cout << "transformer_width: " << transformer_width<< std::endl;
	
	LnFinal = RCLayerNorm(std::vector<int64_t>() = { (int64_t)transformer_width });
	TextProjection = torch::empty({ transformer_width, embed_dim });	//!!!type, device
	LogitScale = torch::ones({}) * logf(1.f / 0.07f);

	register_module("visual", Visual);
	register_module("transformer", NVTransformer);
	register_module("token_embedding", TokenEmbedding);
	register_module("ln_final", LnFinal);
	register_buffer("positional_embedding", PositionalEmbedding);
	register_buffer("text_projection", TextProjection);
	register_buffer("logit_scale", LogitScale);

	InitializeParameters();
}

void CLIPImpl :: InitializeParameters()
{
	torch::nn::init::normal_(TokenEmbedding->weight, 0., 0.02);
	torch::nn::init::normal_(PositionalEmbedding, 0., 0.01);
	NVTransformer->InitializeParameters();
	if (TextProjection.defined() && TextProjection.numel() != 0)
		torch::nn::init::normal_(TextProjection, 0., pow(TransformerWidth, -0.5));
}

torch::Tensor CLIPImpl :: BuildAttentionMask()
{
	auto mask = torch::empty({ ContextLength, ContextLength });
	mask.fill_(-std::numeric_limits<float>::infinity());
	mask.triu_(1);
	return mask;
}

///pixel_values : torch::Tensor	Processed images from RuCLIPProcessor class, out: image_latents : torch::Tensor Image embeddings
torch::Tensor CLIPImpl :: EncodeImage(torch::Tensor pixel_values)
{
	return Visual(pixel_values);
}

///input_ids : torch::Tensor Tokenized texts from RuCLIPProcessor class, out: text_latents : torch::Tensor Text embeddings
torch::Tensor CLIPImpl :: EncodeText(torch::Tensor input_ids)
{
	auto x = TokenEmbedding(input_ids);	//.type(dtype())  // [batch_size, n_ctx, d_model]
	x = x + PositionalEmbedding;				//.type(dtype())
	x = x.permute({ 1, 0, 2 });					//NLD->LND
	x = NVTransformer(x);
	x = x.permute({ 1, 0, 2 });					//LND->NLD
	x = LnFinal(x);											// type(self.dtype) //x.shape = [batch_size, n_ctx, transformer.width]
	x = torch::mm(x.index({ torch::arange(x.sizes()[0]), torch::where(input_ids == EosId)[1] }), TextProjection);
	return x;
}

torch::Tensor CLIPImpl :: forward(torch::Tensor input_ids, torch::Tensor pixel_values)
{
	auto image_features = EncodeImage(pixel_values);
	auto text_features = EncodeText(input_ids);

	//normalize features
	image_features = image_features / image_features.norm(2/*L2*/, -1, true);
	text_features = text_features / text_features.norm(2/*L2*/, -1, true);

	//cosine similarity as logits
	auto scale = LogitScale.exp();
	auto logits_per_image = scale * torch::mm(image_features, text_features.t());
	auto logits_per_text = logits_per_image.t();

	return logits_per_image;
}