#include "TorchHeader.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <filesystem>
#include <string>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "RuCLIP.h"
#include "RuCLIPProcessor.h"

int main(int argc, const char* argv[])
{
	setlocale(LC_ALL, "");

	const char* keys =
	{
		"{ imgs             |img1.jpg,img2.jpg   | List of jpegs | }"
		"{ text             |cat,bear,fox        | List of words | }"
		"{ clip             |../data/ruclip-vit-large-patch14-336 | Path to ruClip model | }"
		"{ bpe              |../data/ruclip-vit-large-patch14-336/bpe.model | Play a video to this position (if 0 then played to the end of file) | }"
		"{ img_size         |336                 | Input model size | }"
	};
	
	cv::CommandLineParser parser(argc, argv, keys);
	parser.printMessage();

	std::string imagesStr = parser.get<std::string>("imgs");
	std::string labelsStr = parser.get<std::string>("text");
	std::string pathToClip = parser.get<std::string>("clip");
	std::string pathToBPE = parser.get<std::string>("bpe");
	int INPUT_IMG_SIZE = parser.get<int>("img_size");

	torch::manual_seed(24);

	torch::Device device(torch::kCPU);
	if (torch::cuda::is_available())
	{
		std::cout << "CUDA is available! Running on GPU." << std::endl;
		device = torch::Device(torch::kCUDA);
	}	else {
		std::cout << "CUDA is not available! Running on CPU." << std::endl;
	}

	std::cout << "Load clip from: " << pathToClip << std::endl;
	CLIP clip = FromPretrained(pathToClip);
	clip->to(device);

	std::cout << "Load processor from: " << pathToBPE << std::endl;
	RuCLIPProcessor processor(
		pathToBPE,
		INPUT_IMG_SIZE,
		77,
		{ 0.48145466, 0.4578275, 0.40821073 },
		{ 0.26862954, 0.26130258, 0.27577711 }
	);

	std::vector<cv::Mat> images;
	{
		std::cout << "images: " << std::endl;
		std::regex sep("[,]+");
		std::sregex_token_iterator tokens(imagesStr.cbegin(), imagesStr.cend(), sep, -1);
		std::sregex_token_iterator end;
		for (; tokens != end; ++tokens)
		{
			cv::Mat img = cv::imread(*tokens, cv::IMREAD_COLOR);

			std::cout << (*tokens) << " is loaded: " << !img.empty() << std::endl;

			cv::resize(img, img, cv::Size(INPUT_IMG_SIZE, INPUT_IMG_SIZE), cv::INTER_CUBIC);
			images.emplace_back(img);
		}
	}

	//Завести метки
	std::vector<std::string> labels;
	{
		std::cout << "labels: ";
		std::regex sep("[,]+");
		std::sregex_token_iterator tokens(labelsStr.cbegin(), labelsStr.cend(), sep, -1);
		std::sregex_token_iterator end;
		for (; tokens != end; ++tokens)
		{
			std::cout << (*tokens) << " | ";
			labels.emplace_back(*tokens);
		}
		std::cout << std::endl;
	}

	std::cout << "Running..." << std::endl;
	auto dummy_input = processor(labels, images);
	try {
		torch::Tensor logits_per_image = clip->forward(dummy_input.first.to(device), dummy_input.second.to(device));
		torch::Tensor logits_per_text = logits_per_image.t();
		auto probs = logits_per_image.softmax(/*dim = */-1).detach().cpu();
		std::cout << "probs per image: " << probs << std::endl;
	}	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
	}
	std::cout << "The end!" << std::endl;
}