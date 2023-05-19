/**
MIT License

Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*
*/

#include "trt_utils.h"
#include <NvInferRuntimeCommon.h>

#ifdef HAVE_FILESYSTEM
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include <fstream>
#include <iomanip>

namespace
{
using namespace nvinfer1;
REGISTER_TENSORRT_PLUGIN(MishPluginCreator);
REGISTER_TENSORRT_PLUGIN(ChunkPluginCreator);
REGISTER_TENSORRT_PLUGIN(HardswishPluginCreator);
REGISTER_TENSORRT_PLUGIN(YoloLayerPluginCreator);
}

void blobFromDsImages(const std::vector<DsImage>& inputImages, cv::Mat& blob, const int& inputH, const int& inputW)
{
#if 0
    std::vector<cv::Mat> letterboxStack;
    letterboxStack.reserve(inputImages.size());
    for (uint32_t i = 0; i < inputImages.size(); ++i)
    {
		letterboxStack.emplace_back(inputImages[i].getLetterBoxedImage());
    }
    blob = cv::dnn::blobFromImages(letterboxStack, 1.0, cv::Size(inputW, inputH), cv::Scalar(0.0, 0.0, 0.0), true);

#else
    cv::Size size(inputW, inputH);
    constexpr bool swapRB = true;
    constexpr int ddepth = CV_32F;
    constexpr int nch = 3;
    size_t nimages = inputImages.size();

    int sz[] = { (int)nimages, nch, inputH, inputW };
    blob.create(4, sz, ddepth);
    cv::Mat ch[4];

    for (size_t i = 0; i < nimages; ++i)
    {
        const cv::Mat& image = inputImages[i].getLetterBoxedImage();

        for (int j = 0; j < nch; ++j)
        {
            ch[j] = cv::Mat(size, ddepth, blob.ptr((int)i, j));
        }

        if(swapRB)
            std::swap(ch[0], ch[2]);

        for (int y = 0; y < image.rows; ++y)
        {
            const uchar* imPtr = image.ptr(y);
            float* ch0 = ch[0].ptr<float>(y);
            float* ch1 = ch[1].ptr<float>(y);
            float* ch2 = ch[2].ptr<float>(y);
            constexpr size_t stepSize = 32;
            for (int x = 0; x < image.cols; x += stepSize)
            {
                for (size_t k = 0; k < stepSize; ++k)
                {
                    ch0[k] = static_cast<float>(imPtr[0 + 3 * k]);
                    ch1[k] = static_cast<float>(imPtr[1 + 3 * k]);
                    ch2[k] = static_cast<float>(imPtr[2 + 3 * k]);
                }
                imPtr += 3 * stepSize;
                ch0 += stepSize;
                ch1 += stepSize;
                ch2 += stepSize;
            }
        }
    }
#endif
}

static void leftTrim(std::string& s)
{
    s.erase(s.begin(), find_if(s.begin(), s.end(), [](int ch) { return !isspace(ch); }));
}

static void rightTrim(std::string& s)
{
    s.erase(find_if(s.rbegin(), s.rend(), [](int ch) { return !isspace(ch); }).base(), s.end());
}

std::string trim(std::string s)
{
    leftTrim(s);
    rightTrim(s);
    return s;
}

std::string triml(std::string s,const char* t)
{
	s.erase(0, s.find_first_not_of(t));
	return s;
}
std::string trimr(std::string s, const char* t)
{
	s.erase(s.find_last_not_of(t) + 1);
	return s;
}

float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

bool fileExists(const std::string fileName, bool verbose)
{
    if (!fs::exists(fs::path(fileName)))
    {
        if (verbose)
            std::cout << "File does not exist : " << fileName << std::endl;
        return false;
    }
    return true;
}

BBox convertBBoxNetRes(const float& bx, const float& by, const float& bw, const float& bh,
                       const uint32_t& stride, const uint32_t& netW, const uint32_t& netH)
{
    BBox b;
    // Restore coordinates to network input resolution
    float x = bx * stride;
    float y = by * stride;

    b.x1 = x - bw / 2;
    b.x2 = x + bw / 2;

    b.y1 = y - bh / 2;
    b.y2 = y + bh / 2;

    b.x1 = clamp(b.x1, 0.f, static_cast<float>(netW));
    b.x2 = clamp(b.x2, 0.f, static_cast<float>(netW));
    b.y1 = clamp(b.y1, 0.f, static_cast<float>(netH));
    b.y2 = clamp(b.y2, 0.f, static_cast<float>(netH));

    return b;
}

void convertBBoxImgRes(const float scalingFactor,
	const float xOffset,
	const float yOffset,
    BBox& bbox)
{
	    //// Undo Letterbox
    bbox.x1 -= xOffset;
    bbox.x2 -= xOffset;
    bbox.y1 -= yOffset;
    bbox.y2 -= yOffset;
//// Restore to input resolution
	bbox.x1 /= scalingFactor;
	bbox.x2 /= scalingFactor;
	bbox.y1 /= scalingFactor;
	bbox.y2 /= scalingFactor;
	std::cout << "convertBBoxImgRes" << std::endl;    
}

void printPredictions(const BBoxInfo& b, const std::string& className)
{
    std::cout << " label:" << b.label << "(" << className << ")"
              << " confidence:" << b.prob << " xmin:" << b.box.x1 << " ymin:" << b.box.y1
              << " xmax:" << b.box.x2 << " ymax:" << b.box.y2 << std::endl;
}

std::vector<std::string> loadListFromTextFile(const std::string filename)
{
    assert(fileExists(filename));
    std::vector<std::string> list;

    std::ifstream f(filename);
    if (!f)
    {
        std::cout << "failed to open " << filename;
        assert(0);
    }

    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty())
            continue;
        else
            list.push_back(trim(line));
    }
    return list;
}

std::vector<std::string> loadImageList(const std::string filename, const std::string prefix)
{
    std::vector<std::string> fileList = loadListFromTextFile(filename);
    for (auto& file : fileList)
    {
        if (fileExists(file, false))
            continue;
        else
        {
            std::string prefixed = prefix + file;
            if (fileExists(prefixed, false))
                file = prefixed;
            else
                std::cerr << "WARNING: couldn't find: " << prefixed
                          << " while loading: " << filename << std::endl;
        }
    }
    return fileList;
}


std::vector<BBoxInfo> nmsAllClasses(const float nmsThresh,
	std::vector<BBoxInfo>& binfo,
	const uint32_t numClasses,
	tensor_rt::ModelType model_type)
{
    std::vector<BBoxInfo> result;
    std::vector<std::vector<BBoxInfo>> splitBoxes(numClasses);
    for (auto& box : binfo)
    {
        splitBoxes.at(box.label).push_back(box);
    }

    for (auto& boxes : splitBoxes)
    {
        if (tensor_rt::YOLOV5 == model_type)
            boxes = diou_nms(nmsThresh, boxes);
        else
            boxes = nonMaximumSuppression(nmsThresh, boxes);
        result.insert(result.end(), boxes.begin(), boxes.end());
    }
    return result;
}


std::vector<BBoxInfo> diou_nms(const float nmsThresh, std::vector<BBoxInfo> binfo)
{
	auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float
	{
		if (x1min > x2min)
		{
			std::swap(x1min, x2min);
			std::swap(x1max, x2max);
		}
		return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
	};
	auto computeIoU = [&overlap1D](BBox& bbox1, BBox& bbox2) -> float
	{
		float overlapX = overlap1D(bbox1.x1, bbox1.x2, bbox2.x1, bbox2.x2);
		float overlapY = overlap1D(bbox1.y1, bbox1.y2, bbox2.y1, bbox2.y2);
		float area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
		float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
		float overlap2D = overlapX * overlapY;
		float u = area1 + area2 - overlap2D;
		return u == 0 ? 0 : overlap2D / u;
	};

	//https://arxiv.org/pdf/1911.08287.pdf
	auto R = [](BBox &bbox1,BBox &bbox2) ->float
	{
		float center1_x = (bbox1.x1 + bbox1.x2) / 2.f;
		float center1_y = (bbox1.y1 + bbox1.y2) / 2.f;
		float center2_x = (bbox2.x1 + bbox2.x2) / 2.f;
		float center2_y = (bbox2.y1 + bbox2.y2) / 2.f;

		float d_center = (center1_x - center2_x)* (center1_x - center2_x) 
						+ (center1_y - center2_y)*(center1_y - center2_y);
		//smallest_enclosing box
		float box_x1 = std::min({ bbox1.x1, bbox1.x2, bbox2.x1, bbox2.x2 });
		float box_y1 = std::min({ bbox1.y1, bbox1.y2, bbox2.y1, bbox2.y2 });
		float box_x2 = std::max({ bbox1.x1, bbox1.x2, bbox2.x1, bbox2.x2 });
		float box_y2 = std::max({ bbox1.y1, bbox1.y2, bbox2.y1, bbox2.y2 });

		float d_diagonal = (box_x1 - box_x2) * (box_x1 - box_x2) +
						   (box_y1 - box_y2) * (box_y1 - box_y2);

		return d_center / d_diagonal;
	};
	std::stable_sort(binfo.begin(), binfo.end(),
		[](const BBoxInfo& b1, const BBoxInfo& b2) { return b1.prob > b2.prob; });
	std::vector<BBoxInfo> out;
	for (auto& i : binfo)
	{
		bool keep = true;
		for (auto& j : out)
		{
			if (keep)
			{
				float overlap = computeIoU(i.box, j.box);
				float r = R(i.box, j.box);
				keep = (overlap-r) <= nmsThresh;
			}
			else
				break;
		}
		if (keep) out.push_back(i);
	}
	return out;
}


std::vector<BBoxInfo> nonMaximumSuppression(const float nmsThresh, std::vector<BBoxInfo> binfo)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float 
	{
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };
    auto computeIoU = [&overlap1D](BBox& bbox1, BBox& bbox2) -> float 
	{
        float overlapX = overlap1D(bbox1.x1, bbox1.x2, bbox2.x1, bbox2.x2);
        float overlapY = overlap1D(bbox1.y1, bbox1.y2, bbox2.y1, bbox2.y2);
        float area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
        float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::stable_sort(binfo.begin(), binfo.end(),
                     [](const BBoxInfo& b1, const BBoxInfo& b2) { return b1.prob > b2.prob; });
    std::vector<BBoxInfo> out;
    for (auto& i : binfo)
    {
        bool keep = true;
        for (auto& j : out)
        {
            if (keep)
            {
                float overlap = computeIoU(i.box, j.box);
                keep = overlap <= nmsThresh;
            }
            else
                break;
        }
        if (keep) out.push_back(i);
    }
    return out;
}

nvinfer1::ICudaEngine* loadTRTEngine(const std::string planFilePath, /*PluginFactory* pluginFactory,*/
                                     Logger& logger)
{
    // reading the model in memory
    std::cout << "Loading TRT Engine..." << std::endl;
    assert(fileExists(planFilePath));
    std::stringstream trtModelStream;
    trtModelStream.seekg(0, trtModelStream.beg);
    std::ifstream cache(planFilePath,std::ios::binary | std::ios::in);
    assert(cache.good());
    trtModelStream << cache.rdbuf();
    cache.close();

    // calculating model size
    trtModelStream.seekg(0, std::ios::end);
    const auto modelSize = trtModelStream.tellg();
    trtModelStream.seekg(0, std::ios::beg);
    void* modelMem = malloc(modelSize);
    trtModelStream.read((char*) modelMem, modelSize);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelMem, modelSize);
    free(modelMem);
#if (NV_TENSORRT_MAJOR < 8)
	runtime->destroy();
#else
    delete runtime;
#endif
    std::cout << "Loading Complete!" << std::endl;

    return engine;
}

//{
//	std::ifstream file(weightsFilePath, std::ios_base::binary);
//	assert(file.good());
//	std::string line;
//	file.ignore(4);
//	char buf[2];
//	file.read(buf, 1);
//	if ((int)(unsigned char)buf[0] == 1)
//	{
//		file.ignore(11);
//	}
//	else if ((int)(unsigned char)buf[0] == 2)
//	{
//		file.ignore(15);
//	}
//}
std::vector<float> LoadWeights(const std::string weightsFilePath)
{
    assert(fileExists(weightsFilePath));
    std::cout << "Loading pre-trained weights..." << std::endl;
    std::ifstream file(weightsFilePath, std::ios_base::binary);
	assert(file.good());
	std::string line;
	file.ignore(4);
	char buf[2];
	file.read(buf, 1);
	if ((int)(unsigned char)buf[0] == 1)
	{
		file.ignore(11);
	}
	else if ((int)(unsigned char)buf[0] == 2)
	{
		file.ignore(15);
	}
    else
    {
        std::cout << "Invalid network type" << std::endl;
        assert(0);
    }

    std::vector<float> weights;
    char* floatWeight = new char[4];
    while (!file.eof())
    {
        file.read(floatWeight, 4);
        assert(file.gcount() == 4);
        weights.push_back(*reinterpret_cast<float*>(floatWeight));
        if (file.peek() == std::istream::traits_type::eof()) break;
    }
    std::cout << "Loading complete!" << std::endl;
    delete[] floatWeight;

   // std::cout << "Total Number of weights read : " << weights.size() << std::endl;
    return weights;
}

std::string dimsToString(const nvinfer1::Dims d)
{
    std::stringstream s;
    assert(d.nbDims >= 1);
    for (int i = 0; i < d.nbDims - 1; ++i)
    {
        s << std::setw(4) << d.d[i] << " x";
    }
    s << std::setw(4) << d.d[d.nbDims - 1];

    return s.str();
}

void displayDimType(const nvinfer1::Dims d)
{
    std::cout << "(" << d.nbDims << ") ";
    for (int i = 0; i < d.nbDims; ++i)
    {
		
   //     switch (d.type[i])
   //     {
			////nvinfer1::DimensionOperation::
   //     case nvinfer1::DimensionOperation::kSPATIAL: std::cout << "kSPATIAL "; break;
   //     case nvinfer1::DimensionOperation::kCHANNEL: std::cout << "kCHANNEL "; break;
   //     case nvinfer1::DimensionOperation::kINDEX: std::cout << "kINDEX "; break;
   //     case nvinfer1::DimensionOperation::kSEQUENCE: std::cout << "kSEQUENCE "; break;
   //     }
    }
    std::cout << std::endl;
}

int getNumChannels(nvinfer1::ITensor* t)
{
    nvinfer1::Dims d = t->getDimensions();
    assert(d.nbDims == 3);

    return d.d[0];
}

uint64_t get3DTensorVolume(nvinfer1::Dims inputDims)
{
    assert(inputDims.nbDims == 3);
    return inputDims.d[0] * inputDims.d[1] * inputDims.d[2];
}

nvinfer1::ILayer* netAddMaxpool(int layerIdx, std::map<std::string, std::string>& block,
                                nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
    assert(block.at("type") == "maxpool");
    assert(block.find("size") != block.end());
    assert(block.find("stride") != block.end());

    int size = std::stoi(block.at("size"));
    int stride = std::stoi(block.at("stride"));

    nvinfer1::IPoolingLayer* pool
        = network->addPoolingNd(*input, nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{size, size});
    assert(pool);
    std::string maxpoolLayerName = "maxpool_" + std::to_string(layerIdx);
	int pad = (size - 1) / 2;
	pool->setPaddingNd(nvinfer1::DimsHW{pad,pad});
    pool->setStrideNd(nvinfer1::DimsHW{stride, stride});
    pool->setName(maxpoolLayerName.c_str());

    return pool;
}

nvinfer1::ILayer* netAddConvLinear(int layerIdx, std::map<std::string, std::string>& block,
                                   std::vector<float>& weights,
                                   std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
                                   int& inputChannels, nvinfer1::ITensor* input,
                                   nvinfer1::INetworkDefinition* network)
{
    assert(block.at("type") == "convolutional");
    assert(block.find("batch_normalize") == block.end());
    assert(block.at("activation") == "linear");
    assert(block.find("filters") != block.end());
    assert(block.find("pad") != block.end());
    assert(block.find("size") != block.end());
    assert(block.find("stride") != block.end());

    int filters = std::stoi(block.at("filters"));
    int padding = std::stoi(block.at("pad"));
    int kernelSize = std::stoi(block.at("size"));
    int stride = std::stoi(block.at("stride"));
    int pad;
    if (padding)
        pad = (kernelSize - 1) / 2;
    else
        pad = 0;
    // load the convolution layer bias
    nvinfer1::Weights convBias{nvinfer1::DataType::kFLOAT, nullptr, filters};
    float* val = new float[filters];
    for (int i = 0; i < filters; ++i)
    {
        val[i] = weights[weightPtr];
        weightPtr++;
    }
    convBias.values = val;
    trtWeights.push_back(convBias);
    // load the convolutional layer weights
    int size = filters * inputChannels * kernelSize * kernelSize;
    nvinfer1::Weights convWt{nvinfer1::DataType::kFLOAT, nullptr, size};
    val = new float[size];
    for (int i = 0; i < size; ++i)
    {
        val[i] = weights[weightPtr];
        weightPtr++;
    }
    convWt.values = val;
    trtWeights.push_back(convWt);
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(*input, filters, nvinfer1::DimsHW{kernelSize, kernelSize}, convWt, convBias);
    assert(conv != nullptr);
    std::string convLayerName = "conv_" + std::to_string(layerIdx);
    conv->setName(convLayerName.c_str());
    conv->setStrideNd(nvinfer1::DimsHW{stride, stride});
    conv->setPaddingNd(nvinfer1::DimsHW{pad, pad});

    return conv;
}

nvinfer1::ILayer* net_conv_bn_mish(int layerIdx, 
	std::map<std::string, std::string>& block,
	std::vector<float>& weights,
	std::vector<nvinfer1::Weights>& trtWeights,
	int& weightPtr,
	int& inputChannels,
	nvinfer1::ITensor* input,
	nvinfer1::INetworkDefinition* network)
{
	assert(block.at("type") == "convolutional");
	assert(block.find("batch_normalize") != block.end());
	assert(block.at("batch_normalize") == "1");
	assert(block.at("activation") == "mish");
	assert(block.find("filters") != block.end());
	assert(block.find("pad") != block.end());
	assert(block.find("size") != block.end());
	assert(block.find("stride") != block.end());

#ifdef DEBUG
    bool batchNormalize = false;
    bool bias = false;
	if (block.find("batch_normalize") != block.end())
	{
		batchNormalize = (block.at("batch_normalize") == "1");
		bias = false;
	}
	else
	{
		batchNormalize = false;
		bias = true;
	}
	// all conv_bn_leaky layers assume bias is false
	assert(batchNormalize == true && bias == false);
#endif

	int filters = std::stoi(block.at("filters"));
	int padding = std::stoi(block.at("pad"));
	int kernelSize = std::stoi(block.at("size"));
	int stride = std::stoi(block.at("stride"));
    int pad = padding ? (kernelSize - 1) / 2 : 0;

	/***** CONVOLUTION LAYER *****/
	/*****************************/
	// batch norm weights are before the conv layer
	// load BN biases (bn_biases)
	std::vector<float> bnBiases;
	for (int i = 0; i < filters; ++i)
	{
		bnBiases.push_back(weights[weightPtr]);
		weightPtr++;
	}
	// load BN weights
	std::vector<float> bnWeights;
	for (int i = 0; i < filters; ++i)
	{
		bnWeights.push_back(weights[weightPtr]);
		weightPtr++;
	}
	// load BN running_mean
	std::vector<float> bnRunningMean;
	for (int i = 0; i < filters; ++i)
	{
		bnRunningMean.push_back(weights[weightPtr]);
		weightPtr++;
	}
	// load BN running_var
	std::vector<float> bnRunningVar;
	for (int i = 0; i < filters; ++i)
	{
		// 1e-05 for numerical stability
		bnRunningVar.push_back(sqrt(weights[weightPtr] + 1.0e-5f));
		weightPtr++;
	}
	// load Conv layer weights (GKCRS)
	int size = filters * inputChannels * kernelSize * kernelSize;
	nvinfer1::Weights convWt{ nvinfer1::DataType::kFLOAT, nullptr, size };
	float* val = new float[size];
	for (int i = 0; i < size; ++i)
	{
		val[i] = weights[weightPtr];
		weightPtr++;
	}
	convWt.values = val;
	trtWeights.push_back(convWt);
	nvinfer1::Weights convBias{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
	trtWeights.push_back(convBias);
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(*input, filters, nvinfer1::DimsHW{ kernelSize, kernelSize }, convWt, convBias);
	assert(conv != nullptr);
	std::string convLayerName = "conv_" + std::to_string(layerIdx);
	conv->setName(convLayerName.c_str());
    conv->setStrideNd(nvinfer1::DimsHW{ stride, stride });
    conv->setPaddingNd(nvinfer1::DimsHW{ pad, pad });

	/***** BATCHNORM LAYER *****/
	/***************************/
	size = filters;
	// create the weights
	nvinfer1::Weights shift{ nvinfer1::DataType::kFLOAT, nullptr, size };
	nvinfer1::Weights scale{ nvinfer1::DataType::kFLOAT, nullptr, size };
	nvinfer1::Weights power{ nvinfer1::DataType::kFLOAT, nullptr, size };
	float* shiftWt = new float[size];
	for (int i = 0; i < size; ++i)
	{
        shiftWt[i] = bnBiases.at(i) - ((bnRunningMean.at(i) * bnWeights.at(i)) / bnRunningVar.at(i));
	}
	shift.values = shiftWt;
	float* scaleWt = new float[size];
	for (int i = 0; i < size; ++i)
	{
		scaleWt[i] = bnWeights.at(i) / bnRunningVar[i];
	}
	scale.values = scaleWt;
	float* powerWt = new float[size];
	for (int i = 0; i < size; ++i)
	{
		powerWt[i] = 1.0;
	}
	power.values = powerWt;
	trtWeights.push_back(shift);
	trtWeights.push_back(scale);
	trtWeights.push_back(power);
	// Add the batch norm layers
	nvinfer1::IScaleLayer* bn = network->addScale(
		*conv->getOutput(0), nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
	assert(bn != nullptr);
	std::string bnLayerName = "batch_norm_" + std::to_string(layerIdx);
	bn->setName(bnLayerName.c_str());
	/***** ACTIVATION LAYER *****/
	/****************************/
	auto creator = getPluginRegistry()->getPluginCreator("Mish_TRT", "1");
	const nvinfer1::PluginFieldCollection* pluginData = creator->getFieldNames();
	nvinfer1::IPluginV2 *pluginObj = creator->createPlugin(("mish" + std::to_string(layerIdx)).c_str(), pluginData);
	nvinfer1::ITensor* inputTensors[] = { bn->getOutput(0) };
	auto mish = network->addPluginV2(&inputTensors[0], 1, *pluginObj);
	return mish;
}

nvinfer1::ILayer * layer_split(const int n_layer_index_,
	nvinfer1::ITensor *input_,
	nvinfer1::INetworkDefinition* network)
{
	auto creator = getPluginRegistry()->getPluginCreator("CHUNK_TRT", "1.0");
	const nvinfer1::PluginFieldCollection* pluginData = creator->getFieldNames();
	nvinfer1::IPluginV2 *pluginObj = creator->createPlugin(("chunk" + std::to_string(n_layer_index_)).c_str(), pluginData);
	auto chunk = network->addPluginV2(&input_, 1, *pluginObj);
	return chunk;
}

std::vector<int> parse_int_list(const std::string s_args_)
{
	std::string s_args = s_args_;
	std::vector<int> vec_args;
	while (!s_args.empty())
	{
		auto npos = s_args.find_first_of(',');
		if (npos != std::string::npos)
		{
			int v = std::stoi(trim(s_args.substr(0, npos)));
			vec_args.push_back(v);
			s_args.erase(0, npos + 1);
		}
		else
		{
			int v = std::stoi(trim(s_args));
			vec_args.push_back(v);
			break;
		}
	}
	return vec_args;
}

std::vector<int> dims2chw(const nvinfer1::Dims d)
{
	std::vector<int> chw;
	assert(d.nbDims >= 1);
	for (int i = 0; i < d.nbDims; ++i)
	{
		chw.push_back(d.d[i]);
	}
	return chw;
}

nvinfer1::ILayer* layer_bottleneck(std::vector<nvinfer1::Weights> &trtWeights_,
	std::string s_layer_name_,
	std::map<std::string, std::vector<float>> &map_wts_,
	nvinfer1::INetworkDefinition* network_,
	nvinfer1::ITensor* input_, 
	const int c2_,
	bool shortcut_ = true,
	const int gouup_ = 1,
	const float e_ = 0.5)
{
	int c_ = int(c2_*e_);
	auto cv1 = layer_conv_bn_act(trtWeights_,s_layer_name_ + ".cv1", map_wts_, input_, network_, c_, 1, 1);
	auto cv2 = layer_conv_bn_act(trtWeights_,s_layer_name_ + ".cv2", map_wts_, cv1->getOutput(0), network_, c2_, 3, 1,gouup_);
	if (shortcut_)
	{
		nvinfer1::IElementWiseLayer* ew
			= network_->addElementWise(*input_,
				*cv2->getOutput(0),
				nvinfer1::ElementWiseOperation::kSUM);
		return ew;
	}
	else
	{
		return cv2;
	}
}

nvinfer1::ILayer * layer_concate(nvinfer1::ITensor** concatInputs,
	const int n_size_,
    const int /*n_axis_*/,
	nvinfer1::INetworkDefinition* network_)
{
    nvinfer1::IConcatenationLayer* concat = network_->addConcatenation(concatInputs, n_size_);
	assert(concat != nullptr);
//	concat->setAxis(n_axis_);
	return concat;
}

nvinfer1::ILayer * layer_bn(std::vector<nvinfer1::Weights> &trtWeights_,
	const std::string s_layer_name_,
	std::map<std::string, std::vector<float>>&map_wts_,//conv-bn
	nvinfer1::ITensor* input_,
	const int n_filters_,
	nvinfer1::INetworkDefinition* network_)
{
	std::vector<float> bn_wts = map_wts_[s_layer_name_ + ".bn.weight"];
	std::vector<float> bn_bias = map_wts_[s_layer_name_ + ".bn.bias"];
	std::vector<float> bn_mean = map_wts_[s_layer_name_ + ".bn.running_mean"];
	std::vector<float> bn_var = map_wts_[s_layer_name_ + ".bn.running_var"];
	assert(bn_wts.size() == n_filters_);
	assert(bn_bias.size() == n_filters_);
	assert(bn_mean.size() == n_filters_);
	assert(bn_var.size() == n_filters_);
	for (int i = 0; i < n_filters_; ++i)
	{
		bn_var[i] = sqrt(bn_var[i] + 1.0e-5f);
	}
	//float bn_num_batches_tracked = map_wts_[s_layer_name_ + ".bn.num_batches_tracked.weight"][0];
	// create the weights
	nvinfer1::Weights shift{ nvinfer1::DataType::kFLOAT, nullptr, n_filters_ };
	nvinfer1::Weights scale{ nvinfer1::DataType::kFLOAT, nullptr, n_filters_ };
	nvinfer1::Weights power{ nvinfer1::DataType::kFLOAT, nullptr, n_filters_ };
	float* shiftWt = new float[n_filters_];
	for (int i = 0; i < n_filters_; ++i)
	{
        shiftWt[i] = bn_bias.at(i) - ((bn_mean.at(i) * bn_wts.at(i)) / bn_var.at(i));
	}
	shift.values = shiftWt;
	float* scaleWt = new float[n_filters_];
	for (int i = 0; i < n_filters_; ++i)
	{
		scaleWt[i] = bn_wts.at(i) / bn_var[i];
	}
	scale.values = scaleWt;
	float* powerWt = new float[n_filters_];
	for (int i = 0; i < n_filters_; ++i)
	{
		powerWt[i] = 1.0;
	}
	power.values = powerWt;
	// Add the batch norm layers
	auto bn = network_->addScale(*input_, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
	assert(bn != nullptr);
	trtWeights_.push_back(shift);
	trtWeights_.push_back(scale);
	trtWeights_.push_back(power);
	return bn;
}

nvinfer1::ILayer * layer_act(nvinfer1::ITensor* input_,
	nvinfer1::INetworkDefinition* network_,
	const std::string s_act_)
{
	if (s_act_ == "leaky")
	{
		auto act = network_->addActivation(*input_, nvinfer1::ActivationType::kLEAKY_RELU);
		assert(act != nullptr);
		act->setAlpha(0.1);
		return act;
	}
	else if (s_act_ == "hardswish")
	{
		nvinfer1::IPluginV2 *hardswish_plugin = new nvinfer1::Hardswish();
		auto act = network_->addPluginV2(&input_, 1, *hardswish_plugin);
		assert(act != nullptr);
		return act;
	}
	else if (s_act_ == "silu")
	{
		auto sig = network_->addActivation(*input_, nvinfer1::ActivationType::kSIGMOID);
		assert(sig != nullptr);
		auto act = network_->addElementWise(*input_, *sig->getOutput(0), ElementWiseOperation::kPROD);
		assert(act != nullptr);
		return act;
	}
	return nullptr;
}

nvinfer1::ILayer * layer_conv(std::vector<nvinfer1::Weights> &trtWeights_,
	const std::string s_layer_name_,
	std::map<std::string, std::vector<float>>&map_wts_,//conv-bn
	nvinfer1::ITensor* input_,
	nvinfer1::INetworkDefinition* network_,
	const int n_filters_,
	const int n_kernel_size_,
	const int n_stride_,
	const bool b_bias_,
	const int group_ ,
	const bool b_padding_)
{
	int pad = b_padding_ ? ((n_kernel_size_ - 1) / 2) : 0;
	std::vector<int> chw = dims2chw(input_->getDimensions());

	//conv
	int size = n_filters_ * chw[0] * n_kernel_size_ * n_kernel_size_;
	nvinfer1::Weights convWt{ nvinfer1::DataType::kFLOAT, nullptr, size };
	float *conv_wts = new float[size];
	std::vector<float> &vec_wts = map_wts_[s_layer_name_ + ".weight"];
	for (int i = 0; i < size; ++i)
	{
		conv_wts[i] = vec_wts[i];
	}
	assert(size == (map_wts_[s_layer_name_ + ".weight"].size()));
	convWt.values = conv_wts;
	nvinfer1::Weights convBias{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
	if (b_bias_)
	{
		int size_bias = n_filters_;
		float *conv_bias = new float[size_bias];
		std::vector<float> &vec_bias = map_wts_[s_layer_name_ + ".bias"];
		for (int i = 0; i < size_bias; ++i)
		{
			conv_bias[i] = vec_bias[i];
		}
		assert(size_bias == vec_bias.size());
		convBias.values = conv_bias;
		convBias.count = size_bias;
	}
	nvinfer1::IConvolutionLayer* conv = network_->addConvolutionNd(
		*input_,
		n_filters_,
		nvinfer1::DimsHW{ n_kernel_size_, n_kernel_size_ },
		convWt,
		convBias);
	assert(conv != nullptr);
	conv->setPaddingNd(nvinfer1::DimsHW{ pad,pad });
	conv->setStrideNd(nvinfer1::DimsHW{ n_stride_ ,n_stride_ });
	if (!b_bias_)
	{
		conv->setNbGroups(group_);
	}
	trtWeights_.push_back(convWt);
	trtWeights_.push_back(convBias);
	return conv;
}

nvinfer1::ILayer * C3(std::vector<nvinfer1::Weights> &trtWeights_,
    std::string s_model_name_,
    std::map<std::string, std::vector<float>> &map_wts_,
    nvinfer1::INetworkDefinition* network_,
    nvinfer1::ITensor* input_,
    const int c2_,
    const int n_depth_,
    const bool b_short_cut_,
    const int group_,
    const float e_ )
{
    int c_ = (int)((float)c2_ * e_);
    auto cv1 = layer_conv_bn_act(trtWeights_, s_model_name_ +".cv1", map_wts_, input_, network_, c_, 1, 1, 1, true, true, "silu");
    auto cv2 = layer_conv_bn_act(trtWeights_, s_model_name_ +".cv2", map_wts_, input_, network_, c_, 1, 1, 1, true, true, "silu");
    auto out = cv1;
    for (int d = 0; d < n_depth_; ++d) {
        std::string m_name = s_model_name_ + ".m." + std::to_string(d);
	out = layer_bottleneck(trtWeights_, m_name, map_wts_, network_, out->getOutput(0), c_, b_short_cut_, group_, 1.f);
    }
    nvinfer1::ITensor** concatInputs = reinterpret_cast<nvinfer1::ITensor**>(malloc(sizeof(nvinfer1::ITensor*) *2));
    concatInputs[0] = out->getOutput(0);
    concatInputs[1] = cv2->getOutput(0);
    auto cat = layer_concate(concatInputs, 2, 0, network_);

    auto cv3 = layer_conv_bn_act(trtWeights_, s_model_name_ +".cv3", map_wts_, cat->getOutput(0), network_, c2_, 1, 1, 1, true, true, "silu");
    return cv3;
}

nvinfer1::ILayer * layer_bottleneck_csp(std::vector<nvinfer1::Weights> &trtWeights_,
	std::string s_model_name_,
	std::map<std::string, std::vector<float>> &map_wts_,
	nvinfer1::INetworkDefinition* network_,
	nvinfer1::ITensor* input_, 
	const int c2_,
	const int n_depth_,
	const bool b_short_cut_ ,
	const int group_ ,
    const float /*e_*/)
{
	std::vector<int> chw=dims2chw(input_->getDimensions());
	int c_ = int(c2_*0.5);
	//cv1
	auto out = layer_conv_bn_act(trtWeights_, s_model_name_ +".cv1", map_wts_, input_, network_, c_, 1);
	//m
	for (int d = 0; d < n_depth_; ++d)
	{
		std::string m_name = s_model_name_ + ".m." + std::to_string(d);
		out = layer_bottleneck(trtWeights_, m_name, map_wts_, network_, out->getOutput(0), c_, b_short_cut_, group_, 1.f);
	}
	//cv3
	auto cv3 = layer_conv(trtWeights_, s_model_name_ + ".cv3", map_wts_, out->getOutput(0), network_, c_, 1);
	//cv2
	auto cv2 = layer_conv(trtWeights_, s_model_name_ + ".cv2", map_wts_, input_, network_, c_, 1);
	//concate
	nvinfer1::ITensor** concatInputs
		= reinterpret_cast<nvinfer1::ITensor**>(malloc(sizeof(nvinfer1::ITensor*) *2));
	concatInputs[0] = cv3->getOutput(0);
	concatInputs[1] = cv2->getOutput(0);
	auto cat = layer_concate(concatInputs, 2, 0,network_);
	auto bn = layer_bn(trtWeights_, s_model_name_, map_wts_, cat->getOutput(0), 2 * c_, network_);
	auto act = layer_act(bn->getOutput(0), network_,"leaky");
	//cv4
	auto cv4 = layer_conv_bn_act(trtWeights_, s_model_name_ + ".cv4", map_wts_, act->getOutput(0), network_, c2_, 1);
	return cv4;
}

nvinfer1::ILayer * layer_spp(std::vector<nvinfer1::Weights> &trtWeights_,
	std::string s_model_name_,
	std::map<std::string, std::vector<float>> &map_wts_,
	nvinfer1::INetworkDefinition* network_,
	nvinfer1::ITensor* input_,
	const int c2_,
	const std::vector<int> &vec_args_)
{
	std::vector<int> chw=dims2chw(input_->getDimensions());
	int c1 = chw[0];//dims2chw(input_->getDimensions())[0];
	int c_ = c1 / 2;
	nvinfer1::ILayer * x = layer_conv_bn_act(trtWeights_, s_model_name_ + ".cv1", map_wts_, input_, network_, c_, 1);
	nvinfer1::ITensor** concatInputs
		= reinterpret_cast<nvinfer1::ITensor**>(malloc(sizeof(nvinfer1::ITensor*) * (vec_args_.size()+1)));
	concatInputs[0] = x->getOutput(0);
    for (size_t ind = 0; ind < vec_args_.size(); ++ind)
	{
		nvinfer1::IPoolingLayer* pool
			= network_->addPoolingNd(*x->getOutput(0),
				nvinfer1::PoolingType::kMAX,
				nvinfer1::DimsHW{ vec_args_[ind], vec_args_[ind] });
		assert(pool);
		int pad = vec_args_[ind] / 2;
		pool->setPaddingNd(nvinfer1::DimsHW{ pad,pad });
		pool->setStrideNd(nvinfer1::DimsHW{1, 1});
		concatInputs[ind + 1] = pool->getOutput(0);
	}
	nvinfer1::IConcatenationLayer* concat = network_->addConcatenation(concatInputs, static_cast<int>(vec_args_.size()+1));
	//concat->setAxis(0);
	assert(concat != nullptr);
	nvinfer1::ILayer *cv2 = layer_conv_bn_act(trtWeights_, s_model_name_ + ".cv2", map_wts_, concat->getOutput(0), network_, c2_, 1);
	assert(cv2 != nullptr);
	return cv2;
}

nvinfer1::ILayer * layer_sppf(std::vector<nvinfer1::Weights> &trtWeights_,
	std::string s_model_name_,
	std::map<std::string, std::vector<float>> &map_wts_,
	nvinfer1::INetworkDefinition* network_,
	nvinfer1::ITensor* input_,
	const int c2_,
	int k_)
{
	std::vector<int> chw = dims2chw(input_->getDimensions());
	int c1 = chw[0];//dims2chw(input_->getDimensions())[0];
	int c_ = c1 / 2;
    nvinfer1::ILayer* x = layer_conv_bn_act(trtWeights_, s_model_name_ + ".cv1", map_wts_, input_, network_, c_, 1);
	nvinfer1::ITensor** concatInputs
		= reinterpret_cast<nvinfer1::ITensor**>(malloc(sizeof(nvinfer1::ITensor*) * 4));
	concatInputs[0] = x->getOutput(0);

	//y1
    nvinfer1::IPoolingLayer* y1 = network_->addPoolingNd(*x->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{ k_,k_ });
	assert(y1);
	int pad = k_ / 2;
	y1->setPaddingNd(nvinfer1::DimsHW{ pad,pad });
	y1->setStrideNd(nvinfer1::DimsHW{ 1, 1 });
	concatInputs[1] = y1->getOutput(0);

	//y2
    nvinfer1::IPoolingLayer* y2 = network_->addPoolingNd(*y1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{ k_,k_ });
	assert(y2);
	y2->setPaddingNd(nvinfer1::DimsHW{ pad,pad });
	y2->setStrideNd(nvinfer1::DimsHW{ 1, 1 });
	concatInputs[2] = y2->getOutput(0);

	//y3
    nvinfer1::IPoolingLayer* y3 = network_->addPoolingNd(*y2->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{ k_,k_ });
	assert(y3);
	y3->setPaddingNd(nvinfer1::DimsHW{ pad,pad });
	y3->setStrideNd(nvinfer1::DimsHW{ 1, 1 });
	concatInputs[3] = y3->getOutput(0);

	nvinfer1::IConcatenationLayer* concat
		= network_->addConcatenation(concatInputs, 4);
	//concat->setAxis(0);
	assert(concat != nullptr);
	nvinfer1::ILayer *cv2 = layer_conv_bn_act(trtWeights_, s_model_name_ + ".cv2", map_wts_, concat->getOutput(0), network_, c2_, 1);
	assert(cv2 != nullptr);
	return cv2;
}

nvinfer1::ILayer *layer_upsample(std::string /*s_model_name_*/,
    std::map<std::string, std::vector<float>>& /*map_wts_*/,
	nvinfer1::INetworkDefinition* network_,
	nvinfer1::ITensor* input_, 
	const int n_scale_)
{
	std::vector<int> chw=dims2chw(input_->getDimensions());
	int c1 = chw[0];//dims2chw(input_->getDimensions())[0];
	float *deval = new float[c1*n_scale_*n_scale_];
	for (int i = 0; i < c1*n_scale_*n_scale_; i++)
	{
		deval[i] = 1.0;
	}
	nvinfer1::Weights wts{ DataType::kFLOAT, deval, c1*n_scale_*n_scale_ };
	nvinfer1::Weights bias{ DataType::kFLOAT, nullptr, 0 };
	IDeconvolutionLayer* upsample = network_->addDeconvolutionNd(*input_,c1, DimsHW{ n_scale_, n_scale_ }, wts, bias);
	upsample->setStrideNd(DimsHW{ n_scale_, n_scale_ });
	upsample->setNbGroups(c1);
	return upsample;
}

nvinfer1::ILayer * layer_conv_bn_act(std::vector<nvinfer1::Weights> &trtWeights_,
	const std::string s_layer_name_,
	std::map<std::string, std::vector<float>>&map_wts_,//conv-bn
	nvinfer1::ITensor* input_,
	nvinfer1::INetworkDefinition* network_,
	const int n_filters_,
	const int n_kernel_size_,
	const int n_stride_,
    const int /*group_*/,
    const bool /*b_padding_*/,
    const bool /*b_bn_*/,
	const std::string s_act_)
{
	std::vector<int> chw = dims2chw(input_->getDimensions());

	//conv
    nvinfer1::ILayer* conv = layer_conv(trtWeights_, s_layer_name_ + ".conv", map_wts_, input_, network_, n_filters_, n_kernel_size_, n_stride_);
	nvinfer1::ILayer* bn = layer_bn(trtWeights_, s_layer_name_, map_wts_, conv->getOutput(0), n_filters_, network_);
    nvinfer1::ILayer* act = layer_act(bn->getOutput(0), network_,s_act_);
	return act;
}



nvinfer1::ILayer* layer_focus(std::vector<nvinfer1::Weights> &trtWeights_,
	std::string s_model_name_,
	std::map<std::string,std::vector<float>>& map_wts_,
	nvinfer1::ITensor* input,
	const int out_channels_,
	const int kernel_size_,
    std::vector<nvinfer1::Weights>& /*trtWeights*/,
	nvinfer1::INetworkDefinition* network)
{
	std::vector<int> chw = dims2chw(input->getDimensions());
	ISliceLayer *s1 = network->addSlice(*input, Dims3{ 0, 0, 0 }, Dims3{ chw[0], chw[1] / 2, chw[2] / 2 }, Dims3{ 1, 2, 2 });
	ISliceLayer *s2 = network->addSlice(*input, Dims3{ 0, 1, 0 }, Dims3{ chw[0], chw[1] / 2, chw[2] / 2 }, Dims3{ 1, 2, 2 });
	ISliceLayer *s3 = network->addSlice(*input, Dims3{ 0, 0, 1 }, Dims3{ chw[0], chw[1] / 2, chw[2] / 2 }, Dims3{ 1, 2, 2 });
	ISliceLayer *s4 = network->addSlice(*input, Dims3{ 0, 1, 1 }, Dims3{ chw[0], chw[1] / 2, chw[2] / 2 }, Dims3{ 1, 2, 2 });
	ITensor* inputTensors[] = { s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0) };
	auto cat = network->addConcatenation(inputTensors, 4);
	auto cat_out = cat->getOutput(0);
	auto out = layer_conv_bn_act(trtWeights_,
		s_model_name_ +".conv",
		map_wts_,
		cat_out,
		network,
		out_channels_,
		kernel_size_);
	return out;
}


nvinfer1::ILayer* netAddConvBNLeaky(int layerIdx,
									std::map<std::string, std::string>& block,
                                    std::vector<float>& weights,
                                    std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
                                    int& inputChannels, nvinfer1::ITensor* input,
                                    nvinfer1::INetworkDefinition* network)
{
    assert(block.at("type") == "convolutional");
    assert(block.find("batch_normalize") != block.end());
    assert(block.at("batch_normalize") == "1");
    assert(block.at("activation") == "leaky");
    assert(block.find("filters") != block.end());
    assert(block.find("pad") != block.end());
    assert(block.find("size") != block.end());
    assert(block.find("stride") != block.end());

#ifdef DEBUG
    bool batchNormalize = false;
    bool bias = false;
    if (block.find("batch_normalize") != block.end())
    {
        batchNormalize = (block.at("batch_normalize") == "1");
        bias = false;
    }
    else
    {
        batchNormalize = false;
        bias = true;
    }
    // all conv_bn_leaky layers assume bias is false
    assert(batchNormalize == true && bias == false);
#endif

    int filters = std::stoi(block.at("filters"));
    int padding = std::stoi(block.at("pad"));
    int kernelSize = std::stoi(block.at("size"));
    int stride = std::stoi(block.at("stride"));
    int pad;
    if (padding)
        pad = (kernelSize - 1) / 2;
    else
        pad = 0;

    /***** CONVOLUTION LAYER *****/
    /*****************************/
    // batch norm weights are before the conv layer
    // load BN biases (bn_biases)
    std::vector<float> bnBiases;
    for (int i = 0; i < filters; ++i)
    {
        bnBiases.push_back(weights[weightPtr]);
        weightPtr++;
    }
    // load BN weights
    std::vector<float> bnWeights;
    for (int i = 0; i < filters; ++i)
    {
        bnWeights.push_back(weights[weightPtr]);
        weightPtr++;
    }
    // load BN running_mean
    std::vector<float> bnRunningMean;
    for (int i = 0; i < filters; ++i)
    {
        bnRunningMean.push_back(weights[weightPtr]);
        weightPtr++;
    }
    // load BN running_var
    std::vector<float> bnRunningVar;
    for (int i = 0; i < filters; ++i)
    {
        // 1e-05 for numerical stability
        bnRunningVar.push_back(sqrt(weights[weightPtr] + 1.0e-5f));
        weightPtr++;
    }
    // load Conv layer weights (GKCRS)
    int size = filters * inputChannels * kernelSize * kernelSize;
    nvinfer1::Weights convWt{nvinfer1::DataType::kFLOAT, nullptr, size};
    float* val = new float[size];
    for (int i = 0; i < size; ++i)
    {
        val[i] = weights[weightPtr];
        weightPtr++;
    }
    convWt.values = val;
    trtWeights.push_back(convWt);
    nvinfer1::Weights convBias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    trtWeights.push_back(convBias);
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(*input, filters, nvinfer1::DimsHW{kernelSize, kernelSize}, convWt, convBias);
    assert(conv != nullptr);
    std::string convLayerName = "conv_" + std::to_string(layerIdx);
    conv->setName(convLayerName.c_str());
    conv->setStrideNd(nvinfer1::DimsHW{stride, stride});
    conv->setPaddingNd(nvinfer1::DimsHW{pad, pad});

    /***** BATCHNORM LAYER *****/
    /***************************/
    size = filters;
    // create the weights
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, nullptr, size};
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, nullptr, size};
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, size};
    float* shiftWt = new float[size];
    for (int i = 0; i < size; ++i)
    {
        shiftWt[i]
            = bnBiases.at(i) - ((bnRunningMean.at(i) * bnWeights.at(i)) / bnRunningVar.at(i));
    }
    shift.values = shiftWt;
    float* scaleWt = new float[size];
    for (int i = 0; i < size; ++i)
    {
        scaleWt[i] = bnWeights.at(i) / bnRunningVar[i];
    }
    scale.values = scaleWt;
    float* powerWt = new float[size];
    for (int i = 0; i < size; ++i)
    {
        powerWt[i] = 1.0;
    }
    power.values = powerWt;
    trtWeights.push_back(shift);
    trtWeights.push_back(scale);
    trtWeights.push_back(power);
    // Add the batch norm layers
    nvinfer1::IScaleLayer* bn = network->addScale(
        *conv->getOutput(0), nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    assert(bn != nullptr);
    std::string bnLayerName = "batch_norm_" + std::to_string(layerIdx);
    bn->setName(bnLayerName.c_str());
    /***** ACTIVATION LAYER *****/
    /****************************/
	auto leaky = network->addActivation(*bn->getOutput(0),nvinfer1::ActivationType::kLEAKY_RELU);
	assert(leaky != nullptr);
	leaky->setAlpha(0.1f);
	/*nvinfer1::IPlugin* leakyRELU = nvinfer1::plugin::createPReLUPlugin(0.1);
	assert(leakyRELU != nullptr);
	nvinfer1::ITensor* bnOutput = bn->getOutput(0);
	nvinfer1::IPluginLayer* leaky = network->addPlugin(&bnOutput, 1, *leakyRELU);*/
	std::string leakyLayerName = "leaky_" + std::to_string(layerIdx);
	leaky->setName(leakyLayerName.c_str());

    return leaky;
}

nvinfer1::ILayer* netAddUpsample(int /*layerIdx*/, std::map<std::string, std::string>& block,
                                 std::vector<float>& /*weights*/,
                                 std::vector<nvinfer1::Weights>& /*trtWeights*/, int& /*inputChannels*/,
                                 nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
    assert(block.at("type") == "upsample");
    nvinfer1::Dims inpDims = input->getDimensions();
    assert(inpDims.nbDims == 3);
   // assert(inpDims.d[1] == inpDims.d[2]);
    int n_scale = std::stoi(block.at("stride"));

	int c1 = inpDims.d[0];
	float *deval = new float[c1*n_scale*n_scale];
	for (int i = 0; i < c1*n_scale*n_scale; i++)
	{
		deval[i] = 1.0;
	}
	nvinfer1::Weights wts{ DataType::kFLOAT, deval, c1*n_scale*n_scale };
	nvinfer1::Weights bias{ DataType::kFLOAT, nullptr, 0 };
	IDeconvolutionLayer* upsample = network->addDeconvolutionNd(*input, c1, DimsHW{ n_scale, n_scale }, wts, bias);
	upsample->setStrideNd(DimsHW{ n_scale, n_scale });
	upsample->setNbGroups(c1);
	return upsample;

    #if 0
// add pre multiply matrix as a constant
    nvinfer1::Dims preDims{3,
                           {1, stride * h, w},
                           {nvinfer1::DimensionType::kCHANNEL, nvinfer1::DimensionType::kSPATIAL,
                            nvinfer1::DimensionType::kSPATIAL}};
    int size = stride * h * w;
    nvinfer1::Weights preMul{nvinfer1::DataType::kFLOAT, nullptr, size};
    float* preWt = new float[size];
    /* (2*h * w)
    [ [1, 0, ..., 0],
      [1, 0, ..., 0],
      [0, 1, ..., 0],
      [0, 1, ..., 0],
      ...,
      ...,
      [0, 0, ..., 1],
      [0, 0, ..., 1] ]
    */
    for (int i = 0, idx = 0; i < h; ++i)
    {
        for (int s = 0; s < stride; ++s)
        {
            for (int j = 0; j < w; ++j, ++idx)
            {
                preWt[idx] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
    preMul.values = preWt;
    trtWeights.push_back(preMul);
    nvinfer1::IConstantLayer* preM = network->addConstant(preDims, preMul);
    assert(preM != nullptr);
    std::string preLayerName = "preMul_" + std::to_string(layerIdx);
    preM->setName(preLayerName.c_str());
    // add post multiply matrix as a constant
    nvinfer1::Dims postDims{3,
                            {1, h, stride * w},
                            {nvinfer1::DimensionType::kCHANNEL, nvinfer1::DimensionType::kSPATIAL,
                             nvinfer1::DimensionType::kSPATIAL}};
    size = stride * h * w;
    nvinfer1::Weights postMul{nvinfer1::DataType::kFLOAT, nullptr, size};
    float* postWt = new float[size];
    /* (h * 2*w)
    [ [1, 1, 0, 0, ..., 0, 0],
      [0, 0, 1, 1, ..., 0, 0],
      ...,
      ...,
      [0, 0, 0, 0, ..., 1, 1] ]
    */
    for (int i = 0, idx = 0; i < h; ++i)
    {
        for (int j = 0; j < stride * w; ++j, ++idx)
        {
            postWt[idx] = (j / stride == i) ? 1.0f : 0.0f;
        }
    }
    postMul.values = postWt;
    trtWeights.push_back(postMul);
    nvinfer1::IConstantLayer* post_m = network->addConstant(postDims, postMul);
    assert(post_m != nullptr);
    std::string postLayerName = "postMul_" + std::to_string(layerIdx);
    post_m->setName(postLayerName.c_str());
    // add matrix multiply layers for upsampling
    nvinfer1::IMatrixMultiplyLayer* mm1
        = network->addMatrixMultiply(*preM->getOutput(0), nvinfer1::MatrixOperation::kNONE, *input,
                                     nvinfer1::MatrixOperation::kNONE);
    assert(mm1 != nullptr);
    std::string mm1LayerName = "mm1_" + std::to_string(layerIdx);
    mm1->setName(mm1LayerName.c_str());
    nvinfer1::IMatrixMultiplyLayer* mm2
        = network->addMatrixMultiply(*mm1->getOutput(0), nvinfer1::MatrixOperation::kNONE,
                                     *post_m->getOutput(0), nvinfer1::MatrixOperation::kNONE);
    assert(mm2 != nullptr);
    std::string mm2LayerName = "mm2_" + std::to_string(layerIdx);
    mm2->setName(mm2LayerName.c_str());
    return mm2;
#endif
}

void printLayerInfo(std::string layerIndex, std::string layerName, std::string layerInput,
                    std::string layerOutput, std::string weightPtr)
{
    std::cout << std::setw(6) << std::left << layerIndex << std::setw(15) << std::left << layerName;
    std::cout << std::setw(20) << std::left << layerInput << std::setw(20) << std::left
              << layerOutput;
    std::cout << std::setw(6) << std::left << weightPtr << std::endl;
}
