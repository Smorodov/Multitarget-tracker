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

#include <experimental/filesystem>
#include <fstream>
#include <iomanip>

cv::Mat blobFromDsImages(const std::vector<DsImage>& inputImages,
						const int& inputH,
                         const int& inputW)
{
    std::vector<cv::Mat> letterboxStack(inputImages.size());
    for (uint32_t i = 0; i < inputImages.size(); ++i)
    {
        inputImages.at(i).getLetterBoxedImage().copyTo(letterboxStack.at(i));
    }
    return cv::dnn::blobFromImages(letterboxStack, 1.0, cv::Size(inputW, inputH),
                                   cv::Scalar(0.0, 0.0, 0.0),true);
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

float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

bool fileExists(const std::string fileName, bool verbose)
{
    if (!std::experimental::filesystem::exists(std::experimental::filesystem::path(fileName)))
    {
        if (verbose) std::cout << "File does not exist : " << fileName << std::endl;
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

    b.x1 = clamp(b.x1, 0, netW);
    b.x2 = clamp(b.x2, 0, netW);
    b.y1 = clamp(b.y1, 0, netH);
    b.y2 = clamp(b.y2, 0, netH);

    return b;
}

void convertBBoxImgRes(const float scalingFactor,
	//const float& xOffset,
//	const float& yOffset,
	const uint32_t &input_w_,
	const uint32_t &input_h_,
	const uint32_t &image_w_,
	const uint32_t &image_h_,
                       BBox& bbox)
{
    //// Undo Letterbox
    //bbox.x1 -= xOffset;
    //bbox.x2 -= xOffset;
    //bbox.y1 -= yOffset;
    //bbox.y2 -= yOffset;

    //// Restore to input resolution
    //bbox.x1 /= scalingFactor;
    //bbox.x2 /= scalingFactor;
    //bbox.y1 /= scalingFactor;
    //bbox.y2 /= scalingFactor;
	bbox.x1 = ((float)bbox.x1 / (float)input_w_)*(float)image_w_;
	bbox.y1 = ((float)bbox.y1 / (float)input_h_)*(float)image_h_;
	bbox.x2 = ((float)bbox.x2 / (float)input_w_)*(float)image_w_;
	bbox.y2 = ((float)bbox.y2 / (float)input_h_)*(float)image_h_;
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

std::vector<BBoxInfo> nmsAllClasses(const float nmsThresh, std::vector<BBoxInfo>& binfo,
                                    const uint32_t numClasses)
{
    std::vector<BBoxInfo> result;
    std::vector<std::vector<BBoxInfo>> splitBoxes(numClasses);
    for (auto& box : binfo)
    {
        splitBoxes.at(box.label).push_back(box);
    }

    for (auto& boxes : splitBoxes)
    {
        boxes = nonMaximumSuppression(nmsThresh, boxes);
        result.insert(result.end(), boxes.begin(), boxes.end());
    }

    return result;
}

std::vector<BBoxInfo> nonMaximumSuppression(const float nmsThresh, std::vector<BBoxInfo> binfo)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };
    auto computeIoU = [&overlap1D](BBox& bbox1, BBox& bbox2) -> float {
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

nvinfer1::ICudaEngine* loadTRTEngine(const std::string planFilePath, PluginFactory* pluginFactory,
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
    const int modelSize = trtModelStream.tellg();
    trtModelStream.seekg(0, std::ios::beg);
    void* modelMem = malloc(modelSize);
    trtModelStream.read((char*) modelMem, modelSize);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine
        = runtime->deserializeCudaEngine(modelMem, modelSize, pluginFactory);
    free(modelMem);
    runtime->destroy();
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
std::vector<float> loadWeights(const std::string weightsFilePath, const std::string& networkType)
{
    assert(fileExists(weightsFilePath));
    std::cout << "Loading pre-trained weights..." << std::endl;
    //std::ifstream file(weightsFilePath, std::ios_base::binary);
    //assert(file.good());
    //std::string line;

    //if (networkType == "yolov2")
    //{
    //    // Remove 4 int32 bytes of data from the stream belonging to the header
    //    file.ignore(4 * 4);
    //}
    //else if ((networkType == "yolov3") || (networkType == "yolov3-tiny")
    //         || (networkType == "yolov2-tiny"))
    //{
    //    // Remove 5 int32 bytes of data from the stream belonging to the header
    //    file.ignore(4 * 5);
    //}
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
        switch (d.type[i])
        {
        case nvinfer1::DimensionType::kSPATIAL: std::cout << "kSPATIAL "; break;
        case nvinfer1::DimensionType::kCHANNEL: std::cout << "kCHANNEL "; break;
        case nvinfer1::DimensionType::kINDEX: std::cout << "kINDEX "; break;
        case nvinfer1::DimensionType::kSEQUENCE: std::cout << "kSEQUENCE "; break;
        }
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
        = network->addPooling(*input, nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{size, size});
    assert(pool);
    std::string maxpoolLayerName = "maxpool_" + std::to_string(layerIdx);
    pool->setStride(nvinfer1::DimsHW{stride, stride});
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
    nvinfer1::IConvolutionLayer* conv = network->addConvolution(
        *input, filters, nvinfer1::DimsHW{kernelSize, kernelSize}, convWt, convBias);
    assert(conv != nullptr);
    std::string convLayerName = "conv_" + std::to_string(layerIdx);
    conv->setName(convLayerName.c_str());
    conv->setStride(nvinfer1::DimsHW{stride, stride});
    conv->setPadding(nvinfer1::DimsHW{pad, pad});

    return conv;
}

nvinfer1::ILayer* netAddConvBNLeaky(int layerIdx, std::map<std::string, std::string>& block,
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

    bool batchNormalize, bias;
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
        bnRunningVar.push_back(sqrt(weights[weightPtr] + 1.0e-5));
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
    nvinfer1::IConvolutionLayer* conv = network->addConvolution(
        *input, filters, nvinfer1::DimsHW{kernelSize, kernelSize}, convWt, convBias);
    assert(conv != nullptr);
    std::string convLayerName = "conv_" + std::to_string(layerIdx);
    conv->setName(convLayerName.c_str());
    conv->setStride(nvinfer1::DimsHW{stride, stride});
    conv->setPadding(nvinfer1::DimsHW{pad, pad});

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
    nvinfer1::IPlugin* leakyRELU = nvinfer1::plugin::createPReLUPlugin(0.1);
    assert(leakyRELU != nullptr);
    nvinfer1::ITensor* bnOutput = bn->getOutput(0);
    nvinfer1::IPluginLayer* leaky = network->addPlugin(&bnOutput, 1, *leakyRELU);
    assert(leaky != nullptr);
    std::string leakyLayerName = "leaky_" + std::to_string(layerIdx);
    leaky->setName(leakyLayerName.c_str());

    return leaky;
}

nvinfer1::ILayer* netAddUpsample(int layerIdx, std::map<std::string, std::string>& block,
                                 std::vector<float>& weights,
                                 std::vector<nvinfer1::Weights>& trtWeights, int& inputChannels,
                                 nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
    assert(block.at("type") == "upsample");
    nvinfer1::Dims inpDims = input->getDimensions();
    assert(inpDims.nbDims == 3);
    assert(inpDims.d[1] == inpDims.d[2]);
    int h = inpDims.d[1];
    int w = inpDims.d[2];
    int stride = std::stoi(block.at("stride"));
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
                preWt[idx] = (i == j) ? 1.0 : 0.0;
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
            postWt[idx] = (j / stride == i) ? 1.0 : 0.0;
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
}

void printLayerInfo(std::string layerIndex, std::string layerName, std::string layerInput,
                    std::string layerOutput, std::string weightPtr)
{
    std::cout << std::setw(6) << std::left << layerIndex << std::setw(15) << std::left << layerName;
    std::cout << std::setw(20) << std::left << layerInput << std::setw(20) << std::left
              << layerOutput;
    std::cout << std::setw(6) << std::left << weightPtr << std::endl;
}
