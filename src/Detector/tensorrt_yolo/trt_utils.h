
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

#ifndef __TRT_UTILS_H__
#define __TRT_UTILS_H__

#include <set>
#include <math.h>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "mish.h"
#include "chunk.h"
#include "hardswish.h"
#include "NvInfer.h"
#include "class_detector.h"

#include "ds_image.h"
#include "plugin_factory.h"

class DsImage;
struct BBox
{
    float x1 = 0;
    float y1 = 0;
    float x2 = 0;
    float y2 = 0;
};

struct BBoxInfo
{
    BBox box;
    int label = 0;
    int classId = 0; // For coco benchmarking
    float prob = 0.f;
};

class Logger : public nvinfer1::ILogger
{
public:
	Logger(Severity severity = Severity::kWARNING)
	{
		severity = severity;
	}

	~Logger() = default;

	nvinfer1::ILogger& getTRTLogger()
	{
		return *this;
	}

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity == Severity::kINFO) return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: " << msg << std::endl; break;
		case Severity::kERROR: std::cerr << "ERROR: " << msg << std::endl; break;
        case Severity::kWARNING: std::cerr << "WARNING: " << msg << std::endl; break;
        case Severity::kINFO: std::cerr << "INFO: " << msg << std::endl; break;
        case Severity::kVERBOSE: break;
      //  default: std::cerr <<"UNKNOW:"<< msg << std::endl;break;
        }
    }
};

//class YoloTinyMaxpoolPaddingFormula : public nvinfer1::IOutputDimensionsFormula
//{
//
//private:
//    std::set<std::string> m_SamePaddingLayers;
//
//    nvinfer1::DimsHW compute(nvinfer1::DimsHW inputDims, nvinfer1::DimsHW kernelSize,
//                             nvinfer1::DimsHW stride, nvinfer1::DimsHW padding,
//                             nvinfer1::DimsHW dilation, const char* layerName) const override
//    {
//     //   assert(inputDims.d[0] == inputDims.d[1]);
//        assert(kernelSize.d[0] == kernelSize.d[1]);
//        assert(stride.d[0] == stride.d[1]);
//        assert(padding.d[0] == padding.d[1]);
//
//		int output_h, output_w;
//        // Only layer maxpool_12 makes use of same padding
//        if (m_SamePaddingLayers.find(layerName) != m_SamePaddingLayers.end())
//        {
//            output_h = (inputDims.d[0] + 2 * padding.d[0]) / stride.d[0];
//            output_w = (inputDims.d[1] + 2 * padding.d[1]) / stride.d[1];
//        }
//        // Valid Padding
//        else
//        {
//            output_h = (inputDims.d[0] - kernelSize.d[0]) / stride.d[0] + 1;
//            output_w = (inputDims.d[1] - kernelSize.d[1]) / stride.d[1] + 1;
//        }
//        return nvinfer1::DimsHW{output_h, output_w};
//    }
//
//public:
//    void addSamePaddingLayer(std::string input) { m_SamePaddingLayers.insert(input); }
//};

// Common helper functions
void blobFromDsImages(const std::vector<DsImage>& inputImages, cv::Mat& blob, const int& inputH, const int& inputW);
std::string trim(std::string s);
std::string triml(std::string s, const char* t);
std::string trimr(std::string s, const char* t);
float clamp(const float val, const float minVal, const float maxVal);
bool fileExists(const std::string fileName, bool verbose = true);
BBox convertBBoxNetRes(const float& bx, const float& by, const float& bw, const float& bh,
                       const uint32_t& stride, const uint32_t& netW, const uint32_t& netH);
void convertBBoxImgRes(const float scalingFactor,
	const float xOffset,
	const float yOffset,
	BBox& bbox);
void printPredictions(const BBoxInfo& info, const std::string& className);
std::vector<std::string> loadListFromTextFile(const std::string filename);
std::vector<std::string> loadImageList(const std::string filename, const std::string prefix);
std::vector<BBoxInfo> diou_nms(const float numThresh, std::vector<BBoxInfo> binfo);
std::vector<BBoxInfo> nmsAllClasses(const float nmsThresh, std::vector<BBoxInfo>& binfo,
                                    const uint32_t numClasses, tensor_rt::ModelType model_type);
std::vector<BBoxInfo> nonMaximumSuppression(const float nmsThresh, std::vector<BBoxInfo> binfo);
nvinfer1::ICudaEngine* loadTRTEngine(const std::string planFilePath,/* PluginFactory* pluginFactory,*/
                                     Logger& logger);
std::vector<float> LoadWeights(const std::string weightsFilePath);
std::string dimsToString(const nvinfer1::Dims d);
void displayDimType(const nvinfer1::Dims d);
int getNumChannels(nvinfer1::ITensor* t);
uint64_t get3DTensorVolume(nvinfer1::Dims inputDims);

// Helper functions to create yolo engine
nvinfer1::ILayer* netAddMaxpool(int layerIdx, std::map<std::string, std::string>& block,
                                nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network);
nvinfer1::ILayer* netAddConvLinear(int layerIdx, std::map<std::string, std::string>& block,
                                   std::vector<float>& weights,
                                   std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
                                   int& inputChannels, nvinfer1::ITensor* input,
                                   nvinfer1::INetworkDefinition* network);

nvinfer1::ILayer* net_conv_bn_mish(int layerIdx,
	std::map<std::string, std::string>& block,
	std::vector<float>& weights,
	std::vector<nvinfer1::Weights>& trtWeights,
	int& weightPtr,
	int& inputChannels,
	nvinfer1::ITensor* input,
	nvinfer1::INetworkDefinition* network);

nvinfer1::ILayer* netAddConvBNLeaky(int layerIdx, std::map<std::string, std::string>& block,
                                    std::vector<float>& weights,
                                    std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
                                    int& inputChannels, nvinfer1::ITensor* input,
                                    nvinfer1::INetworkDefinition* network);
nvinfer1::ILayer* netAddUpsample(int layerIdx, std::map<std::string, std::string>& block,
                                 std::vector<float>& weights,
                                 std::vector<nvinfer1::Weights>& trtWeights, int& inputChannels,
                                 nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network);
void printLayerInfo(std::string layerIndex, std::string layerName, std::string layerInput,
                    std::string layerOutput, std::string weightPtr);

nvinfer1::ILayer * layer_split(const int n_layer_index_,
	nvinfer1::ITensor *input_,
	nvinfer1::INetworkDefinition* network);

std::vector<int> parse_int_list(const std::string s_args_);

nvinfer1::ILayer* layer_focus(std::vector<nvinfer1::Weights> &trtWeights_,
	std::string s_model_name_,
	std::map<std::string, std::vector<float>>& map_wts_,
	nvinfer1::ITensor* input,
	const int out_channels_,
	const int kernel_size_,
	std::vector<nvinfer1::Weights>& trtWeights,
	nvinfer1::INetworkDefinition* network);

nvinfer1::ILayer * layer_conv_bn_act(std::vector<nvinfer1::Weights> &trtWeights_,
	const std::string s_layer_name_,
	std::map<std::string, std::vector<float>> &vec_wts_,//conv-bn
	nvinfer1::ITensor* input_,
	nvinfer1::INetworkDefinition* network_,
	const int n_filters_,
	const int n_kernel_size_ = 3,
	const int n_stride_ = 1,
	const int group_ =1,
	const bool b_padding_ = true,
	const bool b_bn_ = true,
	const std::string s_act_ = "silu");

nvinfer1::ILayer * layer_act(nvinfer1::ITensor* input_,
	nvinfer1::INetworkDefinition* network_,
	const std::string s_act_ = "silu");

nvinfer1::ILayer * C3(std::vector<nvinfer1::Weights> &trtWeights_,
    std::string s_model_name_,
    std::map<std::string, std::vector<float>> &map_wts_,
    nvinfer1::INetworkDefinition* network_,
    nvinfer1::ITensor* input_,
    const int c2_,
    const int n_depth_ = 1,
    const bool b_short_cut_ = true,
    const int group_ = 1,
    const float e_ = 0.5);

nvinfer1::ILayer * layer_bottleneck_csp(std::vector<nvinfer1::Weights> &trtWeights_,
	std::string s_model_name_,
	std::map<std::string, std::vector<float>> &map_wts_,
	nvinfer1::INetworkDefinition* network_,
	nvinfer1::ITensor* input_,
	const int c2_,
	const int n_depth_ = 1,
	const bool b_short_cut_ = true,
	const int group_ = 1,
	const float e_ = 0.5);

nvinfer1::ILayer * layer_spp(std::vector<nvinfer1::Weights> &trtWeights_,
	std::string s_model_name_,
	std::map<std::string, std::vector<float>> &map_wts_,
	nvinfer1::INetworkDefinition* network_,
	nvinfer1::ITensor* input_,
	const int c2_,
	const std::vector<int> &vec_args_);

nvinfer1::ILayer * layer_sppf(std::vector<nvinfer1::Weights> &trtWeights_,
	std::string s_model_name_,
	std::map<std::string, std::vector<float>> &map_wts_,
	nvinfer1::INetworkDefinition* network_,
	nvinfer1::ITensor* input_,
	const int c2_,
	int k_);

nvinfer1::ILayer *layer_upsample(std::string s_model_name_,
	std::map<std::string, std::vector<float>> &map_wts_,
	nvinfer1::INetworkDefinition* network_,
	nvinfer1::ITensor* input_,
	const int n_scale_);

nvinfer1::ILayer * layer_conv(std::vector<nvinfer1::Weights> &trtWeights_,
	const std::string s_layer_name_,
	std::map<std::string, std::vector<float>>&vec_wts_,//conv-bn
	nvinfer1::ITensor* input_,
	nvinfer1::INetworkDefinition* network_,
	const int n_filters_,
	const int n_kernel_size_,
	const int n_stride_ = 1,
	const bool b_bias_ = false,
	const int group_ = 1,
	const bool b_padding_ = true);
std::vector<int> dims2chw(const nvinfer1::Dims d);



#endif
