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

#ifndef _YOLO_H_
#define _YOLO_H_

#include "calibrator.h"
#include "trt_utils.h"

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "cuda_runtime_api.h"
#include <algorithm>
#include <cmath>
#include <stdint.h>
#include <string>
#include <vector>
#include "class_timer.hpp"
#include "opencv2/opencv.hpp"
#include "detect.h"

/**
 * Holds all the file paths required to build a network.
 */
struct NetworkInfo
{
    std::string configFilePath;
    std::string wtsFilePath;
    std::string labelsFilePath;
    std::string precision;
    std::string deviceType;
    std::string calibrationTablePath;
    std::string enginePath;
    std::string inputBlobName;
	std::string data_path;
    tensor_rt::ModelType m_networkType;
};

/**
 * Holds information about runtime inference params.
 */
struct InferParams
{
    bool printPerfInfo = false;
    bool printPredictionInfo = false;
    std::string calibImages;
    std::string calibImagesPath;
    float probThresh = 0.5f;
    float nmsThresh = 0.5f;
    uint32_t batchSize = 1;
    size_t videoMemory = 0;
};

/**
 * Holds information about an output tensor of the yolo network.
 */
struct TensorInfo
{
    std::string blobName;
    uint32_t stride{0};
    uint32_t stride_h{0};
    uint32_t stride_w{0};
    uint32_t gridSize{0};
	uint32_t grid_h{ 0 };
	uint32_t grid_w{ 0 };
    uint32_t numClasses{0};
    uint32_t numBBoxes{0};
    uint64_t volume{0};
    std::vector<uint32_t> masks;
    std::vector<float> anchors;
    int bindingIndex{-1};
    float* hostBuffer{nullptr};
};

///
/// \brief The Yolo class
///
class Yolo
{
public:
    float getNMSThresh() const { return m_NMSThresh; }
    std::string getClassName(const int& label) const { return m_ClassNames.at(label); }
    int getClassId(const int& label) const { return m_ClassIds.at(label); }
    uint32_t getInputH() const { return m_InputH; }
    uint32_t getInputW() const { return m_InputW; }
    uint32_t getNumClasses() const { return static_cast<uint32_t>(m_ClassNames.size()); }
    bool isPrintPredictions() const { return m_PrintPredictions; }
    bool isPrintPerfInfo() const { return m_PrintPerfInfo; }
    void doInference(const unsigned char* input, const uint32_t batchSize);
    std::vector<BBoxInfo> decodeDetections(const int& imageIdx, const int& imageH, const int& imageW);

    virtual ~Yolo();

protected:
    Yolo(const NetworkInfo& networkInfo, const InferParams& inferParams);
    std::string m_EnginePath;
    tensor_rt::ModelType m_NetworkType;
    const std::string m_ConfigFilePath;
    const std::string m_WtsFilePath;
    const std::string m_LabelsFilePath;
    std::string m_Precision;
    const std::string m_DeviceType;
    const std::string m_CalibImages;
    const std::string m_CalibImagesFilePath;
    std::string m_CalibTableFilePath;
    const std::string m_InputBlobName;
    std::vector<TensorInfo> m_OutputTensors;
    std::vector<std::map<std::string, std::string>> m_configBlocks;
    uint32_t m_InputH = 0;
    uint32_t m_InputW = 0;
    uint32_t m_InputC = 0;
    uint64_t m_InputSize = 0;
	uint32_t _n_classes = 0;
	float _f_depth_multiple = 0;
	float _f_width_multiple = 0;
    const float m_ProbThresh = 0.5f;
    const float m_NMSThresh = 0.5f;
    std::vector<std::string> m_ClassNames;
    // Class ids for coco benchmarking
    const std::vector<int> m_ClassIds{
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
        67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};
    const bool m_PrintPerfInfo = false;
    const bool m_PrintPredictions = false;
    // TRT specific members
    uint32_t m_BatchSize = 1;
    size_t m_videoMemory = 0;
    nvinfer1::INetworkDefinition* m_Network = nullptr;
    nvinfer1::IBuilder* m_Builder = nullptr;
    nvinfer1::IHostMemory* m_ModelStream = nullptr;
    nvinfer1::ICudaEngine* m_Engine = nullptr;
    nvinfer1::IExecutionContext* m_Context = nullptr;
    std::vector<void*> m_DeviceBuffers;
    int m_InputBindingIndex = -1;
    cudaStream_t m_CudaStream = nullptr;

    virtual std::vector<BBoxInfo> decodeTensor(const int imageIdx, const int imageH, const int imageW, const TensorInfo& tensor) = 0;

    inline void addBBoxProposal(const float bx, const float by, const float bw, const float bh,
                                const uint32_t stride, const float /*scalingFactor*/,
                                const float /*xOffset*/, const float /*yOffset*/,
                                const int maxIndex, const float maxProb,
                                const uint32_t /*image_w*/, const uint32_t /*image_h*/,
                                std::vector<BBoxInfo>& binfo)
    {
        BBoxInfo bbi;
        bbi.box = convertBBoxNetRes(bx, by, bw, bh, stride, m_InputW, m_InputH);
        if ((bbi.box.x1 > bbi.box.x2) || (bbi.box.y1 > bbi.box.y2))
            return;

        //  convertBBoxImgRes(scalingFactor, m_InputW,m_InputH,image_w,image_h, bbi.box);
        bbi.label = maxIndex;
        bbi.prob = maxProb;
        bbi.classId = getClassId(maxIndex);
        binfo.push_back(bbi);
    }

	void calcuate_letterbox_message(const int imageH, const int imageW,
		float &sh,float &sw,
		int &xOffset,int &yOffset) const
	{
        float r = std::min(static_cast<float>(m_InputH) / static_cast<float>(imageH), static_cast<float>(m_InputW) / static_cast<float>(imageW));
        int resizeH = (std::round(imageH*r));
        int resizeW = (std::round(imageW*r));

		sh = r;
		sw = r;
		if ((m_InputW - resizeW) % 2) resizeW--;
		if ((m_InputH - resizeH) % 2) resizeH--;
		assert((m_InputW - resizeW) % 2 == 0);
		assert((m_InputH - resizeH) % 2 == 0);
        xOffset = (m_InputW - resizeW) / 2;
        yOffset = (m_InputH - resizeH) / 2;
	}
	BBox convert_bbox_res(const float& bx, const float& by, const float& bw, const float& bh,
		const uint32_t& stride_h_, const uint32_t& stride_w_, const uint32_t& netW, const uint32_t& netH)
	{
		BBox b;
		// Restore coordinates to network input resolution
		float x = bx * stride_w_;
		float y = by * stride_h_;

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

	inline void cvt_box(const float sh,
		const float sw,
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
		bbox.x1 /= sw;
		bbox.x2 /= sw;
		bbox.y1 /= sh;
		bbox.y2 /= sh;
	}

	inline void add_bbox_proposal(const float bx, const float by, const float bw, const float bh,
		const uint32_t stride_h_, const uint32_t stride_w_, const float scaleH, const float scaleW, const float xoffset_, const float yoffset, const int maxIndex, const float maxProb,
		const uint32_t 	image_w, const uint32_t image_h,
		std::vector<BBoxInfo>& binfo)
	{
		BBoxInfo bbi;
		bbi.box = convert_bbox_res(bx, by, bw, bh, stride_h_, stride_w_, m_InputW, m_InputH);
		if ((bbi.box.x1 > bbi.box.x2) || (bbi.box.y1 > bbi.box.y2))
			return;

        if (tensor_rt::ModelType::YOLOV5 == m_NetworkType)
		{
			cvt_box(scaleH, scaleW, xoffset_, yoffset, bbi.box);
		}
		else
		{
			bbi.box.x1 = ((float)bbi.box.x1 / (float)m_InputW)*(float)image_w;
			bbi.box.y1 = ((float)bbi.box.y1 / (float)m_InputH)*(float)image_h;
			bbi.box.x2 = ((float)bbi.box.x2 / (float)m_InputW)*(float)image_w;
			bbi.box.y2 = ((float)bbi.box.y2 / (float)m_InputH)*(float)image_h;
		}
		
		bbi.label = maxIndex;
		bbi.prob = maxProb;
		bbi.classId = getClassId(maxIndex);
		binfo.push_back(bbi);
	};

private:
    Logger m_Logger;
    void createYOLOEngine(const nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT, Int8EntropyCalibrator* calibrator = nullptr);
    void create_engine_yolov5(const nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT, Int8EntropyCalibrator* calibrator = nullptr);
    std::vector<std::map<std::string, std::string>> parseConfigFile(const std::string cfgFilePath);
    void parseConfigBlocks();
	void parse_cfg_blocks_v5(const  std::vector<std::map<std::string, std::string>> &vec_block_);
    void allocateBuffers();
    bool verifyYoloEngine();
    void destroyNetworkUtils(std::vector<nvinfer1::Weights>& trtWeights);
    void writePlanFileToDisk();

private:
    Timer _timer;
    void load_weights_v5(const std::string s_weights_path_, std::map<std::string, std::vector<float>> &vec_wts_);

    int _n_yolo_ind = 0;
};

#endif // _YOLO_H_
