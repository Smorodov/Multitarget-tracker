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

#include "yoloplugin_lib.h"
#include "yolo_config_parser.h"
#include "yolov2.h"
#include "yolov3.h"

#include <iomanip>
#include <chrono>

static void decodeBatchDetections(const YoloPluginCtx* ctx, std::vector<YoloPluginOutput*>& outputs)
{
    for (uint32_t p = 0; p < ctx->batchSize; ++p)
    {
        YoloPluginOutput* out = new YoloPluginOutput;
        std::vector<BBoxInfo> binfo = ctx->inferenceNetwork->decodeDetections(
            p, ctx->initParams.processingHeight, ctx->initParams.processingWidth);
        std::vector<BBoxInfo> remaining = nmsAllClasses(
            ctx->inferenceNetwork->getNMSThresh(), binfo, ctx->inferenceNetwork->getNumClasses());
        out->numObjects = remaining.size();
        assert(out->numObjects <= MAX_OBJECTS_PER_FRAME);
        for (uint32_t j = 0; j < remaining.size(); ++j)
        {
            BBoxInfo b = remaining.at(j);
            YoloPluginObject obj;
            obj.left = static_cast<int>(b.box.x1);
            obj.top = static_cast<int>(b.box.y1);
            obj.width = static_cast<int>(b.box.x2 - b.box.x1);
            obj.height = static_cast<int>(b.box.y2 - b.box.y1);
            strcpy(obj.label, ctx->inferenceNetwork->getClassName(b.label).c_str());
            out->object[j] = obj;

            if (ctx->inferParams.printPredictionInfo)
            {
                printPredictions(b, ctx->inferenceNetwork->getClassName(b.label));
            }
        }
        outputs.at(p) = out;
    }
}

static void dsPreProcessBatchInput(const std::vector<cv::Mat*>& cvmats, cv::Mat& batchBlob,
                                   const int& processingHeight, const int& processingWidth,
                                   const int& inputH, const int& inputW)
{

    std::vector<cv::Mat> batch_images(
        cvmats.size(), cv::Mat(cv::Size(processingWidth, processingHeight), CV_8UC3));
    for (uint32_t i = 0; i < cvmats.size(); ++i)
    {
        cv::Mat imageResize, imageBorder, inputImage;
        inputImage = *cvmats.at(i);
        int maxBorder = std::max(inputImage.size().width, inputImage.size().height);

        assert((maxBorder - inputImage.size().height) % 2 == 0);
        assert((maxBorder - inputImage.size().width) % 2 == 0);

        int yOffset = (maxBorder - inputImage.size().height) / 2;
        int xOffset = (maxBorder - inputImage.size().width) / 2;

        // Letterbox and resize to maintain aspect ratio
        cv::copyMakeBorder(inputImage, imageBorder, yOffset, yOffset, xOffset, xOffset,
                           cv::BORDER_CONSTANT, cv::Scalar(127.5, 127.5, 127.5));
        cv::resize(imageBorder, imageResize, cv::Size(inputW, inputH), 0, 0, cv::INTER_CUBIC);
        batch_images.at(i) = imageResize;
    }

    batchBlob = cv::dnn::blobFromImages(batch_images, 1.0, cv::Size(inputW, inputH),
                                        cv::Scalar(0.0, 0.0, 0.0), false);
}

YoloPluginCtx* YoloPluginCtxInit(YoloPluginInitParams* initParams, size_t batchSize)
{
    char** gArgV = new char*[2];
    gArgV[0] = new char[64];
    gArgV[1] = new char[512];
    strcpy(gArgV[0], "yolo_plugin_ctx");
    strcpy(gArgV[1], std::string("--flagfile=" + initParams->configFilePath).c_str());
    yoloConfigParserInit(2, gArgV);

    YoloPluginCtx* ctx = new YoloPluginCtx;
    ctx->initParams = *initParams;
    ctx->batchSize = batchSize;
    ctx->networkInfo = getYoloNetworkInfo();
    ctx->inferParams = getYoloInferParams();
    uint32_t configBatchSize = getBatchSize();

    // Check if config batchsize matches buffer batch size in the pipeline
    if (ctx->batchSize != configBatchSize)
    {
        std::cerr
            << "WARNING: Batchsize set in config file overriden by pipeline. New batchsize is "
            << ctx->batchSize << std::endl;
        int npos = ctx->networkInfo.wtsFilePath.find(".weights");
        assert(npos != std::string::npos
               && "wts file file not recognised. File needs to be of '.weights' format");
        std::string dataPath = ctx->networkInfo.wtsFilePath.substr(0, npos);
        ctx->networkInfo.enginePath = dataPath + "-" + ctx->networkInfo.precision + "-batch"
            + std::to_string(ctx->batchSize) + ".engine";
    }

    if ((ctx->networkInfo.networkType == "yolov2")
        || (ctx->networkInfo.networkType == "yolov2-tiny"))
    {
        ctx->inferenceNetwork = new YoloV2(batchSize, ctx->networkInfo, ctx->inferParams);
    }
    else if ((ctx->networkInfo.networkType == "yolov3")
             || (ctx->networkInfo.networkType == "yolov3-tiny"))
    {
        ctx->inferenceNetwork = new YoloV3(batchSize, ctx->networkInfo, ctx->inferParams);
    }
    else
    {
        std::cerr << "ERROR: Unrecognized network type " << ctx->networkInfo.networkType
                  << std::endl;
        std::cerr << "Network Type has to be one among the following : yolov2, yolov2-tiny, yolov3 "
                     "and yolov3-tiny"
                  << std::endl;
        return nullptr;
    }

    delete[] gArgV;
    return ctx;
}

std::vector<YoloPluginOutput*> YoloPluginProcess(YoloPluginCtx* ctx, std::vector<cv::Mat*>& cvmats)
{
    assert((cvmats.size() <= ctx->batchSize) && "Image batch size exceeds TRT engines batch size");
    std::vector<YoloPluginOutput*> outputs = std::vector<YoloPluginOutput*>(cvmats.size(), nullptr);
    cv::Mat preprocessedImages;
    std::chrono::duration<double> preElapsed, inferElapsed, postElapsed;
	std::chrono::time_point<std::chrono::steady_clock> preStart, preEnd, inferStart, inferEnd, postStart, postEnd;

    if (cvmats.size() > 0)
    {
//		preStart = std::chrono::high_resolution_clock::now();
        dsPreProcessBatchInput(cvmats, preprocessedImages, ctx->initParams.processingWidth,
                               ctx->initParams.processingHeight, ctx->inferenceNetwork->getInputH(),
                               ctx->inferenceNetwork->getInputW());
//		preEnd = std::chrono::high_resolution_clock::now();

//		inferStart = std::chrono::high_resolution_clock::now();
        ctx->inferenceNetwork->doInference(preprocessedImages.data, cvmats.size());
  //      inferEnd = std::chrono::high_resolution_clock::now();

//		postStart = std::chrono::high_resolution_clock::now();
        decodeBatchDetections(ctx, outputs);
//		postEnd = std::chrono::high_resolution_clock::now();
    }

    // Perf calc
    if (ctx->inferParams.printPerfInfo)
    {
        preElapsed
            = ((preEnd - preStart) + (preEnd - preStart) / 1000000.0)
            * 1000;
        inferElapsed = ((inferEnd - inferStart)
                        + (inferEnd - inferStart) / 1000000.0)
            * 1000;
        postElapsed = ((postEnd - postStart)
                       + (postEnd - postStart) / 1000000.0)
            * 1000;

        ctx->inferTime += inferElapsed.count();
        ctx->preTime += preElapsed.count();
        ctx->postTime += postElapsed.count();
        ctx->imageCount += cvmats.size();
    }
    return outputs;
}

void YoloPluginCtxDeinit(YoloPluginCtx* ctx)
{
    if (ctx->inferParams.printPerfInfo)
    {
        std::cout << "Yolo Plugin Perf Summary " << std::endl;
        std::cout << "Batch Size : " << ctx->batchSize << std::endl;
        std::cout << std::fixed << std::setprecision(4)
                  << "PreProcess : " << ctx->preTime / ctx->imageCount
                  << " ms Inference : " << ctx->inferTime / ctx->imageCount
                  << " ms PostProcess : " << ctx->postTime / ctx->imageCount << " ms Total : "
                  << (ctx->preTime + ctx->postTime + ctx->inferTime) / ctx->imageCount
                  << " ms per Image" << std::endl;
    }

    delete ctx->inferenceNetwork;
    delete ctx;
}
