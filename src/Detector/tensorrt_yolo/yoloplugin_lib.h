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
#ifndef __YOLOPLUGIN_LIB__
#define __YOLOPLUGIN_LIB__

//#include <glib.h>

#include "calibrator.h"
#include "trt_utils.h"
#include "yolo.h"

//#ifdef __cplusplus
//extern "C" {
//#endif

#define MAX_OBJECTS_PER_FRAME 32
typedef struct YoloPluginCtx YoloPluginCtx;
typedef struct YoloPluginOutput YoloPluginOutput;
// Init parameters structure as input, required for instantiating yoloplugin_lib
typedef struct
{
    // Width at which frame/object will be scaled
    int processingWidth;
    // height at which frame/object will be scaled
    int processingHeight;
    // Flag to indicate whether operating on crops of full frame
    int fullFrame;
    // Plugin config file
    std::string configFilePath;
} YoloPluginInitParams;

struct YoloPluginCtx
{
    YoloPluginInitParams initParams;
    NetworkInfo networkInfo;
    InferParams inferParams;
    Yolo* inferenceNetwork;

    // perf vars
    float inferTime = 0.0, preTime = 0.0, postTime = 0.0;
    uint32_t batchSize = 0;
    uint64_t imageCount = 0;
};

// Detected/Labelled object structure, stores bounding box info along with label
typedef struct
{
    int left;
    int top;
    int width;
    int height;
    char label[64];
} YoloPluginObject;

// Output data returned after processing
struct YoloPluginOutput
{
    int numObjects;
    YoloPluginObject object[MAX_OBJECTS_PER_FRAME];
};

// Initialize library context
YoloPluginCtx* YoloPluginCtxInit(YoloPluginInitParams* initParams, size_t batchSize);

// Dequeue processed output
std::vector<YoloPluginOutput*> YoloPluginProcess(YoloPluginCtx* ctx, std::vector<cv::Mat*>& cvmats);

// Deinitialize library context
void YoloPluginCtxDeinit(YoloPluginCtx* ctx);

//#ifdef __cplusplus
//}
//#endif

#endif
