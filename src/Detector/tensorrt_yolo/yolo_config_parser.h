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

#ifndef _YOLO_CONFIG_PARSER_
#define _YOLO_CONFIG_PARSER_

#include "yolo.h"

#include <ctime>

// Init to be called at the very beginning to verify all config params are valid
void yoloConfigParserInit(int argc, char** argv);

NetworkInfo getYoloNetworkInfo();
InferParams getYoloInferParams();
uint64_t getSeed();
std::string getNetworkType();
std::string getPrecision();
std::string getTestImages();
std::string getTestImagesPath();
bool getDecode();
bool getDoBenchmark();
bool getViewDetections();
bool getSaveDetections();
std::string getSaveDetectionsPath();
uint32_t getBatchSize();
bool getShuffleTestSet();

#endif //_YOLO_CONFIG_PARSER_