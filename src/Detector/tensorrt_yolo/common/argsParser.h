/*
 * Copyright (c) 1993-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TENSORRT_ARGS_PARSER_H
#define TENSORRT_ARGS_PARSER_H

#include <string>
#include <vector>
#include <iostream>

namespace samplesCommon
{

//!
//! \brief The SampleParams structure groups the basic parameters required by
//!        all sample networks.
//!
struct SampleParams
{
    int32_t batchSize{1};              //!< Number of inputs in a batch
    int32_t dlaCore{-1};               //!< Specify the DLA core to run network on.
    bool int8{false};                  //!< Allow runnning the network in Int8 mode.
    bool fp16{false};                  //!< Allow running the network in FP16 mode.
    std::vector<std::string> dataDirs; //!< Directory paths where sample data files are stored
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
};

//!
//! \brief The OnnxSampleParams structure groups the additional parameters required by
//!         networks that use ONNX
//!
struct OnnxSampleParams : public SampleParams
{
    std::string onnxFileName; //!< Filename of ONNX file of a network
};

//!
//! /brief Struct to maintain command-line arguments.
//!
struct Args
{
    bool runInInt8{false};
    bool runInFp16{false};
    bool help{false};
    int32_t useDLACore{-1};
    int32_t batch{1};
    std::vector<std::string> dataDirs;
    std::string saveEngine;
    std::string loadEngine;
    bool useILoop{false};
};

} // namespace samplesCommon

#endif // TENSORRT_ARGS_PARSER_H
