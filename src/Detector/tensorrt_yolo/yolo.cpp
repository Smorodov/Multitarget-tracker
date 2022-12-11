#include "yolo.h"
#include <memory>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <NvInfer.h>

namespace nvinfer1
{
REGISTER_TENSORRT_PLUGIN(DetectPluginCreator);
}
///
/// \brief Yolo::Yolo
/// \param networkInfo
/// \param inferParams
///
Yolo::Yolo(const NetworkInfo& networkInfo, const InferParams& inferParams)
    : m_NetworkType(networkInfo.m_networkType),
      m_ConfigFilePath(networkInfo.configFilePath),
      m_WtsFilePath(networkInfo.wtsFilePath),
      m_LabelsFilePath(networkInfo.labelsFilePath),
      m_Precision(networkInfo.precision),
      m_DeviceType(networkInfo.deviceType),
      m_CalibImages(inferParams.calibImages),
      m_CalibImagesFilePath(inferParams.calibImagesPath),
      m_CalibTableFilePath(networkInfo.calibrationTablePath),
      m_InputBlobName(networkInfo.inputBlobName),
      m_ProbThresh(inferParams.probThresh),
      m_NMSThresh(inferParams.nmsThresh),
      m_PrintPerfInfo(inferParams.printPerfInfo),
      m_PrintPredictions(inferParams.printPredictionInfo),
      m_BatchSize(inferParams.batchSize),
      m_videoMemory(inferParams.videoMemory)
{
	// m_ClassNames = loadListFromTextFile(m_LabelsFilePath);

	m_configBlocks = parseConfigFile(m_ConfigFilePath);
    if (m_NetworkType == tensor_rt::ModelType::YOLOV5)
		parse_cfg_blocks_v5(m_configBlocks);
	else
		parseConfigBlocks();

	m_EnginePath = networkInfo.data_path + "-" + m_Precision + "-batch" + std::to_string(m_BatchSize) + ".engine";
	if (m_Precision == "kFLOAT")
	{
		if (tensor_rt::ModelType::YOLOV5 == m_NetworkType)
			create_engine_yolov5();
		else
			createYOLOEngine();
	}
	else if (m_Precision == "kINT8")
	{
		Int8EntropyCalibrator calibrator(m_BatchSize, m_CalibImages, m_CalibImagesFilePath,
			m_CalibTableFilePath, m_InputSize, m_InputH, m_InputW,
			m_InputBlobName, m_NetworkType);
		if (tensor_rt::ModelType::YOLOV5 == m_NetworkType)
			create_engine_yolov5(nvinfer1::DataType::kINT8, &calibrator);
		else
			createYOLOEngine(nvinfer1::DataType::kINT8, &calibrator);
	}
	else if (m_Precision == "kHALF")
	{
		if (tensor_rt::ModelType::YOLOV5 == m_NetworkType)
			create_engine_yolov5(nvinfer1::DataType::kHALF, nullptr);
		else
			createYOLOEngine(nvinfer1::DataType::kHALF, nullptr);
	}
	else
	{
		std::cout << "Unrecognized precision type " << m_Precision << std::endl;
		assert(0);
	}

	//assert(m_PluginFactory != nullptr);
	m_Engine = loadTRTEngine(m_EnginePath,/* m_PluginFactory,*/ m_Logger);
	assert(m_Engine != nullptr);
	m_Context = m_Engine->createExecutionContext();
	assert(m_Context != nullptr);
	m_InputBindingIndex = m_Engine->getBindingIndex(m_InputBlobName.c_str());
	assert(m_InputBindingIndex != -1);
	assert(m_BatchSize <= static_cast<uint32_t>(m_Engine->getMaxBatchSize()));
	allocateBuffers();
	NV_CUDA_CHECK(cudaStreamCreate(&m_CudaStream));
	assert(verifyYoloEngine());
}

///
/// \brief Yolo::~Yolo
///
Yolo::~Yolo()
{
    for (auto& tensor : m_OutputTensors)
		NV_CUDA_CHECK(cudaFreeHost(tensor.hostBuffer));
    for (auto& deviceBuffer : m_DeviceBuffers)
		NV_CUDA_CHECK(cudaFree(deviceBuffer));
    NV_CUDA_CHECK(cudaStreamDestroy(m_CudaStream));
    if (m_Context)
    {
#if (NV_TENSORRT_MAJOR < 8)
		m_Context->destroy();
#else
        delete m_Context;
#endif
        m_Context = nullptr;
    }

    if (m_Engine)
    {
#if (NV_TENSORRT_MAJOR < 8)
		m_Engine->destroy();
#else
        delete m_Engine;
#endif
        m_Engine = nullptr;
    }
}

///
/// \brief split_layer_index
/// \param s_
/// \param delimiter_
/// \return
///
std::vector<int> split_layer_index(const std::string &s_,const std::string &delimiter_)
{
	std::vector<int> index;
	std::string s = s_;
	size_t pos = 0;
	std::string token;
	while ((pos = s.find(delimiter_)) != std::string::npos) 
	{
		token = s.substr(0, pos);
		index.push_back(std::stoi(trim(token)));
		s.erase(0, pos + delimiter_.length());
	}
	index.push_back(std::stoi(trim(s)));
	return index;
}

///
/// \brief Yolo::createYOLOEngine
/// \param dataType
/// \param calibrator
///
void Yolo::createYOLOEngine(const nvinfer1::DataType dataType, Int8EntropyCalibrator* calibrator)
{
    if (fileExists(m_EnginePath))
        return;

    std::vector<float> weights = LoadWeights(m_WtsFilePath);
    std::vector<nvinfer1::Weights> trtWeights;
    int weightPtr = 0;
    int channels = m_InputC;
	m_Builder = nvinfer1::createInferBuilder(m_Logger);
	nvinfer1::IBuilderConfig* config = m_Builder->createBuilderConfig();
    m_Network = m_Builder->createNetworkV2(0U);
    if ((dataType == nvinfer1::DataType::kINT8 && !m_Builder->platformHasFastInt8())
        || (dataType == nvinfer1::DataType::kHALF && !m_Builder->platformHasFastFp16()))
    {
        std::cout << "Platform doesn't support this precision." << std::endl;
        assert(0);
    }

    nvinfer1::ITensor* data = m_Network->addInput(
        m_InputBlobName.c_str(), nvinfer1::DataType::kFLOAT,
        nvinfer1::Dims{ 3, static_cast<int>(m_InputC), static_cast<int>(m_InputH), static_cast<int>(m_InputW) });
    assert(data != nullptr);
    // Add elementwise layer to normalize pixel values 0-1
    nvinfer1::Dims divDims{
        3,
        {static_cast<int>(m_InputC), static_cast<int>(m_InputH), static_cast<int>(m_InputW)}
        /*{nvinfer1::DimensionType::kCHANNEL, nvinfer1::DimensionType::kSPATIAL,
         nvinfer1::DimensionType::kSPATIAL}*/};
    nvinfer1::Weights divWeights{nvinfer1::DataType::kFLOAT, nullptr,
                                 static_cast<int64_t>(m_InputSize)};
    float* divWt = new float[m_InputSize];
    for (uint32_t w = 0; w < m_InputSize; ++w) divWt[w] = 255.0;
    divWeights.values = divWt;
    trtWeights.push_back(divWeights);
    nvinfer1::IConstantLayer* constDivide = m_Network->addConstant(divDims, divWeights);
    assert(constDivide != nullptr);
    nvinfer1::IElementWiseLayer* elementDivide = m_Network->addElementWise(
        *data, *constDivide->getOutput(0), nvinfer1::ElementWiseOperation::kDIV);
    assert(elementDivide != nullptr);

    nvinfer1::ITensor* previous = elementDivide->getOutput(0);
    std::vector<nvinfer1::ITensor*> tensorOutputs;
    uint32_t outputTensorCount = 0;

    // build the network using the network API
    for (uint32_t i = 0; i < m_configBlocks.size(); ++i)
    {
		// check if num. of channels is correct
        assert(getNumChannels(previous) == channels);
        std::string layerIndex = "(" + std::to_string(i) + ")";

        if (m_configBlocks.at(i).at("type") == "net")
        {
            printLayerInfo("", "layer", "     inp_size", "     out_size", "weightPtr");
        }
        else if (m_configBlocks.at(i).at("type") == "convolutional")
        {
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::ILayer* out;
            std::string layerType;
			//check activation
			std::string activation = "";
			if (m_configBlocks.at(i).find("activation") != m_configBlocks.at(i).end())
			{
				activation = m_configBlocks[i]["activation"];
			}
            // check if batch_norm enabled
            if ((m_configBlocks.at(i).find("batch_normalize") != m_configBlocks.at(i).end()) &&
				("leaky" == activation))
            {
                out = netAddConvBNLeaky(i, m_configBlocks.at(i), weights, trtWeights, weightPtr,
                                        channels, previous, m_Network);
                layerType = "conv-bn-leaky";
            }
			else if ((m_configBlocks.at(i).find("batch_normalize") != m_configBlocks.at(i).end()) &&
				("mish" == activation))
			{
				out = net_conv_bn_mish(i, m_configBlocks.at(i), weights, trtWeights, weightPtr,
										channels, previous, m_Network);
				layerType = "conv-bn-mish";
			}
            else// if("linear" == activation)
            {
                out = netAddConvLinear(i, m_configBlocks.at(i), weights, trtWeights, weightPtr,
                                       channels, previous, m_Network);
                layerType = "conv-linear";
            }
            previous = out->getOutput(0);
            assert(previous != nullptr);
            channels = getNumChannels(previous);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(out->getOutput(0));
            printLayerInfo(layerIndex, layerType, inputVol, outputVol, std::to_string(weightPtr));
        }
        else if (m_configBlocks.at(i).at("type") == "shortcut")
        {
            assert(m_configBlocks.at(i).at("activation") == "linear");
            assert(m_configBlocks.at(i).find("from") != m_configBlocks.at(i).end());
            int from = stoi(m_configBlocks.at(i).at("from"));

            std::string inputVol = dimsToString(previous->getDimensions());
            // check if indexes are correct
            assert((i - 2 >= 0) && (i - 2 < tensorOutputs.size()));
            assert((i + from - 1 >= 0) && (i + from - 1 < tensorOutputs.size()));
            assert(i + from - 1 < i - 2);
            nvinfer1::IElementWiseLayer* ew
                = m_Network->addElementWise(*tensorOutputs[i - 2], *tensorOutputs[i + from - 1],
                                            nvinfer1::ElementWiseOperation::kSUM);
            assert(ew != nullptr);
            std::string ewLayerName = "shortcut_" + std::to_string(i);
            ew->setName(ewLayerName.c_str());
            previous = ew->getOutput(0);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(ew->getOutput(0));
            printLayerInfo(layerIndex, "skip", inputVol, outputVol, "    -");
        }
        else if (m_configBlocks.at(i).at("type") == "yolo")
        {
            nvinfer1::Dims prevTensorDims = previous->getDimensions();
           // assert(prevTensorDims.d[1] == prevTensorDims.d[2]);
            TensorInfo& curYoloTensor = m_OutputTensors.at(outputTensorCount);
            curYoloTensor.gridSize = prevTensorDims.d[1];
            curYoloTensor.grid_h = prevTensorDims.d[1];
            curYoloTensor.grid_w = prevTensorDims.d[2];
            curYoloTensor.stride = m_InputW / curYoloTensor.gridSize;
            curYoloTensor.stride_h = m_InputH / curYoloTensor.grid_h;
            curYoloTensor.stride_w = m_InputW / curYoloTensor.grid_w;
            m_OutputTensors.at(outputTensorCount).volume = curYoloTensor.grid_h * curYoloTensor.grid_w * (curYoloTensor.numBBoxes * (5 + curYoloTensor.numClasses));
            std::string layerName = "yolo_" + std::to_string(outputTensorCount);
            curYoloTensor.blobName = layerName;
            nvinfer1::IPluginV2* yoloPlugin = new nvinfer1::YoloLayer(m_OutputTensors.at(outputTensorCount).numBBoxes,
                                                                      m_OutputTensors.at(outputTensorCount).numClasses,
                                                                      m_OutputTensors.at(outputTensorCount).grid_h,
                                                                      m_OutputTensors.at(outputTensorCount).grid_w);
            assert(yoloPlugin != nullptr);
            nvinfer1::IPluginV2Layer* yolo = m_Network->addPluginV2(&previous, 1, *yoloPlugin);

            assert(yolo != nullptr);
            yolo->setName(layerName.c_str());
            std::string inputVol = dimsToString(previous->getDimensions());
            previous = yolo->getOutput(0);
            assert(previous != nullptr);
            previous->setName(layerName.c_str());
            std::string outputVol = dimsToString(previous->getDimensions());
            m_Network->markOutput(*previous);
            channels = getNumChannels(previous);
            tensorOutputs.push_back(yolo->getOutput(0));
            printLayerInfo(layerIndex, "yolo", inputVol, outputVol, std::to_string(weightPtr));
            ++outputTensorCount;
        }
        else if (m_configBlocks.at(i).at("type") == "route")
        {
            size_t found = m_configBlocks.at(i).at("layers").find(",");
            if (found != std::string::npos)//concate multi layers 
            {
				std::vector<int> vec_index = split_layer_index(m_configBlocks.at(i).at("layers"), ",");
				for (auto &ind_layer:vec_index)
				{
					if (ind_layer < 0)
						ind_layer = static_cast<int>(tensorOutputs.size()) + ind_layer;
					assert(ind_layer < static_cast<int>(tensorOutputs.size()) && ind_layer >= 0);
				}
                nvinfer1::ITensor** concatInputs
                    = reinterpret_cast<nvinfer1::ITensor**>(malloc(sizeof(nvinfer1::ITensor*) * vec_index.size()));
                for (size_t ind = 0; ind < vec_index.size(); ++ind)
				{
					concatInputs[ind] = tensorOutputs[vec_index[ind]];
				}
                nvinfer1::IConcatenationLayer* concat
                    = m_Network->addConcatenation(concatInputs, static_cast<int>(vec_index.size()));
                assert(concat != nullptr);
                std::string concatLayerName = "route_" + std::to_string(i - 1);
                concat->setName(concatLayerName.c_str());
                // concatenate along the channel dimension
                concat->setAxis(0);
                previous = concat->getOutput(0);
                assert(previous != nullptr);
                //nvinfer1::Dims debug = previous->getDimensions();
                std::string outputVol = dimsToString(previous->getDimensions());
				int nums = 0;
				for (auto &indx:vec_index)
				{
					nums += getNumChannels(tensorOutputs[indx]);
				}
				channels = nums;
                tensorOutputs.push_back(concat->getOutput(0));
                printLayerInfo(layerIndex, "route", "        -", outputVol,std::to_string(weightPtr));
            }
            else //single layer
            {
                int idx = std::stoi(trim(m_configBlocks.at(i).at("layers")));
                if (idx < 0)
                    idx = static_cast<int>(tensorOutputs.size()) + idx;
                assert(idx < static_cast<int>(tensorOutputs.size()) && idx >= 0);

				//route
				if (m_configBlocks.at(i).find("groups") == m_configBlocks.at(i).end())
				{
					previous = tensorOutputs[idx];
					assert(previous != nullptr);
					std::string outputVol = dimsToString(previous->getDimensions());
					// set the output volume depth
					channels = getNumChannels(tensorOutputs[idx]);
					tensorOutputs.push_back(tensorOutputs[idx]);
					printLayerInfo(layerIndex, "route", "        -", outputVol, std::to_string(weightPtr));
				}
				//yolov4-tiny route split layer
				else
				{
					if (m_configBlocks.at(i).find("group_id") == m_configBlocks.at(i).end())
						assert(0);
					int chunk_idx = std::stoi(trim(m_configBlocks.at(i).at("group_id")));
					nvinfer1::ILayer* out = layer_split(i, tensorOutputs[idx], m_Network);
					std::string inputVol = dimsToString(previous->getDimensions());
					previous = out->getOutput(chunk_idx);
					assert(previous != nullptr);
					channels = getNumChannels(previous);
					std::string outputVol = dimsToString(previous->getDimensions());
					tensorOutputs.push_back(out->getOutput(chunk_idx));
					printLayerInfo(layerIndex,"chunk", inputVol, outputVol, std::to_string(weightPtr));
				}
            }
        }
        else if (m_configBlocks.at(i).at("type") == "upsample")
        {
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::ILayer* out = netAddUpsample(i - 1, m_configBlocks[i], weights, trtWeights,
                                                   channels, previous, m_Network);
            previous = out->getOutput(0);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(out->getOutput(0));
            printLayerInfo(layerIndex, "upsample", inputVol, outputVol, "    -");
        }
        else if (m_configBlocks.at(i).at("type") == "maxpool")
        {
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::ILayer* out = netAddMaxpool(i, m_configBlocks.at(i), previous, m_Network);
            previous = out->getOutput(0);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(out->getOutput(0));
            printLayerInfo(layerIndex, "maxpool", inputVol, outputVol, std::to_string(weightPtr));
        }
        else
        {
            std::cout << "Unsupported layer type --> \"" << m_configBlocks.at(i).at("type") << "\""
                      << std::endl;
            assert(0);
        }
    }

    if (static_cast<int>(weights.size()) != weightPtr)
    {
        std::cout << "Number of unused weights left : " << static_cast<int>(weights.size()) - weightPtr << std::endl;
        assert(0);
    }

 //   std::cout << "Output blob names :" << std::endl;
 //   for (auto& tensor : m_OutputTensors) std::cout << tensor.blobName << std::endl;

    // Create and cache the engine if not already present
    if (fileExists(m_EnginePath))
    {
        std::cout << "Using previously generated plan file located at " << m_EnginePath << std::endl;
        destroyNetworkUtils(trtWeights);
        return;
    }

    //std::cout << "Unable to find cached TensorRT engine for network : " << m_NetworkType << " precision : " << m_Precision << " and batch size :" << m_BatchSize << std::endl;

#if (NV_TENSORRT_MAJOR < 8)
    m_Builder->setMaxBatchSize(m_BatchSize);
    config->setMaxWorkspaceSize(m_videoMemory ? m_videoMemory : (1 << 20));
#else
    size_t workspaceSize = config->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE);
    size_t dlaManagedSRAMSize = config->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kDLA_MANAGED_SRAM);
    size_t dlaLocalDRAMSize = config->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kDLA_LOCAL_DRAM);
    size_t dlaGlobalDRAMSize = config->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kDLA_GLOBAL_DRAM);
    std::cout << "workspaceSize = " << workspaceSize << ", dlaManagedSRAMSize = " << dlaManagedSRAMSize << ", dlaLocalDRAMSize = " << dlaLocalDRAMSize << ", dlaGlobalDRAMSize = " << dlaGlobalDRAMSize << std::endl;

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, m_videoMemory ? m_videoMemory : (1 << 20));
#endif

    if (dataType == nvinfer1::DataType::kINT8)
    {
        assert((calibrator != nullptr) && "Invalid calibrator for INT8 precision");
      //  m_Builder->setInt8Mode(true);
		config->setFlag(nvinfer1::BuilderFlag::kINT8);
     //   m_Builder->setInt8Calibrator(calibrator);
		config->setInt8Calibrator(calibrator);
	//	config->setTacticSources(1U << static_cast<uint32_t>(TacticSource::kCUBLAS) | 1U << static_cast<uint32_t>(TacticSource::kCUBLAS_LT));
    }
    else if (dataType == nvinfer1::DataType::kHALF)
    {
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
     //   m_Builder->setHalf2Mode(true);
    }

  //  m_Builder->allowGPUFallback(true);
#if 0
    int nbLayers = m_Network->getNbLayers();
    int layersOnDLA = 0;
    std::cout << "Total number of layers: " << nbLayers << std::endl;
    for (int i = 0; i < nbLayers; i++)
    {
        nvinfer1::ILayer* curLayer = m_Network->getLayer(i);
        if (m_DeviceType == "kDLA" && m_Builder->canRunOnDLA(curLayer))
        {
            m_Builder->setDeviceType(curLayer, nvinfer1::DeviceType::kDLA);
            layersOnDLA++;
            std::cout << "Set layer " << curLayer->getName() << " to run on DLA" << std::endl;
        }
    }
    std::cout << "Total number of layers on DLA: " << layersOnDLA << std::endl;
#endif

    // Build the engine
    std::cout << "Building the TensorRT Engine..." << std::endl;
    m_Engine = m_Builder->buildEngineWithConfig(*m_Network, *config);
    assert(m_Engine != nullptr);
    std::cout << "Building complete!" << std::endl;

    // Serialize the engine
    writePlanFileToDisk();

    // destroy
    destroyNetworkUtils(trtWeights);
}

int make_division(const float f_in_, const int n_divisor_)
{
	return ceil(f_in_ / n_divisor_)*n_divisor_;
}

void parse_c3_args(const std::string s_args_, int &n_out_ch_, bool &b_shourt_cut_)
{
	std::string s_args = s_args_;
	while (!s_args.empty())
	{
		auto npos = s_args.find_first_of(',');
		if (npos != std::string::npos)
		{
			n_out_ch_ = std::stoi(trim(s_args.substr(0, npos)));
			s_args.erase(0, npos + 1);
		}
		else
		{
			try
			{
				n_out_ch_ = std::stoi(trim(s_args.substr(0, npos)));
			}
			catch (const std::exception&)
			{
			}
			if ("False" == trim(s_args))
			{
				b_shourt_cut_ = false;
			}
			else if ("True" == trim(s_args))
			{
				b_shourt_cut_ = true;
			}
			break;
		}
	}
}

void parse_bottleneck_args(const std::string s_args_, int &n_out_ch_, bool &b_shourt_cut_)
{
	std::string s_args = s_args_;
	while (!s_args.empty())
	{
		auto npos = s_args.find_first_of(',');
		if (npos != std::string::npos)
		{
			n_out_ch_ = std::stoi(trim(s_args.substr(0, npos)));
			s_args.erase(0, npos + 1);
		}
		else
		{
			try
			{
				n_out_ch_ = std::stoi(trim(s_args.substr(0, npos)));
			}
			catch (const std::exception&)
			{
			}
			if ("False" == trim(s_args))
			{
				b_shourt_cut_ = false;
			}
			else if ("True" == trim(s_args))
			{
				b_shourt_cut_ = true;
			}
			break;
		}
	}
}

void parse_spp_args(const std::string s_args_, int &n_filters_, std::vector<int> &vec_k_)
{
	std::string s_args = s_args_;
	vec_k_.clear();
	size_t pos = 0;
	std::string token;
	std::string delimiter = ",";
	bool w = false;
	while ((pos = s_args.find(delimiter)) != std::string::npos) 
	{
		token = s_args.substr(0, pos);
		if (!w)
		{
			n_filters_ = std::stoi(triml(trim(token), "["));
			w = true;
		}
		else
		{
			vec_k_.push_back(std::stoi(triml(trim(token), "[")));
		}
		s_args.erase(0, pos + delimiter.length());
	}
	vec_k_.push_back(std::stoi(triml(trim(s_args), "]")));
}

std::vector<std::string> parse_str_list(const std::string s_args_)
{
	std::string s_args = s_args_;
	std::vector<std::string> vec_args;
	while (!s_args.empty())
	{
		auto npos = s_args.find_first_of(',');
		if (npos != std::string::npos)
		{
			std::string v =trimr( triml(trim(s_args.substr(0, npos)),"'"),"'");
			vec_args.push_back(v);
			s_args.erase(0, npos + 1);
		}
		else
		{
			std::string v =trimr( triml(trim(s_args.substr(0, npos)),"'"),"'");
			vec_args.push_back(v);
			break;
		}
	}
	return vec_args;
}

void parse_upsample(const std::string s_args_, int &n_filters_)
{
	std::string s_args = s_args_;
	size_t pos = 0;
	std::string token;
	std::string delimiter = ",";
	while ((pos = s_args.find(delimiter)) != std::string::npos) 
	{
		token = s_args.substr(0, pos);
		try
		{
			n_filters_ = std::stoi(trim(token));
		}
		catch (const std::exception&)
		{
		}
		s_args.erase(0, pos + delimiter.length());
	}
}

float round_f(const float in_, const int precision_)
{
	float out;
	std::stringstream ss;
	ss << std::setprecision(precision_) << in_;
	ss >> out;
	return out;
}

void Yolo::create_engine_yolov5(const nvinfer1::DataType dataType, Int8EntropyCalibrator* calibrator)
{
	if (fileExists(m_EnginePath))
		return;
	std::map<std::string, std::vector<float>> model_wts;
	load_weights_v5(m_WtsFilePath, model_wts);
	std::vector<nvinfer1::Weights> trtWeights;
	int channels = m_InputC;
	m_Builder = nvinfer1::createInferBuilder(m_Logger);

	m_Network = m_Builder->createNetworkV2(0);
	if ((dataType == nvinfer1::DataType::kINT8 && !m_Builder->platformHasFastInt8())
		|| (dataType == nvinfer1::DataType::kHALF && !m_Builder->platformHasFastFp16()))
	{
		std::cout << "Platform doesn't support this precision." << std::endl;
		assert(0);
	}
	nvinfer1::ITensor* data = m_Network->addInput(
		m_InputBlobName.c_str(),
		nvinfer1::DataType::kFLOAT,
        nvinfer1::Dims{3, static_cast<int>(m_InputC), static_cast<int>(m_InputH), static_cast<int>(m_InputW) });
	assert(data != nullptr);
	// Add elementwise layer to normalize pixel values 0-1
	nvinfer1::Dims divDims{
		3,
		{ static_cast<int>(m_InputC), static_cast<int>(m_InputH), static_cast<int>(m_InputW) }/*,
		{ nvinfer1::DimensionType::kCHANNEL, nvinfer1::DimensionType::kSPATIAL,
		nvinfer1::DimensionType::kSPATIAL }*/ };

	nvinfer1::Weights divWeights{ nvinfer1::DataType::kFLOAT,
		nullptr,
		static_cast<int64_t>(m_InputSize) };
	float* divWt = new float[m_InputSize];
	for (uint32_t w = 0; w < m_InputSize; ++w) divWt[w] = 255.0;
	divWeights.values = divWt;
	trtWeights.push_back(divWeights);
	nvinfer1::IConstantLayer* constDivide = m_Network->addConstant(divDims, divWeights);
	assert(constDivide != nullptr);
	nvinfer1::IElementWiseLayer* elementDivide = m_Network->addElementWise(
		*data, *constDivide->getOutput(0), nvinfer1::ElementWiseOperation::kDIV);
	assert(elementDivide != nullptr);

	nvinfer1::ITensor* previous = elementDivide->getOutput(0);
	std::vector<nvinfer1::ITensor*> tensorOutputs;
	int n_layer_wts_index = 0;
	int n_output = 3 * (_n_classes + 5);
	for (uint32_t i = 0; i < m_configBlocks.size(); ++i)
	{
		assert(getNumChannels(previous) == channels);
		std::string layerIndex = "(" + std::to_string(i) + ")";

		if ("net" == m_configBlocks.at(i).at("type") )
		{
			printLayerInfo("", "layer", "     inp_size", "     out_size","");
		}
		else if ("Focus" == m_configBlocks.at(i).at("type"))
		{
			std::string inputVol = dimsToString(previous->getDimensions());
			std::vector<int> args = parse_int_list(m_configBlocks[i]["args"]);
			int filters = args[0];
			int kernel_size = args[1];
			filters = (n_output != filters) ? make_division(filters*_f_width_multiple, 8) : filters;
			nvinfer1::ILayer* out = layer_focus(trtWeights,
				"model." + std::to_string(i - 1),
				model_wts,
				previous,
				filters,
				kernel_size,
				trtWeights,
				m_Network);
			previous = out->getOutput(0);
			assert(previous != nullptr);
			channels = getNumChannels(previous);
			std::string outputVol = dimsToString(previous->getDimensions());
			tensorOutputs.push_back(out->getOutput(0));
			printLayerInfo(layerIndex,"Focus", inputVol, outputVol, "");
		}//end focus
		else if ("Conv" == m_configBlocks.at(i).at("type"))
		{
			std::string inputVol = dimsToString(previous->getDimensions());
			std::vector<int> args = parse_int_list(m_configBlocks[i]["args"]);
			int filters = args[0];
			int kernel_size = args[1];
			int stride = args[2];
			int n_out_channel = (n_output != filters) ? make_division(filters*_f_width_multiple, 8) : filters;
			nvinfer1::ILayer * out = layer_conv_bn_act(trtWeights,
				"model."+std::to_string(i-1), model_wts, previous, m_Network, n_out_channel, kernel_size, stride);
			previous = out->getOutput(0);
			assert(previous != nullptr);
			channels = getNumChannels(previous);
			std::string outputVol = dimsToString(previous->getDimensions());
			tensorOutputs.push_back(out->getOutput(0));
			printLayerInfo(layerIndex, "Conv", inputVol, outputVol, "");
		}//end Conv
		else if ("C3" == m_configBlocks.at(i).at("type"))
		{
			std::string inputVol = dimsToString(previous->getDimensions());
			int filters = 0;
			bool short_cut =true;
			int number = std::stoi(m_configBlocks[i]["number"]);
			parse_bottleneck_args(m_configBlocks[i]["args"], filters, short_cut);
			int n_out_channel = (n_output != filters) ? make_division(filters*_f_width_multiple, 8) : filters;
			int n_depth = (number > 1) ? (std::max(int(round(_f_depth_multiple *number)), 1)) : number;
			std::string s_model_name = "model." + std::to_string(i- 1);
			auto out = C3(trtWeights,s_model_name, model_wts, m_Network, previous, n_out_channel, n_depth, short_cut);
			previous = out->getOutput(0);
			assert(previous != nullptr);
			channels = getNumChannels(previous);
			std::string outputVol = dimsToString(previous->getDimensions());
			tensorOutputs.push_back(out->getOutput(0));
			printLayerInfo(layerIndex, "C3", inputVol, outputVol, "");
		}// end C3
		else if ("BottleneckCSP" == m_configBlocks.at(i).at("type"))
		{
			std::string inputVol = dimsToString(previous->getDimensions());
			int filters = 0;
			bool short_cut =true;
			int number = std::stoi(m_configBlocks[i]["number"]);
			parse_bottleneck_args(m_configBlocks[i]["args"], filters, short_cut);
			int n_out_channel = (n_output != filters) ? make_division(filters*_f_width_multiple, 8) : filters;
			int n_depth = (number > 1) ? (std::max(int(round(_f_depth_multiple *number)), 1)) : number;
			std::string s_model_name = "model." + std::to_string(i- 1);
			auto out = layer_bottleneck_csp(trtWeights,s_model_name, model_wts, m_Network, previous, n_out_channel, n_depth, short_cut);
			previous = out->getOutput(0);
			assert(previous != nullptr);
			channels = getNumChannels(previous);
			std::string outputVol = dimsToString(previous->getDimensions());
			tensorOutputs.push_back(out->getOutput(0));
			printLayerInfo(layerIndex, "BottleneckCSP", inputVol, outputVol, "");
		}// bottleneckCSP
		else if ("SPP" == m_configBlocks.at(i).at("type"))
		{
			std::string inputVol = dimsToString(previous->getDimensions());
			int filters = 0;
			std::vector<int> vec_k;
			parse_spp_args(m_configBlocks[i]["args"], filters, vec_k);
			int n_out_channel = (n_output != filters) ? make_division(filters*_f_width_multiple, 8) : filters;
			std::string s_model_name = "model." + std::to_string(i- 1);
			auto out = layer_spp(trtWeights, s_model_name, model_wts, m_Network, previous, n_out_channel, vec_k);
			previous = out->getOutput(0);
			assert(previous != nullptr);
			channels = getNumChannels(previous);
			std::string outputVol = dimsToString(previous->getDimensions());
			tensorOutputs.push_back(out->getOutput(0));
			printLayerInfo(layerIndex, "SPP", inputVol, outputVol, "");
		}//end SPP
		else if ("SPPF" == m_configBlocks.at(i).at("type"))
		{
			std::string inputVol = dimsToString(previous->getDimensions());
			int filters = 0;
			std::vector<int> vec_k;
			//parse_spp_args(m_configBlocks[i]["args"], filters, vec_k);
			std::vector<int> args = parse_int_list(m_configBlocks[i]["args"]);
			filters = args[0];
			int n_out_channel = (n_output != filters) ? make_division(filters*_f_width_multiple, 8) : filters;
			std::string s_model_name = "model." + std::to_string(i - 1);
			auto out = layer_sppf(trtWeights, s_model_name, model_wts, m_Network, previous, n_out_channel, args[1]);
			previous = out->getOutput(0);
			assert(previous != nullptr);
			channels = getNumChannels(previous);
			std::string outputVol = dimsToString(previous->getDimensions());
			tensorOutputs.push_back(out->getOutput(0));
			printLayerInfo(layerIndex, "SPP", inputVol, outputVol, "");
		}//end SPPF
		else if ("nn.Upsample" == m_configBlocks.at(i).at("type"))
		{
			std::string inputVol = dimsToString(previous->getDimensions());
			int scale = 0;
			parse_upsample(m_configBlocks[i]["args"], scale);
			std::string s_model_name = "model." + std::to_string(i - 1);
			auto out = layer_upsample(s_model_name, model_wts, m_Network, previous, scale);
			previous = out->getOutput(0);
			assert(previous != nullptr);
			channels = getNumChannels(previous);
			std::string outputVol = dimsToString(previous->getDimensions());
			tensorOutputs.push_back(out->getOutput(0));
			printLayerInfo(layerIndex, "Upsample", inputVol, outputVol, "");
		}//end upsample
		else if ("Concat" == m_configBlocks.at(i).at("type"))
		{
			std::string inputVol = dimsToString(previous->getDimensions());
			int n_dimension = std::stoi(m_configBlocks[i]["args"]);
			std::vector<int> vec_from = parse_int_list(m_configBlocks[i]["from"]);
			for (auto &f:vec_from)
			{
				f = f < 0 ? (f + i-1) : f;
			}
			nvinfer1::ITensor** concat_tensor
				= reinterpret_cast<nvinfer1::ITensor**>(malloc(sizeof(nvinfer1::ITensor*) * vec_from.size() ));
			for (size_t j = 0; j < vec_from.size(); ++j)
			{
				concat_tensor[j] = tensorOutputs[vec_from[j]];
			}
			nvinfer1::IConcatenationLayer* concat = m_Network->addConcatenation(concat_tensor, static_cast<int>(vec_from.size()));
			assert(concat != nullptr);
			concat->setAxis(n_dimension-1);
			previous = concat->getOutput(0);
			assert(previous != nullptr);
			channels = getNumChannels(previous);
			std::string outputVol = dimsToString(previous->getDimensions());
			tensorOutputs.push_back(concat->getOutput(0));
			printLayerInfo(layerIndex, "Concat", inputVol, outputVol, "");
		}//end concat
		else if ("Detect" == m_configBlocks.at(i).at("type"))
		{
			std::string inputVol = dimsToString(previous->getDimensions());
			std::vector<int> vec_from = parse_int_list(m_configBlocks[i]["from"]);
			for (auto &f : vec_from)
			{
				f = f < 0 ? (f + i - 1) : f;
			}
			std::vector<std::string> vec_args = parse_str_list(m_configBlocks[i]["args"]);
			std::string s_model_name = "model." + std::to_string(i - 1);
                        for (size_t ind_from = 0; ind_from < vec_from.size(); ++ind_from)
			{
				int n_filters = (5 + _n_classes) * 3;
				int from = vec_from[ind_from];
				auto conv = layer_conv(trtWeights, s_model_name+".m."+std::to_string(ind_from),
					model_wts, tensorOutputs[from], m_Network, n_filters,1,1,true);

				auto tensor_conv = conv->getOutput(0);
				TensorInfo& curYoloTensor = m_OutputTensors.at(ind_from);
				std::vector<int> chw = dims2chw(tensor_conv->getDimensions());
				curYoloTensor.grid_h = chw[1];
				curYoloTensor.grid_w = chw[2];
				curYoloTensor.stride_h = m_InputH / curYoloTensor.grid_h;
				curYoloTensor.stride_w = m_InputW / curYoloTensor.grid_w;
				m_OutputTensors.at(ind_from).volume = curYoloTensor.grid_h
					* curYoloTensor.grid_w
					* (curYoloTensor.numBBoxes * (5 + curYoloTensor.numClasses));
				std::string layerName = "yolo_" + std::to_string(ind_from);
				curYoloTensor.blobName = layerName;
				/*auto creator = getPluginRegistry()->getPluginCreator("DETECT_TRT", "1.0");
				const nvinfer1::PluginFieldCollection* pluginData = creator->getFieldNames();
				nvinfer1::IPluginV2 *yoloPlugin = creator->createPlugin(("detect" + std::to_string(ind_from)).c_str(), pluginData);*/
				nvinfer1::IPluginV2 *yoloPlugin = new nvinfer1::Detect(curYoloTensor.numBBoxes,
					curYoloTensor.numClasses,
					curYoloTensor.grid_h,
					curYoloTensor.grid_w);
				assert(yoloPlugin != nullptr);
				auto yolo = m_Network->addPluginV2(&tensor_conv, 1, *yoloPlugin);
				assert(yolo != nullptr);

				yolo->setName(layerName.c_str());
				inputVol = dimsToString(tensorOutputs[from]->getDimensions());
				previous = yolo->getOutput(0);
				assert(previous != nullptr);
				previous->setName(layerName.c_str());
				std::string outputVol = dimsToString(previous->getDimensions());
				m_Network->markOutput(*yolo->getOutput(0));
				channels = getNumChannels(yolo->getOutput(0));
				tensorOutputs.push_back(yolo->getOutput(0));
				printLayerInfo(layerIndex, "detect"+std::to_string(ind_from), inputVol, outputVol, "");
			}
		}//end detect
		else
		{
			std::cout << "Unsupported layer type --> \"" << m_configBlocks.at(i).at("type") << "\""
				<< std::endl;
			assert(0);
		}
	}
	if (fileExists(m_EnginePath))
	{
		std::cout << "Using previously generated plan file located at " << m_EnginePath
			<< std::endl;
		destroyNetworkUtils(trtWeights);
		return;
	}

    //std::cout << "Unable to find cached TensorRT engine for network : " << m_NetworkType << " precision : " << m_Precision << " and batch size :" << m_BatchSize << std::endl;

	nvinfer1::IBuilderConfig* config = m_Builder->createBuilderConfig();
#if (NV_TENSORRT_MAJOR < 8)
    m_Builder->setMaxBatchSize(m_BatchSize);
    config->setMaxWorkspaceSize(m_videoMemory ? m_videoMemory : (1 << 20));
#else
    size_t workspaceSize = config->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE);
    size_t dlaManagedSRAMSize = config->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kDLA_MANAGED_SRAM);
    size_t dlaLocalDRAMSize = config->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kDLA_LOCAL_DRAM);
    size_t dlaGlobalDRAMSize = config->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kDLA_GLOBAL_DRAM);
    std::cout << "workspaceSize = " << workspaceSize << ", dlaManagedSRAMSize = " << dlaManagedSRAMSize << ", dlaLocalDRAMSize = " << dlaLocalDRAMSize << ", dlaGlobalDRAMSize = " << dlaGlobalDRAMSize << std::endl;

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, m_videoMemory ? m_videoMemory : (1 << 20));
#endif
	if (dataType == nvinfer1::DataType::kINT8)
	{
		assert((calibrator != nullptr) && "Invalid calibrator for INT8 precision");
		//  m_Builder->setInt8Mode(true);
		config->setFlag(nvinfer1::BuilderFlag::kINT8);
		//   m_Builder->setInt8Calibrator(calibrator);
		config->setInt8Calibrator(calibrator);
		//config->setTacticSources(1U << static_cast<uint32_t>(TacticSource::kCUBLAS) | 1U << static_cast<uint32_t>(TacticSource::kCUBLAS_LT));
	}
	else if (dataType == nvinfer1::DataType::kHALF)
	{
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
		//   m_Builder->setHalf2Mode(true);
	}

#if 0
    m_Builder->allowGPUFallback(true);
    int nbLayers = m_Network->getNbLayers();
    int layersOnDLA = 0;
    //   std::cout << "Total number of layers: " << nbLayers << std::endl;
    for (int i = 0; i < nbLayers; i++)
    {
        nvinfer1::ILayer* curLayer = m_Network->getLayer(i);
        if (m_DeviceType == "kDLA" && m_Builder->canRunOnDLA(curLayer))
        {
            m_Builder->setDeviceType(curLayer, nvinfer1::DeviceType::kDLA);
            layersOnDLA++;
            std::cout << "Set layer " << curLayer->getName() << " to run on DLA" << std::endl;
        }
    }
       std::cout << "Total number of layers on DLA: " << layersOnDLA << std::endl;
#endif
	// Build the engine
	std::cout << "Building the TensorRT Engine..." << std::endl;
	m_Engine = m_Builder->buildEngineWithConfig(*m_Network, *config);
	assert(m_Engine != nullptr);
	std::cout << "Building complete!" << std::endl;

	// Serialize the engine
	writePlanFileToDisk();

	// destroy
	destroyNetworkUtils(trtWeights);
}

void Yolo::load_weights_v5(const std::string s_weights_path_,
	std::map<std::string,std::vector<float>> &vec_wts_)
{
	vec_wts_.clear();
	assert(fileExists(s_weights_path_));
	std::cout << "Loading pre-trained weights..." << std::endl;
	std::ifstream file(s_weights_path_, std::ios_base::binary);
	assert(file.good());
	std::string line;
	while (std::getline(file,line))
	{
		if(line.empty())continue;
		std::stringstream iss(line);
		std::string wts_name;
		iss >> wts_name ;
		std::vector<float> weights;
		uint32_t n_str;
		while(iss >> std::hex >> n_str)
		{
			weights.push_back(reinterpret_cast<float&>(n_str));
		}
		vec_wts_[wts_name] = weights;
	}
	std::cout << "Loading complete!" << std::endl;
}

void Yolo::doInference(const unsigned char* input, const uint32_t batchSize)
{
	//Timer timer;
    assert(batchSize <= m_BatchSize && "Image batch size exceeds TRT engines batch size");
    NV_CUDA_CHECK(cudaMemcpyAsync(m_DeviceBuffers.at(m_InputBindingIndex), input,
                                  batchSize * m_InputSize * sizeof(float), cudaMemcpyHostToDevice,
                                  m_CudaStream));

    m_Context->enqueue(batchSize, m_DeviceBuffers.data(), m_CudaStream, nullptr);
    for (auto& tensor : m_OutputTensors)
    {
        NV_CUDA_CHECK(cudaMemcpyAsync(tensor.hostBuffer, m_DeviceBuffers.at(tensor.bindingIndex),
                                      batchSize * tensor.volume * sizeof(float),
                                      cudaMemcpyDeviceToHost, m_CudaStream));
    }
    cudaStreamSynchronize(m_CudaStream);
	//timer.out("inference");
}

std::vector<BBoxInfo> Yolo::decodeDetections(const int& imageIdx,
										     const int& imageH,
                                             const int& imageW)
{
    std::vector<BBoxInfo> binfo;
    for (auto& tensor : m_OutputTensors)
    {
        std::vector<BBoxInfo> curBInfo = decodeTensor(imageIdx, imageH, imageW, tensor);
        binfo.insert(binfo.end(), curBInfo.begin(), curBInfo.end());
    }
    return binfo;
}

std::vector<std::map<std::string, std::string>> Yolo::parseConfigFile(const std::string cfgFilePath)
{
    assert(fileExists(cfgFilePath));
    std::ifstream file(cfgFilePath);
    assert(file.good());
    std::string line;
    std::vector<std::map<std::string, std::string>> blocks;
    std::map<std::string, std::string> block;

    while (getline(file, line))
    {
        line = trim(line);
        if (line.empty() || line == "\r")
            continue;
        if (line.front() == '#')
            continue;
        if (line.front() == '[')
        {
            if (!block.empty())
            {
                blocks.push_back(block);
                block.clear();
            }
            std::string key = "type";
            std::string value = trim(line.substr(1, line.size() - 2));
            block.emplace(key, value);
        }
        else
        {
            size_t cpos = line.find('=');
            std::string key = trim(line.substr(0, cpos));
            std::string value = trim(line.substr(cpos + 1));
            block.emplace(key, value);
        }
    }
    blocks.push_back(block);
    return blocks;
}

void Yolo::parseConfigBlocks()
{
    for (auto block : m_configBlocks)
    {
        if (block.at("type") == "net")
        {
            assert((block.find("height") != block.end())
                   && "Missing 'height' param in network cfg");
            assert((block.find("width") != block.end()) && "Missing 'width' param in network cfg");
            assert((block.find("channels") != block.end())
                   && "Missing 'channels' param in network cfg");
            assert((block.find("batch") != block.end())
                   && "Missing 'batch' param in network cfg");

            m_InputH = std::stoul(trim(block.at("height")));
            m_InputW = std::stoul(trim(block.at("width")));
            m_InputC = std::stoul(trim(block.at("channels")));
			if (m_BatchSize < 1)
                m_BatchSize = std::stoi(trim(block.at("batch")));
         //   assert(m_InputW == m_InputH);
            m_InputSize = m_InputC * m_InputH * m_InputW;
        }
        else if ((block.at("type") == "region") || (block.at("type") == "yolo"))
        {
            assert((block.find("num") != block.end())
                   && std::string("Missing 'num' param in " + block.at("type") + " layer").c_str());
            assert((block.find("classes") != block.end())
                   && std::string("Missing 'classes' param in " + block.at("type") + " layer")
                          .c_str());
            assert((block.find("anchors") != block.end())
                   && std::string("Missing 'anchors' param in " + block.at("type") + " layer")
                          .c_str());

            TensorInfo outputTensor;
            std::string anchorString = block.at("anchors");
            while (!anchorString.empty())
            {
                size_t npos = anchorString.find_first_of(',');
                if (npos != std::string::npos)
                {
                    float anchor = std::stof(trim(anchorString.substr(0, npos)));
                    outputTensor.anchors.push_back(anchor);
                    anchorString.erase(0, npos + 1);
                }
                else
                {
                    float anchor = std::stof(trim(anchorString));
                    outputTensor.anchors.push_back(anchor);
                    break;
                }
            }

            if ((m_NetworkType == tensor_rt::ModelType::YOLOV3) ||
                (m_NetworkType == tensor_rt::ModelType::YOLOV4) ||
                (m_NetworkType == tensor_rt::ModelType::YOLOV4_TINY))
            {
                assert((block.find("mask") != block.end())
                       && std::string("Missing 'mask' param in " + block.at("type") + " layer")
                              .c_str());

                std::string maskString = block.at("mask");
                while (!maskString.empty())
                {
                    size_t npos = maskString.find_first_of(',');
                    if (npos != std::string::npos)
                    {
                        uint32_t mask = std::stoul(trim(maskString.substr(0, npos)));
                        outputTensor.masks.push_back(mask);
                        maskString.erase(0, npos + 1);
                    }
                    else
                    {
                        uint32_t mask = std::stoul(trim(maskString));
                        outputTensor.masks.push_back(mask);
                        break;
                    }
                }
            }

            outputTensor.numBBoxes = outputTensor.masks.size() > 0
                ? outputTensor.masks.size()
                : std::stoul(trim(block.at("num")));
            outputTensor.numClasses = std::stoul(block.at("classes"));
            if (m_ClassNames.empty())
            {
                for (uint32_t i=0;i< outputTensor.numClasses;++i)
                {
                    m_ClassNames.push_back(std::to_string(i));
                }
            }
			outputTensor.blobName = "yolo_" + std::to_string(_n_yolo_ind);
			outputTensor.gridSize = (m_InputH / 32) * pow(2, _n_yolo_ind);
			outputTensor.grid_h = (m_InputH / 32) * pow(2, _n_yolo_ind);
			outputTensor.grid_w = (m_InputW / 32) * pow(2, _n_yolo_ind);
            if (m_NetworkType == tensor_rt::ModelType::YOLOV4)//pan
			{
				outputTensor.gridSize = (m_InputH / 32) * pow(2, 2-_n_yolo_ind);
				outputTensor.grid_h = (m_InputH / 32) * pow(2, 2-_n_yolo_ind);
				outputTensor.grid_w = (m_InputW / 32) * pow(2, 2-_n_yolo_ind);
			}
			outputTensor.stride = m_InputH / outputTensor.gridSize;
			outputTensor.stride_h = m_InputH / outputTensor.grid_h;
			outputTensor.stride_w = m_InputW / outputTensor.grid_w;
            outputTensor.volume = outputTensor.grid_h * outputTensor.grid_w * (outputTensor.numBBoxes * (5 + outputTensor.numClasses));
            m_OutputTensors.push_back(outputTensor);
			_n_yolo_ind++;
        }
    }
}

void Yolo::parse_cfg_blocks_v5(const  std::vector<std::map<std::string, std::string>> &vec_block_)
{
	std::vector<float> vec_anchors;
	for (const auto &block : vec_block_)
	{
		if ("net" == block.at("type"))
		{
			assert((block.find("height") != block.end())
				&& "Missing 'height' param in network cfg");
			assert((block.find("width") != block.end()) && "Missing 'width' param in network cfg");
			assert((block.find("nc") != block.end())
				&& "Missing 'nc' param in network cfg");
			assert((block.find("depth_multiple") != block.end())
				&& "Missing 'depth_multiple' param in network cfg");
			assert((block.find("width_multiple") != block.end())
				&& "Missing 'width_multiple' param in network cfg");
			assert((block.find("anchors") != block.end())
				&& "Missing 'anchors' param in network cfg");
			assert((block.find("channels") != block.end())
				&& "Missing 'channels' param in network cfg");

			m_InputH = std::stoul(trim(block.at("height")));
			m_InputW = std::stoul(trim(block.at("width")));
			m_InputC = std::stoul(trim(block.at("channels")));
			if (m_BatchSize < 1)
                m_BatchSize = std::stoi(trim(block.at("batch")));
			_f_depth_multiple = std::stof(trim(block.at("depth_multiple")));
			_f_width_multiple = std::stof(trim(block.at("width_multiple")));
			_n_classes = std::stoi(trim(block.at("nc")));
			m_InputSize = m_InputC * m_InputH * m_InputW;
			std::string anchorString = block.at("anchors");
			while (!anchorString.empty())
			{
				auto npos = anchorString.find_first_of(',');
				if (npos != std::string::npos)
				{
					float anchor = std::stof(trim(anchorString.substr(0, npos)));
					vec_anchors.push_back(anchor);
					anchorString.erase(0, npos + 1);
				}
				else
				{
					float anchor = std::stof(trim(anchorString));
					vec_anchors.push_back(anchor);
					break;
				}
			}
		}
		else if ("Detect" == block.at("type"))
		{
			assert((block.find("from") != block.end())
				&& "Missing 'from' param in network cfg");
			std::string from = block.at("from");
			std::vector<int> vec_from{};
			while (!from.empty())
			{
				auto npos = from.find_first_of(",");
				if (std::string::npos != npos)
				{
					vec_from.push_back(std::stoi(trim(from.substr(0, npos))));
					from.erase(0, npos + 1);
				}
				else
				{
					vec_from.push_back(std::stoi(trim(from)));
					break;
				}
			}

			for (uint32_t i = 0; i < vec_from.size(); ++i)
			{
				TensorInfo outputTensor;
				outputTensor.anchors = vec_anchors;
				outputTensor.masks = std::vector<uint32_t>{3*i,3*i+1,3*i+2};
				outputTensor.numBBoxes = static_cast<uint32_t>(outputTensor.masks.size());
				outputTensor.numClasses = _n_classes;
				outputTensor.blobName = "yolo_" + std::to_string(i);
				if (i < 3)
				{
					outputTensor.grid_h = (m_InputH / 32) * pow(2 ,2-i);
					outputTensor.grid_w = (m_InputW / 32) * pow(2 ,2-i);
				}
				else
				{
					outputTensor.grid_h = (m_InputH / 32) /2;
					outputTensor.grid_w = (m_InputW / 32) /2;
				}
				outputTensor.stride_h = m_InputH / outputTensor.grid_h;
				outputTensor.stride_w = m_InputW / outputTensor.grid_w;
				outputTensor.volume = outputTensor.grid_h * outputTensor.grid_w
					*(outputTensor.numBBoxes*(5 + outputTensor.numClasses));
				m_OutputTensors.push_back(outputTensor);

				if (m_ClassNames.empty())
				{
					for (uint32_t j = 0; j < outputTensor.numClasses; ++j)
					{
						m_ClassNames.push_back(std::to_string(j));
					}
				}
			}
		}
	}
	std::cout << "Config Done!" << std::endl;
}

void Yolo::allocateBuffers()
{
    m_DeviceBuffers.resize(m_Engine->getNbBindings(), nullptr);
    assert(m_InputBindingIndex != -1 && "Invalid input binding index");
    NV_CUDA_CHECK(cudaMalloc(&m_DeviceBuffers.at(m_InputBindingIndex),
                             m_BatchSize * m_InputSize * sizeof(float)));

    for (auto& tensor : m_OutputTensors)
    {
        tensor.bindingIndex = m_Engine->getBindingIndex(tensor.blobName.c_str());
        assert((tensor.bindingIndex != -1) && "Invalid output binding index");
        NV_CUDA_CHECK(cudaMalloc(&m_DeviceBuffers.at(tensor.bindingIndex),
                                 m_BatchSize * tensor.volume * sizeof(float)));
        NV_CUDA_CHECK(
            cudaMallocHost(&tensor.hostBuffer, tensor.volume * m_BatchSize * sizeof(float)));
    }
}

bool Yolo::verifyYoloEngine()
{
    assert((m_Engine->getNbBindings() == (1 + m_OutputTensors.size())
            && "Binding info doesn't match between cfg and engine file \n"));

    for (const auto& tensor : m_OutputTensors)
    {
        assert(!strcmp(m_Engine->getBindingName(tensor.bindingIndex), tensor.blobName.c_str())
               && "Blobs names dont match between cfg and engine file \n");
        assert(get3DTensorVolume(m_Engine->getBindingDimensions(tensor.bindingIndex))
                   == tensor.volume
               && "Tensor volumes dont match between cfg and engine file \n");
    }

    assert(m_Engine->bindingIsInput(m_InputBindingIndex) && "Incorrect input binding index \n");
    assert(m_Engine->getBindingName(m_InputBindingIndex) == m_InputBlobName
           && "Input blob name doesn't match between config and engine file");
    assert(get3DTensorVolume(m_Engine->getBindingDimensions(m_InputBindingIndex)) == m_InputSize);
    return true;
}

void Yolo::destroyNetworkUtils(std::vector<nvinfer1::Weights>& trtWeights)
{
    if (m_Network)
    {
#if (NV_TENSORRT_MAJOR < 8)
		m_Network->destroy();
#else
        delete m_Network;
#endif
        m_Network = nullptr;
    }
    if (m_Engine)
    {
#if (NV_TENSORRT_MAJOR < 8)
		m_Engine->destroy();
#else
        delete m_Engine;
#endif
        m_Engine = nullptr;
    }
    if (m_Builder)
    {
#if (NV_TENSORRT_MAJOR < 8)
		m_Builder->destroy();
#else
        delete m_Builder;
#endif
        m_Builder = nullptr;
    }
    if (m_ModelStream)
    {
#if (NV_TENSORRT_MAJOR < 8)
		m_ModelStream->destroy();
#else
        delete m_ModelStream;
#endif
        m_ModelStream = nullptr;
    }

    // deallocate the weights
    for (auto & trtWeight : trtWeights)
    {
        if (trtWeight.count > 0)
            free(const_cast<void*>(trtWeight.values));
    }
}

void Yolo::writePlanFileToDisk()
{
    std::cout << "Serializing the TensorRT Engine..." << std::endl;
    assert(m_Engine && "Invalid TensorRT Engine");
    m_ModelStream = m_Engine->serialize();
    assert(m_ModelStream && "Unable to serialize engine");
    assert(!m_EnginePath.empty() && "Enginepath is empty");

    // write data to output file
    std::stringstream gieModelStream;
    gieModelStream.seekg(0, gieModelStream.beg);
    gieModelStream.write(static_cast<const char*>(m_ModelStream->data()), m_ModelStream->size());
    std::ofstream outFile;
    outFile.open(m_EnginePath, std::ios::binary | std::ios::out);
    outFile << gieModelStream.rdbuf();
    outFile.close();

    std::cout << "Serialized plan file cached at location : " << m_EnginePath << std::endl;
}

