#include <cmath>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include "chunk.h"
#include <cuda_runtime.h>
#define NV_CUDA_CHECK(status)                                                                      \
    {                                                                                              \
        if (status != 0)                                                                           \
        {                                                                                          \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) << " in file " << __FILE__ \
                      << " at line " << __LINE__ << std::endl;                                     \
            abort();                                                                               \
        }                                                                                          \
    }


namespace nvinfer1
{
	Chunk::Chunk(const void* buffer, size_t size) 
	{
		assert(size == sizeof(_n_size_split));
		_n_size_split = *reinterpret_cast<const int*>(buffer);
 	}

	int Chunk::getNbOutputs() const noexcept
	{
		return 2;
	}

	Dims Chunk::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)noexcept
	{
		assert(nbInputDims == 1);
		assert(index == 0 || index == 1);
		return Dims3(inputs[0].d[0] / 2, inputs[0].d[1], inputs[0].d[2]);
	}

	int Chunk::initialize() noexcept
	{
		return 0;
	}

	void Chunk::terminate() noexcept
	{
	}

	size_t Chunk::getWorkspaceSize(int maxBatchSize) const noexcept
	{
		return 0;
	}

	int Chunk::enqueue(int batchSize,
		const void* const* inputs,
		void** outputs,
		void* workspace,
		cudaStream_t stream)noexcept
    {
        return enqueue(batchSize, inputs, (void* const*)outputs, workspace, stream);
    }

	int Chunk::enqueue(int batchSize, 
		const void* const* inputs,
		void* const* outputs, 
		void* workspace,
		cudaStream_t stream) noexcept
	{
		//batch
		for (int b = 0; b < batchSize; ++b)
		{
			NV_CUDA_CHECK(cudaMemcpyAsync((char*)outputs[0] + b * _n_size_split, (char*)inputs[0] + b * 2 * _n_size_split, _n_size_split, cudaMemcpyDeviceToDevice, stream));
			NV_CUDA_CHECK(cudaMemcpyAsync((char*)outputs[1] + b * _n_size_split, (char*)inputs[0] + b * 2 * _n_size_split + _n_size_split, _n_size_split, cudaMemcpyDeviceToDevice, stream));
		}
		return 0;
	}

	size_t Chunk::getSerializationSize() const noexcept
	{
		return sizeof(_n_size_split);
	}

	void Chunk::serialize(void *buffer)const noexcept
	{
		*reinterpret_cast<int*>(buffer) = _n_size_split;
	}

	const char* Chunk::getPluginType()const noexcept
	{
		return "CHUNK_TRT";
	}
	const char* Chunk::getPluginVersion() const noexcept
	{
		return "1.0";
	}

	void Chunk::destroy() noexcept
	{
		delete this;
	}
	
	void Chunk::setPluginNamespace(const char* pluginNamespace) noexcept
	{
		_s_plugin_namespace = pluginNamespace;
	}

	const char* Chunk::getPluginNamespace() const noexcept
	{
		return _s_plugin_namespace.c_str();
	}

	DataType Chunk::getOutputDataType(int index,
		const nvinfer1::DataType* inputTypes,
		int nbInputs) const noexcept
	{
		assert(index == 0 || index == 1);
		return DataType::kFLOAT;
	}

	bool Chunk::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
	{
		return false;
	}

	bool Chunk::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
	{
		return false;
	}

	void Chunk::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
	{
	}

	void Chunk::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
	{
		_n_size_split = in->dims.d[0] / 2 * in->dims.d[1] * in->dims.d[2] *sizeof(float);
	}
	void Chunk::detachFromContext()
	{
	}

	bool Chunk::supportsFormat(DataType type, PluginFormat format) const noexcept
	{
		return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
	}

	void Chunk::configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) noexcept
	{
		size_t typeSize = sizeof(float);
		switch (type)
        {
        case DataType::kFLOAT:
            typeSize = sizeof(float);
            break;
        case DataType::kHALF:
            typeSize = sizeof(float) / 2;
            break;
        case DataType::kINT8:
            typeSize = 1;
            break;
        case DataType::kINT32:
            typeSize = 4;
            break;
        case DataType::kBOOL:
            typeSize = 1;
            break;
        }
		_n_size_split = inputDims->d[0] / 2 * inputDims->d[1] * inputDims->d[2] * typeSize;
	}

	// Clone the plugin
	IPluginV2* Chunk::clone() const noexcept
	{
		Chunk *p = new Chunk();
		p->_n_size_split = _n_size_split;
		p->setPluginNamespace(_s_plugin_namespace.c_str());
		return p;
	}

	//----------------------------
	PluginFieldCollection ChunkPluginCreator::_fc{};
	std::vector<PluginField> ChunkPluginCreator::_vec_plugin_attributes;

	ChunkPluginCreator::ChunkPluginCreator()
	{
		_vec_plugin_attributes.clear();
		_fc.nbFields = _vec_plugin_attributes.size();
		_fc.fields = _vec_plugin_attributes.data();
	}

	const char* ChunkPluginCreator::getPluginName() const noexcept
	{
		return "CHUNK_TRT";
	}
	
	const char* ChunkPluginCreator::getPluginVersion() const noexcept
	{
		return "1.0";
	}

	const PluginFieldCollection* ChunkPluginCreator::getFieldNames()noexcept
	{
		return &_fc;
	}

	IPluginV2* ChunkPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)noexcept
	{
		Chunk* obj = new Chunk();
		obj->setPluginNamespace(_s_name_space.c_str());
		return obj;
	}

	IPluginV2* ChunkPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)noexcept
	{
		Chunk* obj = new Chunk(serialData,serialLength);
		obj->setPluginNamespace(_s_name_space.c_str());
		return obj;
	}

	void ChunkPluginCreator::setPluginNamespace(const char* libNamespace)noexcept
	{
		_s_name_space = libNamespace;
	}

	const char* ChunkPluginCreator::getPluginNamespace() const noexcept
	{
		return _s_name_space.c_str();
	}
}//namespace nvinfer1
