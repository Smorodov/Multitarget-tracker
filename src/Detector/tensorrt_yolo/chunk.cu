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
	Chunk::Chunk()
	{

	}
	Chunk::Chunk(const void* buffer, size_t size) 
	{
		assert(size == sizeof(_n_size_split));
		_n_size_split = *reinterpret_cast<const int*>(buffer);
	}
	Chunk::~Chunk()
	{

	}
	int Chunk::getNbOutputs() const
	{
		return 2;
	}

	Dims Chunk::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
	{
		assert(nbInputDims == 1);
		assert(index == 0 || index == 1);
		return Dims3(inputs[0].d[0] / 2, inputs[0].d[1], inputs[0].d[2]);
	}

	int Chunk::initialize()
	{
		return 0;
	}

	void Chunk::terminate()
	{
	}

	size_t Chunk::getWorkspaceSize(int maxBatchSize) const
	{
		return 0;
	}
	
	int Chunk::enqueue(int batchSize,
		const void* const* inputs,
		void** outputs,
		void* workspace,
		cudaStream_t stream)
	{
		//batch
		for (int b = 0; b < batchSize; ++b)
		{
			NV_CUDA_CHECK(cudaMemcpy((char*)outputs[0] + b * _n_size_split, (char*)inputs[0] + b * 2 * _n_size_split, _n_size_split, cudaMemcpyDeviceToDevice));
			NV_CUDA_CHECK(cudaMemcpy((char*)outputs[1] + b * _n_size_split, (char*)inputs[0] + b * 2 * _n_size_split + _n_size_split, _n_size_split, cudaMemcpyDeviceToDevice));
		}
	//	NV_CUDA_CHECK(cudaMemcpy(outputs[0], inputs[0], _n_size_split, cudaMemcpyDeviceToDevice));
	//	NV_CUDA_CHECK(cudaMemcpy(outputs[1], (void*)((char*)inputs[0] + _n_size_split), _n_size_split, cudaMemcpyDeviceToDevice));
		return 0;
	}

	size_t Chunk::getSerializationSize() const
	{
		return sizeof(_n_size_split);
	}

	void Chunk::serialize(void *buffer)const
	{
		*reinterpret_cast<int*>(buffer) = _n_size_split;
	}
	
	const char* Chunk::getPluginType()const
	{
		return "CHUNK_TRT";
	}
	const char* Chunk::getPluginVersion() const
	{	
		return "1.0";
	}

	void Chunk::destroy()
	{
		delete this;
	}
	
	void Chunk::setPluginNamespace(const char* pluginNamespace)
	{
		_s_plugin_namespace = pluginNamespace;
	}

	const char* Chunk::getPluginNamespace() const
	{
		return _s_plugin_namespace.c_str();
	}

	DataType Chunk::getOutputDataType(int index,
		const nvinfer1::DataType* inputTypes,
		int nbInputs) const
	{
		assert(index == 0 || index == 1);
		return DataType::kFLOAT;
	}

	bool Chunk::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
	{
		return false;
	}

	bool Chunk::canBroadcastInputAcrossBatch(int inputIndex) const
	{
		return false;
	}

	void Chunk::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) {}

	void Chunk::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
	{
		_n_size_split = in->dims.d[0] / 2 * in->dims.d[1] * in->dims.d[2] *sizeof(float);
	}
	void Chunk::detachFromContext() {}

	// Clone the plugin
	IPluginV2IOExt* Chunk::clone() const
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

	const char* ChunkPluginCreator::getPluginName() const
	{
		return "CHUNK_TRT";
	}
	
	const char* ChunkPluginCreator::getPluginVersion() const
	{
		return "1.0";
	}

	const PluginFieldCollection* ChunkPluginCreator::getFieldNames()
	{
		return &_fc;
	}

	IPluginV2IOExt* ChunkPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
	{
		Chunk* obj = new Chunk();
		obj->setPluginNamespace(_s_name_space.c_str());
		return obj;
	}

	IPluginV2IOExt* ChunkPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
	{
		Chunk* obj = new Chunk(serialData,serialLength);
		obj->setPluginNamespace(_s_name_space.c_str());
		return obj;
	}

	void ChunkPluginCreator::setPluginNamespace(const char* libNamespace)
	{
		_s_name_space = libNamespace;
	}

	const char* ChunkPluginCreator::getPluginNamespace() const
	{
		return _s_name_space.c_str();
	}
}//namespace nvinfer1
