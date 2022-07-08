#ifndef CHUNK_H_
#define CHUNK_H_

//#include "NvInfer.h"
//#include "NvInferPlugin.h"
//#include "NvInferRuntimeCommon.h"
//#include <cuda_runtime.h>
//#include <iostream>
//#include <memory>
//#include <sstream>
//#include <string>
//#include <cassert>
//#include <vector>

#include <string>
#include <vector>
#include "NvInfer.h"

namespace nvinfer1
{
	class Chunk : public IPluginV2
	{
	public:
        Chunk() = default;
		Chunk(const void* buffer, size_t length);
		~Chunk() = default;
		int getNbOutputs()const noexcept override;
		Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims)noexcept override;
		int initialize()noexcept override;
		void terminate()noexcept override;
		size_t getWorkspaceSize(int maxBatchSize) const noexcept override;

		int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept;
        int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) noexcept;

		size_t getSerializationSize() const noexcept override;
		void serialize(void* buffer) const noexcept override;
		const char* getPluginType() const noexcept override;
		const char* getPluginVersion() const noexcept override;
		void destroy()noexcept override;
		void setPluginNamespace(const char* pluginNamespace)noexcept override;
		const char* getPluginNamespace() const noexcept override;
		DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept;
		bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept;
		bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept;
        void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator);
		void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput);
		void detachFromContext();
        bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int /*nbInputs*/, int /*nbOutputs*/) const
		{
			return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
		}
		IPluginV2* clone() const noexcept override;
		bool supportsFormat(DataType type, PluginFormat format) const noexcept override;
		void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) noexcept override;

	private:
		std::string _s_plugin_namespace;
		int _n_size_split = 0;
	};

	class ChunkPluginCreator : public IPluginCreator
	{
	public:
		ChunkPluginCreator();
		~ChunkPluginCreator() override = default;
		const char* getPluginName()const noexcept override;
		const char* getPluginVersion() const noexcept override;
		const PluginFieldCollection* getFieldNames()noexcept override;
		IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc)noexcept override;
		IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength)noexcept override;
		void setPluginNamespace(const char* libNamespace)noexcept override;
		const char* getPluginNamespace() const noexcept override;
	private:
		std::string _s_name_space;
		static PluginFieldCollection _fc;
		static std::vector<PluginField> _vec_plugin_attributes;
	};

}//nampespace nvinfer1


#endif


