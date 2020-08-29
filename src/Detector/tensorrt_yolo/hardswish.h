
#ifndef _HARDSWISH_H_
#define _HARDSWISH_H_ 

#include <string>
#include <vector>
#include "NvInfer.h"

namespace nvinfer1
{
	template <typename T>
	void w(char*& buffer, const T& val)
	{
		*reinterpret_cast<T*>(buffer) = val;
		buffer += sizeof(T);
	}

	template <typename T>
	void r(const char*& buffer, T& val)
	{
		val = *reinterpret_cast<const T*>(buffer);
		buffer += sizeof(T);
	}

	class Hardswish :public IPluginV2IOExt
	{
	public:
		Hardswish();
		Hardswish(const void* data, size_t length);
		~Hardswish();
		int getNbOutputs()const override
		{
			return 1;
		}
        Dims getOutputDimensions(int /*index*/, const Dims* inputs, int /*nbInputDims*/) override
		{
			return inputs[0];
		}
		int initialize() override
		{
			return 0;
		}
		void terminate() override
		{
		}
        size_t getWorkspaceSize(int /*maxBatchSize*/) const override
		{
			return 0;
		}
		int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)override;
		size_t getSerializationSize() const override;
		void serialize(void* buffer) const override;
		const char* getPluginType() const override
		{
			return "HARDSWISH_TRT";
		}
		const char* getPluginVersion() const override
		{
			return "1.0";
		}
		void destroy() override
		{
			delete this;
		}
		void setPluginNamespace(const char* pluginNamespace) override
		{
			_s_plugin_namespace = pluginNamespace;
		}
		const char* getPluginNamespace() const override
		{
			return _s_plugin_namespace.c_str();
		}
        DataType getOutputDataType(int /*index*/, const nvinfer1::DataType* /*inputTypes*/, int /*nbInputs*/) const override
		{
			return DataType::kFLOAT;
		}
        bool isOutputBroadcastAcrossBatch(int /*outputIndex*/, const bool* /*inputIsBroadcasted*/, int /*nbInputs*/) const override
		{
			return false;
		}
        bool canBroadcastInputAcrossBatch(int /*inputIndex*/) const override
		{
			return false;
		}
		void attachToContext(
            cudnnContext* /*cudnnContext*/, cublasContext* /*cublasContext*/, IGpuAllocator* /*gpuAllocator*/) override
		{}
		void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;
		void detachFromContext() override
		{}
        bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int /*nbInputs*/, int /*nbOutputs*/) const override
		{
			return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
		}
		IPluginV2IOExt* clone() const override;
	private:

		uint32_t _n_max_thread_pre_block;
		uint32_t _n_output_size;
		std::string _s_plugin_namespace;
	}; //end detect

	class HardswishPluginCreator : public IPluginCreator
	{
	public:
		HardswishPluginCreator();
		~HardswishPluginCreator() override = default;
		const char* getPluginName()const override;
		const char* getPluginVersion() const override;
		const PluginFieldCollection* getFieldNames() override;
		IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;
		IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;
		void setPluginNamespace(const char* libNamespace) override;
		const char* getPluginNamespace() const override;
	private:
		std::string _s_name_space;
		static PluginFieldCollection _fc;
		static std::vector<PluginField> _vec_plugin_attributes;
	};//end detect creator

}//end namespace nvinfer1



#endif
