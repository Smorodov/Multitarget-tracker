
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

	class Hardswish :public IPluginV2
	{
	public:
		Hardswish();
		Hardswish(const void* data, size_t length);
        ~Hardswish() = default;
        int getNbOutputs()const noexcept override
		{
			return 1;
		}
        Dims getOutputDimensions(int /*index*/, const Dims* inputs, int /*nbInputDims*/) noexcept override
		{
			return inputs[0];
		}
        int initialize() noexcept override
		{
			return 0;
		}
        void terminate() noexcept override
		{
		}
        size_t getWorkspaceSize(int /*maxBatchSize*/) const noexcept override
		{
			return 0;
		}

		bool supportsFormat(DataType type, PluginFormat format) const noexcept override;
		void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) noexcept override;

		int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) noexcept;
        int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept;

        size_t getSerializationSize() const noexcept override;
        void serialize(void* buffer) const noexcept override;
        const char* getPluginType() const noexcept override
		{
			return "HARDSWISH_TRT";
		}
        const char* getPluginVersion() const noexcept override
		{
			return "1.0";
		}
        void destroy() noexcept override
		{
			delete this;
		}
        void setPluginNamespace(const char* pluginNamespace) noexcept override
		{
			_s_plugin_namespace = pluginNamespace;
		}
        const char* getPluginNamespace() const noexcept override
		{
			return _s_plugin_namespace.c_str();
		}
        DataType getOutputDataType(int /*index*/, const nvinfer1::DataType* /*inputTypes*/, int /*nbInputs*/) const noexcept
		{
			return DataType::kFLOAT;
		}
        bool isOutputBroadcastAcrossBatch(int /*outputIndex*/, const bool* /*inputIsBroadcasted*/, int /*nbInputs*/) const noexcept
		{
			return false;
		}
        bool canBroadcastInputAcrossBatch(int /*inputIndex*/) const noexcept
		{
			return false;
		}
        void attachToContext(cudnnContext* /*cudnnContext*/, cublasContext* /*cublasContext*/, IGpuAllocator* /*gpuAllocator*/) noexcept
        {
        }
		void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) noexcept;
        void detachFromContext() noexcept
        {
        }
        bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int /*nbInputs*/, int /*nbOutputs*/) const noexcept
		{
			return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
		}
		IPluginV2* clone() const  noexcept override;
	private:

        uint32_t _n_max_thread_pre_block = 0;
        uint32_t _n_output_size = 0;
		std::string _s_plugin_namespace;
	}; //end detect

	class HardswishPluginCreator : public IPluginCreator
	{
	public:
		HardswishPluginCreator();
		~HardswishPluginCreator() override = default;
        const char* getPluginName()const noexcept override;
		const char* getPluginVersion() const  noexcept override;
        const PluginFieldCollection* getFieldNames() noexcept override;
        IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
        IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
        void setPluginNamespace(const char* libNamespace) noexcept override;
        const char* getPluginNamespace() const noexcept override;

	private:
		std::string _s_name_space;
		static PluginFieldCollection _fc;
		static std::vector<PluginField> _vec_plugin_attributes;
	};//end detect creator

}//end namespace nvinfer1



#endif
