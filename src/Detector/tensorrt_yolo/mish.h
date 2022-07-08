#ifndef _MISH_PLUGIN_H
#define _MISH_PLUGIN_H

#include <string>
#include <vector>
#include "NvInfer.h"


//https://github.com/wang-xinyu/tensorrtx
namespace nvinfer1
{
    class MishPlugin: public IPluginV2
    {
        public:
            explicit MishPlugin() = default;
            MishPlugin(const void* data, size_t length);

            ~MishPlugin() = default;

            int getNbOutputs() const  noexcept override
            {
                return 1;
            }

            Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;

            int initialize() noexcept override;

            virtual void terminate() noexcept override
            {
            }

            virtual size_t getWorkspaceSize(int /*maxBatchSize*/) const noexcept override
            {
                return 0;
            }

			int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) noexcept;
            int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept;
			bool supportsFormat(DataType type, PluginFormat format) const noexcept override;
			void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) noexcept override;

            virtual size_t getSerializationSize() const noexcept override;

            virtual void serialize(void* buffer) const noexcept override;

            bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int /*nbInputs*/, int /*nbOutputs*/) const noexcept
            {
                return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
            }

            const char* getPluginType() const noexcept override;

            const char* getPluginVersion() const noexcept override;

            void destroy() noexcept override;

            IPluginV2* clone() const noexcept override;

            void setPluginNamespace(const char* pluginNamespace) noexcept override;

            const char* getPluginNamespace() const  noexcept override;

            DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept;

            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept;

            bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept;

            void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept;

            void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) noexcept;

            void detachFromContext()noexcept;

            int input_size_ = 0;

        private:
            void forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize = 1);
            int thread_count_ = 256;
            const char* mPluginNamespace = nullptr;
    };

    class MishPluginCreator : public IPluginCreator
    {
        public:
            MishPluginCreator();

            ~MishPluginCreator() override = default;

            const char* getPluginName() const noexcept override;

            const char* getPluginVersion() const  noexcept override;

            const PluginFieldCollection* getFieldNames() noexcept override;

            IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

            IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

			void setPluginNamespace(const char* libNamespace) noexcept override;

			const char* getPluginNamespace() const noexcept override;

        private:
            std::string mNamespace;
            static PluginFieldCollection mFC;
            static std::vector<PluginField> mPluginAttributes;
    };
}
#endif 
