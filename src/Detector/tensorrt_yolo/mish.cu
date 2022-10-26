#include <cmath>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include "mish.h"

namespace nvinfer1
{
    // create the plugin at runtime from a byte stream
    MishPlugin::MishPlugin(const void* data, size_t length)
    {
        assert(length == sizeof(input_size_));
        input_size_ = *reinterpret_cast<const int*>(data);
    }

    void MishPlugin::serialize(void* buffer) const noexcept
    {
        *reinterpret_cast<int*>(buffer) = input_size_;
    }

    size_t MishPlugin::getSerializationSize() const noexcept
    {
        return sizeof(input_size_);
    }

    int MishPlugin::initialize() noexcept
    {
        return 0;
    }

	bool MishPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
	{
		return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
	}

	void MishPlugin::configureWithFormat(const Dims* inputDims, int nbInputs,
		const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) noexcept
	{

	}

    Dims MishPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)noexcept
    {
        assert(nbInputDims == 1);
        assert(index == 0);
        input_size_ = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
        // Output dimensions
        return Dims3(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

    // Set plugin namespace
    void MishPlugin::setPluginNamespace(const char* pluginNamespace)noexcept
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* MishPlugin::getPluginNamespace() const noexcept
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType MishPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool MishPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool MishPlugin::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
    {
        return false;
    }

    void MishPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)noexcept
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void MishPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)noexcept
    {
    }

    // Detach the plugin object from its execution context.
    void MishPlugin::detachFromContext()noexcept {}

    const char* MishPlugin::getPluginType() const noexcept
    {
        return "Mish_TRT";
    }

    const char* MishPlugin::getPluginVersion() const noexcept
    {
        return "1";
    }

    void MishPlugin::destroy()noexcept
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2* MishPlugin::clone() const noexcept
    {
        MishPlugin *p = new MishPlugin();
        p->input_size_ = input_size_;
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    __device__ float tanh_activate_kernel(float x){return (2/(1 + expf(-2*x)) - 1);}

    __device__ float softplus_kernel(float x, float threshold = 20) 
	{
        if (x > threshold) return x;                // too large
        else if (x < -threshold) return expf(x);    // too small
        return logf(expf(x) + 1);
    }

    __global__ void mish_kernel(const float *input, float *output, int num_elem) 
	{

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_elem) return;

        //float t = exp(input[idx]);
        //if (input[idx] > 20.0) {
        //    t *= t;
        //    output[idx] = (t - 1.0) / (t + 1.0);
        //} else {
        //    float tt = t * t;
        //    output[idx] = (tt + 2.0 * t) / (tt + 2.0 * t + 2.0);
        //}
        //output[idx] *= input[idx];
        output[idx] = input[idx] * tanh_activate_kernel(softplus_kernel(input[idx]));
    }

    void MishPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize)
	{
        int block_size = thread_count_;
        int grid_size = (input_size_ * batchSize + block_size - 1) / block_size;
        mish_kernel<<<grid_size, block_size, 0, stream>>>(inputs[0], output, input_size_ * batchSize);
    }

	int MishPlugin::enqueue(int batchSize,
		const void* const* inputs,
		void* const* outputs,
		void* workspace,
		cudaStream_t stream) noexcept 
	{
        forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);
        return 0;
    }

    int MishPlugin::enqueue(int batchSize,
		const void* const* inputs,
		void** outputs,
		void* workspace,
		cudaStream_t stream) noexcept 
	{
        return enqueue(batchSize, inputs, (void* const*)outputs, workspace, stream);
    }

    PluginFieldCollection MishPluginCreator::mFC{};
    std::vector<PluginField> MishPluginCreator::mPluginAttributes;

    MishPluginCreator::MishPluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* MishPluginCreator::getPluginName() const noexcept
    {
            return "Mish_TRT";
    }

    const char* MishPluginCreator::getPluginVersion() const noexcept
    {
            return "1";
    }

    const PluginFieldCollection* MishPluginCreator::getFieldNames()noexcept
    {
            return &mFC;
    }

    IPluginV2* MishPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)noexcept
    {
        MishPlugin* obj = new MishPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2* MishPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)noexcept
    {
        // This object will be deleted when the network is destroyed, which will
        // call MishPlugin::destroy()
        MishPlugin* obj = new MishPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }


	void MishPluginCreator::setPluginNamespace(const char* libNamespace)noexcept
	{
		mNamespace = libNamespace;
	}

	const char* MishPluginCreator::getPluginNamespace() const noexcept
	{
		return mNamespace.c_str();
	}


}

