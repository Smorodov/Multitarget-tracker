//sys
#include <cmath>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>
#include <string.h>
//my
#include "hardswish.h"

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
	Hardswish::Hardswish()
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		_n_max_thread_pre_block = prop.maxThreadsPerBlock;
	}

	Hardswish::Hardswish(const void* data, size_t length)
	{
		const char *d = reinterpret_cast<const char*>(data), *a = d;
		r(d, _n_max_thread_pre_block);
		r(d, _n_output_size);
		assert(d == a + length);
	}

	__global__ void kernel_hardswish(const float *input_, float *output_, int n_data_size_)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_data_size_)return;
		if (input_[i] >= 3.0f)
		{
			output_[i] = input_[i];
		}
		else if (input_[i] <= -3.0f)
		{
			output_[i] = 0.0f;
		}
		else
		{
			output_[i] = input_[i] * (input_[i] + 3.0f) / 6.0f;
		}
	}

	cudaError_t cuda_hardswish_layer(const void* input_,
		void* output_,
		const int n_batch_size_,
		const int n_output_size_,
		const int threads_,
		cudaStream_t stream_)
	{
		int n_data_size = n_batch_size_ * n_output_size_;
		kernel_hardswish << <(n_data_size + threads_ -1)/threads_, threads_, 0, stream_ >> >(
				reinterpret_cast<const float*>(input_),
				reinterpret_cast<float*>(output_),
				n_data_size);
		return cudaGetLastError();
	}

	int Hardswish::enqueue(int batchSize,
		const void* const* inputs,
		void* const* outputs,
		void* workspace,
		cudaStream_t stream) noexcept
	{
		//printf("batch_size:%d,output_size:%d,threads:%d\n", batchSize, _n_output_size, _n_max_thread_pre_block);
		NV_CUDA_CHECK(cuda_hardswish_layer(inputs[0], outputs[0], batchSize, _n_output_size , _n_max_thread_pre_block, stream));
		return 0;
	}

    int Hardswish::enqueue(int batchSize,
		const void* const* inputs,
		void** outputs,
		void* workspace,
		cudaStream_t stream) noexcept
	{
		return enqueue(batchSize, inputs, (void* const*)outputs, workspace, stream);
	}

	size_t Hardswish::getSerializationSize() const noexcept
	{
		return sizeof(_n_max_thread_pre_block) +sizeof(_n_output_size);
	}

	void Hardswish::serialize(void *buffer) const noexcept
	{
		char *d = static_cast<char*>(buffer), *a = d;
		w(d, _n_max_thread_pre_block);
		w(d, _n_output_size);
		assert(d == a + getSerializationSize());
	}


	bool Hardswish::supportsFormat(DataType type, PluginFormat format) const noexcept
	{
		return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
	}

	void Hardswish::configureWithFormat(const Dims* inputDims, int nbInputs,
		const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) noexcept
	{
	
	}


	void Hardswish::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) noexcept
	{
		
		_n_output_size = in->dims.d[0] * in->dims.d[1] * in->dims.d[2];
	//	printf("output_size:%d,threads:%d\n", _n_output_size, _n_max_thread_pre_block);
	}
	IPluginV2* Hardswish::clone() const noexcept
	{
		Hardswish *p = new Hardswish();
		p->setPluginNamespace(_s_plugin_namespace.c_str());
		p->_n_max_thread_pre_block = _n_max_thread_pre_block;
		p->_n_output_size = _n_output_size;
		return p;
	}


	//
	PluginFieldCollection HardswishPluginCreator::_fc{};
	std::vector<PluginField> HardswishPluginCreator::_vec_plugin_attributes;

	HardswishPluginCreator::HardswishPluginCreator()
	{
		_vec_plugin_attributes.clear();
		_fc.nbFields = _vec_plugin_attributes.size();
		_fc.fields = _vec_plugin_attributes.data();
	}

	const char* HardswishPluginCreator::getPluginName() const noexcept
	{
		return "HARDSWISH_TRT";
	}

	const char* HardswishPluginCreator::getPluginVersion() const noexcept
	{
		return "1.0";
	}

	const PluginFieldCollection* HardswishPluginCreator::getFieldNames() noexcept
	{
		return &_fc;
	}

	IPluginV2* HardswishPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
	{
		Hardswish* obj = new Hardswish();
		obj->setPluginNamespace(_s_name_space.c_str());
		return obj;
	}

	IPluginV2* HardswishPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
	{
		Hardswish* obj = new Hardswish(serialData, serialLength);
		obj->setPluginNamespace(_s_name_space.c_str());
		return obj;
	}

	void HardswishPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
	{
		_s_name_space = libNamespace;
	}

	const char* HardswishPluginCreator::getPluginNamespace() const noexcept
	{
		return _s_name_space.c_str();
	}
}//end namespace nvinfer1
