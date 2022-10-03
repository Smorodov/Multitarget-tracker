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
#include "detect.h"

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
	Detect::Detect(const void* data, size_t length)
	{
		const char *d = reinterpret_cast<const char*>(data), *a = d;
		read(d,_n_anchor);
		read(d,_n_classes);
		read(d,_n_grid_h);
		read(d, _n_grid_w);
		read(d, _n_output_size);
		//printf("anchor:%d,classes:%d,gh:%d,gw:%d,size:%d\n", _n_anchor, _n_classes, _n_grid_h, _n_grid_w, _n_output_size);
		assert(d == a + length);
	}

	Detect::Detect(const uint32_t n_anchor_, const uint32_t n_classes_,
		const uint32_t n_grid_h_, const uint32_t n_grid_w_/*,
		const uint32_t &n_stride_h_, const uint32_t &n_stride_w_*/):
		_n_anchor(n_anchor_),
		_n_classes(n_classes_),
		_n_grid_h(n_grid_h_),
		_n_grid_w(n_grid_w_)
	{
		_n_output_size = (5 + _n_classes)*_n_anchor*_n_grid_h*_n_grid_w;
	}

    inline __device__ float sigmoidGPU(const float& x)
    {
        return 1.0f / (1.0f + __expf(-x));
    }

	__global__ void gpu_detect_layer(const float *input_,
		float* output_,
		const uint32_t n_grid_h_,
		const uint32_t n_grid_w_,
		const uint32_t n_classes_,
		const uint32_t n_anchor_)
	{
		uint32_t x_id = blockIdx.x * blockDim.x + threadIdx.x;
		uint32_t y_id = blockIdx.y * blockDim.y + threadIdx.y;
		uint32_t z_id = blockIdx.z * blockDim.z + threadIdx.z;

		if ((x_id >= n_grid_w_) || (y_id >= n_grid_h_) || (z_id >= n_anchor_))
		{
			return;
		}
		//	printf("grid_h:%d,grid_w:%d,class:%d,anchor:%d\n", n_grid_h_, n_grid_w_, n_classes_, n_anchor_);
		const int numGridCells = n_grid_h_ * n_grid_w_;
		const int bbindex = y_id * n_grid_w_ + x_id;

		output_[bbindex + numGridCells * (z_id * (5 + n_classes_) + 0)]
			= 2.f * sigmoidGPU(input_[bbindex + numGridCells * (z_id * (5 + n_classes_) + 0)])-0.5f;

		output_[bbindex + numGridCells * (z_id * (5 + n_classes_) + 1)]
			= 2.f * sigmoidGPU(input_[bbindex + numGridCells * (z_id * (5 + n_classes_) + 1)])-0.5f;

		float w = 2.f * sigmoidGPU(input_[bbindex + numGridCells * (z_id * (5 + n_classes_) + 2)]);
		output_[bbindex + numGridCells * (z_id * (5 + n_classes_) + 2)]
			= w*w;

		float h = 2.f* sigmoidGPU(input_[bbindex + numGridCells * (z_id * (5 + n_classes_) + 3)]);
		output_[bbindex + numGridCells * (z_id * (5 + n_classes_) + 3)]
			= h*h;

		output_[bbindex + numGridCells * (z_id * (5 + n_classes_) + 4)]
			= sigmoidGPU(input_[bbindex + numGridCells * (z_id * (5 + n_classes_) + 4)]);
		for (uint32_t i = 0; i < n_classes_; ++i)
		{
			output_[bbindex + numGridCells * (z_id * (5 + n_classes_) + (5 + i))]
				= sigmoidGPU(input_[bbindex + numGridCells * (z_id * (5 + n_classes_) + (5 + i))]);
		}
	}

	cudaError_t cuda_detect_layer(const void* input_, 
		void* output_,
		const uint32_t& batch_size_,
		const uint32_t& grid_h_,
		const uint32_t& grid_w_,
		const uint32_t& n_classes_,
		const uint32_t& n_anchor_,
		uint64_t n_output_size_, 
		cudaStream_t stream_)
	{
		dim3 threads_per_block(16, 16, 4);
		dim3 number_of_blocks((grid_w_ / threads_per_block.x) + 1,
			(grid_h_ / threads_per_block.y) + 1,
			(n_anchor_ / threads_per_block.z) + 1);
		for (int batch = 0; batch < batch_size_; ++batch)
		{
			gpu_detect_layer << <number_of_blocks, threads_per_block, 0, stream_ >> >(
				reinterpret_cast<const float*>(input_) + (batch * n_output_size_),
				reinterpret_cast<float*>(output_) + (batch * n_output_size_),
				grid_h_,
				grid_w_,
				n_classes_,
				n_anchor_);
		}
		return cudaGetLastError();
	}

	int Detect::enqueue(int batchSize,
		const void* const* inputs,
		void* const* outputs,
		void* workspace,
		cudaStream_t stream) noexcept
	{
		NV_CUDA_CHECK(cuda_detect_layer(inputs[0], outputs[0], batchSize, _n_grid_h, _n_grid_w, _n_classes, _n_anchor, _n_output_size, stream));
		return 0;
	}

    int Detect::enqueue(int batchSize,
		const void* const* inputs,
		void** outputs,
		void* workspace,
		cudaStream_t stream) noexcept
	{
		return enqueue(batchSize, inputs, (void* const*)outputs, workspace, stream);
	}

	bool Detect::supportsFormat(DataType type, PluginFormat format) const noexcept
	{
		return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
	}

	void Detect::configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) noexcept
	{

	}

	size_t Detect::getSerializationSize() const noexcept
	{
		return sizeof(_n_anchor) + sizeof(_n_classes) + sizeof(_n_grid_h) + sizeof(_n_grid_w)
			+ sizeof(_n_output_size);
	}

	void Detect::serialize(void *buffer) const noexcept
	{
		char *d = static_cast<char*>(buffer), *a = d;
		write(d,_n_anchor);
		write(d, _n_classes);
		write(d, _n_grid_h);
		write(d, _n_grid_w);
		write(d, _n_output_size);
		assert(d == a + getSerializationSize());
	}

	void Detect::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
	{

	}
	IPluginV2* Detect::clone() const noexcept
	{
		Detect *p = new Detect(_n_anchor,_n_classes,_n_grid_h,_n_grid_w);
		p->setPluginNamespace(_s_plugin_namespace.c_str());
		return p;
	}
	 

	//
	PluginFieldCollection DetectPluginCreator::_fc{};
	std::vector<PluginField> DetectPluginCreator::_vec_plugin_attributes;

	DetectPluginCreator::DetectPluginCreator()
	{
		_vec_plugin_attributes.clear();
		_fc.nbFields = _vec_plugin_attributes.size();
		_fc.fields = _vec_plugin_attributes.data();
	}

	const char* DetectPluginCreator::getPluginName() const noexcept
	{
		return "DETECT_TRT";
	}

	const char* DetectPluginCreator::getPluginVersion() const noexcept
	{
		return "1.0";
	}

	const PluginFieldCollection* DetectPluginCreator::getFieldNames() noexcept
	{
		return &_fc;
	}

	IPluginV2* DetectPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
	{
		Detect* obj = new Detect();
		obj->setPluginNamespace(_s_name_space.c_str());
		return obj;
	}

	IPluginV2* DetectPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
	{
		Detect* obj = new Detect(serialData, serialLength);
		obj->setPluginNamespace(_s_name_space.c_str());
		return obj;
	}

	void DetectPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
	{
		_s_name_space = libNamespace;
	}

	const char* DetectPluginCreator::getPluginNamespace() const noexcept
	{
		return _s_name_space.c_str();
	}
}//end namespace nvinfer1
