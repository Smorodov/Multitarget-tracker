#ifndef DECODETENSORCUDA_H_
#define DECODETENSORCUDA_H_
#include "yolo.h" 

float* decodeTensorCUDA(const int imageIdx, const TensorInfo& tensor);
#endif