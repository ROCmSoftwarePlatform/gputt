/******************************************************************************
MIT License

Copyright (c) 2016 Antti-Pekka Hynninen
Copyright (c) 2016 Oak Ridge National Laboratory (UT-Batelle)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************************/
#ifndef GPUTT_TYPES_H
#define GPUTT_TYPES_H

#include "gputt.h"

#define MAX_REG_STORAGE 8

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#define gpuDeviceGetSharedMemConfig hipDeviceGetSharedMemConfig
#define gpuDeviceProp_t hipDeviceProp_t
#define gpuDeviceReset hipDeviceReset
#define gpuDeviceSetSharedMemConfig hipDeviceSetSharedMemConfig
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuError_t hipError_t
#define gpuEvent_t hipEvent_t
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuFree hipFree
#define gpuFuncSetSharedMemConfig hipFuncSetSharedMemConfig
#define gpuGetDevice hipGetDevice
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuGetErrorString hipGetErrorString
#define gpuGetLastError hipGetLastError
#define gpuMalloc hipMalloc
#define gpuMemcpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyDefault hipMemcpyDefault
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemset hipMemset
#define gpuMemsetAsync hipMemsetAsync
#define gpuOccupancyMaxActiveBlocksPerMultiprocessor                           \
  hipOccupancyMaxActiveBlocksPerMultiprocessor
#define gpuSetDevice hipSetDevice
#define gpuSharedMemBankSizeEightByte hipSharedMemBankSizeEightByte
#define gpuSharedMemBankSizeFourByte hipSharedMemBankSizeFourByte
#define gpuSharedMemConfig hipSharedMemConfig
#define gpuStream hipStream_t
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define gpuSuccess hipSuccess
#else // __HIP_PLATFORM_HCC__
#define gpuDeviceGetSharedMemConfig cudaDeviceGetSharedMemConfig
#define gpuDeviceProp_t cudaDeviceProp
#define gpuDeviceReset cudaDeviceReset
#define gpuDeviceSetSharedMemConfig cudaDeviceSetSharedMemConfig
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuError_t cudaError_t
#define gpuEvent_t cudaEvent_t
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuFree cudaFree
#define gpuFuncSetSharedMemConfig cudaFuncSetSharedMemConfig
#define gpuGetDevice cudaGetDevice
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuGetErrorString cudaGetErrorString
#define gpuGetLastError cudaGetLastError
#define gpuMalloc cudaMalloc
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyDefault cudaMemcpyDefault
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemset cudaMemset
#define gpuMemsetAsync cudaMemsetAsync
#define gpuOccupancyMaxActiveBlocksPerMultiprocessor                           \
  cudaOccupancyMaxActiveBlocksPerMultiprocessor
#define gpuSetDevice cudaSetDevice
#define gpuSharedMemBankSizeEightByte cudaSharedMemBankSizeEightByte
#define gpuSharedMemBankSizeFourByte cudaSharedMemBankSizeFourByte
#define gpuSharedMemConfig cudaSharedMemConfig
#define gpuStream cudaStream_t
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuSuccess cudaSuccess
#endif // __HIP_PLATFORM_HCC__

#ifdef __HIPCC__
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif

template<typename T> gputtDataType gputtGetDataType() { return gputtDataTypeUnknown; }
template<> inline gputtDataType gputtGetDataType<  double>() { return gputtDataTypeFloat64; }
template<> inline gputtDataType gputtGetDataType<   float>() { return gputtDataTypeFloat32; }
template<> inline gputtDataType gputtGetDataType<  __half>() { return gputtDataTypeFloat16; }
template<> inline gputtDataType gputtGetDataType< int64_t>() { return gputtDataTypeInt64; }
template<> inline gputtDataType gputtGetDataType<uint64_t>() { return gputtDataTypeUInt64; }
template<> inline gputtDataType gputtGetDataType< int32_t>() { return gputtDataTypeInt32; }
template<> inline gputtDataType gputtGetDataType<uint32_t>() { return gputtDataTypeUInt32; }
template<> inline gputtDataType gputtGetDataType< int16_t>() { return gputtDataTypeInt16; }
template<> inline gputtDataType gputtGetDataType<uint16_t>() { return gputtDataTypeUInt16; }
template<> inline gputtDataType gputtGetDataType<  int8_t>() { return gputtDataTypeInt8; }
template<> inline gputtDataType gputtGetDataType< uint8_t>() { return gputtDataTypeUInt8; }
template<> inline gputtDataType gputtGetDataType<   char4>() { return gputtDataTypeInt8x4; }
template<> inline gputtDataType gputtGetDataType<  uchar4>() { return gputtDataTypeUInt8x4; }

// Tensor conversion constants
struct TensorConv {
  int c;
  int d;
  int ct;
};

// Tensor conversion constants input & output pair
// TODO Use nested struct TensorConv instead
struct TensorConvInOut {
  int c_in;
  int d_in;
  int ct_in;

  int c_out;
  int d_out;
  int ct_out;
};

#endif // GPUTT_TYPES_H
