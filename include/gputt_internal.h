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
#else // __HIPCC__
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

#include <iostream>

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

namespace gputt {

namespace internal {

// CUDA's __half has eq/neq operators for __device__ only.
struct __half : public ::__half
{
  __host__ __device__
  __half() : ::__half() { }

  __host__ __device__
  __half(int v) : ::__half(v) { }

  __host__ __device__
  __half(const ::__half& v) : ::__half(v) { }

  __host__ __device__
  inline bool operator!=(const __half& y) const
  {
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
    return static_cast<const ::__half>(*this) != static_cast<const ::__half>(y);
#else
    // TODO Does not handle denormals correctly.
    return memcmp(this, &y, sizeof(__half));
#endif
  }

  __host__ __device__
  inline bool operator==(const __half& y) const
  {
    return !(*this != y);
  }

  __device__
  inline __half operator*(const __half& y) const
  {
    return static_cast<const ::__half>(*this) * static_cast<const ::__half>(y);
  }

  __device__
  inline __half operator+(const __half& y) const
  {
    return static_cast<const ::__half>(*this) + static_cast<const ::__half>(y);
  }
};

// CUDA's char4 does not have constructor to initialize
// all items with the same value.
struct char4 : public ::char4
{
  __host__ __device__
  char4() : ::char4() { }

  __host__ __device__
  char4(int v) : ::char4({ static_cast<char>(v), static_cast<char>(v), static_cast<char>(v), static_cast<char>(v) }) { }
};

// CUDA's uchar4 does not have constructor to initialize
// all items with the same value.
struct uchar4 : public ::uchar4
{
  __host__ __device__
  uchar4() : ::uchar4() { }

  __host__ __device__
  uchar4(int v) : ::uchar4({ static_cast<unsigned char>(v), static_cast<unsigned char>(v), static_cast<unsigned char>(v), static_cast<unsigned char>(v) }) { }
};

__host__
inline bool operator!=(const char4& x, const char4& y)
{
  return memcmp(&x, &y, sizeof(char4));
}

__host__
inline bool operator!=(const uchar4& x, const uchar4& y)
{
  return memcmp(&x, &y, sizeof(uchar4));
}

__host__
inline bool operator==(const char4& x, const char4& y)
{
  return !(x != y);
}

__host__
inline bool operator==(const uchar4& x, const uchar4& y)
{
  return !(x != y);
}

__host__ __device__
inline char4 operator*(const char4& x, const char4& y)
{
  char4 v {};
  v.x = x.x * y.x;
  v.y = x.y * y.y;
  v.z = x.z * y.z;
  v.w = x.w * y.w;
  return v;
}

__host__ __device__
inline uchar4 operator*(const uchar4& x, const uchar4& y)
{
  uchar4 v {};
  v.x = x.x * y.x;
  v.y = x.y * y.y;
  v.z = x.z * y.z;
  v.w = x.w * y.w;
  return v;
}

__host__ __device__
inline char4 operator+(const char4& x, const char4& y)
{
  char4 v {};
  v.x = x.x + y.x;
  v.y = x.y + y.y;
  v.z = x.z + y.z;
  v.w = x.w + y.w;
  return v;
}

__host__ __device__
inline uchar4 operator+(const uchar4& x, const uchar4& y)
{
  uchar4 v {};
  v.x = x.x + y.x;
  v.y = x.y + y.y;
  v.z = x.z + y.z;
  v.w = x.w + y.w;
  return v;
}

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

} // namespace internal

} // namespace gputt

inline std::ostream& operator<<(std::ostream& os, const __half& val)
{
  os << static_cast<double>(val);
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const char4& val)
{
  os << static_cast<int>(val.x) << " " << static_cast<int>(val.y) << " " <<
    static_cast<int>(val.z) << " " << static_cast<int>(val.w);
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const uchar4& val)
{
  os << static_cast<int>(val.x) << " " << static_cast<int>(val.y) << " " <<
    static_cast<int>(val.z) << " " << static_cast<int>(val.w);
  return os;
}

inline const char* gputtGetDataTypeString(gputtDataType dtype) {
  switch(dtype) {
  case gputtDataTypeFloat64 : return "gputtDataTypeFloat64";
  case gputtDataTypeFloat32 : return "gputtDataTypeFloat32";
  case gputtDataTypeFloat16 : return "gputtDataTypeFloat16";
  case gputtDataTypeInt64   : return "gputtDataTypeInt64";
  case gputtDataTypeUInt64  : return "gputtDataTypeUInt64";
  case gputtDataTypeInt32   : return "gputtDataTypeInt32";
  case gputtDataTypeUInt32  : return "gputtDataTypeUInt32";
  case gputtDataTypeInt16   : return "gputtDataTypeInt16";
  case gputtDataTypeUInt16  : return "gputtDataTypeUInt16";
  case gputtDataTypeInt8    : return "gputtDataTypeInt8";
  case gputtDataTypeUInt8   : return "gputtDataTypeUInt8";
  case gputtDataTypeInt8x4  : return "gputtDataTypeInt8x4";
  case gputtDataTypeUInt8x4 : return "gputtDataTypeUInt8x4";
  default :                   return "gputtDataTypeUnknown";
  }
}

inline const char* gputtGetTransposeMethodString(gputtTransposeMethod method) {
  switch(method) {
  case gputtTransposeMethodTrivial     : return "gputtTransposeMethodTrivial";
  case gputtTransposeMethodPacked      : return "gputtTransposeMethodPacked";
  case gputtTransposeMethodPackedSplit : return "gputtTransposeMethodPackedSplit";
  case gputtTransposeMethodTiled       : return "gputtTransposeMethodTiled";
  case gputtTransposeMethodTiledCopy   : return "gputtTransposeMethodTiledCopy";
  default                              : return "gputtTransposeMethodUnknown";
  }
}

#endif // GPUTT_TYPES_H
