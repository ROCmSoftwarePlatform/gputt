#ifndef GPUTT_RUNTIME_H
#define GPUTT_RUNTIME_H

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
#define gpuOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define gpuSetDevice hipSetDevice
#define gpuSharedMemBankSizeEightByte hipSharedMemBankSizeEightByte
#define gpuSharedMemBankSizeFourByte hipSharedMemBankSizeFourByte
#define gpuSharedMemConfig hipSharedMemConfig
#define gpuStream_t hipStream_t
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
#define gpuOccupancyMaxActiveBlocksPerMultiprocessor cudaOccupancyMaxActiveBlocksPerMultiprocessor
#define gpuSetDevice cudaSetDevice
#define gpuSharedMemBankSizeEightByte cudaSharedMemBankSizeEightByte
#define gpuSharedMemBankSizeFourByte cudaSharedMemBankSizeFourByte
#define gpuSharedMemConfig cudaSharedMemConfig
#define gpuStream_t cudaStream_t
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuSuccess cudaSuccess
#endif // __HIP_PLATFORM_HCC__

#endif // GPUTT_RUNTIME_H
