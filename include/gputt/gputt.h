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
#ifndef GPUTT_H
#define GPUTT_H

#ifdef _WIN32
#ifdef gputt_EXPORTS
#define GPUTT_API __declspec(dllexport)
#else
#define GPUTT_API __declspec(dllimport)
#endif
#else // _WIN32
#define GPUTT_API
#endif // _WIN32

#include <stdint.h>

// Handle type that is used to store and access gputt plans
typedef struct gputtHandle_t* gputtHandle;

// Execution stream.
typedef void* gputtStream;

// Return value
typedef enum GPUTT_API gputtResult_t {
  GPUTT_SUCCESS,           // Success
  GPUTT_INVALID_PLAN,      // Invalid plan handle
  GPUTT_INVALID_PARAMETER, // Invalid input parameter
  GPUTT_INVALID_DEVICE,    // Execution tried on device different than where plan
                           // was created
  GPUTT_INTERNAL_ERROR,    // Internal error
  GPUTT_UNDEFINED_ERROR,   // Undefined error
  GPUTT_UNSUPPORTED_METHOD // Selected method is not supported for the given
                           // parameters
} gputtResult;

// Transposing methods
typedef enum GPUTT_API gputtTransposeMethod_t {
  gputtTransposeMethodUnknown = 0,
  gputtTransposeMethodTrivial,
  gputtTransposeMethodPacked,
  gputtTransposeMethodPackedSplit,
  gputtTransposeMethodTiled,
  gputtTransposeMethodTiledCopy,
  NumTransposeMethods
} gputtTransposeMethod;

// gpuTT's type system is generic, and does not make assumptions
// about the actual types supported in hardware.
// Internally, gpuTT maps the generic types onto hardware types,
// for example Float16 is mapped to __half for NVIDIA/AMD GPUs.
// We encode the type size in bytes into enum values.
typedef enum GPUTT_API gputtDataType_t {
  gputtDataTypeUnknown = 0,
  gputtDataTypeFloat64 = (sizeof(  double)     | (1 << 8)),
  gputtDataTypeFloat32 = (sizeof(   float)     | (1 << 8)),
  gputtDataTypeFloat16 = (sizeof(   float) / 2 | (1 << 8)),
  gputtDataTypeInt64   = (sizeof( int64_t)     | (2 << 8)),
  gputtDataTypeUInt64  = (sizeof(uint64_t)     | (3 << 8)),
  gputtDataTypeInt32   = (sizeof( int32_t)     | (2 << 8)),
  gputtDataTypeUInt32  = (sizeof(uint32_t)     | (3 << 8)),
  gputtDataTypeInt16   = (sizeof( int16_t)     | (2 << 8)),
  gputtDataTypeUInt16  = (sizeof(uint16_t)     | (3 << 8)),
  gputtDataTypeInt8    = (sizeof(  int8_t)     | (1 << 8)),
  gputtDataTypeUInt8   = (sizeof( uint8_t)     | (2 << 8)),
  gputtDataTypeInt8x4  = (sizeof(  int8_t) * 4 | (4 << 8)),
  gputtDataTypeUInt8x4 = (sizeof( uint8_t) * 4 | (5 << 8)),
} gputtDataType;

//
// Create plan
//
// Parameters
// handle            = Returned handle to gpuTT plan
// rank              = Rank of the tensor
// dim[rank]         = Dimensions of the tensor
// permutation[rank] = Transpose permutation
// sizeofType        = Size of the elements of the tensor in bytes (=2, 4 or 8)
// stream            = CUDA stream (0 if no stream is used)
// method            = Transpose method to use (will be chosen based on
// heuristic, if Unknown - default)
//
//
// Returns
// Success/unsuccess code
//
gputtResult GPUTT_API
gputtPlan(gputtHandle *handle, int rank, const int *dim, const int *permutation,
          gputtDataType dtype, gputtStream stream,
          gputtTransposeMethod method = gputtTransposeMethodUnknown);

//
// Create plan and choose implementation by measuring performance
//
// Parameters
// handle            = Returned handle to gpuTT plan
// rank              = Rank of the tensor
// dim[rank]         = Dimensions of the tensor
// permutation[rank] = Transpose permutation
// sizeofType        = Type of the tensor elements
// stream            = CUDA stream (0 if no stream is used)
// idata             = Input data size product(dim)
// odata             = Output data size product(dim)
//
// Returns
// Success/unsuccess code
//
gputtResult GPUTT_API gputtPlanMeasure(gputtHandle *handle, int rank,
                                       const int *dim, const int *permutation,
                                       gputtDataType dtype, gputtStream stream,
                                       const void *idata, void *odata,
                                       const void *alpha = NULL,
                                       const void *beta = NULL);

//
// Destroy plan
//
// Parameters
// handle            = Handle to the gpuTT plan
//
// Returns
// Success/unsuccess code
//
gputtResult GPUTT_API gputtDestroy(gputtHandle handle);

//
// Execute plan out-of-place; performs a tensor transposition of the form \f[
// \mathcal{B}_{\pi(i_0,i_1,...,i_{d-1})} \gets \alpha *
// \mathcal{A}_{i_0,i_1,...,i_{d-1}} + \beta *
// \mathcal{B}_{\pi(i_0,i_1,...,i_{d-1})}, \f]
//
// Parameters
// handle            = Handle to the gpuTT plan
// idata             = Input data size product(dim)
// odata             = Output data size product(dim)
// alpha             = scalar for input
// beta              = scalar for output
//
// Returns
// Success/unsuccess code
//
gputtResult GPUTT_API gputtExecute(gputtHandle handle, const void *idata,
                                   void *odata, const void *alpha = NULL,
                                   const void *beta = NULL);

//
// Get method used for a plan
//
// Parameters
// handle            = Returned handle to gpuTT plan
// method            = Returned method of gpuTT plan
// 
// Returns
// Success/unsuccess code
//
gputtResult GPUTT_API gputtPlanMethod(gputtHandle handle,
                                      gputtTransposeMethod *method);

#endif // GPUTT_H
