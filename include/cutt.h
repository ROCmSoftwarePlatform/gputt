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
#ifndef CUTT_H
#define CUTT_H

#include <cuda_runtime.h> // cudaStream_t

#ifdef _WIN32
#ifdef cutt_EXPORTS
#define CUTT_API __declspec(dllexport)
#else
#define CUTT_API __declspec(dllimport)
#endif
#else // _WIN32
#define CUTT_API
#endif // _WIN32

// Handle type that is used to store and access cutt plans
typedef unsigned int cuttHandle;

// Return value
typedef enum CUTT_API cuttResult_t {
  CUTT_SUCCESS,            // Success
  CUTT_INVALID_PLAN,       // Invalid plan handle
  CUTT_INVALID_PARAMETER,  // Invalid input parameter
  CUTT_INVALID_DEVICE,     // Execution tried on device different than where plan was created
  CUTT_INTERNAL_ERROR,     // Internal error
  CUTT_UNDEFINED_ERROR,    // Undefined error
} cuttResult;

// Initializes cuTT
//
// This is only needed for the Umpire allocator's lifetime management:
// - if CUTT_HAS_UMPIRE is defined, will grab Umpire's allocator;
// - otherwise this is a no-op
void CUTT_API cuttInitialize();

// Finalizes cuTT
//
// This is currently a no-op
void CUTT_API cuttFinalize();

//
// Create plan
//
// Parameters
// handle            = Returned handle to cuTT plan
// rank              = Rank of the tensor
// dim[rank]         = Dimensions of the tensor
// permutation[rank] = Transpose permutation
// sizeofType        = Size of the elements of the tensor in bytes (=4 or 8)
// stream            = CUDA stream (0 if no stream is used)
//
// Returns
// Success/unsuccess code
// 
cuttResult CUTT_API cuttPlan(cuttHandle* handle, int rank, const int* dim, const int* permutation, size_t sizeofType,
  cudaStream_t stream);

//
// Create plan and choose implementation by measuring performance
//
// Parameters
// handle            = Returned handle to cuTT plan
// rank              = Rank of the tensor
// dim[rank]         = Dimensions of the tensor
// permutation[rank] = Transpose permutation
// sizeofType        = Size of the elements of the tensor in bytes (=4 or 8)
// stream            = CUDA stream (0 if no stream is used)
// idata             = Input data size product(dim)
// odata             = Output data size product(dim)
//
// Returns
// Success/unsuccess code
// 
cuttResult CUTT_API cuttPlanMeasure(cuttHandle* handle, int rank, const int* dim, const int* permutation, size_t sizeofType,
  cudaStream_t stream, const void* idata, void* odata, const void* alpha = NULL, const void* beta = NULL);

//
// Destroy plan
//
// Parameters
// handle            = Handle to the cuTT plan
// 
// Returns
// Success/unsuccess code
//
cuttResult CUTT_API cuttDestroy(cuttHandle handle);

//
// Execute plan out-of-place; performs a tensor transposition of the form \f[ \mathcal{B}_{\pi(i_0,i_1,...,i_{d-1})} \gets \alpha * \mathcal{A}_{i_0,i_1,...,i_{d-1}} + \beta * \mathcal{B}_{\pi(i_0,i_1,...,i_{d-1})}, \f]
//
// Parameters
// handle            = Returned handle to cuTT plan
// idata             = Input data size product(dim)
// odata             = Output data size product(dim)
// alpha             = scalar for input
// beta              = scalar for output
// 
// Returns
// Success/unsuccess code
//
cuttResult CUTT_API cuttExecute(cuttHandle handle, const void* idata, void* odata, const void* alpha = NULL, const void* beta = NULL);

#endif // CUTT_H

