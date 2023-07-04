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

#include "TensorTester.h"
#include "gputtUtils.h"

#ifdef __HIPCC__
#define __shfl_sync(mask, ...) __shfl(__VA_ARGS__)
#endif

// Fill tensor with test a data: simple growing index.
__global__ void setTensorCheckPatternKernel(unsigned int *data,
                                            unsigned int ndata) {
  for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; i < ndata;
       i += blockDim.x * gridDim.x)
    data[i] = i;
}

// Check the transposed kernel elements against the reference.
// TODO More detail
template <typename T>
__global__ void checkTransposeKernel(T *data, unsigned int ndata, int rank,
                                     TensorConv *glTensorConv,
                                     TensorError_t *glError, int *glFail) {
  extern __shared__ unsigned int shPos[];

  // Each warp lane takes care of one dimension
  // (therefore, this algo is limited by warpSize).
  const int warpLane = threadIdx.x & (warpSize - 1);
  TensorConv tc;
  if (warpLane < rank)
    tc = glTensorConv[warpLane];

  TensorError_t error{};
  error.pos = 0xffffffff;

  for (int base = blockIdx.x * blockDim.x; base < ndata;
       base += blockDim.x * gridDim.x) {
    int i = base + threadIdx.x;
    T dataValT = (i < ndata) ? data[i] : -1;

    // Make a sum of all values (TODO what values?)
    int refVal = 0;
    for (int j = 0; j < rank; j++)
      refVal += ((i / __shfl_sync(0xffffffff, tc.c, j)) %
                 __shfl_sync(0xffffffff, tc.d, j)) *
                __shfl_sync(0xffffffff, tc.ct, j);

    // TODO Why dataValT is masked with a value smaller than the type?
    // TODO Why divisions by type size and by 4?
    int dataVal = (dataValT & 0xffffffff) / (sizeof(T) / 4);

    // Skip index with no actual data.
    if (i >= ndata)
      continue;

    // Record error in the smallest index.
    if (i >= error.pos)
      continue;

    // If a mismatch, record an error.
    if (refVal != dataVal) {
      error.pos = i;
      error.refVal = refVal;
      error.dataVal = dataVal;
    }
  }

  // Gather error status from all threads of block, so that the
  // minimum error.pos shall arrive into shPos[0] (or 0xffffffff in case of no
  // error).
  shPos[threadIdx.x] = threadIdx.x;
  __syncthreads();
  for (int d = 1; d < blockDim.x; d *= 2) {
    int t = threadIdx.x + d;

    // TODO This is only needed if block size is not divisible
    // by size of warp, which is very unlikely the case. Remove?
    unsigned int posval = (t < blockDim.x) ? shPos[t] : 0xffffffff;
    __syncthreads();

    shPos[threadIdx.x] = min(posval, shPos[threadIdx.x]);
    __syncthreads();
  }

  // If there is at least one error, the details will be
  // saved by a thread, which holds error data with the
  // corresponding data index.
  if (shPos[0] != 0xffffffff && shPos[0] == error.pos) {
    // Save error details in the global memory.
    glError[blockIdx.x] = error;

    // Set the global failure flag as well.
    *glFail = 1;
  }
}

TensorTester::TensorTester() : maxRank(32), maxNumblock(256) {
  h_tensorConv = new TensorConv[maxRank];
  h_error = new TensorError_t[maxNumblock];
  gpuCheck(gpuMalloc(&d_tensorConv, sizeof(TensorConv) * maxRank));
  gpuCheck(gpuMalloc(&d_error, sizeof(TensorError_t) * maxNumblock));
  gpuCheck(gpuMalloc(&d_fail, sizeof(int)));
}

TensorTester::~TensorTester() {
  delete[] h_tensorConv;
  delete[] h_error;
  gpuCheck(gpuFree(d_tensorConv));
  gpuCheck(gpuFree(d_error));
  gpuCheck(gpuFree(d_fail));
}

void TensorTester::setTensorCheckPattern(unsigned int *data,
                                         unsigned int ndata) {
  int numthread = 512;
  int numblock = min(65535, (ndata - 1) / numthread + 1);
  setTensorCheckPatternKernel<<<numblock, numthread>>>(data, ndata);
  gpuCheck(gpuGetLastError());
}

// Calculates tensor conversion constants. Returns total volume of tensor.
int TensorTester::calcTensorConv(const int rank, const int *dim,
                                 const int *permutation,
                                 TensorConv *tensorConv) {

  int vol = dim[0];

  tensorConv[permutation[0]].c = 1;
  tensorConv[0].ct = 1;
  tensorConv[0].d = dim[0];
  for (int i = 1; i < rank; i++) {
    vol *= dim[i];

    tensorConv[permutation[i]].c =
        tensorConv[permutation[i - 1]].c * dim[permutation[i - 1]];

    tensorConv[i].d = dim[i];
    tensorConv[i].ct = tensorConv[i - 1].ct * dim[i - 1];
  }

  return vol;
}

template <typename T>
bool TensorTester::checkTranspose(int rank, int *dim, int *permutation,
                                  T *data) {

  if (rank > 32) {
    return false;
  }

  int ndata = calcTensorConv(rank, dim, permutation, h_tensorConv);
  copy_HtoD<TensorConv>(h_tensorConv, d_tensorConv, rank);

  // printf("tensorConv\n");
  // for (int i=0;i < rank;i++) {
  //   printf("%d %d %d\n", h_tensorConv[i].c, h_tensorConv[i].d,
  //   h_tensorConv[i].ct);
  // }

  set_device_array<TensorError_t>(d_error, 0, maxNumblock);
  set_device_array<int>(d_fail, 0, 1);

  // Compute grid for a data size with padding
  int numthread = 512;
  int numblock = min(maxNumblock, (ndata - 1) / numthread + 1);

  //
  int szshmem = numthread * sizeof(unsigned int);
  checkTransposeKernel<<<numblock, numthread, szshmem>>>(
      data, ndata, rank, d_tensorConv, d_error, d_fail);
  gpuCheck(gpuGetLastError());

  // Reset the error status flags
  int h_fail;
  copy_DtoH<int>(d_fail, &h_fail, 1);
  gpuCheck(gpuDeviceSynchronize());

  if (h_fail) {
    copy_DtoH_sync<TensorError_t>(d_error, h_error, maxNumblock);
    TensorError_t error;
    error.pos = 0x0fffffff;
    for (int i = 0; i < numblock; i++) {
      // printf("%d %d %d\n", error.pos, error.refVal, error.dataVal);
      if (h_error[i].refVal != h_error[i].dataVal &&
          error.pos > h_error[i].pos) {
        error = h_error[i];
      }
    }
    printf("TensorTester::checkTranspose FAIL at %d ref %d data %d\n",
           error.pos, error.refVal, error.dataVal);
    return false;
  }

  return true;
}

// Explicit instances
template bool TensorTester::checkTranspose<int>(int rank, int *dim,
                                                int *permutation, int *data);
template bool TensorTester::checkTranspose<long long int>(int rank, int *dim,
                                                          int *permutation,
                                                          long long int *data);
