# gpuTT - GPU Tensor Transpose library

gpuTT is a high performance tensor transpose library for NVIDIA and AMD GPUs. It works with Kepler (SM 3.0) and above or gfx900 and above GPUs.

This code implements the following tensor transposing methods: `Trivial`, `Tiled`, `TiledCopy`, `Packed`, and `PackedSplit`. The fastest method is chosen for the given problem, either by measuring the performance, or by using a heuristic (the default).

## Building

Prerequisites:

 * CMake, 3.16+
 * C++ compiler with C++17 compitability
 * HIP or CUDA compiler
 * NVIDIA or AMD GPU (sm30 or above)

To compile gpuTT library as well as test cases and benchmarks, simply do:

```
mkdir build
cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=61 ..
make -j12
```

Note the supported GPU architectures must be specified with `CMAKE_CUDA_ARCHITECTURES` or `CMAKE_HIP_ARCHITECTURES`, otherwise the default architecture setting may result into compile errors due to the missing `__half` type support.

In order to compile with CMake < 3.24, C++ compiler must be specified explicitly:

```
cmake -DCMAKE_CXX_COMPILER=hipcc ..
```

This will create the library itself:

 * `include/gputt.h`
 * `libgputt.a`

as well as the test and benchmarks

 * `gputt_test`
 * `gputt_bench`

In order to use gpuTT, you only need the include `include/gputt.h` and the library `lib/libgputt.a` files.

## Running tests and benchmarks

Tests and benchmark executables are in the bin/ directory and they can be run without any options.
Options to the test executable lets you choose the device ID on which to run:

```
gputt_test [options]
Options:
-device gpuid : use GPU with ID gpuid
```

For the benchmark executable, we have an additional option that lets you run the benchmarks using
plans that are chosen optimally by measuring the performance of every possible implementation and
choosing the best one.

```
gputt_bench [options]
Options:
-device gpuid : use GPU with ID gpuid
-measure      : use gputtPlanMeasure (default is gputtPlan)
```

The following modern GPUs have been tested and passed the correctness test suite successfully:

* RTX 3080 LHR (GA104, CC 8.6)
* Radeon RX Vega 56 (gfx900)
* Radeon RX 7900 XTX (gfx1100)

## Performance

gpuTT was designed with performance as the main goal. Here are performance benchmarks for a random set of tensors with 200M `double` elements with ranks 2 to 7. The benchmarks were run with the measurement flag on `./gputt_bench -measure -bench 3`.

![k20x](doc/k20x_bench.png)

<!-- ![k40m](doc/bw_k40m_july1_2016.png)
 -->

<!-- ![titanx](doc/bw_titanx.png)
 -->

## Usage

gpuTT uses a "plan structure" similar to FFTW and cuFFT libraries, where the
user first creates a plan for the transpose and then executes that plan.
Here is an example code (please see a fully working example code in `src/example/example.cu`).

```c++
#include <gputt.h>

//
// Error checking wrapper for gputt
//
#define gputtCheck(stmt) do {                                 \
  gputtResult err = stmt;                            \
  if (err != GPUTT_SUCCESS) {                          \
    fprintf(stderr, "%s in file %s, function %s\n", #stmt,__FILE__,__FUNCTION__); \
    exit(1); \
  }                                                  \
} while(0)

int main() {

  // Four dimensional tensor
  // Transpose (31, 549, 2, 3) -> (3, 31, 2, 549)
  int dim[4] = {31, 549, 2, 3};
  int permutation[4] = {3, 0, 2, 1};

  .... input and output data is setup here ...
  // double* idata : size product(dim)
  // double* odata : size product(dim)

  // Option 1: Create plan on NULL stream and choose implementation based on heuristics
  gputtHandle plan;
  gputtCheck(gputtPlan(&plan, 4, dim, permutation, sizeof(double), 0));

  // Option 2: Create plan on NULL stream and choose implementation based on performance measurements
  // gputtCheck(gputtPlanMeasure(&plan, 4, dim, permutation, sizeof(double), 0, idata, odata));

  // Execute plan
  gputtCheck(gputtExecute(plan, idata, odata));

  ... do stuff with your output and deallocate data ...

  // Destroy plan
  gputtCheck(gputtDestroy(plan));

  return 0;
}
```

Input (idata) and output (odata) data are both in GPU memory and must point to different
memory areas for correct operation. That is, gpuTT only currently supports out-of-place
transposes. Note that using Option 2 to create the plan can take up some time especially
for high-rank tensors.

## gpuTT API

```c++
//
// Create plan
//
// Parameters
// handle            = Returned handle to gpuTT plan
// rank              = Rank of the tensor
// dim[rank]         = Dimensions of the tensor
// permutation[rank] = Transpose permutation
// sizeofType        = Size of the elements of the tensor in bytes (=4 or 8)
// stream            = CUDA stream (0 if no stream is used)
//
// Returns
// Success/unsuccess code
// 
gputtResult gputtPlan(gputtHandle* handle, int rank, int* dim, int* permutation, size_t sizeofType,
  cudaStream_t stream);

//
// Create plan and choose implementation by measuring performance
//
// Parameters
// handle            = Returned handle to gpuTT plan
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
gputtResult gputtPlanMeasure(gputtHandle* handle, int rank, int* dim, int* permutation, size_t sizeofType,
  cudaStream_t stream, void* idata, void* odata);
  
//
// Destroy plan
//
// Parameters
// handle            = Handle to the gpuTT plan
// 
// Returns
// Success/unsuccess code
//
gputtResult gputtDestroy(gputtHandle handle);

//
// Execute plan out-of-place
//
// Parameters
// handle            = Returned handle to gpuTT plan
// idata             = Input data size product(dim)
// odata             = Output data size product(dim)
// alpha             = scaling-factor for input
// beta              = scaling-factor for output
// 
// Returns
// Success/unsuccess code
//
gputtResult gputtExecute(gputtHandle handle, void* idata, void* odata);
```

## Known Bugs

 * Benchmarks sometime fail due to the stupid algorithm I have now to create
 random tensors with fixed volume.

## TODO

 * Make "tiled" method work with sets of ranks (where ranks in `M_m` and `M_k` remain in same order)

## Troubleshooting

1. HIP compiler might appear not found as shown below, even if it definitely exists in the system. The reason is not a PATH setting. The `check_language(HIP)` command actually finds the compiler, but runs into an issue when testing it, for example [this issue](https://github.com/RadeonOpenCompute/ROCm/issues/1843). The solution is to temporary specify the HIP compiler explicitly and handle an error, which will be displayed upon CMake testing it: `cmake -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ ..`.

```
-- Looking for a HIP compiler - NOTFOUND
```

2. The following error could be fixed by `sudo apt-get install libstdc++-12-dev`, as suggested in this issue:

```
  The HIP compiler

    "/opt/rocm/llvm/bin/clang++"

  is not able to compile a simple test program.

    /opt/rocm-5.5.0/llvm/lib/clang/16.0.0/include/__clang_hip_runtime_wrapper.h:50:10: fatal error: 'cmath' file not found
    #include <cmath>
             ^~~~~~~
```

## Credits

In memory of Antti-Pekka Hynninen.

