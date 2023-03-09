# hipTT - CUDA Tensor Transpose

hipTT is a high performance tensor transpose library for NVIDIA GPUs. It works with Kepler (SM 3.0) and above GPUs.

This code implements the following tensor transposing methods: `Trivial`, `Tiled`, `TiledCopy`, `Packed`, and `PackedSplit`. The fastest method is chosen for the given problem, either by measuring the performance, or by using a heuristic (the default).

## Building

Prerequisites:

 * CMake, 3.16+
 * C++ compiler with C++17 compitability
 * HIP or CUDA compiler
 * NVIDIA or AMD GPU (sm30 or above)

To compile hipTT library as well as test cases and benchmarks, simply do:

```
mkdir build
cd build
cmake ..
make -j12
```

In order to compile with CMake < 3.24, C++ compiler must be specified explicitly:

```
cmake -DCMAKE_CXX_COMPILER=hipcc ..
```

This will create the library itself:

 * `include/hiptt.h`
 * `libhiptt.a`

as well as the test and benchmarks

 * `hiptt_test`
 * `hiptt_bench`

In order to use hipTT, you only need the include `include/hiptt.h` and the library `lib/libhiptt.a` files.

## Running tests and benchmarks

Tests and benchmark executables are in the bin/ directory and they can be run without any options.
Options to the test executable lets you choose the device ID on which to run:

```
hiptt_test [options]
Options:
-device gpuid : use GPU with ID gpuid
```

For the benchmark executable, we have an additional option that lets you run the benchmarks using
plans that are chosen optimally by measuring the performance of every possible implementation and
choosing the best one.

```
hiptt_bench [options]
Options:
-device gpuid : use GPU with ID gpuid
-measure      : use hipttPlanMeasure (default is hipttPlan)
```

## Performance

hipTT was designed with performance as the main goal. Here are performance benchmarks for a random set of tensors with 200M `double` elements with ranks 2 to 7. The benchmarks were run with the measurement flag on `./hiptt_bench -measure -bench 3`.

![k20x](doc/k20x_bench.png)

<!-- ![k40m](doc/bw_k40m_july1_2016.png)
 -->

<!-- ![titanx](doc/bw_titanx.png)
 -->

## Usage

hipTT uses a "plan structure" similar to FFTW and cuFFT libraries, where the
user first creates a plan for the transpose and then executes that plan.
Here is an example code.

```c++
#include <hiptt.h>

//
// Error checking wrapper for hiptt
//
#define hipttCheck(stmt) do {                                 \
  hipttResult err = stmt;                            \
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
  hipttHandle plan;
  hipttCheck(hipttPlan(&plan, 4, dim, permutation, sizeof(double), 0));

  // Option 2: Create plan on NULL stream and choose implementation based on performance measurements
  // hipttCheck(hipttPlanMeasure(&plan, 4, dim, permutation, sizeof(double), 0, idata, odata));

  // Execute plan
  hipttCheck(hipttExecute(plan, idata, odata));

  ... do stuff with your output and deallocate data ...

  // Destroy plan
  hipttCheck(hipttDestroy(plan));

  return 0;
}
```

Input (idata) and output (odata) data are both in GPU memory and must point to different
memory areas for correct operation. That is, hipTT only currently supports out-of-place
transposes. Note that using Option 2 to create the plan can take up some time especially
for high-rank tensors.

## hipTT API

```c++
//
// Create plan
//
// Parameters
// handle            = Returned handle to hipTT plan
// rank              = Rank of the tensor
// dim[rank]         = Dimensions of the tensor
// permutation[rank] = Transpose permutation
// sizeofType        = Size of the elements of the tensor in bytes (=4 or 8)
// stream            = CUDA stream (0 if no stream is used)
//
// Returns
// Success/unsuccess code
// 
hipttResult hipttPlan(hipttHandle* handle, int rank, int* dim, int* permutation, size_t sizeofType,
  cudaStream_t stream);

//
// Create plan and choose implementation by measuring performance
//
// Parameters
// handle            = Returned handle to hipTT plan
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
hipttResult hipttPlanMeasure(hipttHandle* handle, int rank, int* dim, int* permutation, size_t sizeofType,
  cudaStream_t stream, void* idata, void* odata);
  
//
// Destroy plan
//
// Parameters
// handle            = Handle to the hipTT plan
// 
// Returns
// Success/unsuccess code
//
hipttResult hipttDestroy(hipttHandle handle);

//
// Execute plan out-of-place
//
// Parameters
// handle            = Returned handle to hipTT plan
// idata             = Input data size product(dim)
// odata             = Output data size product(dim)
// alpha             = scaling-factor for input
// beta              = scaling-factor for output
// 
// Returns
// Success/unsuccess code
//
hipttResult hipttExecute(hipttHandle handle, void* idata, void* odata);
```

## Known Bugs

 * Benchmarks sometime fail due to the stupid algorithm I have now to create
 random tensors with fixed volume.

## TODO

 * Make "tiled" method work with sets of ranks (where ranks in `M_m` and `M_k` remain in same order)

## Credits

In memory of Antti-Pekka Hynninen.

