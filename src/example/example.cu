#include <gputt.h>
#include <gputt_runtime.h>

#include <vector>

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

  std::vector<double> idata(dim[0] * dim[1] * dim[2] * dim[3]);
  std::generate(idata.begin(), idata.end(), rand);
  std::vector<double> odata(idata.size());

  double* idataGPU;
  gpuMalloc(&idataGPU, idata.size());
  gpuMemcpy(idataGPU, idata.data(), idata.size() * sizeof(idata[0]), gpuMemcpyHostToDevice);

  double* odataGPU;
  gpuMalloc(&odataGPU, odata.size());

  // Option 1: Create plan on NULL stream and choose implementation based on heuristics
  gputtHandle plan;
  gputtCheck(gputtPlan(&plan, 4, dim, permutation, sizeof(idata[0]), 0));

  // Option 2: Create plan on NULL stream and choose implementation based on performance measurements
  // gputtCheck(gputtPlanMeasure(&plan, 4, dim, permutation, sizeof(idata[0]), 0, idata, odata));

  // Execute plan
  gputtCheck(gputtExecute(plan, idataGPU, odataGPU));

  gpuMemcpy(odata.data(), odataGPU, odata.size() * sizeof(odata[0]), gpuMemcpyDeviceToHost);

  // Destroy plan
  gputtCheck(gputtDestroy(plan));

  gpuFree(idataGPU);
  gpuFree(odataGPU);

  return 0;
}

