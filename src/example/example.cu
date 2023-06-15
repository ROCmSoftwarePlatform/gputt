#include <gputt.h>
#include <gputt_runtime.h>

#include <vector>

//
// Error checking wrapper for gpuTT and vendor API.
//

#define GPUTT_ERR_CHECK(stmt) do { \
  gputtResult err = stmt; \
  if (err != GPUTT_SUCCESS) { \
    fprintf(stderr, "%s in file %s, function %s\n", #stmt,__FILE__,__FUNCTION__); \
    exit(-1); \
  } \
} while(0)

#define GPU_ERR_CHECK(x) do { \
  gpuError_t err = x; \
  if (err != gpuSuccess) { \
    fprintf(stderr, "Error \"%s\" at %s :%d \n" , gpuGetErrorString(err), __FILE__ , __LINE__); \
    exit(-1); \
  } \
} while (0)

int main()
{
  // Four dimensional tensor
  // Transpose (31, 549, 2, 3) -> (3, 31, 2, 549)
  int dim[4] = {31, 549, 2, 3};
  int permutation[4] = {3, 0, 2, 1};

  std::vector<double> idata(dim[0] * dim[1] * dim[2] * dim[3]);
  std::generate(idata.begin(), idata.end(), rand);
  std::vector<double> odata(idata.size());

  double* idataGPU;
  GPU_ERR_CHECK(gpuMalloc(&idataGPU, idata.size() * sizeof(idata[0])));
  GPU_ERR_CHECK(gpuMemcpy(idataGPU, idata.data(), idata.size() * sizeof(idata[0]), gpuMemcpyHostToDevice));

  double* odataGPU;
  GPU_ERR_CHECK(gpuMalloc(&odataGPU, odata.size() * sizeof(odata[0])));

  // Option 1: Create plan on NULL stream and choose implementation based on heuristics
  gputtHandle plan;
  GPUTT_ERR_CHECK(gputtPlan(&plan, 4, dim, permutation, sizeof(idata[0]), 0));

  // Option 2: Create plan on NULL stream and choose implementation based on performance measurements
  // GPUTT_ERR_CHECK(gputtPlanMeasure(&plan, 4, dim, permutation, sizeof(idata[0]), 0, idata, odata));

  // Execute plan
  GPUTT_ERR_CHECK(gputtExecute(plan, idataGPU, odataGPU));

  GPU_ERR_CHECK(gpuDeviceSynchronize());

  GPU_ERR_CHECK(gpuMemcpy(odata.data(), odataGPU, odata.size() * sizeof(odata[0]), gpuMemcpyDeviceToHost));

  // Destroy plan
  GPUTT_ERR_CHECK(gputtDestroy(plan));

  GPU_ERR_CHECK(gpuFree(idataGPU));
  GPU_ERR_CHECK(gpuFree(odataGPU));

  // Perform the same pemutation on the CPU.
  std::vector<double> odata2(odata.size());
  for (int d0 = 0; d0 < dim[0]; d0++)
    for (int d1 = 0; d1 < dim[1]; d1++)
      for (int d2 = 0; d2 < dim[2]; d2++)
        for (int d3 = 0; d3 < dim[3]; d3++)
        {
          auto in = idata[d0 * dim[1] * dim[2] * dim[3] + d1 * dim[2] * dim[3] + d2 * dim[3] + d3];
          auto& out2 = odata2[d3 * dim[0] * dim[2] * dim[1] + d0 * dim[2] * dim[1] + d2 * dim[1] + d1];

          out2 = in;

	  // Compare with gpuTT's output element.
          auto out = odata[d3 * dim[0] * dim[2] * dim[1] + d0 * dim[2] * dim[1] + d2 * dim[1] + d1];
	  if (out != out2)
	  {
            fprintf(stderr, "Output elements mismatch: %f != %f\n", out, out2);
	    exit(-1);
	  }
        }

  return 0;
}

