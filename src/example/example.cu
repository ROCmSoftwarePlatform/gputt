#include <gputt.h>
#include <gputt_runtime.h>

#include <vector>

using T = uint64_t;

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
  // Transpose (7, 11, 13) -> (13, 7, 11)
  int dim[3] = {7, 11, 13};
  int permutation[4] = {2, 0, 1};
  int odim[3] = {13, 7, 11};

  std::vector<T> idata(dim[0] * dim[1] * dim[2]);
  //std::generate(idata.begin(), idata.end(), rand);
  for (int i = 0; i < idata.size(); i++)
	  idata[i] = i;
  std::vector<T> odata(idata.size());

  T* idataGPU;
  GPU_ERR_CHECK(gpuMalloc(&idataGPU, idata.size() * sizeof(idata[0])));
  GPU_ERR_CHECK(gpuMemcpy(idataGPU, idata.data(), idata.size() * sizeof(idata[0]), gpuMemcpyHostToDevice));

  T* odataGPU;
  GPU_ERR_CHECK(gpuMalloc(&odataGPU, odata.size() * sizeof(odata[0])));

  // Option 1: Create plan on NULL stream and choose implementation based on heuristics
  gputtHandle plan;
  GPUTT_ERR_CHECK(gputtPlan(&plan, 3, dim, permutation, sizeof(idata[0]), 0));

  // Option 2: Create plan on NULL stream and choose implementation based on performance measurements
  // GPUTT_ERR_CHECK(gputtPlanMeasure(&plan, 3, dim, permutation, sizeof(idata[0]), 0, idata, odata));

  // Execute plan
  GPUTT_ERR_CHECK(gputtExecute(plan, idataGPU, odataGPU));

  GPU_ERR_CHECK(gpuDeviceSynchronize());

  GPU_ERR_CHECK(gpuMemcpy(odata.data(), odataGPU, odata.size() * sizeof(odata[0]), gpuMemcpyDeviceToHost));

  // Destroy plan
  GPUTT_ERR_CHECK(gputtDestroy(plan));

  GPU_ERR_CHECK(gpuFree(idataGPU));
  GPU_ERR_CHECK(gpuFree(odataGPU));

  // Perform the same permutation on the CPU.
  std::vector<T> odata2(odata.size());
  for (int d0 = 0; d0 < dim[0]; d0++)
    for (int d1 = 0; d1 < dim[1]; d1++)
      for (int d2 = 0; d2 < dim[2]; d2++)
        {
          auto in = idata[d2 * dim[1] * dim[0] + d1 * dim[0] + d0];
          auto& out2 = odata2[d1 * dim[0] * dim[2] + d0 * odim[2] + d2];
	  // d1 * 7 * 13 + d0 * 13 + d2
          // int dim[3] = {7, 11, 13};
          // int permutation[4] = {2, 0, 1};

          out2 = in;

          if ((d0 == 2) && (d1 == 3) && (d2 == 5))
          {
	    for (int i = 0; i < odata.size(); i++)
            {
              if (out2 != odata[i]) continue;
              
	      printf("found equal to [%d][%d][%d] at %d!\n", d0, d1, d2, i);

              // Find all x, y, z, w: d0 * x + d1 * y + d2 * z + d3 * w == i
              for (int x = 1, xe = odata.size(); x < xe; x++)
                for (int y = 1, ye = odata.size(); y < ye; y++)
                  for (int z = 1, ze = odata.size(); z < ze; z++)
                    {
                      if ((x != 1) && (y != 1) && (z != 1)) continue;
		      if ((x == y) || (y == z) || (x == z)) continue;
                      //if (x * y * z > odata.size()) continue;

                      int m = std::max(x, std::max(y, z));
		      if ((m % x != 0) || (m % y != 0) || (m % z != 0)) continue; 

		      //d0 * 13 + d1 * 13 * 7 + d2 * 1
		      if ((d0 * x + d1 * y + d2 * z) == i)
                        printf("Possible: %d %d %d\n", x, y, z);
                    }
            }
          }
#if 1
	  // Compare with gpuTT's output element.
          auto out = odata[d1 * dim[0] * dim[2] + d0 * dim[2] + d2];
	  if (out != out2)
	  {
            fprintf(stderr, "Output elements mismatch at [%d][%d][%d]: %lu != %lu\n", d0, d1, d2, out, out2);
	    exit(-1);
	  }
#endif
	}

  return 0;
}

