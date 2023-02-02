import hiptt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

if __name__ == '__main__':
    # Four dimensional tensor
    # Transpose (31, 549, 2, 3) -> (3, 31, 2, 549)
    dim = [ 31, 549, 2, 3 ]
    permutation = [ 3, 0, 2, 1 ]

    # .... input and output data is setup here ...
    idata = np.random.randn(np.prod(dim)).astype(np.float64).reshape(dim)
    idata_gpu = gpuarray.to_gpu(idata)
    odata_gpu = gpuarray.empty(dim, dtype=np.float64)

    option = 1

    if option == 1:
        # Option 1: Create plan on NULL stream and choose implementation based on heuristics
        plan = hiptt.hipTT(4, dim, permutation, None)
    else:
        # Option 2: Create plan on NULL stream and choose implementation based on performance measurements
        plan = hiptt.hipTT(4, dim, permutation, None, idata_gpu, odata_gpu)

    # Execute plan
    result = plan.execute(idata_gpu, odata_gpu)

    # ... do stuff with your output and deallocate data ...
    odata = odata_gpu.get()

    print("idata.avg() = {}, odata.avg() = {}".format(np.average(idata), np.average(odata)))

    # Note: make sure to delete the plan before the CUDA driver is deinitialized, as the
    # plan contains internal GPU memory buffers, that shall otherwise fail to deallocate.
    del plan

