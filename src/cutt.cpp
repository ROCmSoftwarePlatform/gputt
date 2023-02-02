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
#include <cuda.h>
#include <list>
#include <unordered_map>
#include "CudaUtils.h"
#include "CudaMem.h"
#include "hipttplan.h"
#include "hipttkernel.h"
#include "hipttTimer.h"
#include "hiptt.h"
#include <atomic>
#include <mutex>
#include <cstdlib>
// #include <chrono>

// global Umpire allocator
#ifdef CUTT_HAS_UMPIRE
umpire::Allocator hiptt_umpire_allocator;
#endif

// Hash table to store the plans
static std::unordered_map<hipttHandle, hipttPlan_t* > planStorage;
static std::mutex planStorageMutex;

// Current handle
static std::atomic<hipttHandle> curHandle(0);

// Table of devices that have been initialized
static std::unordered_map<int, cudaDeviceProp> deviceProps;
static std::mutex devicePropsMutex;

// Checks prepares device if it's not ready yet and returns device properties
// Also sets shared memory configuration
void getDeviceProp(int& deviceID, cudaDeviceProp &prop) {
  cudaCheck(cudaGetDevice(&deviceID));

  // need to lock this function	
  std::lock_guard<std::mutex> lock(devicePropsMutex);

  auto it = deviceProps.find(deviceID);
  if (it == deviceProps.end()) {
    // Get device properties and store it for later use
    cudaCheck(cudaGetDeviceProperties(&prop, deviceID));
    hipttKernelSetSharedMemConfig();
    deviceProps.insert({deviceID, prop});
  } else {
    prop = it->second;
  }
}

static hipttResult hipttPlanCheckInput(int rank, const int* dim, const int* permutation, size_t sizeofType) {
  // Check sizeofType
  if (sizeofType != 4 && sizeofType != 8) return CUTT_INVALID_PARAMETER;
  // Check rank
  if (rank <= 1) return CUTT_INVALID_PARAMETER;
  // Check dim[]
  for (int i=0;i < rank;i++) {
    if (dim[i] <= 1) return CUTT_INVALID_PARAMETER;
  }
  // Check permutation
  bool permutation_fail = false;
  int* check = new int[rank];
  for (int i=0;i < rank;i++) check[i] = 0;
  for (int i=0;i < rank;i++) {
    if (permutation[i] < 0 || permutation[i] >= rank || check[permutation[i]]++) {
      permutation_fail = true;
      break;
    }
  }
  delete [] check;
  if (permutation_fail) return CUTT_INVALID_PARAMETER;  

  return CUTT_SUCCESS;
}

hipttResult hipttPlan(hipttHandle* handle, int rank, const int* dim, const int* permutation, size_t sizeofType,
  cudaStream_t stream) {

#ifdef ENABLE_NVTOOLS
  gpuRangeStart("init");
#endif

  // Check that input parameters are valid
  hipttResult inpCheck = hipttPlanCheckInput(rank, dim, permutation, sizeofType);
  if (inpCheck != CUTT_SUCCESS) return inpCheck;

  // Create new handle
  *handle = curHandle;
  curHandle++;

  // Check that the current handle is available (it better be!)
  {
    std::lock_guard<std::mutex> lock(planStorageMutex);
    if (planStorage.count(*handle) != 0) return CUTT_INTERNAL_ERROR;
  }

  // Prepare device
  int deviceID;
  cudaDeviceProp prop;
  getDeviceProp(deviceID, prop);

  // Reduce ranks
  std::vector<int> redDim;
  std::vector<int> redPermutation;
  reduceRanks(rank, dim, permutation, redDim, redPermutation);

  // Create plans from reduced ranks
  std::list<hipttPlan_t> plans;
  // if (rank != redDim.size()) {
  //   if (!createPlans(redDim.size(), redDim.data(), redPermutation.data(), sizeofType, prop, plans)) return CUTT_INTERNAL_ERROR;
  // }

  // // Create plans from non-reduced ranks
  // if (!createPlans(rank, dim, permutation, sizeofType, prop, plans)) return CUTT_INTERNAL_ERROR;

#if 0
  if (!hipttKernelDatabase(deviceID, prop)) return CUTT_INTERNAL_ERROR;
#endif

#ifdef ENABLE_NVTOOLS
  gpuRangeStop();
  gpuRangeStart("createPlans");
#endif

  // std::chrono::high_resolution_clock::time_point plan_start;
  // plan_start = std::chrono::high_resolution_clock::now();

  if (!hipttPlan_t::createPlans(rank, dim, permutation, redDim.size(), redDim.data(), redPermutation.data(), 
    sizeofType, deviceID, prop, plans)) return CUTT_INTERNAL_ERROR;

  // std::chrono::high_resolution_clock::time_point plan_end;
  // plan_end = std::chrono::high_resolution_clock::now();
  // double plan_duration = std::chrono::duration_cast< std::chrono::duration<double> >(plan_end - plan_start).count();
  // printf("createPlans took %lf ms\n", plan_duration*1000.0);

#ifdef ENABLE_NVTOOLS
  gpuRangeStop();
  gpuRangeStart("countCycles");
#endif

  // Count cycles
  for (auto it=plans.begin();it != plans.end();it++) {
    if (!it->countCycles(prop, 10)) return CUTT_INTERNAL_ERROR;
  }

#ifdef ENABLE_NVTOOLS
  gpuRangeStop();
  gpuRangeStart("rest");
#endif

  // Choose the plan
  std::list<hipttPlan_t>::iterator bestPlan = choosePlanHeuristic(plans);
  if (bestPlan == plans.end()) return CUTT_INTERNAL_ERROR;

  // bestPlan->print();

  // Create copy of the plan outside the list
  hipttPlan_t* plan = new hipttPlan_t();
  // NOTE: No deep copy needed here since device memory hasn't been allocated yet
  *plan = *bestPlan;
  // Set device pointers to NULL in the old copy of the plan so
  // that they won't be deallocated later when the object is destroyed
  bestPlan->nullDevicePointers();

  // Set stream
  plan->setStream(stream);

  // Activate plan
  plan->activate();

  // Insert plan into storage
  {
    std::lock_guard<std::mutex> lock(planStorageMutex);
    planStorage.insert( {*handle, plan} );
  }

#ifdef ENABLE_NVTOOLS
  gpuRangeStop();
#endif

  return CUTT_SUCCESS;
}

hipttResult hipttPlanMeasure(hipttHandle* handle, int rank, const int* dim, const int* permutation, size_t sizeofType,
  cudaStream_t stream, const void* idata, void* odata, const void* alpha, const void *beta) {

  // Check that input parameters are valid
  hipttResult inpCheck = hipttPlanCheckInput(rank, dim, permutation, sizeofType);
  if (inpCheck != CUTT_SUCCESS) return inpCheck;

  if (idata == odata) return CUTT_INVALID_PARAMETER;

  // Create new handle
  *handle = curHandle;
  curHandle++;

  // Check that the current handle is available (it better be!)
  {
    std::lock_guard<std::mutex> lock(planStorageMutex);
    if (planStorage.count(*handle) != 0) return CUTT_INTERNAL_ERROR;
  }

  // Prepare device
  int deviceID;
  cudaDeviceProp prop;
  getDeviceProp(deviceID, prop);

  // Reduce ranks
  std::vector<int> redDim;
  std::vector<int> redPermutation;
  reduceRanks(rank, dim, permutation, redDim, redPermutation);

  // Create plans from reduced ranks
  std::list<hipttPlan_t> plans;
#if 0
  // if (rank != redDim.size()) {
    if (!createPlans(redDim.size(), redDim.data(), redPermutation.data(), sizeofType, prop, plans)) return CUTT_INTERNAL_ERROR;
  // }

  // Create plans from non-reduced ranks
  // if (!createPlans(rank, dim, permutation, sizeofType, prop, plans)) return CUTT_INTERNAL_ERROR;
#else
  if (!hipttPlan_t::createPlans(rank, dim, permutation, redDim.size(), redDim.data(), redPermutation.data(), 
    sizeofType, deviceID, prop, plans)) return CUTT_INTERNAL_ERROR;
#endif

  // // Count cycles
  // for (auto it=plans.begin();it != plans.end();it++) {
  //   if (!it->countCycles(prop, 10)) return CUTT_INTERNAL_ERROR;
  // }

  // // Count the number of elements
  size_t numBytes = sizeofType;
  for (int i=0;i < rank;i++) numBytes *= dim[i];

  // Choose the plan
  double bestTime = 1.0e40;
  auto bestPlan = plans.end();
  Timer timer;
  std::vector<double> times;
  for (auto it=plans.begin();it != plans.end();it++) {
    // Activate plan
    it->activate();
    // Clear output data to invalidate caches
    set_device_array<char>((char *)odata, -1, numBytes);
    cudaCheck(cudaDeviceSynchronize());
    timer.start();
    // Execute plan
    if (!hipttKernel(*it, idata, odata, alpha, beta)) return CUTT_INTERNAL_ERROR;
    timer.stop();
    double curTime = timer.seconds();
    // it->print();
    // printf("curTime %1.2lf\n", curTime*1000.0);
    times.push_back(curTime);
    if (curTime < bestTime) {
      bestTime = curTime;
      bestPlan = it;
    }
  }
  if (bestPlan == plans.end()) return CUTT_INTERNAL_ERROR;

  // bestPlan = plans.begin();

  // printMatlab(prop, plans, times);
  // findMispredictionBest(plans, times, bestPlan, bestTime);
  // bestPlan->print();

  // Create copy of the plan outside the list
  hipttPlan_t* plan = new hipttPlan_t();
  *plan = *bestPlan;
  // Set device pointers to NULL in the old copy of the plan so
  // that they won't be deallocated later when the object is destroyed
  bestPlan->nullDevicePointers();

  // Set stream
  plan->setStream(stream);

  // Activate plan
  plan->activate();

  // Insert plan into storage
  {
    std::lock_guard<std::mutex> lock(planStorageMutex);
    planStorage.insert( {*handle, plan} );
  }

  return CUTT_SUCCESS;
}

void CUDART_CB hipttDestroy_callback(cudaStream_t stream, cudaError_t status, void *userData){
  hipttPlan_t* plan = (hipttPlan_t*) userData;
  delete plan;
}

hipttResult hipttDestroy(hipttHandle handle) {
  std::lock_guard<std::mutex> lock(planStorageMutex);
  auto it = planStorage.find(handle);
  if (it == planStorage.end()) return CUTT_INVALID_PLAN;
#ifdef CUTT_HAS_UMPIRE
  // get the pointer hipttPlan_t
  hipttPlan_t* plan = it->second;
  cudaStream_t stream = plan->stream;
  // Delete entry from plan storage
  planStorage.erase(it);
  // register callback to deallocate plan
  cudaStreamAddCallback(stream, hipttDestroy_callback, plan, 0);
#else
  // Delete instance of hipttPlan_t	 
  delete it->second;	  
  // Delete entry from plan storage	  
  planStorage.erase(it);
#endif
  return CUTT_SUCCESS;
}

hipttResult hipttExecute(hipttHandle handle, const void* idata, void* odata, const void* alpha, const void* beta) {
  // prevent modification when find
  std::lock_guard<std::mutex> lock(planStorageMutex);
  auto it = planStorage.find(handle);
  if (it == planStorage.end()) return CUTT_INVALID_PLAN;

  if (idata == odata) return CUTT_INVALID_PARAMETER;

  hipttPlan_t& plan = *(it->second);

  int deviceID;
  cudaCheck(cudaGetDevice(&deviceID));
  if (deviceID != plan.deviceID) return CUTT_INVALID_DEVICE;

  if (!hipttKernel(plan, idata, odata, alpha, beta)) return CUTT_INTERNAL_ERROR;
  return CUTT_SUCCESS;
}

void hipttInitialize() {
#ifdef CUTT_HAS_UMPIRE
  const char* alloc_env_var = std::getenv("CUTT_USES_THIS_UMPIRE_ALLOCATOR");
#define __CUTT_STRINGIZE(x) #x
#define __CUTT_XSTRINGIZE(x) __CUTT_STRINGIZE(x)
  const char* alloc_cstr = alloc_env_var ? alloc_env_var : __CUTT_XSTRINGIZE(CUTT_USES_THIS_UMPIRE_ALLOCATOR);
  hiptt_umpire_allocator = umpire::ResourceManager::getInstance().getAllocator(alloc_cstr);
#endif
}

void hipttFinalize() {
}
