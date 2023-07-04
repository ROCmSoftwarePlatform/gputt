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
#include <list>
#include <unordered_map>
#include "gputtUtils.h"
#include "gputtplan.h"
#include "gputtkernel.h"
#include "gputtTimer.h"
#include "gputt.h"
#include <atomic>
#include <mutex>
#include <cstdlib>
// #include <chrono>

// Hash table to store the plans
static std::unordered_map<gputtHandle, gputtPlan_t* > planStorage;
static std::mutex planStorageMutex;

// Current handle
static std::atomic<gputtHandle> curHandle(0);

// Table of devices that have been initialized
static std::unordered_map<int, gpuDeviceProp_t> deviceProps;
static std::mutex devicePropsMutex;

// Checks prepares device if it's not ready yet and returns device properties
// Also sets shared memory configuration
void getDeviceProp(int& deviceID, gpuDeviceProp_t &prop) {
  gpuCheck(gpuGetDevice(&deviceID));

  // need to lock this function	
  std::lock_guard<std::mutex> lock(devicePropsMutex);

  auto it = deviceProps.find(deviceID);
  if (it == deviceProps.end()) {
    // Get device properties and store it for later use
    gpuCheck(gpuGetDeviceProperties(&prop, deviceID));
    gputtKernelSetSharedMemConfig();
    deviceProps.insert({deviceID, prop});
  } else {
    prop = it->second;
  }
}

static gputtResult gputtPlanCheckInput(int rank, const int* dim, const int* permutation, size_t sizeofType) {
  // Check sizeofType
  if (sizeofType != 2 && sizeofType != 4 && sizeofType != 8) return GPUTT_INVALID_PARAMETER;
  // Check rank
  if (rank <= 1) return GPUTT_INVALID_PARAMETER;
  // Check dim[]
  for (int i = 0; i < rank; i++) {
    if (dim[i] <= 1) return GPUTT_INVALID_PARAMETER;
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
  if (permutation_fail) return GPUTT_INVALID_PARAMETER;  

  return GPUTT_SUCCESS;
}

//
// Returns best plan according to heuristic criteria
// Returns plans.end() on invalid input or when nothing can be chosen
//
static std::list<gputtPlan_t>::const_iterator choosePlanHeuristic(
  const std::list<gputtPlan_t>& plans) {
  // Choose the "largest" plan
  auto bestIt = plans.end();
  for (auto it = plans.begin(); it != plans.end(); it++) {
    if (bestIt == plans.end() || *bestIt < *it) {
      bestIt = it;
    }
  }

  return bestIt;
}

gputtResult gputtPlan(gputtHandle* handle, int rank, const int* dim, const int* permutation, size_t sizeofType,
  gpuStream_t stream, gputtTransposeMethod method) {

#ifdef ENABLE_NVTOOLS
  gpuRangeStart("init");
#endif

  // Check that input parameters are valid
  gputtResult inpCheck = gputtPlanCheckInput(rank, dim, permutation, sizeofType);
  if (inpCheck != GPUTT_SUCCESS) return inpCheck;

  // Create new handle
  *handle = curHandle;
  curHandle++;

  // Check that the current handle is available (it better be!)
  {
    std::lock_guard<std::mutex> lock(planStorageMutex);
    if (planStorage.count(*handle) != 0) return GPUTT_INTERNAL_ERROR;
  }

  // Prepare device
  int deviceID;
  gpuDeviceProp_t prop;
  getDeviceProp(deviceID, prop);

  // Reduce ranks
  std::vector<int> redDim;
  std::vector<int> redPermutation;
  reduceRanks(rank, dim, permutation, redDim, redPermutation);

#ifdef ENABLE_NVTOOLS
    gpuRangeStop();
    gpuRangeStart("createPlans");
#endif

  // Create all supported plans for the specified problem dimensions.
  std::list<gputtPlan_t> plans;
  if (!gputtPlan_t::createPlans(rank, dim, permutation,
    redDim.size(), redDim.data(), redPermutation.data(), 
    sizeofType, deviceID, prop, plans))
    return GPUTT_INTERNAL_ERROR;

#if 0
  if (!gputtKernelDatabase(deviceID, prop)) return GPUTT_INTERNAL_ERROR;
#endif

  // Use heuristic to choose the best plan, if not already
  // provided by the user.
  gputtPlan_t bestPlan;
  if (method == gputtTransposeMethodUnknown) {
#ifdef ENABLE_NVTOOLS
    gpuRangeStop();
    gpuRangeStart("countCycles");
#endif

    // Calculate the estimated number of cycles for each plan.
    for (auto& plan : plans) {
      if (!plan.countCycles(prop, 10)) return GPUTT_INTERNAL_ERROR;
    }

    // Choose the best plan based on the number of cycles.
    auto it = choosePlanHeuristic(plans);
    if (it == plans.end()) return GPUTT_INTERNAL_ERROR;
    
    bestPlan = *it;
  }
  else {
    // One method could be a fit for a given problem multiple times,
    // with different parameters. Therefore, we use heuristic to select
    // the best plan out of multiple plans with the same method.

    // Find all plans for the chosen method.
    std::list<gputtPlan_t> plansForMethod;
    for (auto& plan : plans) {
      if (plan.tensorSplit.method == method)
        plansForMethod.emplace_back(plan);
    }
    
    if (plansForMethod.size() == 0)
      return GPUTT_UNSUPPORTED_METHOD;
    else if (plansForMethod.size() == 1)
      bestPlan = plansForMethod.front();
    else {
      // Calculate the estimated number of cycles for each plan.
      for (auto& plan : plans) {
        if (!plan.countCycles(prop, 10)) return GPUTT_INTERNAL_ERROR;
      }
      
      // Choose the best plan based on the number of cycles.
      auto it = choosePlanHeuristic(plansForMethod);
      if (it == plans.end()) return GPUTT_INTERNAL_ERROR;
    
      bestPlan = *it;      
    }
  }

#ifdef ENABLE_NVTOOLS
  gpuRangeStop();
  gpuRangeStart("rest");
#endif

#if 1
  bestPlan.print();
#endif
  // Create copy of the plan outside the list
  gputtPlan_t* plan = new gputtPlan_t();
  // NOTE: No deep copy needed here since device memory hasn't been allocated yet
  *plan = bestPlan;
  // Set device pointers to NULL in the old copy of the plan so
  // that they won't be deallocated later when the object is destroyed
  bestPlan.nullDevicePointers();

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

  return GPUTT_SUCCESS;
}

gputtResult gputtPlanMeasure(gputtHandle* handle, int rank, const int* dim, const int* permutation, size_t sizeofType,
  gpuStream_t stream, const void* idata, void* odata, const void* alpha, const void *beta) {

  // Check that input parameters are valid
  gputtResult inpCheck = gputtPlanCheckInput(rank, dim, permutation, sizeofType);
  if (inpCheck != GPUTT_SUCCESS) return inpCheck;

  if (idata == odata) return GPUTT_INVALID_PARAMETER;

  // Create new handle
  *handle = curHandle;
  curHandle++;

  // Check that the current handle is available (it better be!)
  {
    std::lock_guard<std::mutex> lock(planStorageMutex);
    if (planStorage.count(*handle) != 0) return GPUTT_INTERNAL_ERROR;
  }

  // Prepare device
  int deviceID;
  gpuDeviceProp_t prop;
  getDeviceProp(deviceID, prop);

  // Reduce ranks
  std::vector<int> redDim;
  std::vector<int> redPermutation;
  reduceRanks(rank, dim, permutation, redDim, redPermutation);

  // Create plans from reduced ranks
  std::list<gputtPlan_t> plans;
#if 0
  // if (rank != redDim.size()) {
    if (!createPlans(redDim.size(), redDim.data(), redPermutation.data(), sizeofType, prop, plans)) return GPUTT_INTERNAL_ERROR;
  // }

  // Create plans from non-reduced ranks
  // if (!createPlans(rank, dim, permutation, sizeofType, prop, plans)) return GPUTT_INTERNAL_ERROR;
#else
  if (!gputtPlan_t::createPlans(rank, dim, permutation, redDim.size(), redDim.data(), redPermutation.data(), 
    sizeofType, deviceID, prop, plans)) return GPUTT_INTERNAL_ERROR;
#endif

  // // Count cycles
  // for (auto it=plans.begin();it != plans.end();it++) {
  //   if (!it->countCycles(prop, 10)) return GPUTT_INTERNAL_ERROR;
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
    gpuCheck(gpuDeviceSynchronize());
    timer.start();
    // Execute plan
    if (!gputtKernel(*it, idata, odata, alpha, beta)) return GPUTT_INTERNAL_ERROR;
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
  if (bestPlan == plans.end()) return GPUTT_INTERNAL_ERROR;

  // bestPlan = plans.begin();

  // printMatlab(prop, plans, times);
  // findMispredictionBest(plans, times, bestPlan, bestTime);
  // bestPlan->print();

  // Create copy of the plan outside the list
  gputtPlan_t* plan = new gputtPlan_t();
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

  return GPUTT_SUCCESS;
}

void gputtDestroy_callback(gpuStream_t stream, gpuError_t status, void *userData){
  gputtPlan_t* plan = (gputtPlan_t*) userData;
  delete plan;
}

gputtResult gputtDestroy(gputtHandle handle) {
  std::lock_guard<std::mutex> lock(planStorageMutex);
  auto it = planStorage.find(handle);
  if (it == planStorage.end()) return GPUTT_INVALID_PLAN;
  // Delete instance of gputtPlan_t	 
  delete it->second;	  
  // Delete entry from plan storage	  
  planStorage.erase(it);
  return GPUTT_SUCCESS;
}

gputtResult gputtExecute(gputtHandle handle, const void* idata, void* odata, const void* alpha, const void* beta) {
  // prevent modification when find
  std::lock_guard<std::mutex> lock(planStorageMutex);
  auto it = planStorage.find(handle);
  if (it == planStorage.end()) return GPUTT_INVALID_PLAN;

  if (idata == odata) return GPUTT_INVALID_PARAMETER;

  gputtPlan_t& plan = *(it->second);

  int deviceID;
  gpuCheck(gpuGetDevice(&deviceID));
  if (deviceID != plan.deviceID) return GPUTT_INVALID_DEVICE;

  if (!gputtKernel(plan, idata, odata, alpha, beta)) return GPUTT_INTERNAL_ERROR;
  return GPUTT_SUCCESS;
}

