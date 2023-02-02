
#include "HipMem.h"

//----------------------------------------------------------------------------------------
//
// Allocate gpu memory
// pp = memory pointer
// len = length of the array
//
void allocate_device_T(void **pp, const size_t len, const size_t sizeofT) {
  hipCheck(hipMalloc(pp, sizeofT*len));
}

//----------------------------------------------------------------------------------------
//
// Deallocate gpu memory
// pp = memory pointer
//
void deallocate_device_T(void **pp) {
  if (*pp != NULL) {
    hipCheck(hipFree((void *)(*pp)));
    *pp = NULL;
  }
}
