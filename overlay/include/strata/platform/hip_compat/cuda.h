#pragma once
// gfx906 HIP compat: the poly engine's growing-cache path (engine.cpp) speaks CUDA's driver VMM
// API.  Map the subset it uses onto HIP's equivalents; libamdhip64 exports these symbols, and a
// card whose driver lacks VMM fails the engine's capability probe at startup, which routes
// around vmm_ (the same fallback a Windows/HIP machine takes).
// Handles and device addresses stay integral (the engine keeps them in uint64_t piece tables and
// does its own arithmetic on base addresses); the wrappers cast at the HIP boundary.
#include <hip/hip_runtime_api.h>

using CUresult = hipError_t;
using CUdevice = int;
using CUdeviceptr = unsigned long long;
using CUmemGenericAllocationHandle = unsigned long long;

#define CUDA_SUCCESS hipSuccess
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED hipDeviceAttributeVirtualMemoryManagementSupported
#define CU_MEM_ALLOCATION_TYPE_PINNED hipMemAllocationTypePinned
#define CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#define CU_MEM_ALLOC_GRANULARITY_MINIMUM hipMemAllocationGranularityMinimum

// CUDA field names (type/location.type/location.id) kept, values pass through numerically.
struct CUmemLocation {
    hipMemLocationType type;
    int id;
};
struct CUmemAllocationProp {
    hipMemAllocationType type;
    CUmemLocation location;
};
struct CUmemAccessDesc {
    CUmemLocation location;
    hipMemAccessFlags flags;
};

inline CUresult cuGetErrorString(CUresult r, const char** s) {
    return hipDrvGetErrorString(r, s);
}

inline CUresult cuDeviceGet(CUdevice* dev, int ordinal) {
    return hipDeviceGet(dev, ordinal);
}

inline CUresult cuDeviceGetAttribute(int* v, hipDeviceAttribute_t a, CUdevice dev) {
    return hipDeviceGetAttribute(v, a, dev);
}

inline CUresult cuMemAddressReserve(CUdeviceptr* p, size_t size, size_t alignment, CUdeviceptr addr,
                                    unsigned long long flags) {
    return hipMemAddressReserve((void**) p, size, alignment, (void*) addr, flags);
}

inline CUresult cuMemAddressFree(CUdeviceptr p, size_t size) {
    return hipMemAddressFree((void*) p, size);
}

inline hipMemAllocationProp to_hip(const CUmemAllocationProp& p) {
    hipMemAllocationProp h = {};
    h.type = p.type;
    h.location.type = p.location.type;
    h.location.id = p.location.id;
    return h;
}

inline CUresult cuMemCreate(CUmemGenericAllocationHandle* h, size_t size, const CUmemAllocationProp* prop,
                            unsigned long long flags) {
    hipMemGenericAllocationHandle_t hh = nullptr;
    const hipMemAllocationProp hp = to_hip(*prop);
    const hipError_t r = hipMemCreate(&hh, size, &hp, flags);
    *h = (CUmemGenericAllocationHandle) hh;
    return r;
}

inline CUresult cuMemGetAllocationGranularity(size_t* grain, const CUmemAllocationProp* prop,
                                              hipMemAllocationGranularity_flags g) {
    const hipMemAllocationProp hp = to_hip(*prop);
    return hipMemGetAllocationGranularity(grain, &hp, g);
}

inline CUresult cuMemMap(CUdeviceptr at, size_t size, size_t offset, CUmemGenericAllocationHandle h,
                         unsigned long long flags) {
    return hipMemMap((void*) at, size, offset, (hipMemGenericAllocationHandle_t) h, flags);
}

inline CUresult cuMemSetAccess(CUdeviceptr at, size_t size, const CUmemAccessDesc* desc, size_t count) {
    hipMemAccessDesc hd = {};
    hd.location.type = desc->location.type;
    hd.location.id = desc->location.id;
    hd.flags = desc->flags;
    return hipMemSetAccess((void*) at, size, &hd, count);
}

inline CUresult cuMemUnmap(CUdeviceptr at, size_t size) {
    return hipMemUnmap((void*) at, size);
}

inline CUresult cuMemRelease(CUmemGenericAllocationHandle h) {
    return hipMemRelease((hipMemGenericAllocationHandle_t) h);
}

// Remaining CUDA-runtime spellings the poly sources use, not covered by strata_hip.h
#define cudaGraphInstantiateWithFlags hipGraphInstantiateWithFlags
#define cudaMemcpy2D hipMemcpy2D
#define cudaFuncAttributes hipFuncAttributes
#define cudaFuncGetAttributes hipFuncGetAttributes
