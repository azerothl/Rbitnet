// SPDX-License-Identifier: MIT
// One ledger for Rust allocations and native contexts, including failed creates.
// CUDA graph/driver allocations are opaque; a configurable free-device margin
// is separate from the exact bytes controlled by this library.
#include <mutex>
#include <algorithm>
#include <unordered_map>
#include <limits>
#include <new>
namespace {
enum MemoryCategory : unsigned { MemoryWeights,MemoryKv,MemoryActivation,MemoryPrefix,MemoryExperts,MemoryScratch,MemoryOther,MemoryCategories };
thread_local unsigned native_memory_category=MemoryActivation;
struct MemoryCategoryScope {
    unsigned previous;
    explicit MemoryCategoryScope(unsigned kind):previous(native_memory_category) {native_memory_category=kind;}
    ~MemoryCategoryScope() {native_memory_category=previous;}
};
struct MemoryAllocation {size_t bytes;unsigned category;int device;};
struct MemoryLedger {
    std::mutex mutex;
    std::unordered_map<void*,MemoryAllocation> pointers;
    RbitnetCudaMemoryStats stats={1,0,0,0,0,0,{}};
    int limited_device=-1;
};
MemoryLedger &memory_ledger() {
    // Keep the ledger alive through CUDA thread-local scratch destruction.
    static auto *ledger=new MemoryLedger;
    return *ledger;
}
cudaError_t memory_allocate(void **pointer,size_t bytes,unsigned category) {
    if(!pointer)return cudaErrorInvalidValue;
    *pointer=nullptr;
    if(category>=MemoryCategories)return cudaErrorInvalidValue;
    if(!bytes)return cudaSuccess;
    auto &l=memory_ledger();std::lock_guard<std::mutex> lock(l.mutex);
    int device=0;auto status=cudaGetDevice(&device);if(status!=cudaSuccess)return status;
    if((l.stats.limit && device!=l.limited_device) || bytes>std::numeric_limits<uint64_t>::max()-l.stats.live
        || (l.stats.limit && bytes>l.stats.limit-l.stats.live)) {
        l.stats.refusals++;return cudaErrorMemoryAllocation;
    }
    // Serialize admission with allocation and free. No other allocation can
    // exceed the cap while this cudaMalloc is pending or fails.
    status=cudaMalloc(pointer,bytes);
    if(status!=cudaSuccess) {l.stats.refusals++;return status;}
    try {l.pointers.emplace(*pointer,MemoryAllocation{bytes,category,device});}
    catch(const std::bad_alloc&) {cudaFree(*pointer);*pointer=nullptr;l.stats.refusals++;return cudaErrorMemoryAllocation;}
    l.stats.live+=bytes;l.stats.categories[category]+=bytes;l.stats.allocations++;
    l.stats.peak=std::max(l.stats.peak,l.stats.live);
    return cudaSuccess;
}
cudaError_t memory_release(void *pointer) {
    if(!pointer)return cudaSuccess;
    auto &l=memory_ledger();std::lock_guard<std::mutex> lock(l.mutex);
    auto entry=l.pointers.find(pointer);
    if(entry==l.pointers.end())return cudaErrorInvalidDevicePointer;
    int device=0;auto status=cudaGetDevice(&device);if(status!=cudaSuccess)return status;
    if(device!=entry->second.device)return cudaErrorInvalidDevicePointer;
    status=cudaFree(pointer);
    if(status==cudaSuccess) {
        l.stats.live-=entry->second.bytes;l.stats.categories[entry->second.category]-=entry->second.bytes;
        l.pointers.erase(entry);
    }
    // A failed free retains the charge, rather than admitting unbounded memory.
    return status;
}
cudaError_t memory_native_allocate(void **pointer,size_t bytes) {
    return memory_allocate(pointer,bytes,native_memory_category);
}
}
extern "C" {
int rbitnet_cuda_memory_configure(uint64_t limit,uint64_t margin) {
    auto &l=memory_ledger();std::lock_guard<std::mutex> lock(l.mutex);
    if(!limit) {
        if(l.stats.live)return 1;
        if(l.stats.limit)l.stats.peak=0;
        l.stats.limit=0;l.limited_device=-1;return 0;
    }
    int device=0;if(cudaGetDevice(&device)!=cudaSuccess)return 2;
    for(const auto &entry:l.pointers)if(entry.second.device!=device)return 3;
    size_t free=0,total=0;if(cudaMemGetInfo(&free,&total)!=cudaSuccess)return 4;
    uint64_t available=free>margin?uint64_t(free)-margin:0;
    // cudaMemGetInfo already excludes our live allocations; add them exactly
    // once when deriving the cap. Never raise an active process's existing cap.
    uint64_t cap=std::min(limit,l.stats.live+available);
    if(l.stats.live && l.stats.limit)cap=std::min(cap,l.stats.limit);
    if(!cap || cap<l.stats.live)return 5;
    // Peak belongs to this cap epoch. A lower cap must not be compared with
    // a historical peak observed under an older, larger cap.
    if(l.stats.limit!=cap)l.stats.peak=l.stats.live;
    l.stats.limit=cap;l.limited_device=device;return 0;
}
int rbitnet_cuda_memory_stats(RbitnetCudaMemoryStats *out) {
    if(!out)return 1;auto &l=memory_ledger();std::lock_guard<std::mutex> lock(l.mutex);
    *out=l.stats;return 0;
}
int rbitnet_cuda_memory_alloc(void **pointer,size_t bytes,unsigned category) {
    return int(memory_allocate(pointer,bytes,category));
}
int rbitnet_cuda_memory_free(void *pointer) {return int(memory_release(pointer));}
}
// All subsequent native contexts and optional snapshots use this ledger. The
// wrappers above were defined before these macros and call the real CUDA API.
#define cudaMalloc memory_native_allocate
#define cudaFree memory_release
