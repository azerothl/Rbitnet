// SPDX-License-Identifier: MIT
// Diagnostic: a deliberately early API return, without causing a GPU fault.
namespace {
__global__ void completion_delay_check(float *out) {
    unsigned long long began=clock64();
    while(clock64()-began<2000000ULL) {}
    if(!threadIdx.x)*out=123.25f;
}
}
extern "C" RBITNET_CUDA_API int rbitnet_cuda_native_completion_check(unsigned early,float *out) {
    if(early>2 || !out)return 1;
    ResidentLlama owner;float *device=nullptr,*pinned=nullptr;
    if(cudaStreamCreateWithFlags(&owner.stream,cudaStreamNonBlocking)!=cudaSuccess || !owner.alloc(device,1))return 2;
    if(cudaHostAlloc(reinterpret_cast<void**>(&pinned),sizeof(float),cudaHostAllocDefault)!=cudaSuccess)return 3;
    *pinned=-1;
    int status=[&] {
        NativeCallCompletion completion(owner.stream);
        completion_delay_check<<<1,32,0,owner.stream>>>(device);
        if(cudaGetLastError()!=cudaSuccess || cudaMemcpyAsync(pinned,device,sizeof(float),cudaMemcpyDeviceToHost,owner.stream)!=cudaSuccess)return 4;
        if(early==2) {
            if(cudaStreamBeginCapture(owner.stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess)return 7;
            completion_delay_check<<<1,32,0,owner.stream>>>(device);
            // Abandon an otherwise valid capture. The guard must end/discard it
            // and finish the pinned copy queued before capture, without replay.
            return 37;
        }
        if(early)return 37; // destructor must wait before its caller reuses pinned.
        return completion.complete(0,5);
    }();
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    bool uncaptured=cudaStreamIsCapturing(owner.stream,&capture)==cudaSuccess && capture==cudaStreamCaptureStatusNone;
    bool completed=uncaptured && cudaStreamQuery(owner.stream)==cudaSuccess;
    bool value=completed && *pinned==123.25f;
    if(completed)*out=*pinned;
    // Always wait before cleanup even when diagnosing a broken completion guard.
    cudaStreamSynchronize(owner.stream);cudaFreeHost(pinned);
    return completed && value && status==(early?37:0)?0:6;
}
