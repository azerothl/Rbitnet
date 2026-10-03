// SPDX-License-Identifier: MIT
// Complete host/lease lifetimes even when a CUDA call exits early.
// Include after device_memory.cuh; does not modify CUDA kernels or reduction order.
namespace {
struct NativeCallCompletion {
    cudaStream_t stream;
    bool done=false;
    explicit NativeCallCompletion(cudaStream_t s):stream(s) {}
    NativeCallCompletion(const NativeCallCompletion&)=delete;
    NativeCallCompletion& operator=(const NativeCallCompletion&)=delete;
    // A failed graph construction can leave a capture active on this thread.
    // End and discard it before waiting for the work queued before capture.
    cudaError_t wait() {
        cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
        if(cudaStreamIsCapturing(stream,&capture)==cudaSuccess && capture!=cudaStreamCaptureStatusNone) {
            cudaGraph_t discarded=nullptr;
            cudaStreamEndCapture(stream,&discarded);
            if(discarded)cudaGraphDestroy(discarded);
        }
        return cudaStreamSynchronize(stream);
    }
    ~NativeCallCompletion() {if(!done)wait();}
    // Only dismiss when synchronization already completed, or when the enclosing
    // synchronous API owns and explicitly completes all subsequently queued work.
    void dismiss() {done=true;}
    int complete(int status,int wait_failure) {
        auto waited=wait();done=true;
        return status?status:(waited==cudaSuccess?0:wait_failure);
    }
};
}
