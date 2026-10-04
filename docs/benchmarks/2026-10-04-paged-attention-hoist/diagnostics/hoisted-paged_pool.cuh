// SPDX-License-Identifier: MIT
// Native physical F32 KV pages, shared by matching context owners.
#include <memory>
#include <mutex>
#include <limits>
#include <vector>
namespace {
constexpr unsigned llama_page_tokens=32;
uint64_t paged_pool_identity() {
    static std::mutex mutex;static uint64_t next=0;
    std::lock_guard<std::mutex> lock(mutex);
    return next==std::numeric_limits<uint64_t>::max()?0:++next;
}
struct PhysicalKvPage {
    float *k=nullptr,*v=nullptr;
    size_t elements=0;
    ~PhysicalKvPage() {if(k)cudaFree(k);if(v)cudaFree(v);}
    bool init(size_t n) {
        if(!n || n>std::numeric_limits<size_t>::max()/sizeof(float))return false;
        MemoryCategoryScope category(MemoryKv);elements=n;
        return cudaMalloc(reinterpret_cast<void**>(&k),n*sizeof(float))==cudaSuccess
            && cudaMalloc(reinterpret_cast<void**>(&v),n*sizeof(float))==cudaSuccess;
    }
};
struct PhysicalKvPool {
    const uint64_t identity=paged_pool_identity();
    const unsigned layers,stride,limit;
    std::mutex mutex;
    std::vector<unsigned char> model_key;
    bool split=false;int tf32_mode=-1;
    // One pool reference per physical allocation. A page with only this reference
    // is recyclable; any active sequence/snapshot/queued copy holds another.
    std::vector<std::shared_ptr<PhysicalKvPage>> pages;
    uint64_t allocations=0,reuses=0,copies=0,refusals=0,peak_pages=0;
    PhysicalKvPool(unsigned l,unsigned s,unsigned n):layers(l),stride(s),limit(n) {}
    std::shared_ptr<PhysicalKvPage> acquire() {
        std::lock_guard<std::mutex> lock(mutex);
        for(auto &p:pages)if(p && p.use_count()==1) {reuses++;return p;}
        if(pages.size()>=limit) {refusals++;return {};}
        try {
            auto p=std::shared_ptr<PhysicalKvPage>(new(std::nothrow) PhysicalKvPage);
            if(!p || !p->init(size_t(layers)*stride*llama_page_tokens)) {refusals++;return {};}
            pages.push_back(p);allocations++;peak_pages=std::max(peak_pages,uint64_t(pages.size()));return p;
        } catch(const std::bad_alloc&) {refusals++;return {};}
    }
    void record_copy() {std::lock_guard<std::mutex> lock(mutex);copies++;}
    void trim() {
        std::lock_guard<std::mutex> lock(mutex);
        for(auto it=pages.begin();it!=pages.end();) {
            if(it->use_count()==1)it=pages.erase(it);else ++it;
        }
    }
};
struct PagedKvSnapshot {
    std::shared_ptr<PhysicalKvPool> pool;
    std::vector<std::shared_ptr<PhysicalKvPage>> pages;
    unsigned length=0;
};
struct PagedKvState {
    std::shared_ptr<PhysicalKvPool> pool;
    std::vector<std::shared_ptr<PhysicalKvPage>> active;
    std::vector<float*> host_k,host_v;
    float **table_k=nullptr,**table_v=nullptr;
    bool tables_valid=false;
    const unsigned logical_pages;
    explicit PagedKvState(unsigned capacity):logical_pages((capacity+llama_page_tokens-1)/llama_page_tokens) {}
    ~PagedKvState() {if(table_k)cudaFree(table_k);if(table_v)cudaFree(table_v);}
    bool init(std::shared_ptr<PhysicalKvPool> p) {
        if(!p || !p->identity || !logical_pages)return false;
        pool=std::move(p);
        try {active.resize(logical_pages);host_k.resize(logical_pages);host_v.resize(logical_pages);}
        catch(const std::bad_alloc&) {return false;}
        MemoryCategoryScope category(MemoryKv);
        return cudaMalloc(reinterpret_cast<void**>(&table_k),size_t(logical_pages)*sizeof(float*))==cudaSuccess
            && cudaMalloc(reinterpret_cast<void**>(&table_v),size_t(logical_pages)*sizeof(float*))==cudaSuccess;
    }
    bool install(cudaStream_t stream) {
        for(unsigned i=0;i<logical_pages;i++) {host_k[i]=active[i]?active[i]->k:nullptr;host_v[i]=active[i]?active[i]->v:nullptr;}
        // Persistent host table storage also survives an early return. Complete
        // before its next mutation and before any retained pages can be released.
        NativeCallCompletion completion(stream);
        tables_valid=false;
        int status=cudaMemcpyAsync(table_k,host_k.data(),size_t(logical_pages)*sizeof(float*),cudaMemcpyHostToDevice,stream)!=cudaSuccess
            || cudaMemcpyAsync(table_v,host_v.data(),size_t(logical_pages)*sizeof(float*),cudaMemcpyHostToDevice,stream)!=cudaSuccess;
        tables_valid=completion.complete(status,1)==0;return tables_valid;
    }
    bool prepare(unsigned pos,unsigned count,cudaStream_t stream) {
        if(!count || pos/llama_page_tokens>=logical_pages
            || size_t(pos)+count>size_t(logical_pages)*llama_page_tokens)return false;
        // Calls are synchronous at token/block boundaries. Keep the old pages
        // through all proposed COW copies; failed admission leaves the state intact.
        std::vector<std::shared_ptr<PhysicalKvPage>> next;
        try {next=active;}catch(const std::bad_alloc&) {return false;}
        NativeCallCompletion completion(stream); // destroyed before next on error
        unsigned first=pos/llama_page_tokens,last=(pos+count-1)/llama_page_tokens;
        bool changed=false;
        for(unsigned i=first;i<=last;i++) {
            // References: pool, active and next. Additional holders are snapshots
            // or sibling sequences, whose bytes must never be modified in place.
            bool shared=active[i] && active[i].use_count()>3;
            if(!active[i] || shared) {
                auto page=pool->acquire();if(!page)return false;
                if(active[i]) {
                    size_t bytes=active[i]->elements*sizeof(float);
                    if(cudaMemcpyAsync(page->k,active[i]->k,bytes,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess
                        || cudaMemcpyAsync(page->v,active[i]->v,bytes,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess) {
                        // The local page is destroyed before completion's guard.
                        completion.complete(1,1);return false;
                    }
                    pool->record_copy();
                }
                next[i]=std::move(page);changed=true;
            }
        }
        if(completion.complete(0,1)!=0)return false;
        if(changed) {
            // If a table update fails, keep both page sets until the stream is
            // drained; this context becomes unusable until reset/restore.
            active.swap(next);return install(stream);
        }
        return tables_valid || install(stream);
    }
    bool reset(cudaStream_t stream) {
        NativeCallCompletion completion(stream);
        if(completion.complete(0,1)!=0)return false;
        active.assign(logical_pages,{});return install(stream);
    }
    std::unique_ptr<PagedKvSnapshot> snapshot(unsigned length) const {
        if(!length || size_t(length)>size_t(logical_pages)*llama_page_tokens)return {};
        try {
            auto saved=std::unique_ptr<PagedKvSnapshot>(new(std::nothrow) PagedKvSnapshot);
            if(!saved)return {};saved->pool=pool;saved->length=length;
            unsigned count=(length+llama_page_tokens-1)/llama_page_tokens;
            saved->pages.assign(active.begin(),active.begin()+count);
            for(auto &p:saved->pages)if(!p)return {};
            return saved;
        } catch(const std::bad_alloc&) {return {};}
    }
    bool restore(const PagedKvSnapshot &saved,unsigned length,cudaStream_t stream) {
        if(!saved.pool || saved.pool->identity!=pool->identity || !length || length>saved.length
            || size_t(length)>size_t(logical_pages)*llama_page_tokens)return false;
        NativeCallCompletion completion(stream);
        if(completion.complete(0,1)!=0)return false;
        unsigned count=(length+llama_page_tokens-1)/llama_page_tokens;
        active.assign(logical_pages,{});
        for(unsigned i=0;i<count;i++)active[i]=saved.pages[i];
        return install(stream);
    }
    bool truncate(unsigned length,cudaStream_t stream) {
        if(size_t(length)>size_t(logical_pages)*llama_page_tokens)return false;
        NativeCallCompletion completion(stream);
        if(completion.complete(0,1)!=0)return false;
        unsigned keep=(length+llama_page_tokens-1)/llama_page_tokens;
        for(unsigned i=keep;i<logical_pages;i++)active[i].reset();
        return install(stream);
    }
};
}
