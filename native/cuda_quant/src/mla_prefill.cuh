// SPDX-License-Identifier: MIT
// Fixed-bank MLA block prefill with grouped routed FFNs. Include after mla_full.cuh.
namespace {
__global__ void mla_rope_queries_batch(float *q, const float *phase, const unsigned *position,
    unsigned heads, unsigned dim, unsigned rotary, float magnitude, unsigned count) {
    unsigned token = blockIdx.y;
    if (token >= count) return;
    unsigned pos = *position + token;
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x, half = rotary / 2;
    if (i >= heads * half) return;
    unsigned head = i / half, j = i % half;
    float *row = q + size_t(token) * heads * dim + size_t(head) * dim + dim - rotary;
    size_t offset = (size_t(pos) * half + j) * 2;
    float s = phase[offset], c = phase[offset + 1], a = row[2 * j], b = row[2 * j + 1];
    row[2 * j] = __fmul_rn(__fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s)), magnitude);
    row[2 * j + 1] = __fmul_rn(__fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c)), magnitude);
}

__global__ void mla_write_latent_batch(const float *normalized, const float *kv, const float *phase,
    const unsigned *position, unsigned rank, unsigned rotary, float magnitude, float *cache, unsigned count) {
    unsigned token = blockIdx.y;
    if (token >= count) return;
    unsigned pos = *position + token;
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    float *out = cache + size_t(pos) * (rank + rotary);
    if (i < rank) out[i] = normalized[size_t(token) * rank + i];
    if (i < rotary / 2) {
        size_t offset = (size_t(pos) * (rotary / 2) + i) * 2;
        float s = phase[offset], c = phase[offset + 1];
        const float *src = kv + size_t(token) * (rank + rotary);
        float a = src[rank + 2 * i], b = src[rank + 2 * i + 1];
        out[rank + 2 * i] = __fmul_rn(__fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s)), magnitude);
        out[rank + 2 * i + 1] = __fmul_rn(__fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c)), magnitude);
    }
}

template<QuantKind kind>
__global__ void mla_head_matrix_batch(const uint8_t *weights, size_t row_bytes, unsigned cols, unsigned rows,
    const float *x, unsigned token_stride, unsigned head_stride, unsigned count, float *y, unsigned out_token_stride,
    unsigned out_head_stride) {
    unsigned row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    unsigned head = blockIdx.y, token = blockIdx.z;
    if (token >= count || row >= rows) return;
    const uint8_t *w = weights + (size_t(head) * rows + row) * row_bytes;
    float dot = quant_row_dot<kind>(w, x + size_t(token) * token_stride + size_t(head) * head_stride, cols);
    if (!(threadIdx.x & 31))
        y[size_t(token) * out_token_stride + size_t(head) * out_head_stride + row] = dot;
}

void mla_launch_head_batch(const RbitnetLlamaMatrix &m, unsigned heads, unsigned rows, const float *x,
    unsigned token_stride, unsigned head_stride, unsigned count, float *y, unsigned out_token_stride,
    unsigned out_head_stride, cudaStream_t stream) {
    QuantKind kind;
    resident_kind(m.type, kind);
    dim3 grid((rows + 7) / 8, heads, count);
#define MLA_HEAD_BATCH(K) case QuantKind::K: mla_head_matrix_batch<QuantKind::K><<<grid, 256, 0, stream>>>(static_cast<const uint8_t *>(m.weights), m.row_bytes, m.cols, rows, x, token_stride, head_stride, count, y, out_token_stride, out_head_stride); break
    switch (kind) {
        MLA_HEAD_BATCH(F32);
        MLA_HEAD_BATCH(Q4_0);
        MLA_HEAD_BATCH(Q5_0);
        MLA_HEAD_BATCH(Q8_0);
        MLA_HEAD_BATCH(Q4_K);
        MLA_HEAD_BATCH(Q5_K);
        MLA_HEAD_BATCH(Q6_K);
        MLA_HEAD_BATCH(MXFP4);
    }
#undef MLA_HEAD_BATCH
}

__global__ void mla_query_tail_batch(const float *q, unsigned heads, unsigned dim, unsigned rotary, unsigned rank,
    float *absorbed, unsigned count) {
    unsigned token = blockIdx.y;
    if (token >= count) return;
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < heads * rotary) {
        absorbed[size_t(token) * heads * (rank + rotary) + size_t(i / rotary) * (rank + rotary) + rank + i % rotary] =
            q[size_t(token) * heads * dim + size_t(i / rotary) * dim + dim - rotary + i % rotary];
    }
}

__global__ void mla_attention_batch(const float *cache, const float *q, const unsigned *position, unsigned heads,
    unsigned rank, unsigned rotary, float scale, unsigned count, float *out) {
    extern __shared__ float scores[];
    __shared__ float reductions[4];
    unsigned tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    unsigned head = blockIdx.x, token = blockIdx.y;
    if (token >= count) return;
    unsigned seq = *position + token + 1, width = rank + rotary;
    q += size_t(token) * heads * width + size_t(head) * width;
    out += size_t(token) * heads * rank + size_t(head) * rank;
    for (unsigned p = warp; p < seq; p += 4) {
        float dot = 0;
        for (unsigned i = lane; i < width; i += 32) dot = fmaf(q[i], cache[size_t(p) * width + i], dot);
        for (int shift = 16; shift; shift /= 2) dot += __shfl_down_sync(0xffffffff, dot, shift);
        if (!lane) scores[p] = dot * scale;
    }
    __syncthreads();
    float maximum = -CUDART_INF_F;
    for (unsigned p = tid; p < seq; p += 128) maximum = fmaxf(maximum, scores[p]);
    for (int shift = 16; shift; shift /= 2) maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, shift));
    if (!lane) reductions[warp] = maximum;
    __syncthreads();
    maximum = fmaxf(fmaxf(reductions[0], reductions[1]), fmaxf(reductions[2], reductions[3]));
    __syncthreads();
    float sum = 0;
    for (unsigned p = tid; p < seq; p += 128) {
        scores[p] = expf(scores[p] - maximum);
        sum += scores[p];
    }
    for (int shift = 16; shift; shift /= 2) sum += __shfl_down_sync(0xffffffff, sum, shift);
    if (!lane) reductions[warp] = sum;
    __syncthreads();
    sum = reductions[0] + reductions[1] + reductions[2] + reductions[3];
    for (unsigned i = tid; i < rank; i += 128) {
        float value = 0;
        for (unsigned p = 0; p < seq; p++) value = fmaf(scores[p] / sum, cache[size_t(p) * width + i], value);
        out[i] = value;
    }
}

__global__ void mla_attention_partials_batch(const float *cache, const float *q, const unsigned *position,
    unsigned heads, unsigned rank, unsigned rotary, unsigned parts, float scale, unsigned count, float *scratch) {
    __shared__ float scores[attention_tile], reductions[4];
    unsigned tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    unsigned head = blockIdx.x, part = blockIdx.y, token = blockIdx.z;
    if (token >= count) return;
    unsigned width = rank + rotary, seq = *position + token + 1;
    unsigned begin = part * attention_tile, end = min(seq, begin + attention_tile);
    float *out = scratch + ((size_t(token) * heads + head) * parts + part) * (rank + 2);
    if (begin >= end) {
        if (!tid) {
            out[0] = -CUDART_INF_F;
            out[1] = 0;
        }
        return;
    }
    q += size_t(token) * heads * width + size_t(head) * width;
    for (unsigned p = begin + warp; p < end; p += 4) {
        float dot = 0;
        for (unsigned i = lane; i < width; i += 32) dot = fmaf(q[i], cache[size_t(p) * width + i], dot);
        for (int shift = 16; shift; shift /= 2) dot += __shfl_down_sync(0xffffffff, dot, shift);
        if (!lane) scores[p - begin] = dot * scale;
    }
    __syncthreads();
    float maximum = -CUDART_INF_F;
    for (unsigned p = tid; p < end - begin; p += 128) maximum = fmaxf(maximum, scores[p]);
    for (int shift = 16; shift; shift /= 2) maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, shift));
    if (!lane) reductions[warp] = maximum;
    __syncthreads();
    maximum = fmaxf(fmaxf(reductions[0], reductions[1]), fmaxf(reductions[2], reductions[3]));
    __syncthreads();
    float sum = 0;
    for (unsigned p = tid; p < end - begin; p += 128) {
        scores[p] = expf(scores[p] - maximum);
        sum += scores[p];
    }
    for (int shift = 16; shift; shift /= 2) sum += __shfl_down_sync(0xffffffff, sum, shift);
    if (!lane) reductions[warp] = sum;
    __syncthreads();
    sum = reductions[0] + reductions[1] + reductions[2] + reductions[3];
    if (!tid) {
        out[0] = maximum;
        out[1] = sum;
    }
    for (unsigned i = tid; i < rank; i += 128) {
        float value = 0;
        for (unsigned p = begin; p < end; p++) value = fmaf(scores[p - begin], cache[size_t(p) * width + i], value);
        out[i + 2] = value;
    }
}

__global__ void mla_router_batch(const float *raw, const float *bias, unsigned expert_count, unsigned used,
    unsigned groups, unsigned groups_used, unsigned sigmoid, unsigned normalize, float scale, unsigned *ids,
    float *probabilities) {
    if (threadIdx.x) return;
    unsigned token = blockIdx.x;
    raw += size_t(token) * expert_count;
    ids += size_t(token) * used;
    probabilities += size_t(token) * used;
    float p[128], selection[128], group_scores[128];
    unsigned selected_groups[128];
    if (sigmoid) {
        for (unsigned e = 0; e < expert_count; e++) p[e] = 1.0f / __fadd_rn(1.0f, float(exp(double(-raw[e]))));
    } else {
        float maximum = -CUDART_INF_F;
        for (unsigned e = 0; e < expert_count; e++) maximum = fmaxf(maximum, raw[e]);
        float total = 0;
        for (unsigned e = 0; e < expert_count; e++) {
            p[e] = float(exp(double(__fsub_rn(raw[e], maximum))));
            total = __fadd_rn(total, p[e]);
        }
        for (unsigned e = 0; e < expert_count; e++) p[e] /= total;
    }
    for (unsigned e = 0; e < expert_count; e++) {
        float b = bias ? bias[e] : 0.0f;
        selection[e] = isnan(p[e]) ? p[e] : (isnan(b) ? b : __fadd_rn(p[e], b));
    }
    if (groups > 1) {
        unsigned width = expert_count / groups;
        for (unsigned g = 0; g < groups; g++) {
            const unsigned begin = g * width, end = begin + width;
            unsigned first = begin, second = begin;
            for (unsigned e = begin + 1; e < end; e++)
                if (gpt_total_key(selection[e]) > gpt_total_key(selection[first])) first = e;
            bool found = false;
            for (unsigned e = g * width; e < (g + 1) * width; e++)
                if (e != first && (!found || gpt_total_key(selection[e]) > gpt_total_key(selection[second]))) {
                    second = e;
                    found = true;
                }
            group_scores[g] = __fadd_rn(selection[first], found ? selection[second] : 0.0f);
        }
        for (unsigned s = 0; s < groups_used; s++) {
            unsigned best = 0;
            bool found = false;
            for (unsigned g = 0; g < groups; g++) {
                bool seen = false;
                for (unsigned j = 0; j < s; j++) seen |= selected_groups[j] == g;
                if (!seen && (!found || gpt_total_key(group_scores[g]) > gpt_total_key(group_scores[best]))) {
                    best = g;
                    found = true;
                }
            }
            selected_groups[s] = best;
        }
        for (unsigned g = 0; g < groups; g++) {
            bool allowed = false;
            for (unsigned s = 0; s < groups_used; s++) allowed |= selected_groups[s] == g;
            if (!allowed)
                for (unsigned e = g * width; e < (g + 1) * width; e++) selection[e] = -CUDART_INF_F;
        }
    }
    for (unsigned s = 0; s < used; s++) {
        unsigned best = 0;
        bool found = false;
        for (unsigned e = 0; e < expert_count; e++) {
            bool seen = false;
            for (unsigned j = 0; j < s; j++) seen |= ids[j] == e;
            if (!seen && (!found || gpt_total_key(selection[e]) > gpt_total_key(selection[best]))) {
                best = e;
                found = true;
            }
        }
        ids[s] = best;
        probabilities[s] = p[best];
    }
    float total = 0;
    for (unsigned s = 0; s < used; s++) total = __fadd_rn(total, probabilities[s]);
    total = fmaxf(total, 1.0f / 16384.0f);
    for (unsigned s = 0; s < used; s++) probabilities[s] = __fmul_rn(normalize ? probabilities[s] / total : probabilities[s], scale);
}

struct MlaBlockWorkspace {
    ResidentMla *runtime = nullptr;
    unsigned capacity = 0;
    std::vector<void *> allocations;
    float *x = nullptr, *h = nullptr, *qa = nullptr, *qa_normed = nullptr, *q = nullptr, *kv = nullptr, *latent = nullptr;
    float *absorbed = nullptr, *values = nullptr, *attended = nullptr, *projection = nullptr, *shared = nullptr;
    float *sg = nullptr, *su = nullptr, *router = nullptr, *probabilities = nullptr, *scratch = nullptr;
    float *all_logits = nullptr, *all_maxima = nullptr, *all_maximum = nullptr;
    unsigned *ids = nullptr, *all_ids = nullptr, *all_tokens = nullptr;
    unsigned qrank = 0, shared_width = 0;
    std::unique_ptr<MoeGroupedWorkspace> moe;
    cudaGraph_t graphs[3][33] = {};
    cudaGraphExec_t executable[3][33] = {};
    ~MlaBlockWorkspace() {
        if (runtime) cudaStreamSynchronize(runtime->stream);
        for (auto &mode : executable)
            for (auto e : mode)
                if (e) cudaGraphExecDestroy(e);
        for (auto &mode : graphs)
            for (auto g : mode)
                if (g) cudaGraphDestroy(g);
        moe.reset();
        for (auto p : allocations) cudaFree(p);
    }
    template<typename T> bool alloc(T *&pointer, size_t n) {
        if (!n || n > std::numeric_limits<size_t>::max() / sizeof(T)) return false;
        if (cudaMalloc(reinterpret_cast<void **>(&pointer), n * sizeof(T)) != cudaSuccess) return false;
        try {
            allocations.push_back(pointer);
        } catch (const std::bad_alloc &) {
            cudaFree(pointer);
            pointer = nullptr;
            return false;
        }
        return true;
    }
    bool init(ResidentMla *r, unsigned count) {
        if (!r || !count || count > 32 || r->layers.empty()) return false;
        runtime = r;
        capacity = count;
        const auto &c = r->cfg;
        for (const auto &layer : r->layers) {
            qrank = max(qrank, layer.qa.rows);
            if (layer.shared_gate.weights) shared_width = max(shared_width, layer.shared_gate.rows);
        }
        ResidentMoe *first = nullptr;
        for (unsigned il = c.dense_layers; il < c.layers; il++) {
            auto *m = static_cast<ResidentMoe *>(r->layers[il].moe);
            if (!m || m->dynamic) return false;
            if (!first) first = m;
            else if (m->cfg.ffn != first->cfg.ffn || m->cfg.embd != first->cfg.embd || m->cfg.experts != first->cfg.experts
                || m->cfg.used != first->cfg.used || m->cfg.oai) return false;
        }
        if (first) {
            moe = std::unique_ptr<MoeGroupedWorkspace>(new (std::nothrow) MoeGroupedWorkspace);
            if (!moe || !moe->init(first, count)) return false;
        }
        const size_t qs = size_t(count) * c.heads * c.head_dim;
        const size_t width = c.rank + c.rotary;
        const size_t absorbed_stride = size_t(count) * c.heads * width;
        const size_t heads = size_t(count) * ((c.vocab + 255) / 256);
        return alloc(x, size_t(count) * c.embd) && alloc(h, size_t(count) * c.embd) && alloc(qa, size_t(count) * qrank)
            && alloc(qa_normed, size_t(count) * qrank) && alloc(q, qs) && alloc(kv, size_t(count) * width)
            && alloc(latent, size_t(count) * c.rank) && alloc(absorbed, absorbed_stride)
            && alloc(values, size_t(count) * c.heads * c.rank) && alloc(attended, size_t(count) * c.heads * c.value_dim)
            && alloc(projection, size_t(count) * c.embd) && alloc(shared, size_t(count) * c.embd)
            && (shared_width == 0 || (alloc(sg, size_t(count) * shared_width) && alloc(su, size_t(count) * shared_width)))
            && alloc(router, size_t(count) * c.experts) && alloc(probabilities, size_t(count) * c.used)
            && alloc(ids, size_t(count) * c.used) && alloc(all_logits, size_t(count) * c.vocab)
            && alloc(all_maxima, heads) && alloc(all_ids, heads) && alloc(all_maximum, count) && alloc(all_tokens, count)
            && (!c.split || alloc(scratch, size_t(count) * c.heads * ((c.capacity + attention_tile - 1) / attention_tile) * (c.rank + 2)));
    }
    void matrix(const RbitnetLlamaMatrix &m, const float *input, float *output, unsigned count) {
        QuantKind kind;
        resident_kind(m.type, kind);
        launch_ordered_gemm(kind, static_cast<const uint8_t *>(m.weights), m.row_bytes, input, m.cols, m.rows, count,
            output, runtime->stream, 0);
    }
    void norm(float *input, const float *weights, float *output, unsigned count, const float *residual = nullptr) {
        launch_gpt_ordered_norm(input, weights, runtime->cfg.epsilon, runtime->cfg.embd, output, residual, count,
            runtime->stream);
    }
    bool enqueue(unsigned count, unsigned mode, bool all) {
        auto *r = runtime;
        const auto &c = r->cfg;
        const auto stream = r->stream;
        const unsigned width = c.rank + c.rotary;
        for (unsigned il = 0; il < c.layers; il++) {
            const auto &l = r->layers[il];
            norm(x, l.attn_norm, h, count);
            matrix(l.qa, h, qa, count);
            launch_gpt_ordered_norm(qa, l.qa_norm, c.epsilon, l.qa.rows, qa_normed, nullptr, count, stream);
            matrix(l.qb, qa_normed, q, count);
            matrix(l.kva, h, kv, count);
            launch_gpt_ordered_norm(kv, l.kv_norm, c.epsilon, c.rank, latent, nullptr, count, stream);
            if (c.rotary) {
                dim3 rope((c.heads * (c.rotary / 2) + 255) / 256, count);
                mla_rope_queries_batch<<<rope, 256, 0, stream>>>(q, r->phases, r->position, c.heads, c.head_dim, c.rotary,
                    c.rope_magnitude, count);
            }
            mla_write_latent_batch<<<(max(c.rank, c.rotary / 2) + 255) / 256, 256, 0, stream>>>(latent, kv, r->phases,
                r->position, c.rank, c.rotary, c.rope_magnitude, r->cache[il], count);
            mla_launch_head_batch(l.kb, c.heads, c.rank, q, c.heads * c.head_dim, c.head_dim, count, absorbed,
                c.heads * width, width, stream);
            if (c.rotary)
                mla_query_tail_batch<<<(c.heads * c.rotary + 255) / 256, 256, 0, stream>>>(q, c.heads, c.head_dim, c.rotary,
                    c.rank, absorbed, count);
            const float scale = 1.0f / sqrtf(float(c.head_dim));
            if (c.split) {
                const unsigned parts = (c.capacity + attention_tile - 1) / attention_tile;
                mla_attention_partials_batch<<<dim3(c.heads, parts, count), 128, 0, stream>>>(r->cache[il], absorbed,
                    r->position, c.heads, c.rank, c.rotary, parts, scale, count, scratch);
                attention_merge<<<dim3(c.heads, count), 128, parts * sizeof(float), stream>>>(scratch, c.heads, c.rank, parts,
                    values);
            } else {
                mla_attention_batch<<<dim3(c.heads, count), 128, c.capacity * sizeof(float), stream>>>(r->cache[il], absorbed,
                    r->position, c.heads, c.rank, c.rotary, scale, count, values);
            }
            mla_launch_head_batch(l.vb, c.heads, c.value_dim, values, c.heads * c.rank, c.rank, count, attended,
                c.heads * c.value_dim, c.value_dim, stream);
            matrix(l.out, attended, projection, count);
            norm(x, l.ffn_norm, h, count, projection);
            if (l.shared_gate.weights) {
                matrix(l.shared_gate, h, sg, count);
                matrix(l.shared_up, h, su, count);
                resident_silu<<<(count * l.shared_gate.rows + 255) / 256, 256, 0, stream>>>(sg, su, count * l.shared_gate.rows);
                matrix(l.shared_down, sg, shared, count);
            }
            if (il >= c.dense_layers) {
                if (c.ordered && l.router.type == 0)
                    gpt_router_matrix_ordered<<<dim3((c.experts + 7) / 8, count), 256, 0, stream>>>(
                        static_cast<const float *>(l.router.weights), l.router.row_bytes, h, c.embd, c.experts, c.ordered,
                        router);
                else matrix(l.router, h, router, count);
                mla_router_batch<<<count, 1, 0, stream>>>(router, l.selection_bias, c.experts, c.used, c.groups,
                    c.groups_used, c.sigmoid, c.weight_norm, c.weight_scale, ids, probabilities);
                auto *source = static_cast<ResidentMoe *>(l.moe);
                const auto previous = source->stream;
                source->stream = stream;
                moe->source = source;
                if (!moe->enqueue(h, ids, probabilities, projection, count, 0)) return false;
                source->stream = previous;
                if (l.shared_gate.weights)
                    resident_add<<<(count * c.embd + 255) / 256, 256, 0, stream>>>(projection, shared, count * c.embd);
                resident_add<<<(count * c.embd + 255) / 256, 256, 0, stream>>>(x, projection, count * c.embd);
            } else {
                resident_add<<<(count * c.embd + 255) / 256, 256, 0, stream>>>(x, shared, count * c.embd);
            }
        }
        if (cudaMemcpyAsync(r->x, x + size_t(count - 1) * c.embd, c.embd * sizeof(float), cudaMemcpyDeviceToDevice, stream)
            != cudaSuccess)
            return false;
        if (mode) {
            if (all) {
                norm(x, r->norm, h, count);
                matrix(r->head, h, all_logits, count);
            } else {
                launch_gpt_ordered_norm(r->x, r->norm, c.epsilon, c.embd, h, nullptr, 1, stream);
                matrix(r->head, h, r->logits, 1);
            }
        }
        if (mode == 2) {
            const unsigned blocks = (c.vocab + 255) / 256;
            if (all) {
                resident_argmax<<<dim3(blocks, count), 256, 0, stream>>>(all_logits, nullptr, c.vocab, all_maxima, all_ids);
                resident_argmax<<<dim3(1, count), 256, 0, stream>>>(all_maxima, all_ids, blocks, all_maximum, all_tokens);
            } else {
                resident_argmax<<<blocks, 256, 0, stream>>>(r->logits, nullptr, c.vocab, r->maxima, r->ids);
                resident_argmax<<<1, 256, 0, stream>>>(r->maxima, r->ids, blocks, r->maximum, r->token);
            }
        }
        return true;
    }
};

int mla_prefill_impl(void *context, const float *embeddings, unsigned pos, unsigned count, unsigned mode, float *logits,
    unsigned *tokens, bool all) {
    auto *r = static_cast<ResidentMla *>(context);
    auto *b = r ? static_cast<MlaBlockWorkspace *>(r->block) : nullptr;
    if (!r || !b || !embeddings || !count || count > b->capacity || mode > 2 || pos >= r->cfg.capacity
        || count > r->cfg.capacity - pos || (mode == 1 && !logits) || (mode == 2 && !tokens) || (all && !mode))
        return 1;
    if (pos == 0) r->filled = 0;
    if (pos != r->filled || r->token_started || r->prepared) return 2;
    r->output_ready = false;
    NativeCallCompletion completion(r->stream);
    if (cudaMemcpyAsync(b->x, embeddings, size_t(count) * r->cfg.embd * sizeof(float), cudaMemcpyHostToDevice, r->stream)
            != cudaSuccess
        || cudaMemcpyAsync(r->position, &pos, sizeof(pos), cudaMemcpyHostToDevice, r->stream) != cudaSuccess)
        return 3;
    if (r->cfg.graphs) {
        auto &graph = b->graphs[mode][count];
        auto &exec = b->executable[mode][count];
        if (!exec) {
            if (cudaStreamBeginCapture(r->stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return 4;
            if (!b->enqueue(count, mode, all) || cudaStreamEndCapture(r->stream, &graph) != cudaSuccess
                || cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess)
                return 4;
        }
        if (cudaGraphLaunch(exec, r->stream) != cudaSuccess) return 5;
    } else if (!b->enqueue(count, mode, all)) {
        return 5;
    }
    if (cudaGetLastError() != cudaSuccess
        || (mode == 1
            && cudaMemcpyAsync(logits, all ? b->all_logits : r->logits, size_t(all ? count : 1) * r->cfg.vocab * sizeof(float),
                cudaMemcpyDeviceToHost, r->stream)
                != cudaSuccess)
        || (mode == 2
            && cudaMemcpyAsync(tokens, all ? b->all_tokens : r->token, size_t(all ? count : 1) * sizeof(unsigned),
                cudaMemcpyDeviceToHost, r->stream)
                != cudaSuccess))
        return 6;
    if (completion.complete(0, 7) != 0) return 7;
    r->filled = pos + count;
    r->host_position = pos + count - 1;
    r->next_layer = r->cfg.layers;
    r->prepared = false;
    r->token_started = false;
    r->output_ready = true;
    return 0;
}
void mla_block_destroy(void *block) { delete static_cast<MlaBlockWorkspace *>(block); }
}
extern "C" {
int rbitnet_cuda_mla_configure_prefill(void *context, unsigned capacity) {
    auto *r = static_cast<ResidentMla *>(context);
    if (!r || r->filled || r->token_started || r->prepared || r->block || !capacity || capacity > 32) return 1;
    auto *b = new (std::nothrow) MlaBlockWorkspace;
    if (!b) return 2;
    if (!b->init(r, capacity)) {
        delete b;
        return 2;
    }
    r->block = b;
    return 0;
}
unsigned rbitnet_cuda_mla_prefill_capacity(void *context) {
    auto *r = static_cast<ResidentMla *>(context);
    auto *b = r ? static_cast<MlaBlockWorkspace *>(r->block) : nullptr;
    return b ? b->capacity : 0;
}
int rbitnet_cuda_mla_full_prefill(void *context, const float *input, unsigned pos, unsigned count, unsigned mode,
    float *out, unsigned *ids) {
    return mla_prefill_impl(context, input, pos, count, mode, out, ids, false);
}
}
