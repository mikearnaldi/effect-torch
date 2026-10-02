// A table header per sequence stores [retained_start, cursor, end, row_offset].
// Each logical row stores immutable K/V pointers and optional U8 scale pointers.
// Only rows at or beyond cursor point into this invocation's private transaction.
__device__ const et_u64 *et_cache_row(const CudaKernelArgs &a, et_u64 sequence, et_u64 position) {
    const et_u64 *table = (const et_u64 *)a.inputs[3], *header = table + sequence * 4;
    return table + header[3] + (position - header[0]) * 4;
}
__device__ void et_cache_append(const CudaKernelArgs &a, et_u64 lane, et_u64 token, et_u64 heads, et_u64 tokens, et_u64 dim) {
    const unsigned int *cursors = (const unsigned int *)a.inputs[7];
    et_u64 sequence = lane / a.integers[2], position = cursors[lane] + token;
    const et_u64 *header = (const et_u64 *)a.inputs[3] + sequence * 4;
    if (position < header[1] || position >= header[2]) { et_error(a, 1); return; }
    const et_u64 *pointers = et_cache_row(a, sequence, position);
    unsigned int dtype = a.integers[1];
    for (et_u64 head = 0; head < heads; ++head) {
        et_u64 source = ((lane * heads + head) * tokens + token) * dim;
        for (int role = 0; role < 2; ++role) {
            float scale = 1;
            if (dtype == 6) {
                float maximum = 0;
                for (et_u64 d = 0; d < dim; ++d) maximum = fmaxf(maximum, fabsf(et_load<float>(a.inputs[role + 1], a.input_dtypes[role + 1], source + d)));
                scale = maximum == 0 ? 1.0f : maximum / 127.0f;
                ((float *)pointers[role + 2])[head] = scale;
            }
            for (et_u64 d = 0; d < dim; ++d) {
                float value = et_load<float>(a.inputs[role + 1], a.input_dtypes[role + 1], source + d);
                if (dtype == 6) {
                    int code = __float2int_rn(value / scale); code = code < -127 ? -127 : code > 127 ? 127 : code;
                    ((unsigned char *)pointers[role])[head * dim + d] = (unsigned char)(code + 128);
                } else if (a.input_dtypes[role + 1] == 0) {
                    et_store(pointers[role], dtype, head * dim + d, et_load<double>(a.inputs[role + 1], 0, source + d));
                } else et_store(pointers[role], dtype, head * dim + d, value);
            }
        }
    }
}
__device__ float et_cache_load(const CudaKernelArgs &a, int role, et_u64 sequence, et_u64 position, et_u64 head, et_u64 d, et_u64 dim) {
    const et_u64 *p = et_cache_row(a, sequence, position);
    if (a.integers[1] == 6) return ((int)((const unsigned char *)p[role])[head * dim + d] - 128) * ((const float *)p[role + 2])[head];
    return et_load<float>(p[role], a.integers[1], head * dim + d);
}
__device__ float et_kv_round(float value, unsigned int dtype) {
    if (dtype == 2) return et_half_float(et_to16(value, false));
    if (dtype == 3) return et_bfloat_float(et_to16(value, true));
    return value;
}
template<class T> __device__ T et_kv_score(const CudaKernelArgs &a, et_u64 query, et_u64 sequence, et_u64 pos, et_u64 head, et_u64 dim) {
    T score = 0;
    for (et_u64 j = 0; j < dim; ++j) score += et_load<T>(a.inputs[0], a.input_dtypes[0], query + j) * T(et_cache_load(a, 0, sequence, pos, head, j, dim));
    if (a.integers[6]) {
        // The scalar scale stays F32. Rounding belongs after QK and scaling,
        // after normalized probabilities, and after the weighted-value sum.
        float dot = et_kv_round(float(score), a.integers[8]);
        return T(et_kv_round(dot * float(a.scalars[0]), a.integers[8]));
    }
    return score * T(a.scalars[0]);
}
template<class T> __device__ T et_kv_warp_sum(T value) {
#ifndef ET_HOST_TEST
    for (unsigned int offset = 16; offset; offset >>= 1) value += __shfl_down_sync(0xffffffffU, value, offset);
    value = __shfl_sync(0xffffffffU, value, 0);
#endif
    return value;
}
template<class T> __device__ T et_kv_warp_max(T value) {
#ifndef ET_HOST_TEST
    for (unsigned int offset = 16; offset; offset >>= 1) value = fmax(value, __shfl_down_sync(0xffffffffU, value, offset));
    value = __shfl_sync(0xffffffffU, value, 0);
#endif
    return value;
}
// Stream order publishes all invocation-owned current rows before any query
// warp reads them. In particular, later canvas rows are visible bidirectionally.
extern "C" __global__ void et_kv_store(CudaKernelArgs a) {
    unsigned int rank = et_meta(a)[1];
    const et_u64 *qs = et_shape(a, 0), *ks = et_shape(a, 1);
    et_u64 tokens = qs[rank - 2], heads = ks[rank - 3], dim = qs[rank - 1];
    if (a.integers[1] != 6) {
        et_u64 lane = blockIdx.x / tokens, token = blockIdx.x % tokens;
        if (lane >= a.integers[5] * a.integers[2]) return;
        unsigned int count = ((const unsigned int *)a.scratch[0])[lane];
        if (count > tokens) { et_error(a, 1); return; }
        if (token >= count) return;
        const unsigned int *cursors = (const unsigned int *)a.inputs[7];
        et_u64 sequence = lane / a.integers[2], position = cursors[lane] + token;
        const et_u64 *header = (const et_u64 *)a.inputs[3] + sequence * 4;
        if (position < header[1] || position >= header[2]) { et_error(a, 1); return; }
        const et_u64 *pointers = et_cache_row(a, sequence, position);
        for (et_u64 hd = threadIdx.x; hd < heads * dim; hd += blockDim.x) {
            et_u64 head = hd / dim, d = hd % dim;
            et_u64 source = ((lane * heads + head) * tokens + token) * dim + d;
            for (int role = 0; role < 2; ++role) {
                if (a.input_dtypes[role + 1] == 0)
                    et_store(pointers[role], a.integers[1], hd, et_load<double>(a.inputs[role + 1], 0, source));
                else
                    et_store(pointers[role], a.integers[1], hd, et_load<float>(a.inputs[role + 1], a.input_dtypes[role + 1], source));
            }
        }
        return;
    }
    et_u64 lane = et_thread() / tokens, token = et_thread() % tokens;
    if (lane >= a.integers[5] * a.integers[2]) return;
    unsigned int count = ((const unsigned int *)a.scratch[0])[lane];
    if (count > tokens) { et_error(a, 1); return; }
    if (token < count) et_cache_append(a, lane, token, heads, tokens, dim);
}

template<class T> __device__ void et_kv_attention_impl(const CudaKernelArgs &a) {
#ifdef ET_HOST_TEST
    const unsigned int warp_lane = 0, warp_width = 1;
#else
    const unsigned int warp_lane = threadIdx.x & 31U, warp_width = 32;
#endif
    unsigned int rank = et_meta(a)[1]; const et_u64 *qs = et_shape(a, 0), *ks = et_shape(a, 1);
    et_u64 dim = qs[rank - 1], tokens = qs[rank - 2], qheads = qs[rank - 3], kheads = ks[rank - 3];
    et_u64 item = et_thread() / 32, lane = item / (qheads * tokens);
    if (lane >= a.integers[5] * a.integers[2]) return;
    et_u64 head = (item / tokens) % qheads, t = item % tokens, sequence = lane / a.integers[2];
    et_u64 query = item * dim;
    et_u64 output = a.operation ? ((lane * tokens + t) * qheads + head) * dim : query;
    const unsigned int *valid = (const unsigned int *)a.scratch[0], *cursors = (const unsigned int *)a.inputs[7];
    if (valid[lane] > tokens) { et_error(a, 1); return; }
    if (t >= valid[lane]) {
        for (et_u64 d = warp_lane; d < dim; d += warp_width) et_store(a.output, a.output_dtype, output + d, 0.0f);
        return;
    }
    const et_u64 *header = (const et_u64 *)a.inputs[3] + sequence * 4;
    et_u64 end = a.integers[4] ? header[2] : cursors[lane] + t + 1, start = header[0], window = a.integers[3];
    if (window && end > window && end - window > start) start = end - window;
    if (end < start || end - start > a.integers[9]) { et_error(a, 1); return; }
    et_u64 kh = head * kheads / qheads;
    // One score row per query/head, in invocation-owned planned workspace.
    // Keep the old position-strided reduction order and every rounding boundary.
    // Reusing this row removes the former full QK recomputation per four V dims.
    T *probabilities = (T *)a.inputs[4] + item * a.integers[9];
    T maximum = -T(1) / T(0);
    for (et_u64 pos = start + warp_lane; pos < end; pos += warp_width) {
        T score = et_kv_score<T>(a, query, sequence, pos, kh, dim);
        probabilities[pos - start] = score;
        maximum = fmax(maximum, score);
    }
    maximum = et_kv_warp_max(maximum);
    T denominator = 0;
    for (et_u64 pos = start + warp_lane; pos < end; pos += warp_width) {
        T weight = exp(probabilities[pos - start] - maximum);
        probabilities[pos - start] = weight;
        denominator += weight;
    }
    denominator = et_kv_warp_sum(denominator);
    for (et_u64 pos = start + warp_lane; pos < end; pos += warp_width) {
        T probability = probabilities[pos - start] / denominator;
        if (a.integers[6]) probability = T(et_kv_round(float(probability), a.integers[8]));
        probabilities[pos - start] = probability;
    }
    for (et_u64 d = 0; d < dim; d += 4) {
        T result[4] = {0, 0, 0, 0};
        for (et_u64 pos = start + warp_lane; pos < end; pos += warp_width) {
            T probability = probabilities[pos - start];
            for (unsigned int j = 0; j < 4 && d + j < dim; ++j)
                result[j] += probability * T(et_cache_load(a, 1, sequence, pos, kh, d + j, dim));
        }
        for (unsigned int j = 0; j < 4 && d + j < dim; ++j) {
            T value = et_kv_warp_sum(result[j]);
            if (a.integers[6]) value = T(et_kv_round(float(value), a.integers[8]));
            if (warp_lane == 0) et_store(a.output, a.output_dtype, output + d + j, value);
        }
    }
}
extern "C" __global__ void et_kv_attention(CudaKernelArgs a) {
    if (a.compute_dtype == 0 && !a.integers[6]) et_kv_attention_impl<double>(a);
    else et_kv_attention_impl<float>(a);
}

// Dense BF16 operands for the F32-only stepwise GEMM realization. Each call
// packs the actual retained+current rows, not the pool capacity. Prefix pages
// remain immutable; these copies belong only to the invocation workspace.
extern "C" __global__ void et_kv_gemm_gather(CudaKernelArgs a) {
    et_u64 i=et_thread(), positions=a.integers[13], lane=a.integers[14];
    et_u64 heads=a.integers[11], tokens=a.integers[15], dim=a.integers[10];
    if (i<heads*tokens*dim) {
        et_u64 source=((lane*heads+i/(tokens*dim))*a.integers[7]+(i/dim)%tokens)*dim+i%dim;
        et_store(a.inputs[5],3,i,et_load<float>(a.inputs[0],a.input_dtypes[0],source));
    }
    if (i>=heads*positions*dim) return;
    et_u64 head=i/(positions*dim), p=(i/dim)%positions, d=i%dim;
    et_u64 sequence=lane/a.integers[2], kh=head*et_shape(a,1)[et_meta(a)[1]-3]/heads;
    const et_u64 *header=(const et_u64*)a.inputs[3]+sequence*4;
    et_store(a.inputs[6],3,i,et_cache_load(a,0,sequence,header[0]+p,kh,d,dim));
    et_store(a.scratch[1],3,i,et_cache_load(a,1,sequence,header[0]+p,kh,d,dim));
}
extern "C" __global__ void et_kv_gemm_softmax(CudaKernelArgs a) {
    et_u64 item=et_thread()/32, positions=a.integers[13], lane=a.integers[14];
    et_u64 heads=a.integers[11], tokens=a.integers[15];
    if (item>=heads*tokens) return;
    unsigned int wl=threadIdx.x&31U;
    for (et_u64 p=wl;p<positions;p+=32) et_store(a.inputs[1],3,item*positions+p,0.0f);
    et_u64 t=item%tokens;
    if (t>=((const unsigned int*)a.scratch[0])[lane]) return;
    et_u64 sequence=lane/a.integers[2];
    const et_u64 *header=(const et_u64*)a.inputs[3]+sequence*4;
    et_u64 end=a.integers[4]?header[2]:((const unsigned int*)a.inputs[7])[lane]+t+1;
    et_u64 start=header[0], window=a.integers[3];
    if (window && end>window && end-window>start) start=end-window;
    if (end<start || end-header[0]>positions) { et_error(a,1); return; }
    float *scores=(float*)a.inputs[4]+item*positions;
    float maximum=-1.0f/0.0f;
    for (et_u64 p=start+wl;p<end;p+=32) {
        float score=et_kv_round(et_kv_round(scores[p-header[0]],3)*float(a.scalars[0]),3);
        scores[p-header[0]]=score; maximum=fmaxf(maximum,score);
    }
    maximum=et_kv_warp_max(maximum);
    float total=0;
    for (et_u64 p=start+wl;p<end;p+=32) {
        float value=exp(scores[p-header[0]]-maximum);
        scores[p-header[0]]=value; total+=value;
    }
    total=et_kv_warp_sum(total);
    for (et_u64 p=start+wl;p<end;p+=32)
        et_store(a.inputs[1],3,item*positions+p-header[0],scores[p-header[0]]/total);
}
extern "C" __global__ void et_kv_gemm_round(CudaKernelArgs a) {
    et_u64 i=et_thread(); if (i>=a.elements) return;
    et_u64 dim=a.integers[10], tokens=a.integers[15], head=i/(tokens*dim), t=(i/dim)%tokens;
    et_u64 row=a.operation?t*a.integers[11]+head:head*a.integers[7]+t;
    et_store(a.output,a.output_dtype,row*dim+i%dim,et_kv_round(((float*)a.inputs[2])[i],3));
}
