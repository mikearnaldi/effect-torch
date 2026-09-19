__device__ void short_conv_f64(
    unsigned int op,
    const double *x,
    const double *weight,
    const double *g,
    double *out,
    unsigned int len,
    unsigned int outer,
    unsigned int time,
    unsigned int channels,
    unsigned int kernel,
    const float *initial_state,
    float *next_state,
    const unsigned int *valid,
    unsigned int stateful,
    unsigned int dtype
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    double total = 0.0;
    if (op == 0) {
        unsigned int c = index % channels;
        unsigned int t = (index / channels) % time;
        unsigned int batch = index / (channels * time);
        if (t < (valid ? valid[batch] : time)) {
            for (unsigned int j = 0; j < kernel; ++j) {
                long long source = (long long)t - kernel + 1 + j;
                if (source >= 0) {
                    total += x[(batch * time + source) * channels + c] * weight[c * kernel + j];
                } else if (stateful != 0) {
                    unsigned int history = (unsigned int)(source + kernel - 1);
                    total += initial_state[(batch * (kernel - 1) + history) * channels + c] * weight[c * kernel + j];
                }
            }
        }
        if (stateful != 0 && t == 0) {
            unsigned int advance = valid ? valid[batch] : time;
            for (unsigned int j = 0; j + 1 < kernel; ++j) {
                long long source = (long long)advance - kernel + 1 + j;
                unsigned int history = (unsigned int)(source + kernel - 1);
                double value = source >= 0
                    ? x[(batch * time + source) * channels + c]
                    : initial_state[(batch * (kernel - 1) + history) * channels + c];
                next_state[(batch * (kernel - 1) + j) * channels + c] = value;
            }
        }
    } else if (op == 1) {
        unsigned int c = index % channels;
        unsigned int source = (index / channels) % time;
        unsigned int batch = index / (channels * time);
        for (unsigned int j = 0; j < kernel; ++j) {
            long long t = (long long)source + kernel - 1 - j;
            if (t >= 0 && t < time) total += g[(batch * time + t) * channels + c] * weight[c * kernel + j];
        }
    } else {
        unsigned int j = index % kernel;
        unsigned int c = index / kernel;
        for (unsigned int batch = 0; batch < outer; ++batch) for (unsigned int t = 0; t < time; ++t) {
            long long source = (long long)t - kernel + 1 + j;
            if (source >= 0) total += g[(batch * time + t) * channels + c] * x[(batch * time + source) * channels + c];
        }
    }
    out[index] = cast_dtype(total, dtype);
}

template<class State> __device__ void kda_forward_f64(
    const double *q, const double *k, const double *v, const double *decay, const double *beta,
    double *out, State *state, unsigned int outer, unsigned int time, unsigned int dk, unsigned int dv,
    double scale, const unsigned int *valid, unsigned int heads, unsigned int stateful, unsigned int dtype
) {
    unsigned int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= outer) return;
    State *s = state + (unsigned long long)batch * dk * dv;
    if (stateful == 0) for (unsigned int i = 0; i < dk * dv; ++i) s[i] = 0.0;
    for (unsigned int t = 0; t < time; ++t) {
        unsigned long long qbase = ((unsigned long long)batch * time + t) * dk;
        unsigned long long vbase = ((unsigned long long)batch * time + t) * dv;
        if (t >= (valid ? valid[batch / heads] : time)) {
            for (unsigned int j = 0; j < dv; ++j) out[vbase + j] = 0.0;
            continue;
        }
        for (unsigned int d = 0; d < dk; ++d) for (unsigned int j = 0; j < dv; ++j) s[d * dv + j] *= exp(decay[qbase + d]);
        for (unsigned int j = 0; j < dv; ++j) {
            double dot = 0.0;
            for (unsigned int d = 0; d < dk; ++d) dot += k[qbase + d] * s[d * dv + j];
            for (unsigned int d = 0; d < dk; ++d) s[d * dv + j] += beta[(unsigned long long)batch * time + t] * k[qbase + d] * (v[vbase + j] - dot);
            double result = 0.0;
            for (unsigned int d = 0; d < dk; ++d) result += s[d * dv + j] * q[qbase + d];
            out[vbase + j] = cast_dtype(scale * result, dtype);
        }
    }
}

__device__ void kda_backward_f64(
    unsigned int output_kind, const double *q, const double *k, const double *v, const double *decay,
    const double *beta, const double *g, double *out, double *history, double *grad_state,
    unsigned int outer, unsigned int time, unsigned int dk, unsigned int dv, double scale, unsigned int dtype
) {
    unsigned int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= outer) return;
    unsigned long long state_size = (unsigned long long)dk * dv;
    double *states = history + (unsigned long long)batch * (time + 1) * state_size;
    double *gs = grad_state + (unsigned long long)batch * state_size * 2;
    double *next_gs = gs + state_size;
    for (unsigned long long i = 0; i < state_size; ++i) states[i] = 0.0;
    for (unsigned int t = 0; t < time; ++t) {
        double *previous = states + (unsigned long long)t * state_size;
        double *current = previous + state_size;
        unsigned long long qbase = ((unsigned long long)batch * time + t) * dk;
        unsigned long long vbase = ((unsigned long long)batch * time + t) * dv;
        for (unsigned int d = 0; d < dk; ++d) for (unsigned int j = 0; j < dv; ++j) current[d * dv + j] = previous[d * dv + j] * exp(decay[qbase + d]);
        for (unsigned int j = 0; j < dv; ++j) {
            double dot = 0.0; for (unsigned int d = 0; d < dk; ++d) dot += k[qbase + d] * current[d * dv + j];
            for (unsigned int d = 0; d < dk; ++d) current[d * dv + j] += beta[(unsigned long long)batch * time + t] * k[qbase + d] * (v[vbase + j] - dot);
        }
    }
    for (unsigned long long i = 0; i < state_size; ++i) gs[i] = 0.0;
    for (int ti = (int)time - 1; ti >= 0; --ti) {
        unsigned int t = (unsigned int)ti;
        double *previous = states + (unsigned long long)t * state_size;
        double *current = previous + state_size;
        unsigned long long qbase = ((unsigned long long)batch * time + t) * dk;
        unsigned long long vbase = ((unsigned long long)batch * time + t) * dv;
        for (unsigned int d = 0; d < dk; ++d) for (unsigned int j = 0; j < dv; ++j) gs[d * dv + j] += scale * q[qbase + d] * g[vbase + j];
        if (output_kind == 0) for (unsigned int d = 0; d < dk; ++d) { double total = 0.0; for (unsigned int j = 0; j < dv; ++j) total += current[d * dv + j] * g[vbase + j]; out[qbase + d] = cast_dtype(scale * total, dtype); }
        if (output_kind == 2) for (unsigned int j = 0; j < dv; ++j) { double total = 0.0; for (unsigned int d = 0; d < dk; ++d) total += gs[d * dv + j] * k[qbase + d]; out[vbase + j] = cast_dtype(beta[(unsigned long long)batch * time + t] * total, dtype); }
        double beta_grad = 0.0;
        for (unsigned int d = 0; d < dk; ++d) {
            double k_grad = 0.0; double decay_grad = 0.0;
            for (unsigned int j = 0; j < dv; ++j) {
                double decayed = previous[d * dv + j] * exp(decay[qbase + d]);
                double dot = 0.0; double kg = 0.0;
                for (unsigned int i = 0; i < dk; ++i) { dot += k[qbase + i] * previous[i * dv + j] * exp(decay[qbase + i]); kg += k[qbase + i] * gs[i * dv + j]; }
                beta_grad += gs[d * dv + j] * k[qbase + d] * (v[vbase + j] - dot);
                k_grad += beta[(unsigned long long)batch * time + t] * (gs[d * dv + j] * (v[vbase + j] - dot) - decayed * kg);
                double ddecayed = gs[d * dv + j] - beta[(unsigned long long)batch * time + t] * k[qbase + d] * kg;
                decay_grad += ddecayed * decayed;
            }
            if (output_kind == 1) out[qbase + d] = cast_dtype(k_grad, dtype);
            if (output_kind == 3) out[qbase + d] = cast_dtype(decay_grad, dtype);
        }
        if (output_kind == 4) out[(unsigned long long)batch * time + t] = cast_dtype(beta_grad, dtype);
        for (unsigned int d = 0; d < dk; ++d) for (unsigned int j = 0; j < dv; ++j) {
            double kg = 0.0; for (unsigned int i = 0; i < dk; ++i) kg += k[qbase + i] * gs[i * dv + j];
            next_gs[d * dv + j] = (gs[d * dv + j] - beta[(unsigned long long)batch * time + t] * k[qbase + d] * kg) * exp(decay[qbase + d]);
        }
        for (unsigned long long i = 0; i < state_size; ++i) gs[i] = next_gs[i];
    }
}
