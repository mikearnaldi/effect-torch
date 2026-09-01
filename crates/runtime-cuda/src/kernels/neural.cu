extern "C" __global__ void layer_norm_f64(
    unsigned int op,
    const double *x,
    const double *weight,
    const double *other,
    double *out,
    unsigned int len,
    unsigned int width,
    unsigned int rows,
    double eps,
    unsigned int dtype
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    unsigned int column = index % width;
    unsigned int row = index / width;
    if (op == 2 || op == 3) {
        column = index;
        double total = 0.0;
        for (unsigned int r = 0; r < rows; ++r) {
            double mean = 0.0;
            for (unsigned int j = 0; j < width; ++j) mean += x[r * width + j];
            mean /= width;
            double variance = 0.0;
            for (unsigned int j = 0; j < width; ++j) { double centered = x[r * width + j] - mean; variance += centered * centered; }
            variance /= width;
            total += op == 2 ? other[r * width + column] * (x[r * width + column] - mean) / sqrt(variance + eps) : other[r * width + column];
        }
        out[index] = cast_dtype(total, dtype);
        return;
    }
    double mean = 0.0;
    for (unsigned int j = 0; j < width; ++j) mean += x[row * width + j];
    mean /= width;
    double variance = 0.0;
    for (unsigned int j = 0; j < width; ++j) { double centered = x[row * width + j] - mean; variance += centered * centered; }
    variance /= width;
    double inverse_std = 1.0 / sqrt(variance + eps);
    double normalized = (x[index] - mean) * inverse_std;
    if (op == 0) {
        out[index] = cast_dtype(normalized * weight[column] + other[column], dtype);
    } else {
        double mean_h = 0.0;
        double mean_hx = 0.0;
        for (unsigned int j = 0; j < width; ++j) {
            double xhat = (x[row * width + j] - mean) * inverse_std;
            double h = other[row * width + j] * weight[j];
            mean_h += h;
            mean_hx += h * xhat;
        }
        out[index] = cast_dtype((other[index] * weight[column] - mean_h / width - normalized * mean_hx / width) * inverse_std, dtype);
    }
}

__device__ bool attention_visible(unsigned int t, unsigned int s, unsigned int tq, unsigned int sk, unsigned int causal, unsigned int window) {
    if (causal == 0) return true;
    long long boundary = (long long)sk - tq + t;
    if ((long long)s > boundary) return false;
    return window == 0 || (long long)s > boundary - (long long)window;
}

extern "C" __global__ void sdpa_f64(
    unsigned int op,
    const double *q,
    const double *k,
    const double *v,
    const double *g,
    double *out,
    unsigned int len,
    unsigned int rank,
    const unsigned long long *shapes,
    double scale,
    unsigned int causal,
    unsigned int window,
    unsigned int dtype
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    const unsigned long long *qs = shapes;
    const unsigned long long *ks = shapes + rank;
    const unsigned long long *vs = shapes + rank * 2;
    unsigned int tq = qs[rank - 2];
    unsigned int sk = ks[rank - 2];
    unsigned int dk = qs[rank - 1];
    unsigned int dv = vs[rank - 1];
    unsigned int hq = rank >= 3 ? qs[rank - 3] : 1;
    unsigned int hk = rank >= 3 ? ks[rank - 3] : 1;
    unsigned int output_width = op == 0 ? dk : op == 1 ? dk : dv;
    unsigned int position = (index / output_width) % (op == 1 || op == 2 ? sk : tq);
    unsigned int outer = index / (output_width * (op == 1 || op == 2 ? sk : tq));
    unsigned int head = rank >= 3 ? outer % (op == 1 || op == 2 ? hk : hq) : 0;
    unsigned int batch = rank >= 3 ? outer / (op == 1 || op == 2 ? hk : hq) : outer;
    double result = 0.0;
    unsigned int q_head_start = op == 1 || op == 2 ? head * (hq / hk) : head;
    unsigned int q_head_end = op == 1 || op == 2 ? (head + 1) * (hq / hk) : head + 1;
    for (unsigned int qh = q_head_start; qh < q_head_end; ++qh) {
        unsigned int kh = qh / (hq / hk);
        unsigned long long qbase = ((unsigned long long)batch * hq + qh) * tq * dk;
        unsigned long long kbase = ((unsigned long long)batch * hk + kh) * sk * dk;
        unsigned long long vbase = ((unsigned long long)batch * hk + kh) * sk * dv;
        unsigned long long gbase = ((unsigned long long)batch * hq + qh) * tq * dv;
        unsigned int t_start = op == 1 || op == 2 ? 0 : position;
        unsigned int t_end = op == 1 || op == 2 ? tq : position + 1;
        for (unsigned int t = t_start; t < t_end; ++t) {
            double maximum = -1.0 / 0.0;
            for (unsigned int s = 0; s < sk; ++s) if (attention_visible(t, s, tq, sk, causal, window)) {
                double score = 0.0;
                for (unsigned int d = 0; d < dk; ++d) score += q[qbase + t * dk + d] * k[kbase + s * dk + d];
                maximum = fmax(maximum, score * scale);
            }
            double denominator = 0.0;
            double expected = 0.0;
            for (unsigned int s = 0; s < sk; ++s) if (attention_visible(t, s, tq, sk, causal, window)) {
                double score = 0.0;
                for (unsigned int d = 0; d < dk; ++d) score += q[qbase + t * dk + d] * k[kbase + s * dk + d];
                double probability = exp(score * scale - maximum);
                denominator += probability;
                if (op == 0 || op == 1) {
                    double gv = 0.0;
                    for (unsigned int d = 0; d < dv; ++d) gv += g[gbase + t * dv + d] * v[vbase + s * dv + d];
                    expected += probability * gv;
                }
            }
            expected /= denominator;
            if (op == 3) {
                for (unsigned int s = 0; s < sk; ++s) if (attention_visible(t, s, tq, sk, causal, window)) {
                    double score = 0.0;
                    for (unsigned int d = 0; d < dk; ++d) score += q[qbase + t * dk + d] * k[kbase + s * dk + d];
                    result += exp(score * scale - maximum) / denominator * v[vbase + s * dv + index % dv];
                }
            } else if (op == 0) {
                for (unsigned int s = 0; s < sk; ++s) if (attention_visible(t, s, tq, sk, causal, window)) {
                    double score = 0.0;
                    double gv = 0.0;
                    for (unsigned int d = 0; d < dk; ++d) score += q[qbase + t * dk + d] * k[kbase + s * dk + d];
                    for (unsigned int d = 0; d < dv; ++d) gv += g[gbase + t * dv + d] * v[vbase + s * dv + d];
                    result += exp(score * scale - maximum) / denominator * (gv - expected) * k[kbase + s * dk + index % dk] * scale;
                }
            } else if (op == 1 && attention_visible(t, position, tq, sk, causal, window)) {
                double score = 0.0; double gv = 0.0;
                for (unsigned int d = 0; d < dk; ++d) score += q[qbase + t * dk + d] * k[kbase + position * dk + d];
                for (unsigned int d = 0; d < dv; ++d) gv += g[gbase + t * dv + d] * v[vbase + position * dv + d];
                result += exp(score * scale - maximum) / denominator * (gv - expected) * q[qbase + t * dk + index % dk] * scale;
            } else if (op == 2 && attention_visible(t, position, tq, sk, causal, window)) {
                double score = 0.0;
                for (unsigned int d = 0; d < dk; ++d) score += q[qbase + t * dk + d] * k[kbase + position * dk + d];
                result += exp(score * scale - maximum) / denominator * g[gbase + t * dv + index % dv];
            }
        }
    }
    out[index] = cast_dtype(result, dtype);
}

extern "C" __global__ void rotary_f64(
    const double *x,
    double *out,
    unsigned int len,
    unsigned int width,
    unsigned int seq_len,
    const unsigned int *offsets,
    unsigned int groups,
    double theta,
    unsigned int interleaved,
    unsigned int backward,
    unsigned int dtype
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    unsigned int column = index % width;
    unsigned int leading = index / (width * seq_len);
    unsigned int position = offsets[leading / groups] + (index / width) % seq_len;
    unsigned int pair = interleaved ? column / 2 : column % (width / 2);
    unsigned int first = interleaved ? pair * 2 : pair;
    unsigned int second = interleaved ? first + 1 : first + width / 2;
    unsigned int row = index - column;
    double angle = position * pow(theta, -2.0 * pair / width);
    if (backward) angle = -angle;
    double c = cos(angle); double s = sin(angle);
    out[index] = cast_dtype(column == first ? x[row + first] * c - x[row + second] * s : x[row + first] * s + x[row + second] * c, dtype);
}

extern "C" __global__ void optimizer_f64(
    unsigned int kind,
    unsigned int output_kind,
    const double *param,
    const double *grad,
    const double *state1,
    const double *state2,
    const double *lr,
    const double *c1,
    const double *c2,
    double *out,
    unsigned int len,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    double dampening,
    unsigned int nesterov,
    unsigned int dtype
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    double value;
    if (kind == 0) {
        double m = beta1 * state1[index] + (1.0 - beta1) * grad[index];
        double v = beta2 * state2[index] + (1.0 - beta2) * grad[index] * grad[index];
        double updated = param[index] - lr[0] * (m / c1[0]) / (sqrt(v / c2[0]) + eps) - lr[0] * weight_decay * param[index];
        value = output_kind == 0 ? updated : output_kind == 1 ? m : v;
    } else {
        double adjusted = grad[index] + weight_decay * param[index];
        double velocity = state2[0] != 0.0 ? adjusted : beta1 * state1[index] + (1.0 - dampening) * adjusted;
        double update = nesterov != 0 ? adjusted + beta1 * velocity : velocity;
        value = output_kind == 0 ? param[index] - lr[0] * update : velocity;
    }
    out[index] = cast_dtype(value, dtype);
}
