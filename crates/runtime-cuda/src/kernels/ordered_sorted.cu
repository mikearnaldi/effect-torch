// Exact weighted finalization from a private expert-sorted BF16 projection.
// Input3: planner-owned inverse row map, route-major -> expert-sorted.
// This narrow candidate requires weighted_source=1 and valid grouped routing.
__device__ float et_ordered_sorted_source(const CudaKernelArgs &a,
    et_u64 row, et_u64 route, et_u64 feature, et_u64 routes, et_u64 width) {
    if (a.integers[8]) {
        float projected = et_load<float>(a.inputs[0], a.input_dtypes[0],
            ((const unsigned int *)a.inputs[3])[route * a.integers[9] + row] * width + feature);
        float weight = et_load<float>(a.inputs[2], a.input_dtypes[2], row * routes + route);
        // Materialized graph semantics: widen BF16, multiply in F32, narrow
        // to BF16 before the scatter's first positive-zero addition.
        return et_bfloat_float(et_to16(projected * weight, true));
    }
    return et_load<float>(a.inputs[0], a.input_dtypes[0], (row * routes + route) * width + feature);
}
extern "C" __global__ void et_ordered_sorted_reduce(CudaKernelArgs a) {
    et_u64 i = et_thread();
    if (i >= a.elements) return;
    et_u64 routes = a.integers[0], width = a.integers[1];
    et_u64 row = i / width, feature = i % width;
    if (a.integers[7]) {
        // Dispatch guarantees complete 256-thread tiles belonging to one row.
        // Build its inverse permutation once, instead of searching every rank
        // for every output feature. Duplicates retain the general ordered path.
        __shared__ et_i64 shared_indexes[32];
        __shared__ unsigned int source_routes[32];
        __shared__ unsigned int mapping_state;
        if (threadIdx.x < routes)
            shared_indexes[threadIdx.x] = et_load<et_i64>(
                a.inputs[1], a.input_dtypes[1], row * routes + threadIdx.x);
        __syncthreads();
        if (!threadIdx.x) {
            unsigned int seen = 0;
            mapping_state = 1;
            for (unsigned int route = 0; route < routes; ++route) {
                et_i64 destination = shared_indexes[route];
                if (destination < 0 || (et_u64)destination >= routes) {
                    mapping_state = 2;
                    et_error(a, 1);
                    break;
                }
                unsigned int bit = 1U << (unsigned int)destination;
                if (seen & bit) mapping_state = 0;
                seen |= bit;
                source_routes[destination] = route;
            }
        }
        __syncthreads();
        if (mapping_state == 2) return;
        if (mapping_state == 1) {
            float total = 0.0f;
            for (et_u64 destination = 0; destination < routes; ++destination) {
                float value = et_ordered_sorted_source(a, row,
                    source_routes[destination], feature, routes, width);
                // Keep the scatter's positive-zero addition and both BF16
                // narrowing boundaries, including the first destination.
                float selected = et_bfloat_float(et_to16(0.0f + value, true));
                total = et_bfloat_float(et_to16(total + selected, true));
            }
            et_store(a.output, a.output_dtype, i, total);
            return;
        }
    }
    et_i64 indexes[32];
    for (et_u64 route = 0; route < routes; ++route) {
        indexes[route] = et_load<et_i64>(a.inputs[1], a.input_dtypes[1], row * routes + route);
        if (indexes[route] < 0 || (et_u64)indexes[route] >= routes) {
            et_error(a, 1);
            return;
        }
    }
    float total = 0.0f;
    for (et_u64 destination = 0; destination < routes; ++destination) {
        float selected = 0.0f;
        for (et_u64 route = 0; route < routes; ++route) {
            if ((et_u64)indexes[route] != destination) continue;
            float value = et_ordered_sorted_source(a, row, route, feature, routes, width);
            selected = et_bfloat_float(et_to16(selected + value, true));
        }
        total = et_bfloat_float(et_to16(total + selected, true));
    }
    et_store(a.output, a.output_dtype, i, total);
}
