// Compute modules contain only matching F32 or F64 dense storage.
// Boundary conversions live in typed.cu.
__device__ double cast_dtype(double value, unsigned int) { return value; }

__device__ unsigned long long mapped_index(
    unsigned long long index,
    unsigned int rank,
    const unsigned long long *out_shape,
    const unsigned long long *in_shape
) {
    unsigned long long mapped = 0;
    unsigned long long stride = 1;
    for (int axis = (int)rank - 1; axis >= 0; --axis) {
        unsigned long long width = out_shape[axis];
        unsigned long long coordinate = width == 0 ? 0 : index % width;
        index = width == 0 ? 0 : index / width;
        if (in_shape[axis] != 1) mapped += coordinate * stride;
        stride *= in_shape[axis];
    }
    return mapped;
}
