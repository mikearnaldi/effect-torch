// Append after production typed.cu. Arithmetic helpers remain unchanged.
// Fixed wrappers specialize metadata only, preserving BF16 rounding and status.
extern "C" __global__ void et_binary_fixed_0_3_3_3_1(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 0; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 3; a.output_dtype = 3;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 1;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_1_3_3_3_1(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 1; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 3; a.output_dtype = 3;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 1;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_2_3_3_3_1(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 2; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 3; a.output_dtype = 3;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 1;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_3_3_3_3_1(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 3; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 3; a.output_dtype = 3;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 1;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_4_3_3_3_1(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 4; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 3; a.output_dtype = 3;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 1;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_5_3_3_3_1(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 5; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 3; a.output_dtype = 3;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 1;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_0_3_3_3_2(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 0; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 3; a.output_dtype = 3;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 2;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_1_3_3_3_2(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 1; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 3; a.output_dtype = 3;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 2;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_2_3_3_3_2(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 2; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 3; a.output_dtype = 3;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 2;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_3_3_3_3_2(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 3; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 3; a.output_dtype = 3;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 2;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_4_3_3_3_2(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 4; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 3; a.output_dtype = 3;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 2;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_5_3_3_3_2(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 5; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 3; a.output_dtype = 3;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 2;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_0_1_1_1_1(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 0; a.compute_dtype = 1;
    a.input_dtypes[0] = 1; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 1;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_1_1_1_1_1(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 1; a.compute_dtype = 1;
    a.input_dtypes[0] = 1; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 1;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_2_1_1_1_1(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 2; a.compute_dtype = 1;
    a.input_dtypes[0] = 1; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 1;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_3_1_1_1_1(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 3; a.compute_dtype = 1;
    a.input_dtypes[0] = 1; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 1;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_4_1_1_1_1(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 4; a.compute_dtype = 1;
    a.input_dtypes[0] = 1; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 1;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_5_1_1_1_1(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 5; a.compute_dtype = 1;
    a.input_dtypes[0] = 1; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 1;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_0_1_1_1_2(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 0; a.compute_dtype = 1;
    a.input_dtypes[0] = 1; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 2;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_1_1_1_1_2(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 1; a.compute_dtype = 1;
    a.input_dtypes[0] = 1; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 2;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_2_1_1_1_2(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 2; a.compute_dtype = 1;
    a.input_dtypes[0] = 1; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 2;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_3_1_1_1_2(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 3; a.compute_dtype = 1;
    a.input_dtypes[0] = 1; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 2;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_4_1_1_1_2(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 4; a.compute_dtype = 1;
    a.input_dtypes[0] = 1; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 2;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_5_1_1_1_2(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 5; a.compute_dtype = 1;
    a.input_dtypes[0] = 1; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 2;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_0_3_1_1_1(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 0; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 1;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_1_3_1_1_1(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 1; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 1;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_2_3_1_1_1(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 2; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 1;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_3_3_1_1_1(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 3; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 1;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_4_3_1_1_1(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 4; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 1;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_5_3_1_1_1(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 5; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 1;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_0_3_1_1_2(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 0; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 2;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_1_3_1_1_2(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 1; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 2;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_2_3_1_1_2(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 2; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 2;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_3_3_1_1_2(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 3; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 2;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_4_3_1_1_2(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 4; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 2;
    et_binary_impl<float>(a, i);
}
extern "C" __global__ void et_binary_fixed_5_3_1_1_2(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    a.operation = 5; a.compute_dtype = 1;
    a.input_dtypes[0] = 3; a.input_dtypes[1] = 1; a.output_dtype = 1;
    a.integers[0] = 0; a.integers[1] = 0;
    a.integers[2] = 1; a.integers[3] = 2;
    et_binary_impl<float>(a, i);
}
