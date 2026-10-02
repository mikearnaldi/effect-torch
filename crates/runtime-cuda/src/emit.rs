//! CUDA source emission for compiler-selected elementwise regions.

use effect_torch_compiler::KernelExpr;
use effect_torch_runtime::DType;

fn f32_literal(value: f64) -> String {
    let value = value as f32;
    if value.is_infinite() {
        return if value.is_sign_positive() {
            "(INFINITY)".into()
        } else {
            "(-INFINITY)".into()
        };
    }
    if value.is_nan() {
        return "(NAN)".into();
    }
    format!("({value:e}f)")
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for dimension in (0..shape.len().saturating_sub(1)).rev() {
        strides[dimension] = strides[dimension + 1] * shape[dimension + 1];
    }
    strides
}

fn lane_offset(strides: &[usize], moduli: &[usize], shape: &[usize], base: usize) -> String {
    let contiguous = contiguous_strides(shape);
    if strides == contiguous && moduli.iter().all(|&modulus| modulus == 0) {
        return if base == 0 {
            "i".into()
        } else {
            format!("(i + {base}ULL)")
        };
    }
    let mut terms = Vec::new();
    for dimension in 0..shape.len() {
        if shape[dimension] == 1 || strides[dimension] == 0 {
            continue;
        }
        let mut coordinate = if dimension + 1 == shape.len() {
            format!("(i % {}ULL)", shape[dimension])
        } else {
            format!(
                "((i / {}ULL) % {}ULL)",
                contiguous[dimension], shape[dimension]
            )
        };
        if moduli[dimension] != 0 {
            coordinate = format!("({coordinate} % {}ULL)", moduli[dimension]);
        }
        if strides[dimension] == 1 {
            terms.push(coordinate);
        } else {
            terms.push(format!("({coordinate} * {}ULL)", strides[dimension]));
        }
    }
    if base != 0 {
        terms.push(format!("{base}ULL"));
    }
    if terms.is_empty() {
        "0ULL".into()
    } else {
        terms.join(" + ")
    }
}

fn rounded(value: String, dtype: DType) -> Result<String, String> {
    Ok(match dtype {
        DType::F32 => format!("(float)({value})"),
        DType::F16 => format!("et_half_float(et_to16({value}, false))"),
        DType::BF16 => format!("et_bfloat_float(et_to16({value}, true))"),
        DType::U8 => format!("(float)et_uint({value}, 255U)"),
        DType::U32 => format!("(float)et_uint({value}, 0xffffffffU)"),
        DType::I64 => format!("(float)et_int({value})"),
        DType::F64 => return Err("CUDA fused expressions do not support F64".into()),
    })
}

fn emit_expression(
    expression: &KernelExpr,
    lanes: &[String],
    body: &mut String,
) -> Result<String, String> {
    macro_rules! binary {
        ($values:ident, $format:literal) => {{
            let right = $values
                .pop()
                .ok_or("CUDA fused expression operand missing")?;
            let left = $values
                .pop()
                .ok_or("CUDA fused expression operand missing")?;
            format!($format, left, right)
        }};
    }
    macro_rules! unary {
        ($values:ident, $format:literal) => {{
            let value = $values
                .pop()
                .ok_or("CUDA fused expression operand missing")?;
            format!($format, value)
        }};
    }

    let mut work = vec![(expression, false)];
    let mut values = Vec::new();
    let mut temporaries = 0usize;
    while let Some((node, processed)) = work.pop() {
        if !processed {
            match node {
                KernelExpr::Input(lane) => {
                    let offset = lanes
                        .get(*lane as usize)
                        .ok_or("CUDA fused expression lane is out of range")?;
                    values.push(format!(
                        "et_load<float>(a.inputs[{lane}], a.input_dtypes[{lane}], {offset})"
                    ));
                    continue;
                }
                KernelExpr::Scalar(_) => {
                    return Err("CUDA fused scalar packs are unsupported".into());
                }
                KernelExpr::Const(bits) => {
                    values.push(f32_literal(f64::from_bits(*bits)));
                    continue;
                }
                _ => {}
            }
            work.push((node, true));
            match node {
                KernelExpr::Select(condition, left, right) => {
                    work.push((right, false));
                    work.push((left, false));
                    work.push((condition, false));
                }
                KernelExpr::Add(left, right)
                | KernelExpr::Sub(left, right)
                | KernelExpr::Mul(left, right)
                | KernelExpr::Div(left, right)
                | KernelExpr::Min(left, right)
                | KernelExpr::Max(left, right)
                | KernelExpr::Lt(left, right)
                | KernelExpr::Le(left, right)
                | KernelExpr::Gt(left, right)
                | KernelExpr::Ge(left, right)
                | KernelExpr::Eq(left, right)
                | KernelExpr::Ne(left, right) => {
                    work.push((right, false));
                    work.push((left, false));
                }
                KernelExpr::Cast(value, _)
                | KernelExpr::RoundTo(value, _)
                | KernelExpr::Semantic(value, _, _)
                | KernelExpr::Neg(value)
                | KernelExpr::Sqrt(value)
                | KernelExpr::Exp(value)
                | KernelExpr::Sin(value)
                | KernelExpr::Cos(value)
                | KernelExpr::Tanh(value)
                | KernelExpr::Abs(value)
                | KernelExpr::Log(value)
                | KernelExpr::Floor(value)
                | KernelExpr::Ceil(value)
                | KernelExpr::Round(value)
                | KernelExpr::Powf(value, _)
                | KernelExpr::Erf(value)
                | KernelExpr::Gelu(value)
                | KernelExpr::GeluTanh(value) => work.push((value, false)),
                KernelExpr::Input(_) | KernelExpr::Scalar(_) | KernelExpr::Const(_) => {
                    unreachable!()
                }
            }
            continue;
        }

        let result = match node {
            KernelExpr::Input(_) | KernelExpr::Scalar(_) | KernelExpr::Const(_) => unreachable!(),
            KernelExpr::Cast(_, dtype) | KernelExpr::RoundTo(_, dtype) => rounded(
                values
                    .pop()
                    .ok_or("CUDA fused conversion operand missing")?,
                *dtype,
            )?,
            KernelExpr::Semantic(..) => {
                return Err("CUDA fused semantic marker was not legalized".into());
            }
            KernelExpr::Add(..) => binary!(values, "({} + {})"),
            KernelExpr::Sub(..) => binary!(values, "({} - {})"),
            KernelExpr::Mul(..) => binary!(values, "({} * {})"),
            KernelExpr::Div(..) => binary!(values, "({} / {})"),
            KernelExpr::Min(..) => binary!(values, "fminf({}, {})"),
            KernelExpr::Max(..) => binary!(values, "fmaxf({}, {})"),
            KernelExpr::Lt(..) => binary!(values, "({} < {} ? 1.0f : 0.0f)"),
            KernelExpr::Le(..) => binary!(values, "({} <= {} ? 1.0f : 0.0f)"),
            KernelExpr::Gt(..) => binary!(values, "({} > {} ? 1.0f : 0.0f)"),
            KernelExpr::Ge(..) => binary!(values, "({} >= {} ? 1.0f : 0.0f)"),
            KernelExpr::Eq(..) => binary!(values, "({} == {} ? 1.0f : 0.0f)"),
            KernelExpr::Ne(..) => binary!(values, "({} != {} ? 1.0f : 0.0f)"),
            KernelExpr::Select(..) => {
                let right = values.pop().ok_or("CUDA fused select operand missing")?;
                let left = values.pop().ok_or("CUDA fused select operand missing")?;
                let condition = values.pop().ok_or("CUDA fused select condition missing")?;
                format!("({condition} != 0.0f ? {left} : {right})")
            }
            KernelExpr::Neg(..) => unary!(values, "(-{})"),
            KernelExpr::Sqrt(..) => unary!(values, "sqrtf({})"),
            KernelExpr::Exp(..) => unary!(values, "expf({})"),
            KernelExpr::Sin(..) => unary!(values, "sinf({})"),
            KernelExpr::Cos(..) => unary!(values, "cosf({})"),
            KernelExpr::Tanh(..) => unary!(values, "tanhf({})"),
            KernelExpr::Abs(..) => unary!(values, "fabsf({})"),
            KernelExpr::Log(..) => unary!(values, "logf({})"),
            KernelExpr::Floor(..) => unary!(values, "floorf({})"),
            KernelExpr::Ceil(..) => unary!(values, "ceilf({})"),
            KernelExpr::Round(..) => unary!(values, "roundf({})"),
            KernelExpr::Powf(_, exponent) => {
                let value = values.pop().ok_or("CUDA fused pow operand missing")?;
                format!("powf({value}, {})", f32_literal(f64::from_bits(*exponent)))
            }
            KernelExpr::Erf(..) => unary!(values, "erff({})"),
            KernelExpr::Gelu(..) => {
                let value = values.pop().ok_or("CUDA fused GELU operand missing")?;
                format!("(0.5f * {value} * (1.0f + erff({value} * 0.7071067811865475244f)))")
            }
            KernelExpr::GeluTanh(..) => {
                let value = values.pop().ok_or("CUDA fused GELU operand missing")?;
                format!(
                    "(0.5f * {value} * (1.0f + tanhf(0.7978845608028653559f * ({value} + 0.044715f * {value} * {value} * {value}))))"
                )
            }
        };
        let temporary = format!("t{temporaries}");
        temporaries += 1;
        body.push_str(&format!("        float {temporary} = {result};\n"));
        values.push(temporary);
    }
    if values.len() != 1 {
        return Err("CUDA fused expression has invalid arity".into());
    }
    Ok(values.pop().unwrap())
}

#[cfg(test)]
pub(crate) fn elementwise(
    expression: &KernelExpr,
    lane_strides: &[Box<[usize]>],
    lane_moduli: &[Box<[usize]>],
    lane_offsets: &[usize],
    shape: &[usize],
) -> Result<String, String> {
    elementwise_with_sum(
        false,
        false,
        expression,
        lane_strides,
        lane_moduli,
        lane_offsets,
        shape,
    )
}

pub(crate) fn elementwise_with_sum(
    wide_sum: bool,
    wide_arg: bool,
    expression: &KernelExpr,
    lane_strides: &[Box<[usize]>],
    lane_moduli: &[Box<[usize]>],
    lane_offsets: &[usize],
    shape: &[usize],
) -> Result<String, String> {
    if lane_strides.len() != lane_moduli.len() || lane_strides.len() != lane_offsets.len() {
        return Err("CUDA fused expression lane metadata length mismatch".into());
    }
    if lane_strides
        .iter()
        .zip(lane_moduli)
        .any(|(strides, moduli)| strides.len() != shape.len() || moduli.len() != shape.len())
    {
        return Err("CUDA fused expression lane rank mismatch".into());
    }
    let lanes = lane_strides
        .iter()
        .zip(lane_moduli)
        .zip(lane_offsets)
        .map(|((strides, moduli), &base)| lane_offset(strides, moduli, shape, base))
        .collect::<Vec<_>>();
    let mut body = String::new();
    let result = emit_expression(expression, &lanes, &mut body)?;
    if wide_arg {
        return Ok(format!(
            r#"
extern "C" __global__ __launch_bounds__(1024) void et_fused_elementwise(CudaKernelArgs a) {{
    et_u64 width = a.integers[1];
    __shared__ float values[32];
    __shared__ unsigned int indexes[32];
    unsigned int lane = threadIdx.x & 31U, warp = threadIdx.x / 32;
    for (et_u64 row = blockIdx.x; row < a.elements; row += gridDim.x) {{
        float best = a.operation == 0 ? -1.0f / 0.0f : 1.0f / 0.0f;
        unsigned int best_index = 0xffffffffU;
        for (et_u64 column = threadIdx.x; column < width; column += 1024) {{
            et_u64 i = row * width + column;
{body}
            float value = {result};
            bool better = !isnan(value) && (
                (a.operation == 0 && value > best) ||
                (a.operation == 1 && value < best) ||
                (value == best && column < best_index)
            );
            if (better) {{ best = value; best_index = (unsigned int)column; }}
        }}
        for (unsigned int offset = 16; offset; offset >>= 1) {{
            float other = __shfl_down_sync(0xffffffffU, best, offset);
            unsigned int other_index = __shfl_down_sync(0xffffffffU, best_index, offset);
            bool better = (a.operation == 0 && other > best) ||
                (a.operation == 1 && other < best) ||
                (other == best && other_index < best_index);
            if (better) {{ best = other; best_index = other_index; }}
        }}
        if (!lane) {{ values[warp] = best; indexes[warp] = best_index; }}
        __syncthreads();
        if (!warp) {{
            best = values[lane]; best_index = indexes[lane];
            for (unsigned int offset = 16; offset; offset >>= 1) {{
                float other = __shfl_down_sync(0xffffffffU, best, offset);
                unsigned int other_index = __shfl_down_sync(0xffffffffU, best_index, offset);
                bool better = (a.operation == 0 && other > best) ||
                    (a.operation == 1 && other < best) ||
                    (other == best && other_index < best_index);
                if (better) {{ best = other; best_index = other_index; }}
            }}
            if (!lane) {{
                et_u64 i = row * width;
{body}
                if (isnan({result})) best_index = 0;
                et_store(a.output, a.output_dtype, row, (et_i64)best_index);
            }}
        }}
        __syncthreads();
    }}
}}
"#
        ));
    }
    if wide_sum {
        return Ok(format!(
            r#"
extern "C" __global__ __launch_bounds__(1024) void et_fused_elementwise(CudaKernelArgs a) {{
    __shared__ float partials[32];
    unsigned int lane = threadIdx.x & 31U, warp = threadIdx.x / 32;
    et_u64 count = a.integers[1], complete = count - count % 4096;
    for (et_u64 row = blockIdx.x; row < a.elements; row += gridDim.x) {{
        float sum = 0.0f;
        for (et_u64 r = threadIdx.x * 4; r < complete; r += 4096) {{
            #pragma unroll
            for (unsigned int j = 0; j < 4; ++j) {{
                et_u64 i = row * count + r + j;
{body}
                sum += {result};
            }}
        }}
        for (et_u64 r = complete + threadIdx.x; r < count; r += 1024) {{
            et_u64 i = row * count + r;
{body}
            sum += {result};
        }}
        for (unsigned int offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffU, sum, offset);
        if (!lane) partials[warp] = sum;
        __syncthreads();
        if (!warp) {{
            sum = partials[lane];
            for (unsigned int offset = 16; offset; offset >>= 1)
                sum += __shfl_down_sync(0xffffffffU, sum, offset);
            if (!lane) et_store(a.output, a.output_dtype, row, sum);
        }}
        __syncthreads();
    }}
}}
"#
        ));
    }
    Ok(format!(
        "extern \"C\" __global__ void et_fused_elementwise(CudaKernelArgs a) {{\n    for (et_u64 i = et_thread(); i < a.elements; i += (et_u64)gridDim.x * blockDim.x) {{\n{body}        et_store(a.output, a.output_dtype, i, {result});\n    }}\n}}\n"
    ))
}

/// Keep the canonical conversion helpers and expression intact, but make each
/// physical storage tag constant so NVRTC can remove unused switch branches.
/// Call only after lowering has resolved views and legalized storage values.
pub(crate) fn specialize_storage_dtypes(
    mut source: String,
    input_dtypes: &[u32],
    output_dtype: u32,
) -> Result<String, String> {
    if input_dtypes.len() > 8 || output_dtype > 6 || input_dtypes.iter().any(|&dtype| dtype > 6) {
        return Err("CUDA fused storage specialization has invalid dtype metadata".into());
    }
    for (lane, dtype) in input_dtypes.iter().enumerate() {
        source = source.replace(&format!("a.input_dtypes[{lane}]"), &format!("{dtype}U"));
    }
    if source.contains("a.input_dtypes[") {
        return Err("CUDA fused storage specialization is missing an input lane".into());
    }
    Ok(source.replace("a.output_dtype", &format!("{output_dtype}U")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_lane_loads_element_zero() {
        let source = elementwise(
            &KernelExpr::Div(
                Box::new(KernelExpr::Input(0)),
                Box::new(KernelExpr::Input(1)),
            ),
            &[
                vec![17, 1].into_boxed_slice(),
                vec![0, 0].into_boxed_slice(),
            ],
            &[vec![0, 0].into_boxed_slice(), vec![0, 0].into_boxed_slice()],
            &[0, 0],
            &[2, 17],
        )
        .unwrap();
        assert!(source.contains("a.input_dtypes[1], 0ULL)"), "{source}");
    }

    #[test]
    fn repeated_trailing_lane_wraps_its_coordinate() {
        let source = elementwise(
            &KernelExpr::Input(0),
            &[vec![12, 4, 1].into_boxed_slice()],
            &[vec![0, 0, 4].into_boxed_slice()],
            &[0],
            &[2, 3, 8],
        )
        .unwrap();

        assert!(source.contains("((i % 8ULL) % 4ULL)"), "{source}");
    }
    #[test]
    fn static_storage_tags_preserve_mixed_conversion_helpers_and_view_coordinates() {
        let generic = elementwise(
            &KernelExpr::Add(
                Box::new(KernelExpr::Input(0)),
                Box::new(KernelExpr::Input(1)),
            ),
            &[
                vec![12, 4, 1].into_boxed_slice(),
                vec![0, 0, 0].into_boxed_slice(),
            ],
            &[
                vec![0, 0, 4].into_boxed_slice(),
                vec![0, 0, 0].into_boxed_slice(),
            ],
            &[7, 0],
            &[2, 3, 8],
        )
        .unwrap();
        let typed = specialize_storage_dtypes(generic.clone(), &[3, 2], 1).unwrap();
        assert!(typed.contains("et_load<float>(a.inputs[0], 3U,"));
        assert!(typed.contains("et_load<float>(a.inputs[1], 2U, 0ULL)"));
        assert!(typed.contains("((i % 8ULL) % 4ULL)"));
        assert!(typed.contains("7ULL"));
        assert!(typed.contains("et_store(a.output, 1U, i,"));
        assert_eq!(
            typed,
            generic
                .replace("a.input_dtypes[0]", "3U")
                .replace("a.input_dtypes[1]", "2U")
                .replace("a.output_dtype", "1U")
        );
    }

    #[test]
    fn static_storage_tags_preserve_integer_argmax_store_and_reject_missing_lanes() {
        let generic = elementwise_with_sum(
            false,
            true,
            &KernelExpr::Input(0),
            &[vec![4096, 1].into_boxed_slice()],
            &[vec![0, 0].into_boxed_slice()],
            &[0],
            &[2, 4096],
        )
        .unwrap();
        let typed = specialize_storage_dtypes(generic.clone(), &[1], 4).unwrap();
        assert!(typed.contains("et_store(a.output, 4U, row, (et_i64)best_index)"));
        assert!(!typed.contains("a.input_dtypes"));
        assert!(!typed.contains("a.output_dtype"));
        assert!(specialize_storage_dtypes(generic.clone(), &[], 4).is_err());
        assert!(specialize_storage_dtypes(generic.clone(), &[7], 4).is_err());
        assert!(specialize_storage_dtypes(generic, &[1], 7).is_err());
    }
}

#[cfg(test)]
#[path = "emit_storage_tests.rs"]
mod storage_tests;
