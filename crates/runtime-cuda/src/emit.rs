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
                    return Err("CUDA fused scalar packs are unsupported".into())
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
                return Err("CUDA fused semantic marker was not legalized".into())
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
                format!("(0.5f * {value} * (1.0f + tanhf(0.7978845608028653559f * ({value} + 0.044715f * {value} * {value} * {value}))))")
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

pub(crate) fn elementwise(
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
    Ok(format!(
        "extern \"C\" __global__ void et_fused_elementwise(CudaKernelArgs a) {{\n    for (et_u64 i = et_thread(); i < a.elements; i += (et_u64)gridDim.x * blockDim.x) {{\n{body}        et_store(a.output, a.output_dtype, i, {result});\n    }}\n}}\n"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
