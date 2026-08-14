//! First-party fusion IR → MSL emitter.
//!
//! Emits one specialized compute kernel per fused expression set
//! (elementwise) or per reduction. Conventions:
//!
//! - **SSA form.** Expressions compile to flat `float tN = ...;`
//!   temporaries, not nested parentheses: deep chains would exceed MSL's
//!   bracket nesting limit, and temporaries dedupe shared subtrees
//!   (memoized by node address). Emission is an iterative post-order walk
//!   so kernel compilation never multiplies expression depth by the host
//!   call stack.
//! - **Clamped flat index.** Each kernel takes a thread-position parameter
//!   and clamps it to `n - 1` instead of early-returning, so padded grids
//!   are safe without divergent control flow. Past `u32::MAX` elements the
//!   index widens to `ulong` over a 2-D grid of width [`WIDE`].
//! - **Dtypes.** The IR only models float math: f32 lanes load/store
//!   directly, bf16 lanes convert at the boundary (`float(...)` in,
//!   `bfloat(...)` out). Any other storage dtype is an emitter bug
//!   (`unreachable!`).
//! - **Strides baked in.** Per-input lane offsets are emitted as constant
//!   arithmetic on the output coordinates; a layout change is a different
//!   kernel and a different pipeline key.
//! - **Buffer binding order.** `in0..inK`, optional `scs` (packed f32
//!   scalars), then `out0..outJ`, matching what the runners bind.
//! - **Literals.** f32 constants are emitted via `{:e}` with explicit
//!   handling of inf/NaN, so emitted source round-trips bit-exactly and
//!   never depends on locale formatting.

use super::device::MetalDevice;
use crate::fusion::{Expr, ReduceOp};

/// Threadgroup width (and grid padding quantum) for all flat fused
/// kernels.
pub const BLOCK: usize = 256;

/// Grid width of 64-bit kernels: flat index = gid.y * WIDE + gid.x.
pub const WIDE: usize = MetalDevice::WIDE;

/// The shader index type for a tensor of `n` elements: 32-bit math on
/// the fast path; 64-bit once a flat offset can exceed u32 — any model
/// worth training has a logits or weight tensor past 4G elements.
pub fn idx_ty(n: usize) -> &'static str {
    if n <= u32::MAX as usize {
        "uint"
    } else {
        "ulong"
    }
}

/// The gid parameter declaration and clamped flat index for a kernel
/// covering `n` elements: 1-D grid with uint math when small, a 2-D
/// grid with a widened flat index when not.
fn gid_decl(n: usize) -> (String, String) {
    if n <= u32::MAX as usize {
        (
            "    uint gid [[thread_position_in_grid]]".to_string(),
            format!(
                "    const uint clamped = min(gid, {}u);\n",
                n.saturating_sub(1)
            ),
        )
    } else {
        (
            "    uint2 gid2 [[thread_position_in_grid]]".to_string(),
            format!(
                "    const ulong gid = ulong(gid2.y) * {}ul + ulong(gid2.x);\n    const ulong clamped = min(gid, {}ul);\n",
                WIDE,
                n.saturating_sub(1)
            ),
        )
    }
}

// Storage dtype of the lanes and outputs: bf16 kernels load into float,
// compute in float, and store back as bfloat — the fusion IR only models
// float math, so dtype support is purely a load/store concern.
fn storage_ty(dtype: crate::runtime::dtype::DType) -> &'static str {
    match dtype {
        crate::runtime::dtype::DType::F32 => "float",
        crate::runtime::dtype::DType::BF16 => "bfloat",
        other => unreachable!("emit: unsupported storage dtype {other:?}"),
    }
}

fn load_expr(access: String, dtype: crate::runtime::dtype::DType) -> String {
    match dtype {
        crate::runtime::dtype::DType::F32 => access,
        crate::runtime::dtype::DType::BF16 => format!("float({access})"),
        other => unreachable!("emit: unsupported storage dtype {other:?}"),
    }
}

fn store_expr(value: &str, dtype: crate::runtime::dtype::DType) -> String {
    match dtype {
        crate::runtime::dtype::DType::F32 => value.to_string(),
        crate::runtime::dtype::DType::BF16 => format!("bfloat({value})"),
        other => unreachable!("emit: unsupported storage dtype {other:?}"),
    }
}

fn f32_lit(v: f64) -> String {
    let v = v as f32;
    if v.is_infinite() {
        return if v > 0.0 {
            "(INFINITY)".to_string()
        } else {
            "(-INFINITY)".to_string()
        };
    }
    if v.is_nan() {
        return "(NAN)".to_string();
    }
    let s = format!("{v:e}");
    let s = s.replace('e', "e");
    format!("({s}f)")
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    let mut acc = 1usize;
    for d in (0..shape.len()).rev() {
        strides[d] = acc;
        acc *= shape[d];
    }
    strides
}

// SSA form: long fused chains (thousands of ops) exceed MSL's bracket
// nesting limit when emitted as nested parentheses; flat temporaries also
// compile dramatically faster and dedupe shared subtrees.
fn emit_expr_ssa(
    e: &Expr,
    lane: &dyn Fn(u32) -> String,
    indent: &str,
    body: &mut String,
    next: &mut usize,
    memo: &mut std::collections::HashMap<usize, String>,
) -> String {
    // Iterative post-order emission (memoized by node pointer): a long
    // fused chain is one deep Expr, and kernel emission must not
    // multiply its depth by the call stack.
    macro_rules! binary {
        ($stack:ident, $fmt:literal) => {{
            let b = $stack.pop().expect("emit operand");
            let a = $stack.pop().expect("emit operand");
            format!($fmt, a, b)
        }};
    }
    macro_rules! unary {
        ($stack:ident, $fmt:literal) => {{
            let a = $stack.pop().expect("emit operand");
            format!($fmt, a)
        }};
    }
    let mut stack: Vec<(&Expr, bool)> = vec![(e, false)];
    let mut results: Vec<String> = Vec::new();
    while let Some((node, processed)) = stack.pop() {
        if !processed {
            let ptr = node as *const Expr as usize;
            if let Some(t) = memo.get(&ptr) {
                results.push(t.clone());
                continue;
            }
            match node {
                Expr::Input(k) => {
                    results.push(lane(*k));
                    continue;
                }
                Expr::Scalar(k) => {
                    results.push(format!("scs[{}]", *k as usize));
                    continue;
                }
                Expr::Const(bits) => {
                    results.push(f32_lit(f64::from_bits(*bits)));
                    continue;
                }
                _ => {}
            }
            stack.push((node, true));
            match node {
                Expr::Select(c, a, b) => {
                    stack.push((b, false));
                    stack.push((a, false));
                    stack.push((c, false));
                }
                Expr::Add(a, b)
                | Expr::Sub(a, b)
                | Expr::Mul(a, b)
                | Expr::Div(a, b)
                | Expr::Min(a, b)
                | Expr::Max(a, b)
                | Expr::Lt(a, b)
                | Expr::Le(a, b)
                | Expr::Gt(a, b)
                | Expr::Ge(a, b)
                | Expr::Eq(a, b)
                | Expr::Ne(a, b) => {
                    stack.push((b, false));
                    stack.push((a, false));
                }
                Expr::Neg(a)
                | Expr::Sqrt(a)
                | Expr::Exp(a)
                | Expr::Sin(a)
                | Expr::Cos(a)
                | Expr::Tanh(a)
                | Expr::Abs(a)
                | Expr::Log(a)
                | Expr::Floor(a)
                | Expr::Ceil(a)
                | Expr::Round(a)
                | Expr::Powf(a, _)
                | Expr::Erf(a)
                | Expr::Gelu(a)
                | Expr::GeluTanh(a) => stack.push((a, false)),
                Expr::Input(_) | Expr::Scalar(_) | Expr::Const(_) => unreachable!(),
            }
            continue;
        }
        let rhs = match node {
            Expr::Input(_) | Expr::Scalar(_) | Expr::Const(_) => unreachable!(),
            Expr::Add(..) => binary!(results, "({} + {})"),
            Expr::Sub(..) => binary!(results, "({} - {})"),
            Expr::Mul(..) => binary!(results, "({} * {})"),
            Expr::Div(..) => binary!(results, "({} / {})"),
            Expr::Min(..) => binary!(results, "fmin({}, {})"),
            Expr::Max(..) => binary!(results, "fmax({}, {})"),
            Expr::Lt(..) => binary!(results, "({} < {} ? 1.0f : 0.0f)"),
            Expr::Le(..) => binary!(results, "({} <= {} ? 1.0f : 0.0f)"),
            Expr::Gt(..) => binary!(results, "({} > {} ? 1.0f : 0.0f)"),
            Expr::Ge(..) => binary!(results, "({} >= {} ? 1.0f : 0.0f)"),
            Expr::Eq(..) => binary!(results, "({} == {} ? 1.0f : 0.0f)"),
            Expr::Ne(..) => binary!(results, "({} != {} ? 1.0f : 0.0f)"),
            Expr::Select(..) => {
                let b = results.pop().expect("emit operand");
                let a = results.pop().expect("emit operand");
                let c = results.pop().expect("emit operand");
                format!("({} != 0.0f ? {} : {})", c, a, b)
            }
            Expr::Neg(..) => unary!(results, "(-{})"),
            Expr::Sqrt(..) => unary!(results, "sqrt({})"),
            Expr::Exp(..) => unary!(results, "exp({})"),
            Expr::Sin(..) => unary!(results, "sin({})"),
            Expr::Cos(..) => unary!(results, "cos({})"),
            Expr::Tanh(..) => unary!(results, "tanh({})"),
            Expr::Abs(..) => unary!(results, "fabs({})"),
            Expr::Log(..) => unary!(results, "log({})"),
            Expr::Floor(..) => unary!(results, "floor({})"),
            Expr::Ceil(..) => unary!(results, "ceil({})"),
            Expr::Round(..) => unary!(results, "round({})"),
            Expr::Powf(_, e) => {
                let a = results.pop().expect("emit operand");
                format!("pow({}, {})", a, f32_lit(f64::from_bits(*e)))
            }
            Expr::Erf(..) => unary!(results, "erf_as({})"),
            Expr::Gelu(..) => unary!(results, "gelu_as({})"),
            Expr::GeluTanh(..) => unary!(results, "gelu_tanh_as({})"),
        };
        let name = format!("t{}", *next);
        *next += 1;
        body.push_str(&format!("{indent}float {name} = {rhs};\n"));
        memo.insert(node as *const Expr as usize, name.clone());
        results.push(name);
    }
    debug_assert_eq!(results.len(), 1);
    results.pop().expect("emit result")
}

fn lane_offset_expr(strides: &[usize], out_shape: &[usize], index: &str) -> String {
    let contig = contiguous_strides(out_shape);
    if strides == contig {
        return index.to_string();
    }
    let rank = out_shape.len();
    let mut terms: Vec<String> = Vec::new();
    for d in 0..rank {
        if out_shape[d] == 1 || strides[d] == 0 {
            continue;
        }
        let coord = if d == rank - 1 {
            format!("({index} % {})", out_shape[d])
        } else {
            format!("(({index} / {}) % {})", contig[d], out_shape[d])
        };
        if strides[d] == 1 {
            terms.push(coord);
        } else {
            terms.push(format!("({coord} * {})", strides[d]));
        }
    }
    if terms.is_empty() {
        "0u".to_string()
    } else {
        terms.join(" + ")
    }
}

/// The erf/gelu helpers, shared by fused elementwise/reduce kernels
/// (via [`HEADER`]) and kernels that build their own source (gemm
/// epilogues).
pub(crate) const ACT_FNS: &str = r#"
inline float erf_as(float x) {
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float p = 1.061405429f;
    p = p * t - 1.453152027f;
    p = p * t + 1.421413741f;
    p = p * t - 0.284496736f;
    p = p * t + 0.254829592f;
    float tail = 1.0f - p * t * exp(-x * x);
    float sign = x / fmax(ax, 1e-30f);
    return sign * tail;
}

inline float gelu_as(float x) {
    return 0.5f * x * (1.0f + erf_as(x * 0.7071067811865476f));
}

inline float gelu_tanh_as(float x) {
    float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanh(u));
}
"#;

const PREAMBLE: &str = r#"
#include <metal_stdlib>
using namespace metal;
"#;

/// Emits the fused multi-output elementwise kernel source: one thread per
/// output element computes every expression from the strided input lanes.
pub fn emit_elementwise(
    exprs: &[Expr],
    lane_strides: &[Vec<usize>],
    out_shape: &[usize],
    n: usize,
    num_scalars: usize,
    name: &str,
    dtype: crate::runtime::dtype::DType,
) -> String {
    let num_inputs = lane_strides.len();
    let num_outputs = exprs.len();
    let ty = storage_ty(dtype);
    let mut src = format!("{PREAMBLE}{ACT_FNS}");
    src.push_str(&format!("kernel void {name}(\n"));
    let mut idx = 0usize;
    let mut params: Vec<String> = Vec::new();
    for k in 0..num_inputs {
        params.push(format!("    device const {ty}* in{k} [[buffer({idx})]]"));
        idx += 1;
    }
    if num_scalars > 0 {
        params.push(format!("    device const float* scs [[buffer({idx})]]"));
        idx += 1;
    }
    for j in 0..num_outputs {
        params.push(format!("    device {ty}* out{j} [[buffer({idx})]]"));
        idx += 1;
    }
    let (gid_param, clamped_decl) = gid_decl(n);
    params.push(gid_param);
    src.push_str(&params.join(",\n"));
    src.push_str("\n) {\n");
    src.push_str(&clamped_decl);
    let lanes: Vec<String> = lane_strides
        .iter()
        .map(|s| lane_offset_expr(s, out_shape, "clamped"))
        .collect();
    let lane = |k: u32| -> String { load_expr(format!("in{}[{}]", k, lanes[k as usize]), dtype) };
    let mut next = 0usize;
    let mut memo = std::collections::HashMap::new();
    for (j, expr) in exprs.iter().enumerate() {
        let mut body = String::new();
        let v = emit_expr_ssa(expr, &lane, "    ", &mut body, &mut next, &mut memo);
        src.push_str(&body);
        src.push_str(&format!(
            "    out{j}[clamped] = {};\n",
            store_expr(&v, dtype)
        ));
    }
    src.push_str("}\n");
    src
}

/// Emits the fused reduction kernel source: one thread per output element
/// accumulates the expression over the flattened reduced dimensions, in a
/// fixed order (deterministic).
#[allow(clippy::too_many_arguments)]
pub fn emit_reduce(
    op: ReduceOp,
    expr: &Expr,
    lane_strides: &[Vec<usize>],
    in_shape: &[usize],
    dims: &[usize],
    keepdims: bool,
    out_shape: &[usize],
    name: &str,
    dtype: crate::runtime::dtype::DType,
) -> String {
    let num_inputs = lane_strides.len();
    let rank = in_shape.len();
    let out_n: usize = out_shape.iter().product();
    let contig_out = contiguous_strides(out_shape);
    let ty = storage_ty(dtype);
    let mut src = format!("{PREAMBLE}{ACT_FNS}");
    src.push_str(&format!("kernel void {name}(\n"));
    let mut params: Vec<String> = Vec::new();
    let mut idx = 0usize;
    for k in 0..num_inputs {
        params.push(format!("    device const {ty}* in{k} [[buffer({idx})]]"));
        idx += 1;
    }
    params.push(format!("    device {ty}* out [[buffer({idx})]]"));
    let (gid_param, clamped_decl) = gid_decl(out_n);
    params.push(gid_param);
    src.push_str(&params.join(",\n"));
    src.push_str("\n) {\n");
    src.push_str(&clamped_decl);

    // Per-lane base offsets from the non-reduced coordinates of the
    // flattened output index.
    let mut base_offsets: Vec<String> = Vec::with_capacity(num_inputs);
    for strides in lane_strides {
        let mut terms: Vec<String> = Vec::new();
        let mut o = 0;
        for d in 0..rank {
            if dims.contains(&d) {
                continue;
            }
            let out_d = if keepdims { d } else { o };
            o += 1;
            if out_shape[out_d] == 1 || strides[d] == 0 {
                continue;
            }
            let coord = if out_d == out_shape.len() - 1 {
                format!("(clamped % {})", out_shape[out_d])
            } else {
                format!("((clamped / {}) % {})", contig_out[out_d], out_shape[out_d])
            };
            if strides[d] == 1 {
                terms.push(coord);
            } else {
                terms.push(format!("({coord} * {})", strides[d]));
            }
        }
        base_offsets.push(if terms.is_empty() {
            "0u".to_string()
        } else {
            terms.join(" + ")
        });
    }

    let red_sizes: Vec<usize> = dims.iter().map(|&d| in_shape[d]).collect();
    let red_contig = contiguous_strides(&red_sizes);
    let extent: usize = red_sizes.iter().product();
    let init = match op {
        ReduceOp::Sum | ReduceOp::Mean => "0.0f",
        ReduceOp::Max => "(-INFINITY)",
        ReduceOp::Prod => "1.0f",
        ReduceOp::Min => "(INFINITY)",
    };
    src.push_str(&format!("    float acc = {init};\n"));
    let rt = idx_ty(extent);
    let (rz, ru) = if rt == "uint" {
        ("0u", "u")
    } else {
        ("0ul", "ul")
    };
    src.push_str(&format!(
        "    for ({rt} r = {rz}; r < {extent}{ru}; ++r) {{\n"
    ));
    let mut lane_offsets = base_offsets.clone();
    for (j, &d) in dims.iter().enumerate() {
        if red_sizes[j] == 1 {
            continue;
        }
        let rcoord = if j == dims.len() - 1 {
            "(r % {RS})".replace("{RS}", &red_sizes[j].to_string())
        } else {
            format!("((r / {}) % {})", red_contig[j], red_sizes[j])
        };
        for (k, strides) in lane_strides.iter().enumerate() {
            if strides[d] == 0 {
                continue;
            }
            let term = if strides[d] == 1 {
                rcoord.clone()
            } else {
                format!("({rcoord} * {})", strides[d])
            };
            lane_offsets[k] = format!("{} + {term}", lane_offsets[k]);
        }
    }
    let lane =
        |k: u32| -> String { load_expr(format!("in{}[{}]", k, lane_offsets[k as usize]), dtype) };
    let mut body = String::new();
    let mut next = 0usize;
    let mut memo = std::collections::HashMap::new();
    let v = emit_expr_ssa(expr, &lane, "        ", &mut body, &mut next, &mut memo);
    let fold = match op {
        ReduceOp::Sum | ReduceOp::Mean => format!("acc += {v};"),
        ReduceOp::Max => format!("acc = fmax(acc, {v});"),
        ReduceOp::Prod => format!("acc *= {v};"),
        ReduceOp::Min => format!("acc = fmin(acc, {v});"),
    };
    src.push_str(&body);
    src.push_str(&format!("        {fold}\n"));
    src.push_str("    }\n");
    if op == ReduceOp::Mean {
        src.push_str(&format!("    acc /= {extent}.0f;\n"));
    }
    src.push_str(&format!(
        "    out[clamped] = {};\n",
        store_expr("acc", dtype)
    ));
    src.push_str("}\n");
    src
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elementwise_source_shape() {
        let exprs = vec![Expr::Add(
            Box::new(Expr::Input(0)),
            Box::new(Expr::Mul(
                Box::new(Expr::Input(1)),
                Box::new(Expr::Const(2.0f64.to_bits())),
            )),
        )];
        let src = emit_elementwise(
            &exprs,
            &[vec![3, 1], vec![0, 1]],
            &[2, 3],
            6,
            0,
            "fused_test",
            crate::runtime::dtype::DType::F32,
        );
        assert!(src.contains("kernel void fused_test("));
        assert!(src.contains("in0[clamped]"));
        assert!(src.contains("in1[(clamped % 3)]"));
        assert!(src.contains("out0[clamped] ="));
        let nc = emit_elementwise(
            &exprs,
            &[vec![2, 1], vec![3, 1]],
            &[2, 3],
            6,
            0,
            "fused_nc",
            crate::runtime::dtype::DType::F32,
        );
        assert!(nc.contains("in0[(((clamped / 3) % 2) * 2) + (clamped % 3)]"));
    }

    #[test]
    fn reduce_source_shape() {
        let expr = Expr::Mul(Box::new(Expr::Input(0)), Box::new(Expr::Input(0)));
        let src = emit_reduce(
            ReduceOp::Mean,
            &expr,
            &[vec![3, 1]],
            &[2, 3],
            &[1],
            false,
            &[2],
            "reduce_test",
            crate::runtime::dtype::DType::F32,
        );
        assert!(src.contains("for (uint r = 0u; r < 3u; ++r)"));
        assert!(src.contains(
            "        float t0 = (in0[((clamped % 2) * 3) + (r % 3)] * in0[((clamped % 2) * 3) + (r % 3)]);\n        acc += t0;"
        ));
        assert!(src.contains("acc /= 3.0f;"));
        assert!(src.contains("out[clamped] = acc;"));

        let selected = Expr::Select(
            Box::new(Expr::Input(0)),
            Box::new(Expr::Input(1)),
            Box::new(Expr::Input(2)),
        );
        let selected_src = emit_reduce(
            ReduceOp::Sum,
            &selected,
            &[vec![1], vec![1], vec![1]],
            &[2],
            &[0],
            false,
            &[1],
            "reduce_select_bf16",
            crate::runtime::dtype::DType::BF16,
        );
        assert!(selected_src.contains(
            "        float t0 = (float(in0[0u + (r % 2)]) != 0.0f ? float(in1[0u + (r % 2)]) : float(in2[0u + (r % 2)]));\n        acc += t0;"
        ));
        assert!(selected_src.contains("out[clamped] = bfloat(acc);"));
    }

    #[test]
    fn deep_reduce_source_emission_is_stack_safe() {
        std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                for depth in [50_000usize, 100_000] {
                    let mut expr = crate::fusion::KernelExpr::Input(0);
                    for _ in 0..depth {
                        expr = crate::fusion::KernelExpr::Neg(Box::new(expr));
                    }
                    let src = emit_reduce(
                        ReduceOp::Sum,
                        &expr,
                        &[vec![1]],
                        &[2],
                        &[0],
                        false,
                        &[1],
                        "deep_reduce",
                        crate::runtime::dtype::DType::F32,
                    );
                    assert!(src.contains(&format!("float t{} = (-t{});", depth - 1, depth - 2)));
                    assert!(src.contains(&format!("acc += t{};", depth - 1)));
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn wide_indexing_past_u32() {
        // Past u32::MAX elements the flat index widens: 2-D grid,
        // ulong math — a 5G-element tensor is addressable.
        let big = 5_000_000_000usize;
        let exprs = vec![Expr::Input(0)];
        let src = emit_elementwise(
            &exprs,
            &[vec![1]],
            &[big],
            big,
            0,
            "wide_test",
            crate::runtime::dtype::DType::F32,
        );
        assert!(src.contains("uint2 gid2 [[thread_position_in_grid]]"));
        assert!(src.contains("const ulong gid = ulong(gid2.y) * 1073741824ul + ulong(gid2.x);"));
        assert!(src.contains("const ulong clamped = min(gid, 4999999999ul);"));
        assert!(src.contains("out0[clamped]"));
        // The small path is untouched: uint math, 1-D grid.
        let small = emit_elementwise(
            &exprs,
            &[vec![1]],
            &[6],
            6,
            0,
            "narrow_test",
            crate::runtime::dtype::DType::F32,
        );
        assert!(small.contains("uint gid [[thread_position_in_grid]]"));
        assert!(small.contains("const uint clamped = min(gid, 5u);"));
        let red = emit_reduce(
            ReduceOp::Sum,
            &Expr::Input(0),
            &[vec![1]],
            &[big],
            &[0],
            false,
            &[1],
            "wide_reduce",
            crate::runtime::dtype::DType::F32,
        );
        assert!(red.contains("for (ulong r = 0ul; r < 5000000000ul; ++r)"));
    }
}
