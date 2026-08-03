use crate::fusion::{Expr, ReduceOp};

pub const BLOCK: usize = 256;

fn f32_lit(v: f64) -> String {
    let v = v as f32;
    if v.is_infinite() {
        return if v > 0.0 { "(INFINITY)".to_string() } else { "(-INFINITY)".to_string() };
    }
    if v.is_nan() {
        return "(NAN)".to_string();
    }
    let s = format!("{v:e}");
    let s = s.replace('e', "e");
    format!("({s}f)")
}

fn emit_expr(e: &Expr, lane: &dyn Fn(u32) -> String, num_inputs: usize) -> String {
    match e {
        Expr::Input(k) => lane(*k),
        Expr::Scalar(k) => format!("scs[{}]", *k as usize),
        Expr::Const(bits) => f32_lit(f64::from_bits(*bits)),
        Expr::Add(a, b) => format!("({} + {})", emit_expr(a, lane, num_inputs), emit_expr(b, lane, num_inputs)),
        Expr::Sub(a, b) => format!("({} - {})", emit_expr(a, lane, num_inputs), emit_expr(b, lane, num_inputs)),
        Expr::Mul(a, b) => format!("({} * {})", emit_expr(a, lane, num_inputs), emit_expr(b, lane, num_inputs)),
        Expr::Div(a, b) => format!("({} / {})", emit_expr(a, lane, num_inputs), emit_expr(b, lane, num_inputs)),
        Expr::Min(a, b) => format!("fmin({}, {})", emit_expr(a, lane, num_inputs), emit_expr(b, lane, num_inputs)),
        Expr::Max(a, b) => format!("fmax({}, {})", emit_expr(a, lane, num_inputs), emit_expr(b, lane, num_inputs)),
        Expr::Lt(a, b) => format!("({} < {} ? 1.0f : 0.0f)", emit_expr(a, lane, num_inputs), emit_expr(b, lane, num_inputs)),
        Expr::Le(a, b) => format!("({} <= {} ? 1.0f : 0.0f)", emit_expr(a, lane, num_inputs), emit_expr(b, lane, num_inputs)),
        Expr::Gt(a, b) => format!("({} > {} ? 1.0f : 0.0f)", emit_expr(a, lane, num_inputs), emit_expr(b, lane, num_inputs)),
        Expr::Ge(a, b) => format!("({} >= {} ? 1.0f : 0.0f)", emit_expr(a, lane, num_inputs), emit_expr(b, lane, num_inputs)),
        Expr::Eq(a, b) => format!("({} == {} ? 1.0f : 0.0f)", emit_expr(a, lane, num_inputs), emit_expr(b, lane, num_inputs)),
        Expr::Ne(a, b) => format!("({} != {} ? 1.0f : 0.0f)", emit_expr(a, lane, num_inputs), emit_expr(b, lane, num_inputs)),
        Expr::Select(c, a, b) => format!(
            "({} != 0.0f ? {} : {})",
            emit_expr(c, lane, num_inputs),
            emit_expr(a, lane, num_inputs),
            emit_expr(b, lane, num_inputs)
        ),
        Expr::Neg(a) => format!("(-{})", emit_expr(a, lane, num_inputs)),
        Expr::Sqrt(a) => format!("sqrt({})", emit_expr(a, lane, num_inputs)),
        Expr::Exp(a) => format!("exp({})", emit_expr(a, lane, num_inputs)),
        Expr::Sin(a) => format!("sin({})", emit_expr(a, lane, num_inputs)),
        Expr::Cos(a) => format!("cos({})", emit_expr(a, lane, num_inputs)),
        Expr::Tanh(a) => format!("tanh({})", emit_expr(a, lane, num_inputs)),
        Expr::Abs(a) => format!("fabs({})", emit_expr(a, lane, num_inputs)),
        Expr::Log(a) => format!("log({})", emit_expr(a, lane, num_inputs)),
        Expr::Floor(a) => format!("floor({})", emit_expr(a, lane, num_inputs)),
        Expr::Ceil(a) => format!("ceil({})", emit_expr(a, lane, num_inputs)),
        Expr::Round(a) => format!("rint({})", emit_expr(a, lane, num_inputs)),
        Expr::Powf(a, e) => format!("pow({}, {})", emit_expr(a, lane, num_inputs), f32_lit(f64::from_bits(*e))),
        Expr::Erf(a) => format!("erf_as({})", emit_expr(a, lane, num_inputs)),
    }
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

const HEADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float erf_as(float x) {
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float p = 1.061405429f * t;
    p = p * t - 1.453152027f;
    p = p * t + 1.421413741f;
    p = p * t - 0.284496736f;
    p = p * t + 0.254829592f;
    float tail = 1.0f - p * t * exp(-x * x);
    float sign = x / fmax(ax, 1e-30f);
    return sign * tail;
}
"#;

pub fn emit_elementwise(
    exprs: &[Expr],
    lane_strides: &[Vec<usize>],
    out_shape: &[usize],
    n: usize,
    num_scalars: usize,
    name: &str,
) -> String {
    let num_inputs = lane_strides.len();
    let num_outputs = exprs.len();
    let mut src = String::from(HEADER);
    src.push_str(&format!("kernel void {name}(\n"));
    let mut idx = 0usize;
    let mut params: Vec<String> = Vec::new();
    for k in 0..num_inputs {
        params.push(format!("    device const float* in{k} [[buffer({idx})]]"));
        idx += 1;
    }
    if num_scalars > 0 {
        params.push(format!("    device const float* scs [[buffer({idx})]]"));
        idx += 1;
    }
    for j in 0..num_outputs {
        params.push(format!("    device float* out{j} [[buffer({idx})]]"));
        idx += 1;
    }
    params.push("    uint gid [[thread_position_in_grid]]".to_string());
    src.push_str(&params.join(",\n"));
    src.push_str("\n) {\n");
    src.push_str(&format!("    const uint clamped = min(gid, {}u);\n", n.saturating_sub(1)));
    let lanes: Vec<String> = lane_strides
        .iter()
        .map(|s| lane_offset_expr(s, out_shape, "clamped"))
        .collect();
    let lane = |k: u32| -> String { format!("in{}[{}]", k, lanes[k as usize]) };
    for (j, expr) in exprs.iter().enumerate() {
        let v = emit_expr(expr, &lane, num_inputs);
        src.push_str(&format!("    out{j}[clamped] = {v};\n"));
    }
    src.push_str("}\n");
    src
}

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
) -> String {
    let num_inputs = lane_strides.len();
    let rank = in_shape.len();
    let out_n: usize = out_shape.iter().product();
    let contig_out = contiguous_strides(out_shape);
    let mut src = String::from(HEADER);
    src.push_str(&format!("kernel void {name}(\n"));
    let mut params: Vec<String> = Vec::new();
    let mut idx = 0usize;
    for k in 0..num_inputs {
        params.push(format!("    device const float* in{k} [[buffer({idx})]]"));
        idx += 1;
    }
    params.push(format!("    device float* out [[buffer({idx})]]"));
    params.push("    uint gid [[thread_position_in_grid]]".to_string());
    src.push_str(&params.join(",\n"));
    src.push_str("\n) {\n");
    src.push_str(&format!("    const uint clamped = min(gid, {}u);\n", out_n.saturating_sub(1)));

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
        base_offsets.push(if terms.is_empty() { "0u".to_string() } else { terms.join(" + ") });
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
    src.push_str(&format!("    for (uint r = 0u; r < {}u; ++r) {{\n", extent));
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
    let lane = |k: u32| -> String { format!("in{}[{}]", k, lane_offsets[k as usize]) };
    let v = emit_expr(expr, &lane, num_inputs);
    let fold = match op {
        ReduceOp::Sum | ReduceOp::Mean => format!("acc += {v};"),
        ReduceOp::Max => format!("acc = fmax(acc, {v});"),
        ReduceOp::Prod => format!("acc *= {v};"),
        ReduceOp::Min => format!("acc = fmin(acc, {v});"),
    };
    src.push_str(&format!("        {fold}\n"));
    src.push_str("    }\n");
    if op == ReduceOp::Mean {
        src.push_str(&format!("    acc /= {extent}.0f;\n"));
    }
    src.push_str("    out[clamped] = acc;\n");
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
            Box::new(Expr::Mul(Box::new(Expr::Input(1)), Box::new(Expr::Const(2.0f64.to_bits())))),
        )];
        let src = emit_elementwise(&exprs, &[vec![3, 1], vec![0, 1]], &[2, 3], 6, 0, "fused_test");
        assert!(src.contains("kernel void fused_test("));
        assert!(src.contains("in0[clamped]"));
        assert!(src.contains("in1[(clamped % 3)]"));
        assert!(src.contains("out0[clamped] ="));
        let nc = emit_elementwise(&exprs, &[vec![2, 1], vec![3, 1]], &[2, 3], 6, 0, "fused_nc");
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
        );
        assert!(src.contains("for (uint r = 0u; r < 3u; ++r)"));
        assert!(src.contains("acc /= 3.0f;"));
        assert!(src.contains("out[clamped] = acc;"));
    }
}
