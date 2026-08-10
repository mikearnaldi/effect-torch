use crate::Expr;
use effect_torch_graph::Node;
use effect_torch_runtime::{BindingDecl, InvocationSignature};
use std::sync::Arc;

pub type ProgramNode = Node<Expr>;

/// Precision choices that are part of compilation and executable cache keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PrecisionPolicy {
    #[default]
    Strict,
    AllowReducedPrecision,
}

/// Inference-only assumptions explicitly authorized by the caller.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct InferenceOptions {
    pub constant_weights: bool,
}

/// One immutable read of all compiler A/B environment switches.
///
/// The snapshot is carried by `CompileOptions`, so later phases never re-read
/// process state that could change the lowered schedule or memory plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnvironmentOptions {
    pub fusion: bool,
    pub gemm_epilogues: bool,
    pub multi_output_fusion: bool,
    pub optimizer_groups: bool,
    pub fusion_debug: bool,
}

impl Default for EnvironmentOptions {
    fn default() -> Self {
        Self {
            fusion: true,
            gemm_epilogues: true,
            multi_output_fusion: true,
            optimizer_groups: false,
            fusion_debug: false,
        }
    }
}

impl EnvironmentOptions {
    pub fn snapshot() -> Self {
        Self {
            fusion: std::env::var_os("EFFECT_TORCH_NO_FUSION").is_none(),
            gemm_epilogues: std::env::var_os("EFFECT_TORCH_NO_EPILOGUE").is_none(),
            multi_output_fusion: std::env::var_os("EFFECT_TORCH_NO_MULTI_FUSION").is_none(),
            optimizer_groups: std::env::var_os("EFFECT_TORCH_OPT_GROUPS").is_some(),
            fusion_debug: std::env::var_os("EFFECT_TORCH_FUSION_DEBUG").is_some(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompileOptions {
    pub optimize: bool,
    pub precision: PrecisionPolicy,
    pub inference: Option<InferenceOptions>,
    pub environment: EnvironmentOptions,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            optimize: true,
            precision: PrecisionPolicy::Strict,
            inference: None,
            environment: EnvironmentOptions::default(),
        }
    }
}

impl CompileOptions {
    pub fn from_environment() -> Self {
        Self {
            environment: EnvironmentOptions::snapshot(),
            ..Self::default()
        }
    }
}

/// A semantic graph and its complete compile-time invocation contract.
#[derive(Clone)]
pub struct ProgramRequest {
    pub roots: Vec<Arc<ProgramNode>>,
    pub bindings: Vec<BindingDecl>,
    pub invocation: InvocationSignature,
    pub options: CompileOptions,
}

impl ProgramRequest {
    pub fn new(
        roots: Vec<Arc<ProgramNode>>,
        bindings: Vec<BindingDecl>,
        invocation: InvocationSignature,
        options: CompileOptions,
    ) -> Self {
        Self {
            roots,
            bindings,
            invocation,
            options,
        }
    }
}
