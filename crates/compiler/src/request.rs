//! Compile requests, options, and invocation contracts.
//!
//! A [`ProgramRequest`] combines semantic graph roots with caller-visible
//! bindings, runtime scalar and value declarations, output signatures, and
//! [`CompileOptions`]. Options include the environment snapshot and form part
//! of program identity. Requests with different options cannot share a cached
//! executable.
//!
//! Preparation returns a [`PreparedProgram`] with the program's single
//! [`GraphIndex`], resolved [`ProgramSignature`], and optional state cursor.
//! Later phases use this contract without deriving it again.

use crate::schedule::GraphIndex;
use effect_torch_graph::Node;
use effect_torch_runtime::{
    BindingAliasing, BindingDecl, BindingLayoutPolicy, DeviceId, InvocationSignature, Layout,
    LayoutConstraint, OutputSignature, Placement, ProgramSignature, RuntimeValueDecl, ScalarDecl,
    ScalarType,
};
use std::sync::Arc;

/// Default compute-elementwise chunk size of 2^26 elements. This amortizes
/// chunked launches and limits intermediate allocations.
pub const DEFAULT_CE_CHUNK_SIZE: usize = 1 << 26;

/// The semantic node type this compiler consumes.
pub type ProgramNode = Node;

/// Caller-authorized inference assumptions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct InferenceOptions {
    pub constant_weights: bool,
}

/// An immutable snapshot of the compiler A/B environment switches.
///
/// `CompileOptions` carries the snapshot so later phases do not read process
/// state that could change the lowered schedule or memory plan.
/// Every field maps to an `EFFECT_TORCH_*` variable documented by
/// [`EnvironmentOptions::snapshot`]. Unset variables select the defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnvironmentOptions {
    /// Elementwise and reduce fusion. `EFFECT_TORCH_NO_FUSION` disables it.
    pub fusion: bool,
    /// GEMM epilogue regions. `EFFECT_TORCH_NO_EPILOGUE` disables them.
    pub gemm_epilogues: bool,
    /// Multi-output elementwise merges. `EFFECT_TORCH_NO_MULTI_FUSION` disables them.
    pub multi_output_fusion: bool,
    /// Grouped AdamW regions. `EFFECT_TORCH_OPT_GROUPS` enables them.
    pub optimizer_groups: bool,
    /// Verbose fusion decisions on stderr, set by `EFFECT_TORCH_FUSION_DEBUG`.
    pub fusion_debug: bool,
    /// Compute-elementwise chunk size from `EFFECT_TORCH_CE_CHUNK_SIZE`.
    pub ce_chunk_size: usize,
    /// Metal private-storage intermediates, set by `EFFECT_TORCH_PRIVATE_INTERMEDIATES`.
    pub metal_private_intermediates: bool,
    /// Metal MMA lowering. `EFFECT_TORCH_NO_MMA` disables it.
    pub metal_mma: bool,
}

impl Default for EnvironmentOptions {
    fn default() -> Self {
        Self {
            fusion: true,
            gemm_epilogues: true,
            multi_output_fusion: true,
            optimizer_groups: false,
            fusion_debug: false,
            ce_chunk_size: DEFAULT_CE_CHUNK_SIZE,
            metal_private_intermediates: false,
            metal_mma: true,
        }
    }
}

impl EnvironmentOptions {
    /// Reads the process environment once. Malformed or non-positive numeric
    /// values use defaults instead of failing compilation.
    pub fn snapshot() -> Self {
        let positive_usize = |name: &str, default: usize| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(default)
        };
        Self {
            fusion: std::env::var_os("EFFECT_TORCH_NO_FUSION").is_none(),
            gemm_epilogues: std::env::var_os("EFFECT_TORCH_NO_EPILOGUE").is_none(),
            multi_output_fusion: std::env::var_os("EFFECT_TORCH_NO_MULTI_FUSION").is_none(),
            optimizer_groups: std::env::var_os("EFFECT_TORCH_OPT_GROUPS").is_some(),
            fusion_debug: std::env::var_os("EFFECT_TORCH_FUSION_DEBUG").is_some(),
            ce_chunk_size: positive_usize("EFFECT_TORCH_CE_CHUNK_SIZE", DEFAULT_CE_CHUNK_SIZE),
            metal_private_intermediates: std::env::var_os("EFFECT_TORCH_PRIVATE_INTERMEDIATES")
                .is_some(),
            metal_mma: std::env::var_os("EFFECT_TORCH_NO_MMA").is_none(),
        }
    }
}

/// Compile-time choices for one program. Options implement `Eq + Hash` because
/// they form part of the compiled artifact's cache identity. Changing any
/// field requires a new compile.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompileOptions {
    /// Master switch for region selection. When false, the plan covers no
    /// regions, and every node lowers independently.
    pub optimize: bool,
    /// Caller-authorized inference-only assumptions.
    pub inference: Option<InferenceOptions>,
    /// Frozen environment switches.
    pub environment: EnvironmentOptions,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            optimize: true,
            inference: None,
            environment: EnvironmentOptions::default(),
        }
    }
}

impl CompileOptions {
    /// Builds default options from the current process environment.
    pub fn from_environment() -> Self {
        Self {
            environment: EnvironmentOptions::snapshot(),
            ..Self::default()
        }
    }

    /// Whether the caller authorized treating weights as constants.
    pub fn constant_weights(&self) -> bool {
        self.inference
            .as_ref()
            .is_some_and(|inference| inference.constant_weights)
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.environment.ce_chunk_size == 0 {
            return Err("compile: CE chunk size must be positive".to_string());
        }
        Ok(())
    }
}

/// A semantic graph and its compile-time invocation contract.
#[derive(Clone)]
pub struct ProgramRequest {
    pub roots: Vec<Arc<ProgramNode>>,
    pub bindings: Vec<BindingDecl>,
    pub invocation: InvocationSignature,
    pub options: CompileOptions,
    pub state_cursor: Option<StateCursorSlot>,
    derive_contract: bool,
}

/// Declares the input slot for the runtime state cursor, or decode position,
/// and whether it is a scalar or an `i64 [batch]` tensor. The cursor is
/// internal runtime metadata, not a caller-visible binding or scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateCursorSlot {
    pub slot: u32,
    pub tensor: bool,
}

impl StateCursorSlot {
    pub const fn new(slot: u32, tensor: bool) -> Self {
        Self { slot, tensor }
    }
}

impl ProgramRequest {
    /// Creates a request with a caller-supplied binding and invocation contract.
    /// Preparation uses the signature unchanged.
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
            state_cursor: None,
            derive_contract: false,
        }
    }

    /// Creates a native request. Preparation derives its binding and invocation
    /// declarations from the graph index.
    pub fn from_roots(roots: Vec<Arc<ProgramNode>>, options: CompileOptions) -> Self {
        Self {
            roots,
            bindings: Vec::new(),
            invocation: InvocationSignature::default(),
            options,
            state_cursor: None,
            derive_contract: true,
        }
    }

    pub fn with_state_cursor(mut self, state_cursor: StateCursorSlot) -> Self {
        self.state_cursor = Some(state_cursor);
        self
    }

    /// Validates the request, builds the shared graph index, and resolves
    /// the invocation contract. See `driver::prepare_program`.
    pub fn prepare(self) -> Result<PreparedProgram, String> {
        PreparedProgram::from_request(self)
    }
}

/// A prepared compile request with graph analysis for later phases.
///
/// Preparation guarantees non-empty, single-device `roots`. It builds `index`
/// once and resolves the caller-supplied or graph-derived `signature`.
#[derive(Clone)]
pub struct PreparedProgram {
    pub roots: Box<[Arc<ProgramNode>]>,
    pub index: Arc<GraphIndex>,
    pub signature: ProgramSignature,
    pub options: CompileOptions,
    pub state_cursor: Option<StateCursorSlot>,
    /// Timings recorded during preparation for the graph-index build.
    pub(crate) preparation_phases: Box<[effect_torch_runtime::CompilePhaseTiming]>,
}

impl PreparedProgram {
    pub fn from_request(request: ProgramRequest) -> Result<Self, String> {
        crate::driver::prepare_program(request)
    }

    /// The resolved invocation contract.
    pub fn signature(&self) -> &ProgramSignature {
        &self.signature
    }
}

pub(crate) fn request_parts(
    request: ProgramRequest,
) -> (
    Vec<Arc<ProgramNode>>,
    Vec<BindingDecl>,
    InvocationSignature,
    CompileOptions,
    Option<StateCursorSlot>,
    bool,
) {
    (
        request.roots,
        request.bindings,
        request.invocation,
        request.options,
        request.state_cursor,
        request.derive_contract,
    )
}

/// Maps a semantic device and ordinal to its runtime signature placement.
fn placement(device: &effect_torch_graph::Device) -> Placement {
    match device {
        effect_torch_graph::Device::Cpu(ordinal) => {
            Placement::new(DeviceId::new(format!("cpu:{ordinal}")))
        }
        effect_torch_graph::Device::Metal(ordinal) => {
            Placement::with_memory_space(DeviceId::new(format!("metal:{ordinal}")), "shared")
        }
        effect_torch_graph::Device::Cuda(ordinal) => {
            Placement::new(DeviceId::new(format!("cuda:{ordinal}")))
        }
    }
}

/// Derives native binding and invocation declarations from the graph index.
/// Tensor slots become bindings. CPU tensors use an exact contiguous layout.
/// Device tensors use a zero-offset contiguous layout. Scalar slots become
/// typed invocation scalars named `slot_N`. A state cursor slot becomes
/// internal runtime-value metadata, and its graph declaration must match.
pub(crate) fn derive_graph_signature(
    index: &GraphIndex,
    state_cursor: Option<StateCursorSlot>,
) -> Result<ProgramSignature, String> {
    let mut bindings = Vec::new();
    let mut invocation = InvocationSignature::default();
    for (slot, declaration) in index.slots.iter().enumerate() {
        if state_cursor.is_some_and(|cursor| cursor.slot as usize == slot) {
            let cursor = state_cursor.expect("state cursor was matched");
            if declaration.dtype != effect_torch_runtime::DType::I64
                || declaration.scalar == cursor.tensor
                || (cursor.tensor && (declaration.shape.len() != 1 || declaration.shape[0] == 0))
            {
                return Err(format!(
                    "compile: state cursor slot {slot} has an invalid scalar/tensor signature"
                ));
            }
            invocation.runtime_values.push(if cursor.tensor {
                RuntimeValueDecl::u32_array("state_cursors", declaration.shape[0])
            } else {
                RuntimeValueDecl::u64("state_cursor", 0, u32::MAX as u64)
            });
            continue;
        }
        if declaration.scalar {
            let scalar_type = match declaration.dtype {
                effect_torch_runtime::DType::U8 | effect_torch_runtime::DType::U32 => {
                    ScalarType::U32
                }
                effect_torch_runtime::DType::I64 => ScalarType::I64,
                effect_torch_runtime::DType::F64 => ScalarType::F64,
                effect_torch_runtime::DType::F16
                | effect_torch_runtime::DType::BF16
                | effect_torch_runtime::DType::F32 => ScalarType::F32,
            };
            invocation.scalars.push(ScalarDecl {
                name: format!("slot_{slot}"),
                scalar_type,
            });
        } else {
            let layout = if declaration.device.is_cpu() {
                BindingLayoutPolicy::Require(LayoutConstraint::Exact(Layout::contiguous(
                    declaration.shape.clone(),
                )))
            } else {
                BindingLayoutPolicy::Require(LayoutConstraint::ZeroOffsetContiguous)
            };
            bindings.push(BindingDecl {
                shape: declaration.shape.clone(),
                dtype: declaration.dtype,
                placement: placement(&declaration.device),
                layout,
                aliasing: BindingAliasing::MayAlias,
            });
        }
    }
    Ok(signature_with_contract(index, bindings, invocation))
}

/// Adds output declarations from caller-ordered roots to a signature and
/// preserves duplicate roots.
pub(crate) fn signature_with_contract(
    index: &GraphIndex,
    bindings: Vec<BindingDecl>,
    invocation: InvocationSignature,
) -> ProgramSignature {
    let outputs = index
        .roots
        .iter()
        .map(|root| {
            let node = index
                .node(*root)
                .expect("GraphIndex roots always reference indexed nodes");
            OutputSignature {
                shape: node.shape.clone(),
                dtype: node.dtype,
                placement: placement(&node.device),
            }
        })
        .collect();
    ProgramSignature {
        bindings,
        invocation,
        outputs,
    }
}

impl TryFrom<ProgramRequest> for PreparedProgram {
    type Error = String;

    fn try_from(request: ProgramRequest) -> Result<Self, Self::Error> {
        Self::from_request(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use effect_torch_graph::{Device, NodeKind};
    use effect_torch_runtime::DType;

    fn constant(device: Device) -> Arc<ProgramNode> {
        Node::new(NodeKind::Zeros {
            shape: vec![2],
            dtype: DType::F32,
            device,
        })
        .unwrap()
    }

    #[test]
    fn program_request_prepares_one_shared_index_and_retains_its_contract() {
        let root = constant(Device::Cpu(0));
        let invocation = InvocationSignature::default();
        let options = CompileOptions {
            optimize: false,
            ..CompileOptions::default()
        };
        let prepared = ProgramRequest::new(
            vec![root.clone()],
            Vec::new(),
            invocation.clone(),
            options.clone(),
        )
        .prepare()
        .unwrap();

        assert_eq!(prepared.roots.len(), 1);
        assert!(Arc::ptr_eq(&prepared.roots[0], &root));
        assert!(Arc::ptr_eq(&prepared.index.order[0], &root));
        assert_eq!(prepared.index.work.graph_index_builds, 1);
        assert!(prepared.signature.bindings.is_empty());
        assert_eq!(prepared.signature.invocation, invocation);
        assert_eq!(prepared.signature.outputs.len(), 1);
        assert_eq!(prepared.options, options);
    }

    #[test]
    fn placement_preserves_device_ordinals() {
        for (device, device_id, memory_space) in [
            (Device::Cpu(2), "cpu:2", None),
            (Device::Metal(3), "metal:3", Some("shared")),
            (Device::Cuda(4), "cuda:4", None),
        ] {
            let placement = placement(&device);
            assert_eq!(placement.device().as_str(), device_id);
            assert_eq!(placement.memory_space(), memory_space);
        }
    }

    #[test]
    fn native_signature_derives_declarations_and_preserves_caller_output_order() {
        for (device, device_id, memory_space) in [
            (Device::Cpu(0), "cpu:0", None),
            (Device::Metal(0), "metal:0", Some("shared")),
        ] {
            let input = Node::new(NodeKind::Input {
                slot: 0,
                shape: vec![2],
                dtype: DType::F32,
                device: device.clone(),
            })
            .unwrap();
            let scalar = Node::new(NodeKind::ScalarInput {
                slot: 1,
                dtype: DType::F32,
                device,
            })
            .unwrap();

            let prepared = ProgramRequest::from_roots(
                vec![input.clone(), scalar, input],
                CompileOptions::default(),
            )
            .prepare()
            .unwrap();
            let signature = prepared.signature();

            assert_eq!(signature.bindings.len(), 1);
            assert_eq!(signature.bindings[0].shape, [2]);
            assert_eq!(signature.bindings[0].placement.device().as_str(), device_id);
            assert_eq!(signature.bindings[0].placement.memory_space(), memory_space);
            if device_id == "cpu:0" {
                assert_eq!(
                    signature.bindings[0].layout,
                    BindingLayoutPolicy::Require(LayoutConstraint::Exact(Layout::contiguous(
                        vec![2]
                    )))
                );
            } else {
                assert_eq!(
                    signature.bindings[0].layout,
                    BindingLayoutPolicy::Require(LayoutConstraint::ZeroOffsetContiguous)
                );
            }
            assert_eq!(signature.invocation.scalars.len(), 1);
            assert_eq!(signature.invocation.scalars[0].name, "slot_1");
            assert_eq!(signature.invocation.scalars[0].scalar_type, ScalarType::F32);
            assert_eq!(
                signature
                    .outputs
                    .iter()
                    .map(|output| (output.shape.as_slice(), output.dtype))
                    .collect::<Vec<_>>(),
                [
                    (&[2][..], DType::F32),
                    (&[][..], DType::F32),
                    (&[2][..], DType::F32)
                ]
            );
            assert!(signature.outputs.iter().all(|output| {
                output.placement.device().as_str() == device_id
                    && output.placement.memory_space() == memory_space
            }));
        }
    }

    #[test]
    fn stateless_signature_preserves_tensor_and_scalar_slot_order() {
        let scalar_0 = Node::new(NodeKind::ScalarInput {
            slot: 0,
            dtype: DType::F32,
            device: Device::Cpu(0),
        })
        .unwrap();
        let tensor_1 = Node::new(NodeKind::Input {
            slot: 1,
            shape: vec![1],
            dtype: DType::I64,
            device: Device::Cpu(0),
        })
        .unwrap();
        let scalar_2 = Node::new(NodeKind::ScalarInput {
            slot: 2,
            dtype: DType::U32,
            device: Device::Cpu(0),
        })
        .unwrap();
        let tensor_3 = Node::new(NodeKind::Input {
            slot: 3,
            shape: vec![3],
            dtype: DType::F32,
            device: Device::Cpu(0),
        })
        .unwrap();
        let prepared = ProgramRequest::from_roots(
            vec![tensor_3.clone(), scalar_0, tensor_1, scalar_2, tensor_3],
            CompileOptions::default(),
        )
        .prepare()
        .unwrap();

        assert_eq!(
            prepared
                .signature
                .bindings
                .iter()
                .map(|binding| (binding.shape.as_slice(), binding.dtype))
                .collect::<Vec<_>>(),
            [(&[1][..], DType::I64), (&[3][..], DType::F32)]
        );
        assert_eq!(
            prepared
                .signature
                .invocation
                .scalars
                .iter()
                .map(|scalar| (scalar.name.as_str(), scalar.scalar_type))
                .collect::<Vec<_>>(),
            [("slot_0", ScalarType::F32), ("slot_2", ScalarType::U32)]
        );
        assert_eq!(prepared.signature.outputs.len(), 5);
        assert_eq!(prepared.signature.outputs[0], prepared.signature.outputs[4]);
    }

    #[test]
    fn state_cursor_is_internal_runtime_metadata_not_a_caller_argument() {
        for tensor in [false, true] {
            let caller = Node::new(NodeKind::Input {
                slot: 0,
                shape: vec![2],
                dtype: DType::F32,
                device: Device::Cpu(0),
            })
            .unwrap();
            let cursor = if tensor {
                Node::new(NodeKind::Input {
                    slot: 1,
                    shape: vec![3],
                    dtype: DType::I64,
                    device: Device::Cpu(0),
                })
            } else {
                Node::new(NodeKind::ScalarInput {
                    slot: 1,
                    dtype: DType::I64,
                    device: Device::Cpu(0),
                })
            }
            .unwrap();
            let prepared =
                ProgramRequest::from_roots(vec![caller, cursor], CompileOptions::default())
                    .with_state_cursor(StateCursorSlot::new(1, tensor))
                    .prepare()
                    .unwrap();

            assert_eq!(prepared.state_cursor, Some(StateCursorSlot::new(1, tensor)));
            assert_eq!(prepared.signature.bindings.len(), 1);
            assert!(prepared.signature.invocation.scalars.is_empty());
            assert_eq!(prepared.signature.invocation.runtime_values.len(), 1);
            assert_eq!(
                prepared.signature.invocation.runtime_values[0].kind,
                if tensor {
                    effect_torch_runtime::RuntimeValueKind::U32Array {
                        max_len: 3,
                        element_min: 0,
                        element_max: u32::MAX,
                    }
                } else {
                    effect_torch_runtime::RuntimeValueKind::U64 {
                        min: 0,
                        max: u32::MAX as u64,
                    }
                }
            );
        }
    }

    #[test]
    fn environment_compile_choices_are_part_of_option_identity() {
        let baseline = CompileOptions::default();
        let mut changed = baseline.clone();
        changed.environment.ce_chunk_size += 1;
        assert_ne!(baseline, changed);

        let mut changed = baseline.clone();
        changed.environment.metal_private_intermediates = true;
        changed.environment.metal_mma = false;
        assert_ne!(baseline, changed);

        let mut changed = baseline.clone();
        changed.inference = Some(InferenceOptions {
            constant_weights: true,
        });
        assert_ne!(baseline, changed);
        assert!(changed.constant_weights());
    }

    #[test]
    fn preparation_validates_roots_and_devices() {
        let empty = ProgramRequest::new(
            Vec::new(),
            Vec::new(),
            InvocationSignature::default(),
            CompileOptions::default(),
        );
        assert_eq!(
            empty.prepare().err().unwrap(),
            "compile: expected at least one root"
        );

        let mixed = ProgramRequest::new(
            vec![constant(Device::Cpu(0)), constant(Device::Metal(0))],
            Vec::new(),
            InvocationSignature::default(),
            CompileOptions::default(),
        );
        assert_eq!(
            mixed.prepare().err().unwrap(),
            "compile: graph contains multiple devices"
        );
    }
}
