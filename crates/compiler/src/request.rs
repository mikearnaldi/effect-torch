use crate::schedule::GraphIndex;
use effect_torch_graph::Node;
use effect_torch_runtime::{
    BindingAliasing, BindingDecl, BindingLayoutPolicy, DeviceId, InvocationSignature, Layout,
    LayoutConstraint, OutputSignature, Placement, ProgramSignature, RuntimeValueDecl, ScalarDecl,
    ScalarType,
};
use std::sync::Arc;

pub const DEFAULT_CE_CHUNK_SIZE: usize = 1 << 26;

pub type ProgramNode = Node;

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
    pub ce_chunk_size: usize,
    pub metal_private_intermediates: bool,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompileOptions {
    pub optimize: bool,
    pub inference: Option<InferenceOptions>,
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
    pub fn from_environment() -> Self {
        Self {
            environment: EnvironmentOptions::snapshot(),
            ..Self::default()
        }
    }

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

/// A semantic graph and its complete compile-time invocation contract.
#[derive(Clone)]
pub struct ProgramRequest {
    pub roots: Vec<Arc<ProgramNode>>,
    pub bindings: Vec<BindingDecl>,
    pub invocation: InvocationSignature,
    pub options: CompileOptions,
    pub state_cursor: Option<StateCursorSlot>,
    derive_contract: bool,
}

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

    /// Creates a native request whose binding and invocation declarations are
    /// derived from the graph index during preparation.
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

    pub fn prepare(self) -> Result<PreparedProgram, String> {
        PreparedProgram::from_request(self)
    }
}

/// A validated compile request and the graph analysis shared by later phases.
#[derive(Clone)]
pub struct PreparedProgram {
    pub roots: Box<[Arc<ProgramNode>]>,
    pub index: Arc<GraphIndex>,
    pub signature: ProgramSignature,
    pub options: CompileOptions,
    pub state_cursor: Option<StateCursorSlot>,
    pub(crate) preparation_phases: Box<[effect_torch_runtime::CompilePhaseTiming]>,
}

impl PreparedProgram {
    pub fn from_request(request: ProgramRequest) -> Result<Self, String> {
        crate::driver::prepare_program(request)
    }

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

fn placement(device: &effect_torch_graph::Device) -> Placement {
    match device {
        effect_torch_graph::Device::Cpu => Placement::new(DeviceId::new("cpu:0")),
        effect_torch_graph::Device::Metal => {
            Placement::with_memory_space(DeviceId::new("metal:0"), "shared")
        }
    }
}

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
        let root = constant(Device::Cpu);
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
    fn native_signature_derives_declarations_and_preserves_caller_output_order() {
        for (device, device_id, memory_space) in [
            (Device::Cpu, "cpu:0", None),
            (Device::Metal, "metal:0", Some("shared")),
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
            device: Device::Cpu,
        })
        .unwrap();
        let tensor_1 = Node::new(NodeKind::Input {
            slot: 1,
            shape: vec![1],
            dtype: DType::I64,
            device: Device::Cpu,
        })
        .unwrap();
        let scalar_2 = Node::new(NodeKind::ScalarInput {
            slot: 2,
            dtype: DType::U32,
            device: Device::Cpu,
        })
        .unwrap();
        let tensor_3 = Node::new(NodeKind::Input {
            slot: 3,
            shape: vec![3],
            dtype: DType::F32,
            device: Device::Cpu,
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
                device: Device::Cpu,
            })
            .unwrap();
            let cursor = if tensor {
                Node::new(NodeKind::Input {
                    slot: 1,
                    shape: vec![3],
                    dtype: DType::I64,
                    device: Device::Cpu,
                })
            } else {
                Node::new(NodeKind::ScalarInput {
                    slot: 1,
                    dtype: DType::I64,
                    device: Device::Cpu,
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
            vec![constant(Device::Cpu), constant(Device::Metal)],
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
