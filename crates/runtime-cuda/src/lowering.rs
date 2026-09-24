//! CUDA lowering with independent semantic, instruction, and materialized value IDs.
use crate::cublas::{plan_row_bf16_gemm, Bf16GemmPlan, RowGemmKind, CUBLAS_WORKSPACE_BYTES};
use crate::executable::{
    dtype_code, CudaKernelArgs, CudaStateLayout, Instruction, KernelSpec, StateAccess,
};
use crate::workspace::{CudaMemorySpace, CUDA_STORAGE_ALIGNMENT};
use crate::CudaValue;
use effect_torch_compiler::{
    DenseNodeId, ExecutableDTypePlan, ExecutionRealization, GraphIndex, InstructionEffects,
    LegalizationPlan, LoweredInstruction, LoweredProgram, LoweredValue, LoweringUnit,
    OperandInterpretation, OperandPreparation, OperationDTypeSpec, OutputDecl, ResultCompletion,
    ValueDecl, ValueStorage, ValueUse,
};
use effect_torch_graph::{node_children, Node, NodeKind};
use effect_torch_runtime::{
    DType, InstructionId, Location, SegmentOwnership, StorageClass, StorageMetadata,
    StorageRepresentation, ValueId, ValueSpec,
};

pub(super) type CudaLoweredProgram = LoweredProgram<&'static str, CudaMemorySpace, CudaValueMeta>;
pub(super) const BF16_LINEAR_BIAS_KERNEL: &str = "et_linear_bias_f32";

#[cfg(test)]
#[path = "executable_tests.rs"]
mod tests;

fn selected_result(kind: &NodeKind) -> Option<(&std::sync::Arc<Node>, u8)> {
    match kind {
        NodeKind::AdamWOut { step, index } | NodeKind::SgdOut { step, index } => {
            Some((step, *index))
        }
        NodeKind::SdpaBackwardOut { of, index }
        | NodeKind::KdaBackwardOut { of, index }
        | NodeKind::LayerNormBackwardOut { of, index }
        | NodeKind::ChunkedHeadCeBackwardOut { of, index } => Some((of, *index)),
        _ => None,
    }
}

fn kernel_name(name: &'static str, compute: u32) -> Result<&'static str, String> {
    if name == "et_reduce" && compute >= 4 {
        return Ok("et_reduce_integer");
    }
    let pair = match name {
        "et_reduce" => ("et_reduce_f32", "et_reduce_f64"),
        "et_matmul" => ("et_matmul_f32", "et_matmul_f64"),
        "et_rms_norm" => ("et_rms_norm_f32", "et_rms_norm_f64"),
        "et_cross_entropy" => ("et_cross_entropy_f32", "et_cross_entropy_f64"),
        "et_chunked_head_ce" => ("et_chunked_head_ce_f32", "et_chunked_head_ce_f64"),
        "et_conv" => ("et_conv_f32", "et_conv_f64"),
        "et_linalg" => ("et_linalg_f32", "et_linalg_f64"),
        "et_linear" => ("et_linear_f32", "et_linear_f64"),
        "et_linear_bias" => ("et_linear_bias_f32", "et_linear_bias_f64"),
        "et_layer_norm" => ("et_layer_norm_f32", "et_layer_norm_f64"),
        "et_sdpa" => ("et_sdpa_f32", "et_sdpa_f64"),
        "et_rotary" => ("et_rotary_f32", "et_rotary_f64"),
        "et_short_conv" => ("et_short_conv_f32", "et_short_conv_f64"),
        "et_kda" => ("et_kda_f32", "et_kda_f64"),
        "et_random" => ("et_random_f32", "et_random_f64"),
        _ => return Ok(name),
    };
    match compute {
        0 => Ok(pair.1),
        1 => Ok(pair.0),
        _ => Err(format!(
            "compile: {name} requires planned F32 or F64 operands"
        )),
    }
}

#[derive(Clone, Debug)]
pub(super) struct CudaValueMeta {
    pub(super) decl: ValueDecl<CudaMemorySpace>,
    pub(super) shape: Vec<usize>,
    pub(super) dtype: DType,
    pub(super) storage: StorageMetadata,
}
impl CudaValueMeta {
    pub(super) fn spec(&self) -> ValueSpec<'_> {
        ValueSpec {
            semantic_dtype: self.dtype,
            logical_shape: &self.shape,
            storage: self.storage.as_spec(),
        }
    }
}
impl LoweredValue<CudaMemorySpace> for CudaValueMeta {
    fn value_decl(&self) -> &ValueDecl<CudaMemorySpace> {
        &self.decl
    }
}

pub(super) enum CommandKind {
    Prepare,
    Value(CudaValue),
    Input {
        binding: usize,
    },
    Scalar {
        binding: usize,
    },
    Cursor {
        tensor: bool,
    },
    Alias {
        source: ValueId,
    },
    Gemm {
        x: ValueId,
        weight: ValueId,
        weight_transposed: bool,
        plan: Bf16GemmPlan,
        out_f32: bool,
        workspace: ValueId,
    },
    /// Exact active group sizes require bounded host control readback. This
    /// command is non-capturable; all payloads and stable grouping stay on GPU.
    GroupedExpert {
        x: ValueId,
        weight: ValueId,
        indexes: ValueId,
        rows: usize,
        columns: usize,
        inner: usize,
        experts: usize,
        control: ValueId,
        row_map: ValueId,
        gathered: ValueId,
        projected: ValueId,
        workspace: Option<ValueId>,
    },
    /// Infallible numerical epilogue with by-value geometry and no status I/O.
    LinearBias {
        accumulator: ValueId,
        bias: ValueId,
        args: CudaKernelArgs,
    },
    Kernel {
        name: &'static str,
        args: CudaKernelArgs,
        inputs: [Option<ValueId>; 8],
        scratch: [Option<ValueId>; 3],
        status: ValueId,
        metadata: Vec<u64>,
        state: StateAccess,
        state_buffers: [Option<ValueId>; 4],
        kv_matmul: Option<crate::kv_matmul::KvMatmulWorkspace>,
    },
}
pub(super) struct Command {
    pub(super) output: Option<ValueId>,
    pub(super) kind: CommandKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum StateComponent {
    Keys(usize),
    Values(usize),
    KeyScales(usize),
    ValueScales(usize),
    Kda(usize),
    Conv(usize),
}

fn product_checked(shape: &[usize]) -> Result<usize, String> {
    crate::value::element_count(shape)
}

#[derive(Clone, Copy)]
struct NativeGemm {
    geometry: Bf16GemmPlan,
    weight: usize,
    weight_transposed: bool,
}

/// Immutable physical choices derived from the already accepted dtype plan.
/// Transpose views never escape into graph outputs or canonical kernels.
struct CudaGemmLoweringPlan {
    operations: Vec<Option<NativeGemm>>,
    weight_views: Vec<bool>,
}

impl CudaGemmLoweringPlan {
    fn new(index: &GraphIndex, legalization: &LegalizationPlan) -> Result<Self, String> {
        let mut operations = vec![None; index.order.len()];
        for entry in legalization.units() {
            let LoweringUnit::Node(dense) = entry.unit() else {
                continue;
            };
            let execution = entry.disposition().execution();
            let operation = execution
                .operation(dense)
                .ok_or("compile: missing CUDA dtype plan")?;
            if execution.realization != ExecutionRealization::DirectKernel
                || operation.compute_dtype != Some(DType::BF16)
            {
                continue;
            }
            let node = index.node(dense).ok_or("compile: missing GEMM node")?;
            let (kind, x, weight) = match &node.kind {
                NodeKind::Linear { x, weight, .. } => (RowGemmKind::Linear, x, weight),
                NodeKind::Matmul { a, b } => (RowGemmKind::Matmul, a, b),
                _ => continue,
            };
            if operation
                .operands
                .iter()
                .any(|operand| operand.preparation != OperandPreparation::Direct)
                || operation
                    .results
                    .iter()
                    .any(|result| result.completion != ResultCompletion::Direct)
            {
                return Err(
                    "compile: native BF16 GEMM requires direct operands and results".into(),
                );
            }
            let geometry = plan_row_bf16_gemm(kind, &x.shape, &weight.shape, &node.shape)
                .ok_or("compile: accepted native BF16 GEMM has unsupported geometry")?;
            let (source, weight_transposed) = match &weight.kind {
                NodeKind::Permute { a, dims } if a.shape.len() == 2 && dims == &[1, 0] => (a, true),
                _ => (weight, false),
            };
            operations[dense.index()] = Some(NativeGemm {
                geometry,
                weight: index
                    .dense_id(source.id)
                    .ok_or("compile: missing GEMM weight source")?
                    .index(),
                weight_transposed,
            });
        }
        let mut weight_views = vec![false; index.order.len()];
        for (position, node) in index.order.iter().enumerate() {
            let NodeKind::Permute { a, dims } = &node.kind else {
                continue;
            };
            if a.shape.len() != 2
                || dims != &[1, 0]
                || index.roots.iter().any(|root| root.index() == position)
                || index.consumers[position].is_empty()
            {
                continue;
            }
            weight_views[position] = index.consumers[position].iter().all(|consumer| {
                let Some(gemm) = operations[consumer.index()] else {
                    return false;
                };
                if !gemm.weight_transposed {
                    return false;
                }
                let children = node_children(&index.order[consumer.index()].kind);
                children.get(1).is_some_and(|child| child.id == node.id)
                    && children
                        .iter()
                        .enumerate()
                        .all(|(role, child)| role == 1 || child.id != node.id)
            });
        }
        Ok(Self {
            operations,
            weight_views,
        })
    }
}

pub(super) struct CudaProgramBuilder {
    pub(super) values: Vec<CudaValueMeta>,
    pub(super) commands: Vec<Command>,
    lowered: Vec<LoweredInstruction<&'static str>>,
    semantic_values: Vec<Option<ValueId>>,
    semantic_results: Vec<Vec<ValueId>>,
    state_layout: Option<CudaStateLayout>,
    state_values: std::collections::HashMap<StateComponent, (ValueId, Option<ValueId>)>,
    pub(super) conversion_count: usize,
    pub(super) conversion_bytes: usize,
    gemms: CudaGemmLoweringPlan,
    bf16_kv_matmul: bool,
}

impl CudaProgramBuilder {
    pub(super) fn new(
        index: &GraphIndex,
        state_layout: Option<CudaStateLayout>,
        legalization: &LegalizationPlan,
    ) -> Result<Self, String> {
        Ok(Self {
            values: Vec::new(),
            commands: Vec::new(),
            lowered: Vec::new(),
            semantic_values: vec![None; index.order.len()],
            semantic_results: vec![Vec::new(); index.order.len()],
            state_layout,
            state_values: std::collections::HashMap::new(),
            conversion_count: 0,
            conversion_bytes: 0,
            gemms: CudaGemmLoweringPlan::new(index, legalization)?,
            bf16_kv_matmul: legalization
                .target()
                .features
                .iter()
                .any(|f| f == "stepwise-bf16-kv-f32-gemm-active-rows-v1"),
        })
    }

    fn value(
        &mut self,
        shape: Vec<usize>,
        dtype: DType,
        storage: StorageMetadata,
        name: &str,
        allocation: ValueStorage<CudaMemorySpace>,
    ) -> Result<ValueId, String> {
        let id = ValueId::from_index(self.values.len()).ok_or("compile: too many CUDA values")?;
        let bytes = ValueSpec {
            semantic_dtype: dtype,
            logical_shape: &shape,
            storage: storage.as_spec(),
        }
        .canonical_geometry()?
        .byte_len;
        self.values.push(CudaValueMeta {
            decl: ValueDecl {
                id,
                name: format!("{name}_{}", id.get()),
                bytes,
                storage: allocation,
            },
            shape,
            dtype,
            storage,
        });
        Ok(id)
    }
    fn planned(&mut self, shape: Vec<usize>, dtype: DType, name: &str) -> Result<ValueId, String> {
        self.value(
            shape,
            dtype,
            StorageMetadata::dense(),
            name,
            ValueStorage::Planned {
                class: StorageClass::Workspace,
                alignment: CUDA_STORAGE_ALIGNMENT,
                memory_space: CudaMemorySpace::Device,
                ownership: SegmentOwnership::Workspace,
            },
        )
    }
    fn resolve(&self, semantic: usize) -> Result<ValueId, String> {
        self.semantic_values
            .get(semantic)
            .copied()
            .flatten()
            .ok_or_else(|| format!("compile: missing CUDA semantic value {semantic}"))
    }
    fn emit(
        &mut self,
        name: &'static str,
        kind: CommandKind,
        output: Option<ValueId>,
        inputs: Vec<ValueUse>,
        defines: bool,
    ) -> Result<(), String> {
        let id = InstructionId::from_index(self.lowered.len())
            .ok_or("compile: too many CUDA instructions")?;
        let outputs = if defines {
            output.into_iter().map(OutputDecl::new).collect()
        } else {
            Vec::new()
        };
        self.lowered
            .push(LoweredInstruction::new(id, name, inputs, outputs));
        self.commands.push(Command { output, kind });
        Ok(())
    }

    fn conversion(&mut self, source: ValueId, destination: DType) -> Result<ValueId, String> {
        let metadata = &self.values[source.index()];
        if metadata.storage.representation != StorageRepresentation::Dense {
            return Err("compile: dense conversion cannot decode packed storage".into());
        }
        let output = self.planned(metadata.shape.clone(), destination, "convert")?;
        let mut spec = KernelSpec {
            name: "et_convert",
            args: CudaKernelArgs::default(),
            inputs: [None; 8],
            tail: Vec::new(),
            scratch: [None; 3],
            state: StateAccess::None,
        };
        spec.args.compute_dtype = dtype_code(destination);
        let mut inputs = [None; 8];
        inputs[0] = Some(source);
        self.kernel(spec, inputs, output)?;
        self.conversion_count += 1;
        self.conversion_bytes = self
            .conversion_bytes
            .checked_add(self.values[output.index()].decl.bytes)
            .ok_or("compile: conversion bytes overflow")?;
        Ok(output)
    }

    fn state_value(
        &mut self,
        component: StateComponent,
        elements: usize,
        dtype: DType,
        transaction: Option<usize>,
    ) -> Result<(ValueId, Option<ValueId>), String> {
        if let Some(ids) = self.state_values.get(&component) {
            return Ok(*ids);
        }
        let fixed = self.value(
            vec![elements],
            dtype,
            StorageMetadata::dense(),
            "state",
            ValueStorage::Fixed {
                class: StorageClass::PersistentState,
                location: Location::Persistent {
                    slot: u32::try_from(self.values.len())
                        .map_err(|_| "compile: too many state slots")?,
                },
            },
        )?;
        let staging = if let Some(elements) = transaction {
            let id = self.value(
                vec![elements],
                dtype,
                StorageMetadata::dense(),
                "state_transaction",
                ValueStorage::Planned {
                    class: StorageClass::Workspace,
                    alignment: CUDA_STORAGE_ALIGNMENT,
                    memory_space: CudaMemorySpace::Device,
                    ownership: SegmentOwnership::StateTransaction,
                },
            )?;
            let instruction = InstructionId::from_index(self.lowered.len())
                .ok_or("compile: too many state copies")?;
            self.lowered.push(
                LoweredInstruction::new(
                    instruction,
                    "state_prepare",
                    Vec::new(),
                    vec![OutputDecl::new(id)],
                )
                .with_resources(
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    vec![ValueUse::read(fixed)],
                ),
            );
            Some(id)
        } else {
            None
        };
        self.state_values.insert(component, (fixed, staging));
        Ok((fixed, staging))
    }

    fn kernel(
        &mut self,
        mut spec: KernelSpec,
        inputs: [Option<ValueId>; 8],
        output: ValueId,
    ) -> Result<(), String> {
        spec.name = kernel_name(spec.name, spec.args.compute_dtype)?;
        let mut kv_matmul = None;
        let mut state_buffers = [None; 4];
        let mut state_uses = Vec::new();
        match spec.state {
            StateAccess::Kv {
                layer,
                heads,
                dim,
                batch,
            } => {
                let layout = self
                    .state_layout
                    .clone()
                    .ok_or("compile: CUDA KV attention requires an explicit state layout")?;
                let descriptor = layout
                    .kv_layers
                    .iter()
                    .find(|d| d.layer_id as usize == layer)
                    .ok_or("compile: KV layer descriptor missing")?;
                if descriptor.kv_heads != heads || descriptor.head_dim != dim {
                    return Err("compile: KV layer geometry mismatch".into());
                }
                let tokens = spec.args.integers[7] as usize;
                let temporary_rows = product_checked(&[batch, tokens, heads])?;
                let temporary_elements = product_checked(&[temporary_rows, dim])?;
                let table_rows = (layout.capacity as usize)
                    .checked_add(
                        tokens
                            .checked_mul(layout.packed_rows_per_sequence.unwrap_or(1) as usize)
                            .ok_or("compile: KV rows overflow")?,
                    )
                    .and_then(|n| n.checked_add(1))
                    .ok_or("compile: KV table overflow")?;
                // Keep the pointer table and per-query score rows in one planned
                // invocation allocation. Scores become probabilities in place;
                // no persistent KV storage is copied or used as scratch.
                let table_bytes = product_checked(&[layout.slots as usize, table_rows, 4, 8])?;
                let score_width = if spec.args.compute_dtype == 0 && spec.args.integers[6] == 0 {
                    8
                } else {
                    4
                };
                let score_bytes = product_checked(&[
                    batch,
                    tokens,
                    spec.args.integers[11] as usize,
                    table_rows - 1,
                    score_width,
                ])?;
                spec.args.integers[9] = (table_rows - 1) as u64;
                spec.args.integers[12] = table_bytes as u64;
                let mut workspace_bytes = table_bytes
                    .checked_add(score_bytes)
                    .ok_or("compile: KV score workspace overflow")?;
                if self.bf16_kv_matmul
                    && descriptor.dtype == DType::BF16
                    && spec.args.integers[6] == 1
                    && spec.args.integers[8] == 3
                    && spec.args.compute_dtype == 1
                    && self.values[output.index()].dtype == DType::F32
                {
                    kv_matmul = crate::kv_matmul::KvMatmulWorkspace::new(
                        workspace_bytes,
                        spec.args.integers[11] as usize,
                        tokens,
                        table_rows - 1,
                        dim,
                    );
                    if let Some(plan) = kv_matmul {
                        workspace_bytes = plan.bytes;
                    }
                }
                spec.scratch[2] = Some((workspace_bytes, DType::U8));
                layout.validate()?;
                if product_checked(&[
                    layout.slots as usize,
                    layout.packed_rows_per_sequence.unwrap_or(1) as usize,
                ])? != batch
                {
                    return Err("compile: KV state layout batch mismatch".into());
                }
                let rows =
                    product_checked(&[layout.slots as usize, layout.capacity as usize, heads])?;
                let elements = product_checked(&[rows, dim])?;
                for (slot, component) in [
                    StateComponent::Keys(layer),
                    StateComponent::Values(layer),
                    StateComponent::KeyScales(layer),
                    StateComponent::ValueScales(layer),
                ]
                .into_iter()
                .enumerate()
                {
                    if slot >= 2 && descriptor.dtype != DType::U8 {
                        continue;
                    }
                    let (_, transaction) = self.state_value(
                        component,
                        if slot < 2 { elements } else { rows },
                        if slot < 2 {
                            descriptor.dtype
                        } else {
                            DType::F32
                        },
                        Some(if slot < 2 {
                            temporary_elements
                        } else {
                            temporary_rows
                        }),
                    )?;
                    state_buffers[slot] = transaction;
                    state_uses.push(ValueUse::read_write(
                        transaction.ok_or("compile: missing state transaction")?,
                    ));
                }
            }
            StateAccess::Kda {
                layer: Some(layer),
                batch,
                elements_per_sequence,
                ..
            }
            | StateAccess::Conv {
                layer: Some(layer),
                batch,
                elements_per_sequence,
            } => {
                if self.state_layout.as_ref().is_some_and(|layout| {
                    layout.access == effect_torch_runtime::StateAccessMode::ReadOnly
                }) {
                    return Err(
                        "compile: read-only state does not support recurrent mutation".into(),
                    );
                }
                let component = if matches!(spec.state, StateAccess::Kda { .. }) {
                    StateComponent::Kda(layer)
                } else {
                    StateComponent::Conv(layer)
                };
                let (fixed, _) = self.state_value(
                    component,
                    product_checked(&[batch, elements_per_sequence])?,
                    DType::F32,
                    None,
                )?;
                state_uses.push(ValueUse::read_write(fixed));
            }
            _ => {}
        }
        let mut scratch = [None; 3];
        for (slot, requirement) in spec.scratch.iter().enumerate() {
            if let Some((elements, dtype)) = requirement {
                scratch[slot] = Some(self.planned(vec![*elements], *dtype, "scratch")?);
            }
        }
        let status = self.planned(vec![1], DType::U32, "status")?;
        let mut definitions = scratch
            .iter()
            .flatten()
            .copied()
            .map(OutputDecl::new)
            .collect::<Vec<_>>();
        definitions.push(OutputDecl::new(status));
        let prepare_id = InstructionId::from_index(self.lowered.len())
            .ok_or("compile: too many CUDA instructions")?;
        self.lowered.push(LoweredInstruction::new(
            prepare_id,
            "prepare",
            Vec::new(),
            definitions,
        ));
        self.commands.push(Command {
            output: None,
            kind: CommandKind::Prepare,
        });
        let out = &self.values[output.index()];
        spec.args.elements = crate::value::element_count(&out.shape)? as u64;
        if spec.name == "et_rms_norm_f32" {
            spec.args.integers[0] = *out.shape.last().ok_or("compile: RMS input rank")? as u64;
        }
        spec.args.output_dtype = dtype_code(out.dtype);
        let mut metadata = vec![out.shape.len() as u64];
        metadata.extend(
            inputs
                .iter()
                .map(|id| id.map_or(0, |id| self.values[id.index()].shape.len() as u64)),
        );
        metadata.extend(out.shape.iter().map(|d| *d as u64));
        for (slot, id) in inputs.iter().enumerate() {
            if let Some(id) = id {
                let value = &self.values[id.index()];
                metadata.extend(value.shape.iter().map(|d| *d as u64));
                spec.args.input_dtypes[slot] = dtype_code(
                    if value.storage.representation == StorageRepresentation::Dense {
                        value.dtype
                    } else {
                        DType::U8
                    },
                );
            }
        }
        metadata.extend(spec.tail);
        let metadata_bytes = metadata
            .len()
            .checked_mul(8)
            .ok_or("compile: CUDA metadata size overflow")?;
        let metadata_id = self.value(
            vec![metadata_bytes],
            DType::U8,
            StorageMetadata::dense(),
            "metadata",
            ValueStorage::Fixed {
                class: StorageClass::PersistentConstant,
                location: Location::Persistent {
                    slot: u32::try_from(self.values.len())
                        .map_err(|_| "compile: metadata slots overflow")?,
                },
            },
        )?;
        let mut uses = inputs.iter().flatten().copied().collect::<Vec<_>>();
        uses.sort_unstable();
        uses.dedup();
        uses.push(metadata_id);
        let stateful = matches!(
            spec.state,
            StateAccess::Kv { .. }
                | StateAccess::Kda { layer: Some(_), .. }
                | StateAccess::Conv { layer: Some(_), .. }
        );
        let id = InstructionId::from_index(self.lowered.len())
            .ok_or("compile: CUDA instruction count overflow")?;
        self.lowered.push(
            LoweredInstruction::new(
                id,
                if kv_matmul.is_some() {
                    "kv_stepwise_bf16_gemm_active_rows"
                } else {
                    spec.name
                },
                uses.into_iter().map(ValueUse::read).collect::<Vec<_>>(),
                vec![OutputDecl::new(output)],
            )
            .with_resources(
                scratch
                    .iter()
                    .flatten()
                    .copied()
                    .map(ValueUse::read_write)
                    .collect::<Vec<_>>(),
                Vec::new(),
                vec![ValueUse::read_write(status)],
                state_uses,
            )
            .with_effects(InstructionEffects {
                may_fail: true,
                has_side_effects: stateful,
            }),
        );
        self.commands.push(Command {
            output: Some(output),
            kind: CommandKind::Kernel {
                name: spec.name,
                args: spec.args,
                inputs,
                scratch,
                status,
                metadata,
                state: spec.state,
                state_buffers,
                kv_matmul,
            },
        });
        Ok(())
    }

    pub(super) fn add(
        &mut self,
        dense: DenseNodeId,
        node: &Node,
        index: &GraphIndex,
        instruction: Instruction,
        plan: &ExecutableDTypePlan,
    ) -> Result<(), String> {
        let execution = plan.execution();
        if !matches!(
            execution.realization,
            ExecutionRealization::DirectKernel | ExecutionRealization::MaterializedTransforms
        ) {
            return Err(
                "compile: CUDA lowerer requires a direct or materialized execution plan".into(),
            );
        }
        let operation = execution
            .operation(dense)
            .ok_or("compile: CUDA legalization entry is missing")?;
        let boundary_storage = node.storage.clone();
        // A 2D transpose that only feeds a row-oriented BF16 weight position is
        // a physical view. Lowering keeps the original bytes and lets the GEMM
        // consume them through its transpose operand, so no weight transpose
        // copy and no F32 weight conversion is ever materialized.
        if let Instruction::Reindex {
            op: 1,
            a,
            parameters,
        } = &instruction
        {
            if self.gemms.weight_views[dense.index()]
                && parameters.as_slice() == [1, 0]
                && node.shape.len() == 2
            {
                let source = self.resolve(*a)?;
                let id = self.value(
                    node.shape.clone(),
                    node.dtype,
                    boundary_storage,
                    "cublas_transpose_view",
                    ValueStorage::Alias {
                        source,
                        byte_offset: 0,
                    },
                )?;
                self.emit(
                    "cublas_transpose_view",
                    CommandKind::Alias { source },
                    Some(id),
                    vec![ValueUse::read(source)],
                    false,
                )?;
                self.semantic_values[dense.index()] = Some(id);
                self.semantic_results[dense.index()] = vec![id];
                return Ok(());
            }
        }
        if let Some((parent, selected)) = selected_result(&node.kind) {
            let parent = index
                .dense_id(parent.id)
                .ok_or("compile: missing selected producer")?;
            let source = *self.semantic_results[parent.index()]
                .get(selected as usize)
                .ok_or("compile: CUDA producer result missing")?;
            self.semantic_values[dense.index()] = Some(source);
            self.semantic_results[dense.index()] = vec![source];
            return Ok(());
        }
        let native_gemm = self.gemms.operations[dense.index()];
        let output = match instruction {
            Instruction::GroupedExpertLinearRows { x, weight, indexes } => {
                self.grouped_expert(x, weight, indexes)?
            }
            Instruction::Linear { x, bias, .. } if native_gemm.is_some() => {
                self.row_major_gemm(native_gemm.unwrap(), x, node.shape.clone(), Some(bias))?
            }
            Instruction::Matmul { a, .. } if native_gemm.is_some() => {
                self.row_major_gemm(native_gemm.unwrap(), a, node.shape.clone(), None)?
            }
            Instruction::Value(value) => {
                let id = self.value(
                    node.shape.clone(),
                    node.dtype,
                    boundary_storage,
                    "constant",
                    ValueStorage::Fixed {
                        class: StorageClass::PersistentConstant,
                        location: Location::Persistent {
                            slot: self.values.len() as u32,
                        },
                    },
                )?;
                self.emit(
                    "value",
                    CommandKind::Value(value),
                    Some(id),
                    Vec::new(),
                    false,
                )?;
                id
            }
            Instruction::Input {
                binding,
                scalar: false,
                ..
            } => {
                let id = self.value(
                    node.shape.clone(),
                    node.dtype,
                    boundary_storage,
                    "input",
                    ValueStorage::Fixed {
                        class: StorageClass::ExternalInput,
                        location: Location::External {
                            slot: u32::try_from(binding)
                                .map_err(|_| "compile: too many bindings")?,
                        },
                    },
                )?;
                self.emit(
                    "input",
                    CommandKind::Input { binding },
                    Some(id),
                    Vec::new(),
                    false,
                )?;
                id
            }
            Instruction::Input {
                binding,
                scalar: true,
                ..
            } => {
                let id = self.planned(node.shape.clone(), node.dtype, "scalar")?;
                self.emit(
                    "scalar",
                    CommandKind::Scalar { binding },
                    Some(id),
                    Vec::new(),
                    true,
                )?;
                id
            }
            Instruction::StateCursor { tensor, .. } => {
                let id = self.planned(node.shape.clone(), node.dtype, "cursor")?;
                self.emit(
                    "cursor",
                    CommandKind::Cursor { tensor },
                    Some(id),
                    Vec::new(),
                    true,
                )?;
                id
            }
            Instruction::Alias { a, .. } => {
                let source = self.resolve(a)?;
                if boundary_storage.representation != StorageRepresentation::Dense {
                    return Err("compile: packed CUDA aliases are unsupported".into());
                }
                let id = self.value(
                    node.shape.clone(),
                    node.dtype,
                    boundary_storage,
                    "alias",
                    ValueStorage::Alias {
                        source,
                        byte_offset: 0,
                    },
                )?;
                self.emit(
                    "alias",
                    CommandKind::Alias { source },
                    Some(id),
                    vec![ValueUse::read(source)],
                    false,
                )?;
                id
            }
            instruction => {
                let spec = instruction.kernel()?;
                let mut inputs = [None; 8];
                for (slot, semantic) in spec.inputs.iter().enumerate() {
                    if let Some(semantic) = semantic {
                        inputs[slot] = Some(self.resolve(*semantic)?);
                    }
                }
                let children = node_children(&node.kind);
                for operand in &operation.operands {
                    let child = children
                        .get(operand.index as usize)
                        .ok_or("compile: operand missing")?;
                    let semantic = index
                        .dense_id(child.id)
                        .ok_or("compile: operand source missing")?
                        .index();
                    let mut source = self.resolve(semantic)?;
                    if let OperandInterpretation::ScalarCoercion(conversion) =
                        operand.interpretation
                    {
                        if self.values[source.index()].dtype != conversion.source {
                            return Err("compile: scalar coercion source mismatch".into());
                        }
                        source = self.conversion(source, conversion.destination)?;
                    }
                    if let OperandPreparation::Convert(conversion) = operand.preparation {
                        if self.values[source.index()].dtype != conversion.source {
                            return Err("compile: conversion source mismatch".into());
                        }
                        source = self.conversion(source, conversion.destination)?;
                    }
                    for (slot, input) in spec.inputs.iter().enumerate() {
                        if *input == Some(semantic) {
                            inputs[slot] = Some(source);
                        }
                    }
                }
                let semantic_spec = OperationDTypeSpec::new(index, dense)?;
                let mut results = Vec::with_capacity(operation.results.len());
                for (position, result) in operation.results.iter().enumerate() {
                    let boundary = &semantic_spec.results[position].value;
                    let id = self.planned(
                        boundary.logical_shape.to_vec(),
                        result.execution_dtype,
                        "result",
                    )?;
                    let mut selected = spec.clone();
                    if operation.results.len() > 1 {
                        match selected.name {
                            "et_optimizer" => selected.args.integers[0] = position as u64,
                            "et_kda" | "et_sdpa" => selected.args.operation = position as u32,
                            "et_layer_norm" | "et_chunked_head_ce" => {
                                selected.args.operation = position as u32 + 1
                            }
                            _ => {
                                return Err(
                                    "compile: CUDA multi-result family has no selector".into()
                                )
                            }
                        }
                    }
                    if selected.name == "et_kv_attention" {
                        selected.args.integers[8] = u64::from(dtype_code(node.dtype));
                    }
                    selected.args.compute_dtype =
                        dtype_code(operation.compute_dtype.unwrap_or(result.execution_dtype));
                    self.kernel(selected, inputs, id)?;
                    results.push(match result.completion {
                        ResultCompletion::Direct => id,
                        ResultCompletion::ConvertToBoundary(conversion) => {
                            if result.execution_dtype != conversion.source
                                || boundary.semantic_dtype != conversion.destination
                            {
                                return Err("compile: CUDA boundary dtype mismatch".into());
                            }
                            self.conversion(id, conversion.destination)?
                        }
                    });
                }
                let first = results[0];
                self.semantic_results[dense.index()] = results;
                first
            }
        };
        self.semantic_values[dense.index()] = Some(output);
        if self.semantic_results[dense.index()].is_empty() {
            self.semantic_results[dense.index()] = vec![output];
        }
        Ok(())
    }

    fn emit_gemm(
        &mut self,
        plan: &Bf16GemmPlan,
        weight_transposed: bool,
        x: ValueId,
        weight: ValueId,
        output: ValueId,
        out_f32: bool,
    ) -> Result<(), String> {
        let workspace =
            self.planned(vec![CUBLAS_WORKSPACE_BYTES], DType::U8, "cublas_workspace")?;
        self.emit(
            "prepare",
            CommandKind::Prepare,
            Some(workspace),
            Vec::new(),
            true,
        )?;
        let id = InstructionId::from_index(self.lowered.len())
            .ok_or("compile: too many CUDA instructions")?;
        self.lowered.push(
            LoweredInstruction::new(
                id,
                "cublas_bf16_gemm_f32_accum",
                vec![ValueUse::read(x), ValueUse::read(weight)],
                vec![OutputDecl::new(output)],
            )
            .with_resources(
                vec![ValueUse::read_write(workspace)],
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
            .with_effects(InstructionEffects {
                may_fail: true,
                has_side_effects: false,
            }),
        );
        self.commands.push(Command {
            output: Some(output),
            kind: CommandKind::Gemm {
                x,
                weight,
                weight_transposed,
                plan: *plan,
                out_f32,
                workspace,
            },
        });
        Ok(())
    }

    fn grouped_expert(
        &mut self,
        x: usize,
        weight: usize,
        indexes: usize,
    ) -> Result<ValueId, String> {
        let (x, weight, indexes) = (
            self.resolve(x)?,
            self.resolve(weight)?,
            self.resolve(indexes)?,
        );
        let dtype = self.values[x.index()].dtype;
        let rows = self.values[x.index()].shape[0];
        let inner = self.values[x.index()].shape[1];
        let experts = self.values[weight.index()].shape[0];
        let columns = self.values[weight.index()].shape[1];
        let control = self.planned(
            vec![experts.checked_add(2).ok_or("group control overflow")?],
            DType::U32,
            "group_control",
        )?;
        let row_map = self.planned(vec![rows], DType::U32, "group_rows")?;
        let gathered = self.planned(vec![rows, inner], dtype, "group_input")?;
        let projected = self.planned(vec![rows, columns], dtype, "group_output")?;
        let workspace = if dtype == DType::BF16 && rows != 0 && inner != 0 && columns != 0 {
            Some(self.planned(vec![CUBLAS_WORKSPACE_BYTES], DType::U8, "cublas_workspace")?)
        } else {
            None
        };
        let scratch = [
            Some(control),
            Some(row_map),
            Some(gathered),
            Some(projected),
            workspace,
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        for &id in &scratch {
            self.emit("prepare", CommandKind::Prepare, Some(id), Vec::new(), true)?;
        }
        let output = self.planned(vec![rows, columns], dtype, "result")?;
        let id = InstructionId::from_index(self.lowered.len())
            .ok_or("compile: too many CUDA instructions")?;
        self.lowered.push(
            LoweredInstruction::new(
                id,
                "grouped_expert_linear_rows_host_control_non_capturable",
                vec![
                    ValueUse::read(x),
                    ValueUse::read(weight),
                    ValueUse::read(indexes),
                ],
                vec![OutputDecl::new(output)],
            )
            .with_resources(
                scratch
                    .into_iter()
                    .map(ValueUse::read_write)
                    .collect::<Vec<_>>(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
            .with_effects(InstructionEffects {
                may_fail: true,
                has_side_effects: false,
            }),
        );
        self.commands.push(Command {
            output: Some(output),
            kind: CommandKind::GroupedExpert {
                x,
                weight,
                indexes,
                rows,
                columns,
                inner,
                experts,
                control,
                row_map,
                gathered,
                projected,
                workspace,
            },
        });
        Ok(output)
    }

    /// Native BF16 row-major GEMM for a Linear or Matmul.
    fn row_major_gemm(
        &mut self,
        native: NativeGemm,
        activation: usize,
        output_shape: Vec<usize>,
        bias: Option<usize>,
    ) -> Result<ValueId, String> {
        let x = self.resolve(activation)?;
        let w = self.resolve(native.weight)?;
        let plan = native.geometry;
        let weight_transposed = native.weight_transposed;
        if let Some(bias) = bias {
            // Add bias before the single BF16 rounding boundary. The GEMM
            // accumulates into F32 and the bias kernel performs the only
            // rounding to BF16.
            let accumulator =
                self.planned(output_shape.clone(), DType::F32, "linear_accumulator")?;
            self.emit_gemm(&plan, weight_transposed, x, w, accumulator, true)?;
            let bias = self.resolve(bias)?;
            let mut args = CudaKernelArgs {
                elements: product_checked(&output_shape)? as u64,
                compute_dtype: dtype_code(DType::F32),
                output_dtype: dtype_code(DType::BF16),
                ..Default::default()
            };
            args.integers[0] = *output_shape
                .last()
                .ok_or("compile: CUDA linear result has no columns")?
                as u64;
            args.input_dtypes[0] = dtype_code(DType::F32);
            args.input_dtypes[1] = dtype_code(DType::BF16);
            let result = self.planned(output_shape, DType::BF16, "result")?;
            self.emit(
                BF16_LINEAR_BIAS_KERNEL,
                CommandKind::LinearBias {
                    accumulator,
                    bias,
                    args,
                },
                Some(result),
                vec![ValueUse::read(accumulator), ValueUse::read(bias)],
                true,
            )?;
            return Ok(result);
        }
        let result = self.planned(output_shape, DType::BF16, "result")?;
        self.emit_gemm(&plan, weight_transposed, x, w, result, false)?;
        Ok(result)
    }

    pub(super) fn finish(
        mut self,
        index: &GraphIndex,
    ) -> Result<(CudaLoweredProgram, Vec<Command>), String> {
        let mut transactions = self
            .state_values
            .iter()
            .filter_map(|(component, (persistent, transaction))| {
                transaction.map(|transaction| (*component, *persistent, transaction))
            })
            .collect::<Vec<_>>();
        transactions.sort_by_key(|(_, persistent, _)| *persistent);
        for (_component, persistent, transaction) in transactions {
            let id = InstructionId::from_index(self.lowered.len())
                .ok_or("compile: too many state commits")?;
            let append = self.state_layout.as_ref().is_some_and(|layout| {
                layout.access == effect_torch_runtime::StateAccessMode::Append
            });
            self.lowered.push(
                LoweredInstruction::new(
                    id,
                    if append {
                        "state_commit"
                    } else {
                        "state_discard"
                    },
                    Vec::new(),
                    Vec::new(),
                )
                .with_resources(
                    Vec::new(),
                    vec![ValueUse::read(transaction)],
                    Vec::new(),
                    if append {
                        vec![ValueUse::write(persistent)]
                    } else {
                        Vec::new()
                    },
                )
                .with_effects(InstructionEffects {
                    may_fail: false,
                    has_side_effects: append,
                }),
            );
        }
        let outputs = index
            .roots
            .iter()
            .map(|root| self.resolve(root.index()))
            .collect::<Result<Vec<_>, _>>()?;
        for root in &outputs {
            let mut source = *root;
            while let ValueStorage::Alias { source: parent, .. } =
                self.values[source.index()].decl.storage
            {
                source = parent;
            }
            if let ValueStorage::Planned {
                class, ownership, ..
            } = &mut self.values[source.index()].decl.storage
            {
                *class = StorageClass::EscapingOutput;
                *ownership = SegmentOwnership::ProvisionalOutput;
            }
        }
        Ok((
            LoweredProgram::new(self.values, self.lowered, outputs),
            self.commands,
        ))
    }
}
