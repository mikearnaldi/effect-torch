//! Bounded CUDA literal-only lowering. Preparation, legalization, memory
//! validation, uploads, and ordinary invocation ownership remain unchanged.
use super::{CudaLoweredProgram, CudaMemorySpace, CudaValueMeta};
use effect_torch_compiler::{
    DenseNodeId, GraphIndex, LoweredInstruction, LoweredProgram, ValueDecl, ValueStorage,
};
use effect_torch_graph::{Node, NodeKind};
use effect_torch_runtime::{InstructionId, Location, StorageClass, ValueId};
use std::sync::Arc;

pub(crate) const ENV: &str = "EFFECT_TORCH_CUDA_LITERAL_LEAF88";
const MAX_ROOTS: usize = 8;
const MAX_BYTES: usize = 64 * 1024;

pub(crate) fn admitted(roots: &[Arc<Node>], stateless: bool, enabled: bool) -> bool {
    if !enabled || !stateless || roots.is_empty() || roots.len() > MAX_ROOTS {
        return false;
    }
    let mut bytes = 0usize;
    for root in roots {
        if !matches!(
            root.kind,
            NodeKind::Full { .. } | NodeKind::FromBytes { .. }
        ) {
            return false;
        }
        let Ok(geometry) = root.value_spec().canonical_geometry() else {
            return false;
        };
        let Some(total) = bytes.checked_add(geometry.byte_len) else {
            return false;
        };
        if total > MAX_BYTES {
            return false;
        }
        bytes = total;
    }
    true
}

/// Mirrors the ordinary `Instruction::Value` branch without inference-pattern
/// scans or post-lowering transformations. Dense semantic IDs are unchanged.
pub(crate) struct Builder {
    values: Vec<CudaValueMeta>,
    instructions: Vec<LoweredInstruction<&'static str>>,
    semantic_values: Vec<Option<ValueId>>,
}

impl Builder {
    pub(crate) fn new(index: &GraphIndex) -> Self {
        Self {
            values: Vec::with_capacity(index.order.len()),
            instructions: Vec::with_capacity(index.order.len()),
            semantic_values: vec![None; index.order.len()],
        }
    }

    pub(crate) fn add(&mut self, dense: DenseNodeId, node: &Node) -> Result<ValueId, String> {
        if !matches!(
            node.kind,
            NodeKind::Full { .. } | NodeKind::FromBytes { .. }
        ) {
            return Err("compile: literal88 received a non-literal node".into());
        }
        let id = ValueId::from_index(self.values.len()).ok_or("compile: too many CUDA values")?;
        let instruction = InstructionId::from_index(self.instructions.len())
            .ok_or("compile: too many CUDA instructions")?;
        let slot =
            u32::try_from(self.values.len()).map_err(|_| "compile: literal slots overflow")?;
        self.values.push(CudaValueMeta {
            decl: ValueDecl {
                id,
                name: format!("constant_{}", id.get()),
                bytes: node.value_spec().canonical_geometry()?.byte_len,
                storage: ValueStorage::Fixed {
                    class: StorageClass::PersistentConstant,
                    location: Location::Persistent { slot },
                },
            },
            shape: node.shape.clone(),
            dtype: node.dtype,
            storage: node.storage.clone(),
        });
        // Fixed constants are externally initialized, not instruction-defined.
        self.instructions.push(LoweredInstruction::new(
            instruction,
            "value",
            Vec::new(),
            Vec::new(),
        ));
        *self
            .semantic_values
            .get_mut(dense.index())
            .ok_or("compile: literal ID missing")? = Some(id);
        Ok(id)
    }

    pub(crate) fn finish(self, index: &GraphIndex) -> Result<CudaLoweredProgram, String> {
        let outputs = index
            .roots
            .iter()
            .map(|root| self.semantic_values[root.index()].ok_or("compile: literal output missing"))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(LoweredProgram::<_, CudaMemorySpace, _>::new(
            self.values,
            self.instructions,
            outputs,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capabilities::CudaCapabilities;
    use crate::lowering::{CudaProgramBuilder, CUDA_STORAGE_ALIGNMENT};
    use effect_torch_compiler::{
        CompileOptions, CompilerDriver, LoweringUnit, MemoryPlannerConfig, ProgramRequest,
    };
    use effect_torch_graph::Device;
    use effect_torch_runtime::{CancellationFlag, DType};

    fn full(shape: Vec<usize>, dtype: DType, value: f64) -> Arc<Node> {
        Node::new(NodeKind::Full {
            shape,
            dtype,
            value,
            device: Device::Cuda(0),
        })
        .unwrap()
    }
    fn bytes() -> Arc<Node> {
        Node::new(NodeKind::FromBytes {
            shape: vec![1, 256],
            dtype: DType::U32,
            device: Device::Cuda(0),
            data: (0u32..256).flat_map(u32::to_le_bytes).collect(),
        })
        .unwrap()
    }
    fn plan(roots: Vec<Arc<Node>>, shortcut: bool) -> CudaLoweredProgram {
        let prepared = ProgramRequest::from_roots(roots, CompileOptions::default())
            .prepare()
            .unwrap();
        let capabilities = CudaCapabilities::new(0, 12, 0);
        let mut driver = CompilerDriver::new(&prepared, &capabilities).unwrap();
        let program = if shortcut {
            let mut builder = Builder::new(&prepared.index);
            driver
                .lower(|unit, index, _, _| {
                    let LoweringUnit::Node(dense) = unit else {
                        panic!("unexpected region")
                    };
                    builder.add(dense, index.node(dense).unwrap())?;
                    Ok(())
                })
                .unwrap();
            builder.finish(&prepared.index).unwrap()
        } else {
            let mut builder =
                CudaProgramBuilder::new(&prepared.index, None, driver.legalization()).unwrap();
            driver
                .lower(|unit, index, _, _| {
                    let LoweringUnit::Node(dense) = unit else {
                        panic!("unexpected region")
                    };
                    let node = index.node(dense).unwrap();
                    // Exact host table portion of Instruction::Value; uploading and
                    // attaching CudaValue to Command are excluded from both modes.
                    builder.value(
                        node.shape.clone(),
                        node.dtype,
                        node.storage.clone(),
                        "constant",
                        ValueStorage::Fixed {
                            class: StorageClass::PersistentConstant,
                            location: Location::Persistent {
                                slot: builder.values.len() as u32,
                            },
                        },
                    )?;
                    builder.lowered.push(LoweredInstruction::new(
                        InstructionId::from_index(builder.lowered.len()).unwrap(),
                        "value",
                        Vec::new(),
                        Vec::new(),
                    ));
                    builder.semantic_values[dense.index()] =
                        ValueId::from_index(builder.values.len() - 1);
                    Ok(())
                })
                .unwrap();
            builder.finish(&prepared.index).unwrap().0
        };
        let memory = driver
            .plan_memory(
                &program,
                &MemoryPlannerConfig::uniform(
                    CudaMemorySpace::Device,
                    usize::MAX / 2,
                    CUDA_STORAGE_ALIGNMENT,
                    CUDA_STORAGE_ALIGNMENT,
                ),
            )
            .unwrap();
        assert!(memory.segments.is_empty());
        program
    }

    #[test]
    fn literal88_bounds_and_actual_roots_only() {
        let root = full(Vec::new(), DType::F32, 0.5);
        assert!(admitted(std::slice::from_ref(&root), true, true));
        assert!(!admitted(std::slice::from_ref(&root), false, true));
        assert!(!admitted(std::slice::from_ref(&root), true, false));
        assert!(!admitted(&[], true, true));
        assert!(!admitted(&vec![root.clone(); 9], true, true));
        assert!(admitted(
            &[full(vec![MAX_BYTES / 4], DType::F32, 0.)],
            true,
            true
        ));
        assert!(!admitted(
            &[full(vec![MAX_BYTES / 4 + 1], DType::F32, 0.)],
            true,
            true
        ));
        let operation = Node::new(NodeKind::Neg { a: root }).unwrap();
        assert!(!admitted(&[operation], true, true));
        let zero = Node::new(NodeKind::Zeros {
            shape: vec![],
            dtype: DType::F32,
            device: Device::Cuda(0),
        })
        .unwrap();
        assert!(!admitted(&[zero], true, true));
    }

    #[test]
    fn literal88_tables_match_ordinary_lowering_order_duplicates_empty_and_dtypes() {
        for dtype in [
            DType::F64,
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::U8,
            DType::U32,
            DType::I64,
        ] {
            let scalar = full(Vec::new(), dtype, 0.3);
            let roots = vec![
                scalar.clone(),
                bytes(),
                full(vec![0], dtype, f64::NAN),
                scalar,
            ];
            assert!(admitted(&roots, true, true));
            let fast = plan(roots.clone(), true);
            let ordinary = plan(roots, false);
            assert_eq!(fast.instructions, ordinary.instructions);
            assert_eq!(fast.outputs, ordinary.outputs);
            assert_eq!(fast.values.len(), ordinary.values.len());
            for (a, b) in fast.values.iter().zip(ordinary.values.iter()) {
                assert_eq!(a.decl, b.decl);
                assert_eq!(a.shape, b.shape);
                assert_eq!(a.dtype, b.dtype);
                assert_eq!(a.storage, b.storage);
            }
        }
    }

    #[test]
    fn literal88_keeps_native_options_and_target_validation() {
        let mut options = CompileOptions::default();
        options.environment.ce_chunk_size = 0;
        assert!(ProgramRequest::from_roots(vec![bytes()], options)
            .prepare()
            .is_err());
        let prepared = ProgramRequest::from_roots(vec![bytes()], CompileOptions::default())
            .prepare()
            .unwrap();
        assert!(CompilerDriver::new(&prepared, &CudaCapabilities::new(1, 12, 0)).is_err());
    }

    #[test]
    #[ignore = "CPU-only compilation microbenchmark; run explicitly with --nocapture"]
    fn literal88_cpu_compilation_microbenchmark() {
        use std::hint::black_box;
        use std::time::Instant;
        for (name, roots) in [
            ("temperature", vec![full(vec![], DType::F32, 0.7)]),
            ("token-position", vec![bytes(), bytes()]),
        ] {
            for mode in [false, true] {
                for _ in 0..100 {
                    black_box(plan(roots.clone(), mode));
                }
            }
            let mut samples = [Vec::new(), Vec::new()];
            for round in 0..9 {
                for mode in if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let start = Instant::now();
                    for _ in 0..1000 {
                        black_box(plan(roots.clone(), mode));
                    }
                    samples[usize::from(mode)].push(start.elapsed().as_nanos() as f64 / 1000.);
                }
            }
            for sample in &mut samples {
                sample.sort_by(f64::total_cmp);
            }
            println!(
                "literal88_cpu {name}: ordinaryMedianNs={} literalMedianNs={} ratio={}",
                samples[0][4],
                samples[1][4],
                samples[1][4] / samples[0][4]
            );
        }
    }

    #[test]
    #[ignore = "requires CUDA and EFFECT_TORCH_CUDA_LITERAL_LEAF88=1"]
    fn literal88_native_cancel_exact_bytes_and_retained_outputs() {
        assert_eq!(std::env::var(ENV).as_deref(), Ok("1"));
        for dtype in [
            DType::F64,
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::U8,
            DType::U32,
            DType::I64,
        ] {
            for value in [0.3, -0., f64::NAN, f64::INFINITY] {
                let root = full(vec![3], dtype, value);
                let executable = crate::compile(
                    vec![root.clone(), bytes(), root, full(vec![0], dtype, 0.)],
                    0,
                )
                .unwrap();
                let cancelled = CancellationFlag::new();
                cancelled.cancel();
                assert!(executable.execute(&[], &[], &cancelled).is_err());
                let first = executable
                    .execute(&[], &[], &CancellationFlag::new())
                    .unwrap();
                let second = executable
                    .execute(&[], &[], &CancellationFlag::new())
                    .unwrap();
                drop(executable);
                drop(second);
                let expected = crate::value::dense_bytes_from_host(&[value; 3], dtype);
                assert_eq!(first[0].read_storage_bytes().unwrap(), expected);
                assert_eq!(first[2].read_storage_bytes().unwrap(), expected);
                assert_eq!(
                    first[1].read_storage_bytes().unwrap(),
                    (0u32..256).flat_map(u32::to_le_bytes).collect::<Vec<_>>()
                );
                assert!(first[3].read_storage_bytes().unwrap().is_empty());
            }
        }
    }
}
