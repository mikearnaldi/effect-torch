use crate::legalization::{validate_disposition, validate_execution};
use crate::test_target::TestTarget;
use crate::*;
use effect_torch_graph::{Device, Node, NodeKind};
use effect_torch_runtime::{DType, GgmlKQuant, StorageMetadata, StorageRepresentation};
use std::sync::Arc;

fn input(slot: u32, dtype: DType, shape: &[usize]) -> Arc<Node> {
    Node::new(NodeKind::Input {
        slot,
        dtype,
        shape: shape.to_vec(),
        device: Device::Cpu(0),
        storage: StorageMetadata::dense(),
    })
    .unwrap()
}
fn full(value: f64, dtype: DType) -> Arc<Node> {
    Node::new(NodeKind::Full {
        shape: vec![1],
        value,
        dtype,
        device: Device::Cpu(0),
    })
    .unwrap()
}
fn cancellation(dtype: DType) -> Arc<Node> {
    let x = input(0, dtype, &[1]);
    let sum = Node::new(NodeKind::Add {
        a: x.clone(),
        b: full(1.0, dtype),
    })
    .unwrap();
    Node::new(NodeKind::Sub { a: sum, b: x }).unwrap()
}
fn prepared(root: Arc<Node>, optimize: bool) -> PreparedProgram {
    ProgramRequest::from_roots(
        vec![root],
        CompileOptions {
            optimize,
            ..CompileOptions::default()
        },
    )
    .prepare()
    .unwrap()
}
fn evaluate(expression: &KernelExpr, inputs: &[f32]) -> f32 {
    let program = CpuFusionProgram::new(std::slice::from_ref(expression));
    let mut scratch = vec![0.0; program.value_scratch_len()];
    program.evaluate(0, |lane| inputs[lane as usize], |_| 0.0, &mut scratch)
}

#[test]
fn half_regions_round_each_semantic_node_and_keep_graph_identity() {
    for (dtype, value) in [(DType::BF16, 256.0), (DType::F16, 2048.0)] {
        let root = cancellation(dtype);
        let identity = Arc::as_ptr(&root);
        let prepared = prepared(root, true);
        let signature = prepared.signature.clone();
        let mut target = TestTarget::for_index(&prepared.index);
        target.promote_half = true;
        let driver = CompilerDriver::new(&prepared, &target).unwrap();
        assert_eq!(driver.optimization().regions.len(), 1);
        let unit = driver
            .legalization()
            .units()
            .iter()
            .find(|unit| matches!(unit.unit(), LoweringUnit::Region(_)))
            .unwrap();
        assert!(unit.disposition().is_legalized());
        assert_eq!(
            unit.execution().realization,
            ExecutionRealization::KernelLocal
        );
        let expressions =
            legalize_region_expressions(&driver.optimization().regions[0], unit.disposition())
                .unwrap();
        assert_eq!(evaluate(&expressions[0], &[value]), 0.0);
        let unrounded = KernelExpr::Sub(
            Box::new(KernelExpr::Add(
                Box::new(KernelExpr::Input(0)),
                Box::new(KernelExpr::cst(1.0)),
            )),
            Box::new(KernelExpr::Input(0)),
        );
        assert_eq!(evaluate(&unrounded, &[value]), 1.0);
        assert_eq!(identity, Arc::as_ptr(&prepared.roots[0]));
        assert_eq!(signature, prepared.signature);
        assert_eq!(driver.legalization().work().kernel_local_legalizations, 1);
    }
}

#[test]
fn folded_half_constants_obey_creation_rounding() {
    let dtype = DType::BF16;
    let x = input(0, dtype, &[1]);
    let sum = Node::new(NodeKind::Add {
        a: x.clone(),
        b: full(257.0, dtype),
    })
    .unwrap();
    let root = Node::new(NodeKind::Sub {
        a: sum,
        b: full(256.0, dtype),
    })
    .unwrap();
    let prepared = prepared(root, true);
    let mut target = TestTarget::for_index(&prepared.index);
    target.promote_half = true;
    let driver = CompilerDriver::new(&prepared, &target).unwrap();
    let unit = driver
        .legalization()
        .units()
        .iter()
        .find(|unit| matches!(unit.unit(), LoweringUnit::Region(_)))
        .unwrap();
    let expressions =
        legalize_region_expressions(&driver.optimization().regions[0], unit.disposition()).unwrap();
    assert_eq!(evaluate(&expressions[0], &[0.0]), 0.0);
}

#[test]
fn every_unit_is_classified_once_and_region_decisions_are_reused() {
    let prepared = prepared(cancellation(DType::F32), true);
    let target = TestTarget::for_index(&prepared.index);
    let mut driver = CompilerDriver::new(&prepared, &target).unwrap();
    assert_eq!(
        target.region_queries.get(),
        driver.optimization().work.capability_queries
    );
    let independent = driver
        .optimization()
        .lowering_order
        .iter()
        .filter(|unit| matches!(unit, LoweringUnit::Node(_)))
        .count();
    assert_eq!(target.node_queries.get(), independent);
    assert_eq!(
        driver.legalization().units().len(),
        driver.optimization().lowering_order.len()
    );
    let expected = driver.optimization().lowering_order.to_vec();
    let mut received = Vec::new();
    driver
        .lower(|unit, _, _, plan| {
            assert!(!plan.execution().operations.is_empty());
            received.push(unit);
            Ok(())
        })
        .unwrap();
    assert_eq!(received, expected);
}

#[test]
fn rejected_regions_leave_nodes_available_for_independent_lowering() {
    let prepared = prepared(cancellation(DType::F32), true);
    let mut target = TestTarget::for_index(&prepared.index);
    target.reject_regions = true;
    let driver = CompilerDriver::new(&prepared, &target).unwrap();
    assert!(driver.optimization().regions.is_empty());
    assert!(driver
        .optimization()
        .node_region
        .iter()
        .all(Option::is_none));
    assert_eq!(target.node_queries.get(), prepared.index.order.len());
    assert!(driver.legalization().work().rejected_region_candidates > 0);
}

#[test]
fn rejected_gemm_epilogue_does_not_reserve_nodes() {
    let device = Device::Metal(0);
    let make = |slot, shape: &[usize]| {
        Node::new(NodeKind::Input {
            slot,
            shape: shape.to_vec(),
            dtype: DType::F32,
            device: device.clone(),
            storage: StorageMetadata::dense(),
        })
        .unwrap()
    };
    let linear = Node::new(NodeKind::Linear {
        x: make(0, &[2, 3]),
        weight: make(1, &[3, 4]),
        bias: make(2, &[4]),
    })
    .unwrap();
    let root = Node::new(NodeKind::Gelu {
        a: linear,
        approximate: false,
    })
    .unwrap();
    let prepared = prepared(root, true);
    let mut target = TestTarget::for_index(&prepared.index);
    target.reject_regions = true;
    let driver = CompilerDriver::new(&prepared, &target).unwrap();
    assert_eq!(target.node_queries.get(), prepared.index.order.len());
    assert!(driver
        .optimization()
        .node_region
        .iter()
        .all(Option::is_none));
}

#[test]
fn rejected_multi_output_merge_retains_accepted_original_regions() {
    let x = input(0, DType::F32, &[1]);
    let prefix = Node::new(NodeKind::Tanh {
        a: Node::new(NodeKind::Neg { a: x }).unwrap(),
    })
    .unwrap();
    let a = Node::new(NodeKind::Tanh {
        a: Node::new(NodeKind::Neg { a: prefix.clone() }).unwrap(),
    })
    .unwrap();
    let b = Node::new(NodeKind::Exp {
        a: Node::new(NodeKind::Abs { a: prefix }).unwrap(),
    })
    .unwrap();
    let prepared = ProgramRequest::from_roots(vec![a, b], CompileOptions::default())
        .prepare()
        .unwrap();
    let mut target = TestTarget::for_index(&prepared.index);
    target.reject_multi_output = true;
    let driver = CompilerDriver::new(&prepared, &target).unwrap();
    assert_eq!(driver.optimization().regions.len(), 3);
    assert!(driver
        .optimization()
        .regions
        .iter()
        .all(|region| matches!(region, NativeRegion::Elementwise(_))));
    assert_eq!(driver.legalization().work().rejected_region_candidates, 1);
    driver.optimization().validate(&prepared.index).unwrap();
}

#[test]
fn unsupported_independent_node_fails_before_lowering() {
    let prepared = prepared(cancellation(DType::F32), false);
    let mut target = TestTarget::for_index(&prepared.index);
    target.reject_nodes = true;
    let error = CompilerDriver::new(&prepared, &target).err().unwrap();
    assert!(error.contains("node 0"));
    assert!(error.contains("cpu:0"));
    assert!(error.contains("test independent operation rejected"));
}

#[test]
fn malformed_execution_recipes_cannot_enter_the_plan() {
    let prepared = prepared(cancellation(DType::BF16), false);
    let node = *prepared.index.roots.first().unwrap();
    let spec = OperationDTypeSpec::new(&prepared.index, node).unwrap();
    let valid = spec
        .native_execution()
        .promote_half(ExecutionRealization::MaterializedTransforms);
    let check = |execution| {
        validate_disposition(
            std::slice::from_ref(&spec),
            DTypeDisposition::Legalize(execution),
        )
    };
    check(valid.clone()).unwrap();
    let mut invalid = valid.clone();
    invalid.operations[0].rounding_boundaries = Box::new([]);
    assert!(check(invalid).unwrap_err().contains("rounding"));
    let mut invalid = valid.clone();
    invalid.operations[0].results[0].completion = ResultCompletion::Direct;
    assert!(check(invalid).unwrap_err().contains("restore"));
    let mut invalid = valid.clone();
    invalid.operations[0].operands[0].role = ValueRole::Condition;
    assert!(check(invalid).unwrap_err().contains("role"));
    let mut invalid = valid.clone();
    invalid.operations[0].operands[0].preparation = OperandPreparation::Direct;
    assert!(check(invalid).unwrap_err().contains("operand"));
    let mut invalid = valid.clone();
    invalid.operations[0].node = DenseNodeId::new(u32::MAX);
    assert!(check(invalid).unwrap_err().contains("coverage"));
    let mut invalid = valid.clone();
    invalid.realization = ExecutionRealization::DirectKernel;
    assert!(check(invalid).unwrap_err().contains("realization"));
    assert!(
        validate_disposition(std::slice::from_ref(&spec), DTypeDisposition::Native(valid)).is_err()
    );
    let mut empty = spec.native_execution();
    empty.realization = ExecutionRealization::KernelLocal;
    assert!(check(empty).unwrap_err().contains("no transformation"));
}

#[test]
fn reduction_accumulation_and_f64_precision_cannot_be_weakened() {
    let root = Node::new(NodeKind::Sum {
        a: input(0, DType::BF16, &[3]),
        dims: vec![0],
        keepdims: false,
    })
    .unwrap();
    let index = GraphIndex::new(&[root]).unwrap();
    let spec = OperationDTypeSpec::new(&index, index.roots[0]).unwrap();
    assert_eq!(
        spec.required_numerics.accumulation.unwrap().dtype,
        DType::F32
    );
    let mut invalid = spec.native_execution();
    invalid.operations[0].accumulation.as_mut().unwrap().dtype = DType::BF16;
    assert!(
        validate_disposition(&[spec], DTypeDisposition::Native(invalid))
            .unwrap_err()
            .contains("accumulation")
    );
    let prepared = prepared(cancellation(DType::F64), false);
    let spec = OperationDTypeSpec::new(&prepared.index, prepared.index.roots[0]).unwrap();
    let mut invalid = spec.native_execution();
    invalid.operations[0].compute_dtype = Some(DType::F32);
    assert!(
        validate_disposition(&[spec], DTypeDisposition::Native(invalid))
            .unwrap_err()
            .contains("compute")
    );
}

#[test]
fn packed_queries_use_logical_f32_values_and_canonical_access_only() {
    let weight = Node::new(NodeKind::Input {
        slot: 1,
        shape: vec![1, 256],
        dtype: DType::F32,
        device: Device::Cpu(0),
        storage: StorageMetadata::packed(GgmlKQuant::Q4K),
    })
    .unwrap();
    let root = Node::new(NodeKind::QuantizedLinear {
        x: input(0, DType::F32, &[1, 256]),
        weight,
        bias: None,
    })
    .unwrap();
    let prepared = prepared(root, false);
    let target = TestTarget::for_index(&prepared.index);
    CompilerDriver::new(&prepared, &target).unwrap();
    let spec = OperationDTypeSpec::new(&prepared.index, prepared.index.roots[0]).unwrap();
    assert_eq!(spec.operands[1].role, ValueRole::Weight);
    assert_eq!(spec.operands[1].value.semantic_dtype, DType::F32);
    assert_eq!(spec.operands[1].value.logical_shape, &[1, 256]);
    assert!(matches!(
        spec.operands[1].value.storage.representation,
        StorageRepresentation::Packed(_)
    ));
    assert_eq!(
        spec.required_numerics.operand_interpretations[1],
        OperandInterpretation::CanonicalPacked
    );
    let mut execution = spec.native_execution();
    execution.operations[0].operands[1].preparation = OperandPreparation::Direct;
    assert!(validate_disposition(
        std::slice::from_ref(&spec),
        DTypeDisposition::Native(execution)
    )
    .is_err());
    let mut execution = spec.native_execution();
    execution.operations[0].operands[1].preparation =
        OperandPreparation::CanonicalPacked(PackedOperandAccess::TileBounded {
            max_logical_elements: 256,
            max_scratch_bytes: 0,
        });
    assert!(validate_disposition(
        std::slice::from_ref(&spec),
        DTypeDisposition::Native(execution)
    )
    .unwrap_err()
    .contains("bound"));
    let mut execution = spec.native_execution();
    execution.operations[0].operands[1].execution_dtype = DType::U8;
    assert!(validate_disposition(&[spec], DTypeDisposition::Native(execution)).is_err());
}

#[test]
fn multi_result_operations_describe_each_result_and_picker_port() {
    let step = Node::new(NodeKind::SgdStep {
        param: input(0, DType::F32, &[2]),
        grad: input(1, DType::F32, &[2]),
        velocity: input(2, DType::F32, &[2]),
        first: input(3, DType::F32, &[]),
        lr: input(4, DType::F32, &[]),
        momentum: 0.9,
        dampening: 0.0,
        nesterov: false,
        weight_decay: 0.0,
    })
    .unwrap();
    let picker = Node::new(NodeKind::SgdOut {
        step: step.clone(),
        index: 1,
    })
    .unwrap();
    let index = GraphIndex::new(&[picker]).unwrap();
    let spec = OperationDTypeSpec::new(&index, index.dense_id(step.id).unwrap()).unwrap();
    assert_eq!(spec.results.len(), 2);
    assert_eq!(spec.required_numerics.rounding_boundaries.len(), 2);
    let picker = OperationDTypeSpec::new(&index, index.roots[0]).unwrap();
    assert_eq!(picker.operands[0].source_result, 1);
    assert_eq!(picker.required_numerics.compute_dtype, None);
}

#[test]
fn target_identity_and_placement_are_validated_before_lowering() {
    let prepared = prepared(cancellation(DType::F32), true);
    let target = TestTarget::for_index(&prepared.index);
    let driver = CompilerDriver::new(&prepared, &target).unwrap();
    let mut changed = TestTarget::for_index(&prepared.index);
    changed.revision += 1;
    assert!(driver
        .legalization()
        .validate(&prepared.index, driver.optimization(), &changed)
        .unwrap_err()
        .contains("revision"));
    changed.revision = target.revision;
    changed.fingerprint.lowering_abi_revision += 1;
    assert!(driver
        .legalization()
        .validate(&prepared.index, driver.optimization(), &changed)
        .unwrap_err()
        .contains("fingerprint"));
    let other = TestTarget::new(Device::Metal(0));
    assert!(CompilerDriver::new(&prepared, &other)
        .err()
        .unwrap()
        .contains("placement"));
}

#[test]
fn expression_application_rejects_a_missing_semantic_operation() {
    let prepared = prepared(cancellation(DType::BF16), true);
    let mut target = TestTarget::for_index(&prepared.index);
    target.promote_half = true;
    let driver = CompilerDriver::new(&prepared, &target).unwrap();
    let region = &driver.optimization().regions[0];
    let mut execution = RegionDTypeSpec::new(&prepared.index, region)
        .unwrap()
        .native_execution()
        .promote_half(ExecutionRealization::KernelLocal);
    execution.operations = execution
        .operations
        .into_vec()
        .into_iter()
        .skip(1)
        .collect();
    let invalid = ExecutableDTypePlan::Legalize(Arc::new(execution));
    assert!(legalize_region_expressions(region, &invalid)
        .unwrap_err()
        .contains("absent"));
    let specs = RegionDTypeSpec::new(&prepared.index, region).unwrap();
    assert!(validate_execution(&specs.operations, &invalid).is_err());
}

#[test]
fn packed_binding_signature_separates_logical_and_physical_geometry() {
    use effect_torch_runtime::{BindingLayoutPolicy, LayoutConstraint};
    let weight = Node::new(NodeKind::Input {
        slot: 1,
        shape: vec![3, 256],
        dtype: DType::F32,
        device: Device::Cpu(0),
        storage: StorageMetadata::packed(GgmlKQuant::Q4K),
    })
    .unwrap();
    let root = Node::new(NodeKind::QuantizedLinear {
        x: input(0, DType::F32, &[1, 256]),
        weight,
        bias: None,
    })
    .unwrap();
    let prepared = prepared(root, false);
    let binding = &prepared.signature.bindings[1];
    assert_eq!(binding.shape, [3, 256]);
    assert_eq!(binding.dtype, DType::F32);
    let BindingLayoutPolicy::Require(LayoutConstraint::Exact(layout)) = &binding.layout else {
        panic!("expected exact CPU binding geometry")
    };
    assert_eq!(layout.shape(), &[3, 144]);
    assert_eq!(
        prepared.index.slots[1].storage,
        StorageMetadata::packed(GgmlKQuant::Q4K)
    );
}

#[test]
fn packed_roots_are_rejected_even_when_storage_is_supported() {
    let root = Node::new(NodeKind::Input {
        slot: 0,
        shape: vec![1, 256],
        dtype: DType::F32,
        device: Device::Cpu(0),
        storage: StorageMetadata::packed(GgmlKQuant::Q4K),
    })
    .unwrap();
    let prepared = prepared(root, false);
    let target = TestTarget::for_index(&prepared.index);
    assert!(CompilerDriver::new(&prepared, &target)
        .err()
        .unwrap()
        .contains("packed program outputs"));
}

#[test]
fn multi_output_inlining_retains_shared_half_rounding_boundaries() {
    let x = input(0, DType::BF16, &[1]);
    let sum = Node::new(NodeKind::Add {
        a: x.clone(),
        b: full(1.0, DType::BF16),
    })
    .unwrap();
    let prefix = Node::new(NodeKind::Neg { a: sum }).unwrap();
    let a = Node::new(NodeKind::Neg {
        a: Node::new(NodeKind::Add {
            a: prefix.clone(),
            b: x.clone(),
        })
        .unwrap(),
    })
    .unwrap();
    let b = Node::new(NodeKind::Neg {
        a: Node::new(NodeKind::Sub { a: prefix, b: x }).unwrap(),
    })
    .unwrap();
    let prepared = ProgramRequest::from_roots(vec![a, b], CompileOptions::default())
        .prepare()
        .unwrap();
    let mut target = TestTarget::for_index(&prepared.index);
    target.promote_half = true;
    let driver = CompilerDriver::new(&prepared, &target).unwrap();
    assert!(matches!(
        driver.optimization().regions[0],
        NativeRegion::MultiOutput(_)
    ));
    let entry = driver
        .legalization()
        .units()
        .iter()
        .find(|entry| matches!(entry.unit(), LoweringUnit::Region(_)))
        .unwrap();
    let expressions =
        legalize_region_expressions(&driver.optimization().regions[0], entry.disposition())
            .unwrap();
    assert_eq!(evaluate(&expressions[0], &[256.0]), 0.0);
    assert_eq!(evaluate(&expressions[1], &[256.0]), 512.0);
}

#[test]
fn graph_casts_remain_explicit_inside_region_expressions() {
    let x = input(0, DType::F32, &[1]);
    let narrow = Node::new(NodeKind::Cast {
        a: x,
        dtype: DType::BF16,
    })
    .unwrap();
    let wide = Node::new(NodeKind::Cast {
        a: narrow,
        dtype: DType::F32,
    })
    .unwrap();
    let prepared = prepared(wide, true);
    let target = TestTarget::for_index(&prepared.index);
    let driver = CompilerDriver::new(&prepared, &target).unwrap();
    let entry = driver
        .legalization()
        .units()
        .iter()
        .find(|entry| matches!(entry.unit(), LoweringUnit::Region(_)))
        .unwrap();
    let expressions =
        legalize_region_expressions(&driver.optimization().regions[0], entry.disposition())
            .unwrap();
    assert_eq!(evaluate(&expressions[0], &[257.0]), 256.0);
    assert_eq!(prepared.signature.outputs[0].dtype, DType::F32);
}

#[test]
fn optimized_and_independent_half_operations_have_identical_semantic_boundaries() {
    let root = cancellation(DType::BF16);
    let optimized = prepared(root.clone(), true);
    let independent = prepared(root, false);
    let mut target = TestTarget::for_index(&optimized.index);
    target.promote_half = true;
    let fused = CompilerDriver::new(&optimized, &target).unwrap();
    let separate = CompilerDriver::new(&independent, &target).unwrap();
    let boundaries = |plan: &LegalizationPlan| {
        let mut boundaries = plan
            .units()
            .iter()
            .flat_map(|unit| unit.execution().operations.iter())
            .flat_map(|operation| operation.rounding_boundaries.iter().copied())
            .collect::<Vec<_>>();
        boundaries.sort_by_key(|boundary| (boundary.node, boundary.result));
        boundaries
    };
    assert_eq!(
        boundaries(fused.legalization()),
        boundaries(separate.legalization())
    );
    assert_eq!(optimized.signature, independent.signature);
}

#[test]
fn folded_f16_constant_does_not_double_round_through_f32() {
    let x = input(0, DType::F16, &[1]);
    let sum = Node::new(NodeKind::Add {
        a: x,
        b: full(1.0004882821813226, DType::F16),
    })
    .unwrap();
    let root = Node::new(NodeKind::Sub {
        a: sum,
        b: full(1.0, DType::F16),
    })
    .unwrap();
    let prepared = prepared(root, true);
    let mut target = TestTarget::for_index(&prepared.index);
    target.promote_half = true;
    let driver = CompilerDriver::new(&prepared, &target).unwrap();
    let unit = driver
        .legalization()
        .units()
        .iter()
        .find(|unit| matches!(unit.unit(), LoweringUnit::Region(_)))
        .unwrap();
    let expressions =
        legalize_region_expressions(&driver.optimization().regions[0], unit.disposition()).unwrap();
    assert_eq!(evaluate(&expressions[0], &[0.0]), 0.0009765625);
}

#[test]
fn full_driver_legalization_is_stack_safe_for_deep_regions() {
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| {
            let mut root = input(0, DType::BF16, &[1]);
            for _ in 0..20_000 {
                root = Node::new(NodeKind::Neg { a: root }).unwrap();
            }
            let prepared = prepared(root, true);
            let mut target = TestTarget::for_index(&prepared.index);
            target.promote_half = true;
            let driver = CompilerDriver::new(&prepared, &target).unwrap();
            let unit = driver
                .legalization()
                .units()
                .iter()
                .find(|unit| matches!(unit.unit(), LoweringUnit::Region(_)))
                .unwrap();
            let expressions =
                legalize_region_expressions(&driver.optimization().regions[0], unit.disposition())
                    .unwrap();
            assert_eq!(evaluate(&expressions[0], &[256.0]), 256.0);
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn mixed_scalar_coercion_precedes_half_arithmetic_in_both_operand_orders() {
    for (dtype, epsilon, uncoerced) in [
        (DType::BF16, 1.0 / 256.0, 3.015625),
        (DType::F16, 1.0 / 2048.0, 3.001953125),
    ] {
        for scalar_first in [false, true] {
            let tensor = input(0, dtype, &[1]);
            let scalar = input(1, DType::F32, &[]);
            let (a, b) = if scalar_first {
                (scalar, tensor)
            } else {
                (tensor, scalar)
            };
            let product = Node::new(NodeKind::Mul { a, b }).unwrap();
            let root = Node::new(NodeKind::Neg {
                a: Node::new(NodeKind::Neg { a: product.clone() }).unwrap(),
            })
            .unwrap();
            let prepared = prepared(root, true);
            let mut target = TestTarget::for_index(&prepared.index);
            target.promote_half = true;
            let driver = CompilerDriver::new(&prepared, &target).unwrap();
            let product_id = prepared.index.dense_id(product.id).unwrap();
            let spec = OperationDTypeSpec::new(&prepared.index, product_id).unwrap();
            let scalar_operand = usize::from(!scalar_first);
            let coercion = OperandInterpretation::ScalarCoercion(DenseConversionContract::new(
                DType::F32,
                dtype,
            ));
            assert_eq!(
                spec.required_numerics.operand_interpretations[scalar_operand],
                coercion
            );
            let entry = driver
                .legalization()
                .units()
                .iter()
                .find(|entry| matches!(entry.unit(), LoweringUnit::Region(_)))
                .unwrap();
            assert_eq!(
                entry.execution().operation(product_id).unwrap().operands[scalar_operand]
                    .interpretation,
                coercion
            );
            let region = &driver.optimization().regions[0];
            let expressions = legalize_region_expressions(region, entry.disposition()).unwrap();
            let lanes = region
                .inputs()
                .iter()
                .map(|node| {
                    if prepared.index.order[node.index()].shape.is_empty() {
                        1.0 + epsilon
                    } else {
                        3.0
                    }
                })
                .collect::<Vec<f32>>();
            assert_eq!(evaluate(&expressions[0], &lanes), 3.0);
            let wrong = KernelExpr::RoundTo(
                Box::new(KernelExpr::Mul(
                    Box::new(KernelExpr::cst(3.0)),
                    Box::new(KernelExpr::cst(f64::from(1.0 + epsilon))),
                )),
                dtype,
            );
            assert_eq!(evaluate(&wrong, &[]), uncoerced);
            let mut malformed = spec
                .native_execution()
                .promote_half(ExecutionRealization::MaterializedTransforms);
            malformed.operations[0].operands[scalar_operand].interpretation =
                OperandInterpretation::Direct;
            assert!(
                validate_disposition(&[spec], DTypeDisposition::Legalize(malformed))
                    .unwrap_err()
                    .contains("coercion")
            );
        }
    }
}

#[test]
fn half_attention_forward_and_backward_allow_f32_compute_and_require_f32_accumulation() {
    for dtype in [DType::F16, DType::BF16] {
        let q = input(0, dtype, &[1, 1, 2, 2]);
        let k = input(1, dtype, &[1, 1, 2, 2]);
        let v = input(2, dtype, &[1, 1, 2, 2]);
        let fwd = Node::new(NodeKind::Sdpa {
            q: q.clone(),
            k: k.clone(),
            v: v.clone(),
            scale: 0.5,
            causal: false,
            window: effect_torch_graph::AttentionWindow::Inherit,
        })
        .unwrap();
        let backward = Node::new(NodeKind::SdpaBackward {
            q,
            k,
            v,
            g: input(3, dtype, &[1, 1, 2, 2]),
            fwd: fwd.clone(),
            scale: 0.5,
            causal: false,
            window: effect_torch_graph::AttentionWindow::Inherit,
        })
        .unwrap();
        for root in [fwd, backward] {
            let prepared = prepared(root, false);
            let spec = OperationDTypeSpec::new(&prepared.index, prepared.index.roots[0]).unwrap();
            assert!(spec.required_numerics.permits_f32_compute);
            assert_eq!(
                spec.required_numerics.accumulation.unwrap().dtype,
                DType::F32
            );
            let plan = validate_disposition(
                std::slice::from_ref(&spec),
                DTypeDisposition::Legalize(
                    spec.native_execution()
                        .promote_half(ExecutionRealization::MaterializedTransforms),
                ),
            )
            .unwrap();
            assert_eq!(
                plan.execution().operations[0].compute_dtype,
                Some(DType::F32)
            );
            for result in &plan.execution().operations[0].results {
                assert_eq!(
                    result.completion,
                    ResultCompletion::ConvertToBoundary(DenseConversionContract::new(
                        DType::F32,
                        dtype
                    ))
                );
            }
        }
    }
}

#[test]
fn half_model_composites_allow_only_declared_f32_execution_and_result_rounding() {
    for dtype in [DType::F16, DType::BF16] {
        for operation in model_composites(dtype)
            .into_iter()
            .chain(audited_half_composites(dtype))
        {
            let root = Node::new(operation).unwrap();
            let prepared = prepared(root, false);
            let spec = OperationDTypeSpec::new(&prepared.index, prepared.index.roots[0]).unwrap();
            assert!(
                spec.required_numerics.permits_f32_compute,
                "{}",
                operation_name(spec.operation)
            );
            let expected_accumulation = if matches!(
                spec.operation,
                NodeKind::RotaryEmbedding { .. }
                    | NodeKind::RotaryEmbeddingBackward { .. }
                    | NodeKind::AdamWStep { .. }
                    | NodeKind::SgdStep { .. }
            ) {
                None
            } else {
                Some(AccumulationExecution {
                    dtype: DType::F32,
                    order: ReductionOrder::BackendDefined,
                })
            };
            assert_eq!(spec.required_numerics.accumulation, expected_accumulation);
            assert_eq!(
                spec.required_numerics.rounding_boundaries.len(),
                spec.results.len()
            );
            for realization in [
                ExecutionRealization::KernelLocal,
                ExecutionRealization::MaterializedTransforms,
            ] {
                let execution = spec.native_execution().promote_half(realization);
                let plan = validate_disposition(
                    std::slice::from_ref(&spec),
                    DTypeDisposition::Legalize(execution.clone()),
                )
                .unwrap();
                let recipe = &plan.execution().operations[0];
                assert_eq!(recipe.compute_dtype, Some(DType::F32));
                assert_eq!(recipe.accumulation, expected_accumulation);
                for (value, operand) in spec.operands.iter().zip(recipe.operands.iter()) {
                    if value.value.semantic_dtype == dtype {
                        assert_eq!(operand.execution_dtype, DType::F32);
                        assert_eq!(
                            operand.preparation,
                            OperandPreparation::Convert(DenseConversionContract::new(
                                dtype,
                                DType::F32
                            ))
                        );
                    } else {
                        // CE target indices retain their integer storage and interpretation.
                        assert_eq!(operand.execution_dtype, value.value.semantic_dtype);
                        assert_eq!(operand.preparation, OperandPreparation::Direct);
                    }
                }
                for result in &recipe.results {
                    assert_eq!(result.execution_dtype, DType::F32);
                    assert_eq!(
                        result.completion,
                        ResultCompletion::ConvertToBoundary(DenseConversionContract::new(
                            DType::F32,
                            dtype
                        ))
                    );
                }
                let mut invalid = execution.clone();
                invalid.operations[0].rounding_boundaries = Box::new([]);
                assert!(validate_disposition(
                    std::slice::from_ref(&spec),
                    DTypeDisposition::Legalize(invalid)
                )
                .unwrap_err()
                .contains("rounding"));
                for result in 0..spec.results.len() {
                    let mut invalid = execution.clone();
                    invalid.operations[0].results[result].completion = ResultCompletion::Direct;
                    assert!(validate_disposition(
                        std::slice::from_ref(&spec),
                        DTypeDisposition::Legalize(invalid)
                    )
                    .unwrap_err()
                    .contains("restore"));
                }
                let mut invalid = execution;
                invalid.operations[0].compute_dtype = Some(DType::F64);
                let error = validate_disposition(
                    std::slice::from_ref(&spec),
                    DTypeDisposition::Legalize(invalid),
                )
                .unwrap_err();
                assert!(error.contains(operation_name(spec.operation)), "{error}");
                assert!(error.contains("cpu:0"), "{error}");
                assert!(
                    error.contains(&format!("expected {dtype} or f32; actual f64")),
                    "{error}"
                );
            }
            let mut target = TestTarget::for_index(&prepared.index);
            target.promote_half = true;
            let driver = CompilerDriver::new(&prepared, &target).unwrap();
            assert_eq!(driver.legalization().work().legalized_lowering_units, 1);
        }
    }
}

#[test]
fn composite_half_approval_does_not_change_f32_or_f64_compute() {
    for (dtype, invalid_compute) in [(DType::F32, DType::F64), (DType::F64, DType::F32)] {
        for operation in model_composites(dtype)
            .into_iter()
            .chain(audited_half_composites(dtype))
        {
            let root = Node::new(operation).unwrap();
            let prepared = prepared(root, false);
            let spec = OperationDTypeSpec::new(&prepared.index, prepared.index.roots[0]).unwrap();
            assert!(!spec.required_numerics.permits_f32_compute);
            let mut invalid = spec.native_execution();
            invalid.operations[0].compute_dtype = Some(invalid_compute);
            let error = validate_disposition(
                std::slice::from_ref(&spec),
                DTypeDisposition::Native(invalid),
            )
            .unwrap_err();
            assert!(error.contains(operation_name(spec.operation)), "{error}");
            assert!(
                error.contains(&format!("expected {dtype}; actual {invalid_compute}")),
                "{error}"
            );
        }
    }
}

#[test]
fn composite_approval_does_not_admit_unapproved_half_operations() {
    for dtype in [DType::F16, DType::BF16] {
        let root = Node::new(NodeKind::Inverse {
            a: input(0, dtype, &[2, 2]),
        })
        .unwrap();
        let prepared = prepared(root, false);
        let spec = OperationDTypeSpec::new(&prepared.index, prepared.index.roots[0]).unwrap();
        assert!(!spec.required_numerics.permits_f32_compute);
        let execution = spec
            .native_execution()
            .promote_half(ExecutionRealization::MaterializedTransforms);
        let error = validate_disposition(
            std::slice::from_ref(&spec),
            DTypeDisposition::Legalize(execution),
        )
        .unwrap_err();
        assert!(error.contains("inverse on cpu:0"), "{error}");
        assert!(
            error.contains(&format!("expected {dtype} or f64; actual f32")),
            "{error}"
        );
    }
}

fn audited_half_composites(dtype: DType) -> Vec<NodeKind> {
    let x1 = input(0, dtype, &[1, 1, 3]);
    let w1 = input(1, dtype, &[1, 1, 2]);
    let x2 = input(0, dtype, &[1, 1, 3, 3]);
    let w2 = input(1, dtype, &[1, 1, 2, 2]);
    let short_x = input(0, dtype, &[1, 3, 2]);
    let short_w = input(1, dtype, &[2, 3]);
    let q = input(0, dtype, &[1, 2, 3, 4]);
    let k = input(1, dtype, &[1, 2, 3, 4]);
    let v = input(2, dtype, &[1, 2, 3, 5]);
    let decay = input(3, dtype, &[1, 2, 3, 4]);
    let beta = input(4, dtype, &[1, 2, 3, 1]);
    vec![
        NodeKind::Conv1d {
            x: x1.clone(),
            w: w1.clone(),
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
        },
        NodeKind::Conv2d {
            x: x2.clone(),
            w: w2.clone(),
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
        },
        NodeKind::ConvTranspose1d {
            x: x1.clone(),
            w: w1,
            stride: 1,
            padding: 0,
            output_padding: 0,
            dilation: 1,
            groups: 1,
        },
        NodeKind::ConvTranspose2d {
            x: x2.clone(),
            w: w2,
            stride: 1,
            padding: 0,
            output_padding: 0,
            dilation: 1,
            groups: 1,
        },
        NodeKind::Conv1dBackwardW {
            x: x1,
            g: input(1, dtype, &[1, 1, 2]),
            kernel: 2,
            out_channels: 1,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
        },
        NodeKind::Conv2dBackwardW {
            x: x2,
            g: input(1, dtype, &[1, 1, 2, 2]),
            kernel: [2, 2],
            out_channels: 1,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
        },
        NodeKind::ShortConv1d {
            x: short_x.clone(),
            weight: short_w.clone(),
        },
        NodeKind::ShortConv1dBackwardX {
            x: short_x.clone(),
            weight: short_w.clone(),
            g: input(2, dtype, &[1, 3, 2]),
        },
        NodeKind::ShortConv1dBackwardW {
            x: short_x.clone(),
            weight: short_w.clone(),
            g: input(2, dtype, &[1, 3, 2]),
        },
        NodeKind::ConvState {
            x: short_x,
            weight: short_w,
            layer: 0,
        },
        NodeKind::KdaChunk {
            q: q.clone(),
            k: k.clone(),
            v: v.clone(),
            log_decay: decay.clone(),
            beta: beta.clone(),
            scale: 0.5,
        },
        NodeKind::KdaRecurrence {
            q: q.clone(),
            k: k.clone(),
            v: v.clone(),
            log_decay: decay.clone(),
            beta: beta.clone(),
            scale: 0.5,
            layer: 0,
        },
        NodeKind::KdaBackward {
            q,
            k,
            v,
            log_decay: decay,
            beta,
            g: input(5, dtype, &[1, 2, 3, 5]),
            scale: 0.5,
        },
        NodeKind::AdamWStep {
            param: input(0, dtype, &[3]),
            grad: input(1, dtype, &[3]),
            m: input(2, dtype, &[3]),
            v: input(3, dtype, &[3]),
            lr: input(4, dtype, &[]),
            c1: input(5, dtype, &[]),
            c2: input(6, dtype, &[]),
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        },
        NodeKind::SgdStep {
            param: input(0, dtype, &[3]),
            grad: input(1, dtype, &[3]),
            velocity: input(2, dtype, &[3]),
            first: input(3, dtype, &[]),
            lr: input(4, dtype, &[]),
            momentum: 0.9,
            dampening: 0.0,
            nesterov: true,
            weight_decay: 0.01,
        },
        NodeKind::Cumsum {
            a: input(0, dtype, &[2, 3]),
            dim: 1,
        },
    ]
}

fn model_composites(dtype: DType) -> Vec<NodeKind> {
    use effect_torch_graph::{CrossEntropyReduction, PositionOffset, RotaryLayout};
    let x = input(0, dtype, &[2, 4]);
    let weight = input(1, dtype, &[4]);
    let mut operations = vec![
        NodeKind::LayerNorm {
            x: x.clone(),
            weight: weight.clone(),
            bias: input(2, dtype, &[4]),
            eps: 1e-5,
        },
        NodeKind::LayerNormBackward {
            x: x.clone(),
            weight: weight.clone(),
            g: x.clone(),
            eps: 1e-5,
        },
        NodeKind::RmsNorm {
            x: x.clone(),
            weight: Some(weight),
            eps: 1e-5,
        },
        NodeKind::RmsNorm {
            x: x.clone(),
            weight: None,
            eps: 1e-5,
        },
    ];
    for target_dtype in [DType::I64, DType::U32] {
        let target = input(1, target_dtype, &[2]);
        for reduction in [CrossEntropyReduction::Sum, CrossEntropyReduction::Mean] {
            operations.push(NodeKind::CrossEntropy {
                logits: x.clone(),
                target: target.clone(),
                ignore_index: -1,
                reduction,
            });
            operations.push(NodeKind::CrossEntropyBackward {
                logits: x.clone(),
                target: target.clone(),
                ignore_index: -1,
                reduction,
            });
        }
    }
    for layout in [RotaryLayout::HalfSplit, RotaryLayout::InterleavedPairs] {
        operations.push(NodeKind::RotaryEmbedding {
            x: x.clone(),
            seq_len: 2,
            theta: 10000.0,
            offset: PositionOffset::Absolute,
            layout,
        });
        operations.push(NodeKind::RotaryEmbeddingBackward {
            g: x.clone(),
            shape: vec![2, 4],
            seq_len: 2,
            theta: 10000.0,
            layout,
        });
    }
    operations
}

#[test]
fn composite_reductions_reject_narrowed_accumulators() {
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let x = input(0, dtype, &[1, 2]);
        let weight = input(1, dtype, &[2]);
        let bias = input(2, dtype, &[2]);
        let operations = [
            NodeKind::LayerNorm {
                x: x.clone(),
                weight: weight.clone(),
                bias,
                eps: 1e-5,
            },
            NodeKind::RmsNorm {
                x: x.clone(),
                weight: Some(weight.clone()),
                eps: 1e-5,
            },
            NodeKind::LayerNormBackward {
                x: x.clone(),
                weight,
                g: x.clone(),
                eps: 1e-5,
            },
            NodeKind::CrossEntropy {
                logits: x,
                target: input(1, DType::I64, &[1]),
                ignore_index: -1,
                reduction: effect_torch_graph::CrossEntropyReduction::Mean,
            },
            NodeKind::Conv1d {
                x: input(0, dtype, &[1, 1, 3]),
                w: input(1, dtype, &[1, 1, 2]),
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: 1,
            },
        ];
        for operation in operations {
            let root = Node::new(operation).unwrap();
            let prepared = prepared(root, false);
            let spec = OperationDTypeSpec::new(&prepared.index, prepared.index.roots[0]).unwrap();
            let expected = if dtype == DType::F64 {
                DType::F64
            } else {
                DType::F32
            };
            assert_eq!(spec.required_numerics.accumulation.unwrap().dtype, expected);
            validate_disposition(
                std::slice::from_ref(&spec),
                DTypeDisposition::Native(spec.native_execution()),
            )
            .unwrap();
            for replacement in [
                None,
                Some(AccumulationExecution {
                    dtype: DType::F16,
                    order: ReductionOrder::BackendDefined,
                }),
            ] {
                let mut malformed = spec.native_execution();
                malformed.operations[0].accumulation = replacement;
                assert!(validate_disposition(
                    std::slice::from_ref(&spec),
                    DTypeDisposition::Native(malformed)
                )
                .unwrap_err()
                .contains("accumulation"));
            }
        }
    }
}

#[test]
fn half_random_legalization_preserves_sources_stream_mapping_and_sample_counts() {
    for dtype in [DType::F16, DType::BF16] {
        let normal = || {
            Node::new(NodeKind::Randn {
                shape: vec![3],
                dtype,
                device: Device::Cpu(0),
            })
            .unwrap()
        };
        let first = normal();
        let uniform = Node::new(NodeKind::Uniform {
            shape: vec![3],
            lo: -0.5,
            hi: 1.5,
            dtype,
            device: Device::Cpu(0),
        })
        .unwrap();
        let roots = vec![first.clone(), uniform, normal(), first];
        let mut original_sources = None;
        for optimize in [false, true] {
            let prepared = ProgramRequest::from_roots(
                roots.clone(),
                CompileOptions {
                    optimize,
                    ..CompileOptions::default()
                },
            )
            .prepare()
            .unwrap();
            let sources = prepared.index.random_source_order.to_vec();
            assert_eq!(sources.len(), 3);
            if let Some(original) = &original_sources {
                assert_eq!(&sources, original);
            } else {
                original_sources = Some(sources.clone());
            }
            let mut target = TestTarget::for_index(&prepared.index);
            target.promote_half = true;
            let driver = CompilerDriver::new(&prepared, &target).unwrap();
            assert_eq!(driver.legalization().units().len(), 3);
            assert_eq!(driver.legalization().work().legalized_lowering_units, 3);
            for (entry, source) in driver.legalization().units().iter().zip(sources.iter()) {
                let required = RandomExecution {
                    source: *source,
                    stream: RandomStreamMapping::PreserveSemanticSource,
                    samples: 3,
                };
                let spec = OperationDTypeSpec::new(&prepared.index, source.node).unwrap();
                assert_eq!(spec.required_numerics.random, Some(required));
                assert_eq!(entry.execution().operations[0].random, Some(required));
                validate_disposition(
                    std::slice::from_ref(&spec),
                    DTypeDisposition::Native(spec.native_execution()),
                )
                .unwrap();
                for realization in [
                    ExecutionRealization::KernelLocal,
                    ExecutionRealization::MaterializedTransforms,
                ] {
                    let execution = spec.native_execution().promote_half(realization);
                    let recipe = &execution.operations[0];
                    assert_eq!(recipe.compute_dtype, Some(DType::F32));
                    assert_eq!(recipe.random, Some(required));
                    assert!(recipe.operands.is_empty());
                    assert_eq!(
                        recipe.results[0].completion,
                        ResultCompletion::ConvertToBoundary(DenseConversionContract::new(
                            DType::F32,
                            dtype
                        ))
                    );
                    validate_disposition(
                        std::slice::from_ref(&spec),
                        DTypeDisposition::Legalize(execution.clone()),
                    )
                    .unwrap();
                    let mut wrong_count = required;
                    wrong_count.samples += 1;
                    let mut wrong_source = required;
                    wrong_source.source.id = RandomSourceId::new(source.id.get() + 1);
                    let mut wrong_provenance = required;
                    wrong_provenance.source.provenance += 1;
                    let mut wrong_distribution = required;
                    wrong_distribution.source.kind = match source.kind {
                        RandomSourceKind::Randn => RandomSourceKind::Uniform,
                        RandomSourceKind::Uniform => RandomSourceKind::Randn,
                    };
                    for random in [
                        None,
                        Some(wrong_count),
                        Some(wrong_source),
                        Some(wrong_provenance),
                        Some(wrong_distribution),
                    ] {
                        let mut malformed = execution.clone();
                        malformed.operations[0].random = random;
                        assert!(validate_disposition(
                            std::slice::from_ref(&spec),
                            DTypeDisposition::Legalize(malformed)
                        )
                        .unwrap_err()
                        .contains("random stream/counter"));
                    }
                    let mut malformed = execution;
                    malformed.operations[0].results[0].completion = ResultCompletion::Direct;
                    assert!(validate_disposition(
                        std::slice::from_ref(&spec),
                        DTypeDisposition::Legalize(malformed)
                    )
                    .unwrap_err()
                    .contains("restore"));
                }
            }
        }
    }
}

#[test]
fn random_f64_arithmetic_cannot_be_narrowed_by_half_legalization() {
    for kind in [RandomSourceKind::Randn, RandomSourceKind::Uniform] {
        let operation = match kind {
            RandomSourceKind::Randn => NodeKind::Randn {
                shape: vec![1],
                dtype: DType::F64,
                device: Device::Cpu(0),
            },
            RandomSourceKind::Uniform => NodeKind::Uniform {
                shape: vec![1],
                lo: 0.0,
                hi: 1.0,
                dtype: DType::F64,
                device: Device::Cpu(0),
            },
        };
        let prepared = prepared(Node::new(operation).unwrap(), false);
        let spec = OperationDTypeSpec::new(&prepared.index, prepared.index.roots[0]).unwrap();
        assert!(!spec.required_numerics.permits_f32_compute);
        let mut malformed = spec.native_execution();
        malformed.operations[0].compute_dtype = Some(DType::F32);
        let error = validate_disposition(
            std::slice::from_ref(&spec),
            DTypeDisposition::Native(malformed),
        )
        .unwrap_err();
        assert!(error.contains("expected f64; actual f32"), "{error}");
    }
}

#[test]
fn f64_linalg_recipes_declare_local_conversions_and_accumulation() {
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let matrix = input(0, dtype, &[2, 3, 3]);
        for operation in [
            NodeKind::Inverse { a: matrix.clone() },
            NodeKind::Det { a: matrix.clone() },
            NodeKind::Solve {
                a: matrix.clone(),
                b: input(1, dtype, &[2, 3, 2]),
            },
        ] {
            let root = Node::new(operation).unwrap();
            for optimize in [false, true] {
                let prepared = prepared(root.clone(), optimize);
                let signature = prepared.signature.clone();
                let spec =
                    OperationDTypeSpec::new(&prepared.index, prepared.index.roots[0]).unwrap();
                assert_eq!(
                    spec.required_numerics.f64_compute,
                    Some(F64ComputeClass::LinearAlgebra)
                );
                let mut target = TestTarget::for_index(&prepared.index);
                target.f64_algorithms = true;
                let driver = CompilerDriver::new(&prepared, &target).unwrap();
                let plan = driver
                    .legalization()
                    .units()
                    .iter()
                    .find(|unit| unit.unit() == LoweringUnit::Node(spec.node))
                    .unwrap();
                let execution = plan.execution();
                let recipe = &execution.operations[0];
                assert_eq!(recipe.compute_dtype, Some(DType::F64));
                assert_eq!(
                    recipe.accumulation,
                    Some(AccumulationExecution {
                        dtype: DType::F64,
                        order: ReductionOrder::BackendDefined
                    })
                );
                assert_eq!(
                    recipe.rounding_boundaries,
                    spec.required_numerics.rounding_boundaries
                );
                assert_eq!(
                    execution.realization,
                    if dtype == DType::F64 {
                        ExecutionRealization::DirectKernel
                    } else {
                        ExecutionRealization::KernelLocal
                    }
                );
                assert_eq!(plan.disposition().is_legalized(), dtype != DType::F64);
                for operand in &recipe.operands {
                    assert_eq!(operand.execution_dtype, DType::F64);
                    assert_eq!(operand.layout, ExecutionLayout::PreserveBoundary);
                    assert_eq!(
                        operand.preparation,
                        if dtype == DType::F64 {
                            OperandPreparation::Direct
                        } else {
                            OperandPreparation::Convert(DenseConversionContract::new(
                                dtype,
                                DType::F64,
                            ))
                        }
                    );
                }
                assert_eq!(recipe.results[0].execution_dtype, DType::F64);
                assert_eq!(recipe.results[0].layout, ExecutionLayout::PreserveBoundary);
                assert_eq!(
                    recipe.results[0].completion,
                    if dtype == DType::F64 {
                        ResultCompletion::Direct
                    } else {
                        ResultCompletion::ConvertToBoundary(DenseConversionContract::new(
                            DType::F64,
                            dtype,
                        ))
                    }
                );
                assert_eq!(prepared.signature, signature);
                assert!(Arc::ptr_eq(&prepared.roots[0], &root));
                assert_eq!(driver.legalization().work().materialized_conversions, 0);

                for accumulation in [
                    None,
                    Some(AccumulationExecution {
                        dtype: DType::F32,
                        order: ReductionOrder::BackendDefined,
                    }),
                ] {
                    let mut malformed = execution.clone();
                    malformed.operations[0].accumulation = accumulation;
                    let disposition = if dtype == DType::F64 {
                        DTypeDisposition::Native(malformed)
                    } else {
                        DTypeDisposition::Legalize(malformed)
                    };
                    assert!(
                        validate_disposition(std::slice::from_ref(&spec), disposition)
                            .unwrap_err()
                            .contains("accumulation")
                    );
                }
                if dtype == DType::F64 {
                    let mut malformed = execution.clone();
                    malformed.realization = ExecutionRealization::KernelLocal;
                    malformed.operations[0].results[0].completion =
                        ResultCompletion::ConvertToBoundary(DenseConversionContract::new(
                            DType::F64,
                            DType::F64,
                        ));
                    assert!(validate_disposition(
                        std::slice::from_ref(&spec),
                        DTypeDisposition::Legalize(malformed),
                    )
                    .is_err());
                    continue;
                }
                for operand in 0..recipe.operands.len() {
                    let mut malformed = execution.clone();
                    malformed.operations[0].operands[operand].preparation =
                        OperandPreparation::Direct;
                    assert!(validate_disposition(
                        std::slice::from_ref(&spec),
                        DTypeDisposition::Legalize(malformed.clone())
                    )
                    .unwrap_err()
                    .contains("conversion contract"));
                    malformed.operations[0].operands[operand].execution_dtype = dtype;
                    assert!(validate_disposition(
                        std::slice::from_ref(&spec),
                        DTypeDisposition::Legalize(malformed)
                    )
                    .unwrap_err()
                    .contains("expected f64"));
                }
                let mut malformed = execution.clone();
                malformed.operations[0].results[0].completion = ResultCompletion::Direct;
                assert!(validate_disposition(
                    std::slice::from_ref(&spec),
                    DTypeDisposition::Legalize(malformed.clone())
                )
                .unwrap_err()
                .contains("restore"));
                malformed.operations[0].results[0].execution_dtype = dtype;
                assert!(validate_disposition(
                    std::slice::from_ref(&spec),
                    DTypeDisposition::Legalize(malformed)
                )
                .unwrap_err()
                .contains("expected f64"));
                for realization in [
                    ExecutionRealization::DirectKernel,
                    ExecutionRealization::MaterializedTransforms,
                ] {
                    let mut malformed = execution.clone();
                    malformed.realization = realization;
                    let disposition = if realization == ExecutionRealization::DirectKernel {
                        DTypeDisposition::Native(malformed)
                    } else {
                        DTypeDisposition::Legalize(malformed)
                    };
                    assert!(
                        validate_disposition(std::slice::from_ref(&spec), disposition)
                            .unwrap_err()
                            .contains("kernel-local")
                    );
                }
            }
        }
    }
}

#[test]
fn f64_factories_preserve_random_identity_and_canonical_result_conversion() {
    for dtype in [
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
        DType::I64,
        DType::U32,
        DType::U8,
    ] {
        let normal = || {
            Node::new(NodeKind::Randn {
                shape: vec![3],
                dtype,
                device: Device::Cpu(0),
            })
            .unwrap()
        };
        let shared = normal();
        let mut roots = vec![shared.clone(), normal(), shared];
        if dtype.is_float() {
            roots.push(
                Node::new(NodeKind::Uniform {
                    lo: -0.5,
                    hi: 1.5,
                    shape: vec![3],
                    dtype,
                    device: Device::Cpu(0),
                })
                .unwrap(),
            );
        }
        roots.push(
            Node::new(NodeKind::Arange {
                start: -1.0,
                end: 2.0,
                step: 0.5,
                dtype,
                device: Device::Cpu(0),
            })
            .unwrap(),
        );
        let mut sources = None;
        for optimize in [false, true] {
            let prepared = ProgramRequest::from_roots(
                roots.clone(),
                CompileOptions {
                    optimize,
                    ..CompileOptions::default()
                },
            )
            .prepare()
            .unwrap();
            let expected_sources = prepared.index.random_source_order.to_vec();
            assert_eq!(expected_sources.len(), if dtype.is_float() { 3 } else { 2 });
            if let Some(previous) = &sources {
                assert_eq!(previous, &expected_sources);
            }
            sources = Some(expected_sources);
            let mut target = TestTarget::for_index(&prepared.index);
            target.f64_algorithms = true;
            let driver = CompilerDriver::new(&prepared, &target).unwrap();
            assert_eq!(driver.legalization().units().len(), roots.len() - 1);
            assert_eq!(driver.legalization().work().materialized_conversions, 0);
            for unit in driver.legalization().units() {
                let recipe = &unit.execution().operations[0];
                let spec = OperationDTypeSpec::new(&prepared.index, recipe.node).unwrap();
                assert_eq!(recipe.compute_dtype, Some(DType::F64));
                assert!(recipe.operands.is_empty());
                assert_eq!(recipe.accumulation, None);
                assert_eq!(
                    recipe.rounding_boundaries,
                    spec.required_numerics.rounding_boundaries
                );
                assert_eq!(recipe.random, spec.required_numerics.random);
                assert_eq!(
                    spec.required_numerics.f64_compute,
                    Some(if recipe.random.is_some() {
                        F64ComputeClass::RandomSampling
                    } else {
                        F64ComputeClass::Arange
                    })
                );
                assert_eq!(recipe.results[0].execution_dtype, DType::F64);
                assert_eq!(
                    recipe.results[0].completion,
                    if dtype == DType::F64 {
                        ResultCompletion::Direct
                    } else {
                        ResultCompletion::ConvertToBoundary(DenseConversionContract::new(
                            DType::F64,
                            dtype,
                        ))
                    }
                );
                assert_eq!(unit.disposition().is_legalized(), dtype != DType::F64);
                if let Some(random) = recipe.random {
                    assert_eq!(random.samples, 3);
                    let mut wrong_count = random;
                    wrong_count.samples += 1;
                    let mut wrong_source = random;
                    wrong_source.source.provenance += 1;
                    for replacement in [None, Some(wrong_count), Some(wrong_source)] {
                        let mut malformed = unit.execution().clone();
                        malformed.operations[0].random = replacement;
                        let disposition = if dtype == DType::F64 {
                            DTypeDisposition::Native(malformed)
                        } else {
                            DTypeDisposition::Legalize(malformed)
                        };
                        assert!(
                            validate_disposition(std::slice::from_ref(&spec), disposition)
                                .unwrap_err()
                                .contains("random stream/counter")
                        );
                    }
                }
                if dtype != DType::F64 {
                    let mut malformed = unit.execution().clone();
                    malformed.operations[0].results[0].completion = ResultCompletion::Direct;
                    assert!(validate_disposition(
                        std::slice::from_ref(&spec),
                        DTypeDisposition::Legalize(malformed)
                    )
                    .unwrap_err()
                    .contains("restore"));
                }
            }
        }
    }
}

#[test]
fn f64_algorithm_permission_does_not_extend_to_other_composites_or_arithmetic() {
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut operations = model_composites(dtype);
        operations.extend(audited_half_composites(dtype));
        operations.push(NodeKind::Add {
            a: input(0, dtype, &[1]),
            b: input(1, dtype, &[1]),
        });
        operations.push(NodeKind::Matmul {
            a: input(0, dtype, &[2, 2]),
            b: input(1, dtype, &[2, 2]),
        });
        for operation in operations {
            let prepared = prepared(Node::new(operation).unwrap(), false);
            let spec = OperationDTypeSpec::new(&prepared.index, prepared.index.roots[0]).unwrap();
            assert_eq!(spec.required_numerics.f64_compute, None);
            let error = spec.f64_execution().unwrap_err();
            assert!(error.contains(operation_name(spec.operation)), "{error}");
            assert!(error.contains("cpu:0"), "{error}");
        }
    }
}

#[test]
fn half_optimizer_and_kda_recipes_preserve_every_result_role_and_picker() {
    for dtype in [DType::F16, DType::BF16] {
        for operation in audited_half_composites(dtype) {
            let results = match &operation {
                NodeKind::AdamWStep { param, m, v, .. } => {
                    vec![param.clone(), m.clone(), v.clone()]
                }
                NodeKind::SgdStep {
                    param, velocity, ..
                } => vec![param.clone(), velocity.clone()],
                NodeKind::KdaBackward {
                    q,
                    k,
                    v,
                    log_decay,
                    beta,
                    ..
                } => vec![
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    log_decay.clone(),
                    beta.clone(),
                ],
                _ => continue,
            };
            let step = Node::new(operation).unwrap();
            let roots = (0..results.len())
                .map(|index| {
                    Node::new(match &step.kind {
                        NodeKind::AdamWStep { .. } => NodeKind::AdamWOut {
                            step: step.clone(),
                            index: index as u8,
                        },
                        NodeKind::SgdStep { .. } => NodeKind::SgdOut {
                            step: step.clone(),
                            index: index as u8,
                        },
                        _ => NodeKind::KdaBackwardOut {
                            of: step.clone(),
                            index: index as u8,
                        },
                    })
                    .unwrap()
                })
                .collect::<Vec<_>>();
            for optimize in [false, true] {
                let prepared = ProgramRequest::from_roots(
                    roots.clone(),
                    CompileOptions {
                        optimize,
                        ..CompileOptions::default()
                    },
                )
                .prepare()
                .unwrap();
                let node = prepared.index.dense_id(step.id).unwrap();
                let spec = OperationDTypeSpec::new(&prepared.index, node).unwrap();
                let execution = spec
                    .native_execution()
                    .promote_half(ExecutionRealization::KernelLocal);
                validate_disposition(
                    std::slice::from_ref(&spec),
                    DTypeDisposition::Legalize(execution.clone()),
                )
                .unwrap();
                for (port, source) in results.iter().enumerate() {
                    assert_eq!(spec.results[port].value.logical_shape, source.shape);
                    assert_eq!(spec.results[port].role, ValueRole::Result(port as u32));
                    assert_eq!(
                        execution.operations[0].rounding_boundaries[port],
                        RoundingBoundary {
                            node,
                            result: port as u32,
                            dtype
                        }
                    );
                    let picker =
                        OperationDTypeSpec::new(&prepared.index, prepared.index.roots[port])
                            .unwrap();
                    assert_eq!(picker.operands[0].source, node);
                    assert_eq!(picker.operands[0].source_result, port as u32);
                    assert_eq!(picker.operands[0].value.logical_shape, source.shape);
                    assert_eq!(picker.required_numerics.compute_dtype, None);
                    let mut malformed = execution.clone();
                    malformed.operations[0].results[port].role =
                        ValueRole::Result(results.len() as u32);
                    assert!(validate_disposition(
                        std::slice::from_ref(&spec),
                        DTypeDisposition::Legalize(malformed)
                    )
                    .unwrap_err()
                    .contains("result role"));
                }
            }
        }
    }
}

#[test]
fn dense_cast_queries_derive_canonical_modes_for_all_dtype_pairs() {
    use DenseConversionMode::*;
    let dtypes = [
        DType::F64,
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::I64,
        DType::U32,
        DType::U8,
    ];
    let expected = [
        [
            Identity,
            FloatToFloatNearestEven,
            FloatToFloatNearestEven,
            FloatToFloatNearestEven,
            FloatToIntegerTruncateSaturate,
            FloatToIntegerTruncateSaturate,
            FloatToIntegerTruncateSaturate,
        ],
        [
            FloatToFloatNearestEven,
            Identity,
            FloatToFloatNearestEven,
            FloatToFloatNearestEven,
            FloatToIntegerTruncateSaturate,
            FloatToIntegerTruncateSaturate,
            FloatToIntegerTruncateSaturate,
        ],
        [
            FloatToFloatNearestEven,
            FloatToFloatNearestEven,
            Identity,
            FloatToFloatNearestEven,
            FloatToIntegerTruncateSaturate,
            FloatToIntegerTruncateSaturate,
            FloatToIntegerTruncateSaturate,
        ],
        [
            FloatToFloatNearestEven,
            FloatToFloatNearestEven,
            FloatToFloatNearestEven,
            Identity,
            FloatToIntegerTruncateSaturate,
            FloatToIntegerTruncateSaturate,
            FloatToIntegerTruncateSaturate,
        ],
        [
            IntegerToFloatNearestEven,
            IntegerToFloatNearestEven,
            IntegerToFloatNearestEven,
            IntegerToFloatNearestEven,
            Identity,
            IntegerToIntegerWrapping,
            IntegerToIntegerWrapping,
        ],
        [
            IntegerToFloatNearestEven,
            IntegerToFloatNearestEven,
            IntegerToFloatNearestEven,
            IntegerToFloatNearestEven,
            IntegerToIntegerWrapping,
            Identity,
            IntegerToIntegerWrapping,
        ],
        [
            IntegerToFloatNearestEven,
            IntegerToFloatNearestEven,
            IntegerToFloatNearestEven,
            IntegerToFloatNearestEven,
            IntegerToIntegerWrapping,
            IntegerToIntegerWrapping,
            Identity,
        ],
    ];
    for (row, source) in dtypes.into_iter().enumerate() {
        for (column, destination) in dtypes.into_iter().enumerate() {
            let root = Node::new(NodeKind::Cast {
                a: input(0, source, &[1]),
                dtype: destination,
            })
            .unwrap();
            let prepared = prepared(root, false);
            let spec = OperationDTypeSpec::new(&prepared.index, prepared.index.roots[0]).unwrap();
            let contract = DenseConversionContract::new(source, destination);
            assert_eq!(
                contract.mode, expected[row][column],
                "{source} -> {destination}"
            );
            contract.validate().unwrap();
            assert_eq!(
                spec.required_numerics.result_formations[0],
                ResultFormation::Cast(contract)
            );
            for mode in [
                Identity,
                FloatToFloatNearestEven,
                IntegerToFloatNearestEven,
                FloatToIntegerTruncateSaturate,
                IntegerToIntegerWrapping,
            ] {
                if mode == contract.mode {
                    continue;
                }
                let mut invalid = contract;
                invalid.mode = mode;
                assert!(invalid
                    .validate()
                    .unwrap_err()
                    .contains("noncanonical dense conversion"));
            }
        }
    }
}

#[test]
fn legalization_rejects_noncanonical_operand_result_and_scalar_conversion_modes() {
    let root = Node::new(NodeKind::Mul {
        a: input(0, DType::BF16, &[1]),
        b: input(1, DType::F32, &[]),
    })
    .unwrap();
    let prepared = prepared(root, false);
    let spec = OperationDTypeSpec::new(&prepared.index, prepared.index.roots[0]).unwrap();
    let valid = spec
        .native_execution()
        .promote_half(ExecutionRealization::KernelLocal);
    for position in 0..3 {
        let mut invalid = valid.clone();
        let operation = &mut invalid.operations[0];
        let conversion = match position {
            0 => match &mut operation.operands[0].preparation {
                OperandPreparation::Convert(conversion) => conversion,
                _ => unreachable!(),
            },
            1 => match &mut operation.operands[1].interpretation {
                OperandInterpretation::ScalarCoercion(conversion) => conversion,
                _ => unreachable!(),
            },
            _ => match &mut operation.results[0].completion {
                ResultCompletion::ConvertToBoundary(conversion) => conversion,
                _ => unreachable!(),
            },
        };
        conversion.mode = DenseConversionMode::FloatToIntegerTruncateSaturate;
        let error = validate_disposition(
            std::slice::from_ref(&spec),
            DTypeDisposition::Legalize(invalid),
        )
        .unwrap_err();
        assert!(error.contains("noncanonical dense conversion"), "{error}");
    }
}

#[test]
fn compiler_half_casts_round_f64_midpoint_neighbors_directly() {
    for (dtype, largest_finite) in [(DType::F16, 0x7bffu16), (DType::BF16, 0x7f7fu16)] {
        let decode = |bits| match dtype {
            DType::F16 => half::f16::from_bits(bits).to_f64(),
            _ => half::bf16::from_bits(bits).to_f64(),
        };
        for lower in 0..largest_finite {
            let midpoint = (decode(lower) + decode(lower + 1)) * 0.5;
            let cases = [
                (f64::from_bits(midpoint.to_bits() - 1), lower),
                (midpoint, lower + (lower & 1)),
                (f64::from_bits(midpoint.to_bits() + 1), lower + 1),
            ];
            for (value, expected) in cases {
                for sign in [0, 0x8000] {
                    let value = if sign == 0 { value } else { -value };
                    let actual = Scalar::cast(value, dtype);
                    assert_eq!(
                        actual.to_bits(),
                        decode(expected | sign).to_bits(),
                        "{dtype}, lower={lower:04x}, value={value:e}"
                    );
                }
            }
        }
    }
}

#[test]
fn compiler_half_casts_preserve_subnormals_signed_zero_and_nonfinite_classes() {
    for (dtype, minimum, overflow) in [
        (DType::F16, 2f64.powi(-24), 65520.0),
        (
            DType::BF16,
            2f64.powi(-133),
            2f64.powi(128) - 2f64.powi(119),
        ),
    ] {
        for sign in [1.0, -1.0] {
            assert_eq!(
                Scalar::cast(sign * minimum, dtype).to_bits(),
                (sign * minimum).to_bits()
            );
            assert_eq!(
                Scalar::cast(sign * minimum * 0.5, dtype).to_bits(),
                (sign * 0.0).to_bits()
            );
            assert_eq!(
                Scalar::cast(sign * f64::from_bits(1), dtype).to_bits(),
                (sign * 0.0).to_bits()
            );
            assert_eq!(Scalar::cast(sign * overflow, dtype), sign * f64::INFINITY);
            assert!(Scalar::cast(sign * f64::from_bits(overflow.to_bits() - 1), dtype).is_finite());
            assert_eq!(
                Scalar::cast(sign * f64::INFINITY, dtype),
                sign * f64::INFINITY
            );
        }
        assert!(Scalar::cast(f64::NAN, dtype).is_nan());
    }
    for (dtype, value, expected) in [
        (
            DType::F16,
            1.0 + 2f64.powi(-11) + 2f64.powi(-40),
            1.0009765625,
        ),
        (DType::BF16, 1.0 + 2f64.powi(-8) + 2f64.powi(-40), 1.0078125),
    ] {
        assert_eq!(
            evaluate(&KernelExpr::typed_constant(value, dtype), &[]),
            expected
        );
    }
}

#[test]
fn prepared_binding_queries_use_layout_policy_without_mutating_semantic_nodes() {
    use effect_torch_runtime::{
        BindingAliasing, BindingDecl, BindingLayoutPolicy, DeviceId, InvocationSignature, Layout,
        LayoutConstraint, Placement,
    };
    let source = input(0, DType::F32, &[2]);
    let root = Node::new(NodeKind::Neg { a: source.clone() }).unwrap();
    let layout = Layout::new(vec![2], vec![3], 1);
    let binding = BindingDecl {
        shape: vec![2],
        dtype: DType::F32,
        storage: StorageMetadata::dense(),
        placement: Placement::new(DeviceId::new("cpu:0")),
        layout: BindingLayoutPolicy::Require(LayoutConstraint::Exact(layout.clone())),
        aliasing: BindingAliasing::MayAlias,
    };
    let prepared = ProgramRequest::new(
        vec![root],
        vec![binding],
        InvocationSignature::default(),
        CompileOptions::default(),
    )
    .prepare()
    .unwrap();
    let spec = OperationDTypeSpec::new(&prepared.index, prepared.index.roots[0]).unwrap();
    assert_eq!(
        spec.operands[0].value.storage.layout_constraint,
        LayoutConstraintSpec::DenseStrided(&layout)
    );
    assert_eq!(source.storage, StorageMetadata::dense());
    let target = TestTarget::for_index(&prepared.index);
    CompilerDriver::new(&prepared, &target).unwrap();
}
