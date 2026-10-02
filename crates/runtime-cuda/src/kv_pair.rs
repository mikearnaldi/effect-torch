//! Exact opt-in launch pairing of two independent, already-legalized BF16 GEMMs.
use crate::cublas::{ordinary_k16, Bf16GemmPlan};
use crate::lowering::{Command, CommandKind, CommandOverlap, CudaLoweredProgram};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use effect_torch_compiler::{
    normalize_aliases, InstructionEffects, OutputDecl, ValueStorage, ValueUse,
};
use effect_torch_runtime::{DType, StorageRepresentation, ValueId};
use std::sync::Arc;

pub(crate) const ENV: &str = "EFFECT_TORCH_CUDA_KV_PAIR";
pub(crate) const PATH_ENV: &str = "EFFECT_TORCH_CUDA_KV_PAIR_PTX";
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = TEST_POLICY.with(|p| p.get()) {
        return value;
    }
    std::env::var(ENV).as_deref() == Ok("1") && std::env::var(PATH_ENV).is_ok_and(|p| !p.is_empty())
}
#[cfg(test)]
thread_local! { static TEST_POLICY: std::cell::Cell<Option<bool>> = const {std::cell::Cell::new(None)}; }
#[cfg(test)]
pub(crate) fn with_test_policy<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_POLICY.with(|p| p.set(self.0));
        }
    }
    let _restore = Restore(TEST_POLICY.with(|p| p.replace(Some(enabled))));
    run()
}
#[repr(C)]
struct Arguments {
    x: u64,
    key_weight: u64,
    value_weight: u64,
    key_out: u64,
    value_out: u64,
}
// SAFETY: five initialized scalar addresses match the proved standalone ABI.
unsafe impl DeviceRepr for Arguments {}
pub(crate) fn geometry(plan: Bf16GemmPlan, transposed: bool, out_f32: bool) -> bool {
    plan.n == 2048 && ordinary_k16::supports(plan, transposed, out_f32, [16, 32, 48])
}
pub(crate) fn pointers_valid(plan: Bf16GemmPlan, addresses: [u64; 5]) -> bool {
    if !geometry(plan, true, false) {
        return false;
    }
    let sizes = [
        256 * 2816 * 2,
        2048 * 2816 * 2,
        2048 * 2816 * 2,
        256 * 2048 * 2,
        256 * 2048 * 2,
    ];
    let mut ends = [0; 5];
    for i in 0..5 {
        if addresses[i] == 0 || addresses[i] % 16 != 0 {
            return false;
        }
        let Some(end) = addresses[i].checked_add(sizes[i]) else {
            return false;
        };
        ends[i] = end;
    }
    // Read-only X may overlap weights; weights are conservatively distinct.
    for (a, b) in [
        (1, 2),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 4),
        (4, 0),
        (4, 1),
        (4, 2),
    ] {
        if addresses[a] < ends[b] && addresses[b] < ends[a] {
            return false;
        }
    }
    true
}
/// Selection happens before any CUDA submission. The test policy can model a
/// missing optional artifact without changing the shared device or environment.
pub(crate) fn select_kernel(
    kernel: Option<&Kernel>,
    plan: Bf16GemmPlan,
    addresses: [u64; 5],
) -> Option<&Kernel> {
    #[cfg(test)]
    if TEST_POLICY.with(|p| p.get()) == Some(false) {
        return None;
    }
    kernel.filter(|_| pointers_valid(plan, addresses))
}
pub(crate) struct Kernel {
    function: CudaFunction,
}
impl Kernel {
    pub(crate) fn load(context: &Arc<CudaContext>, path: &str) -> Result<Self, String> {
        let source =
            std::fs::read_to_string(path).map_err(|e| format!("CUDA KV pair PTX {path}: {e}"))?;
        let module = context
            .load_module(Ptx::from_src(source))
            .map_err(|e| e.to_string())?;
        let function = module
            .load_function("et_kv_pair57")
            .map_err(|e| e.to_string())?;
        Ok(Self { function })
    }
    /// All five allocations belong to/are retained by the invocation fence.
    pub(crate) unsafe fn launch(
        &self,
        stream: &Arc<CudaStream>,
        p: [u64; 5],
    ) -> Result<(), String> {
        let args = Arguments {
            x: p[0],
            key_weight: p[1],
            value_weight: p[2],
            key_out: p[3],
            value_out: p[4],
        };
        let mut launch = stream.launch_builder(&self.function);
        launch.arg(&args);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (32, 4, 2),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 49152,
            })
        }
        .map_err(|e| e.to_string())?;
        Ok(())
    }
}
#[derive(Clone, Copy)]
struct Gemm {
    x: ValueId,
    weight: ValueId,
    output: ValueId,
    workspace: ValueId,
    plan: Bf16GemmPlan,
}
fn candidate(command: &Command) -> Option<Gemm> {
    let CommandKind::Gemm {
        x,
        weight,
        weight_transposed,
        plan,
        out_f32,
        workspace,
    } = &command.kind
    else {
        return None;
    };
    if !matches!(command.overlap, CommandOverlap::Primary)
        || !geometry(*plan, *weight_transposed, *out_f32)
    {
        return None;
    }
    Some(Gemm {
        x: *x,
        weight: *weight,
        output: command.output?,
        workspace: *workspace,
        plan: *plan,
    })
}
fn host_only(command: &Command) -> bool {
    matches!(
        command.kind,
        CommandKind::Prepare
            | CommandKind::Value(_)
            | CommandKind::Input { .. }
            | CommandKind::Alias { .. }
            | CommandKind::PlannedAlias
    )
}
fn metadata_valid(program: &CudaLoweredProgram, a: Gemm, b: Gemm) -> bool {
    a.workspace != b.workspace
        && [
            (a.x, 256 * 2816 * 2),
            (a.weight, 2048 * 2816 * 2),
            (b.weight, 2048 * 2816 * 2),
            (a.output, 256 * 2048 * 2),
            (b.output, 256 * 2048 * 2),
        ]
        .into_iter()
        .all(|(v, bytes)| {
            let m = &program.values[v.index()];
            m.dtype == DType::BF16
                && m.storage.representation == StorageRepresentation::Dense
                && m.decl.bytes == bytes
        })
        && [a.workspace, b.workspace].iter().all(|v| {
            let m = &program.values[v.index()];
            m.dtype == DType::U8
                && m.decl.bytes == crate::cublas::CUBLAS_WORKSPACE_BYTES
                && matches!(m.decl.storage, ValueStorage::Planned { .. })
        })
}
fn constant_storage(program: &CudaLoweredProgram, value: ValueId) -> bool {
    matches!(
        program.values[value.index()].decl.storage,
        ValueStorage::Fixed {
            class: effect_torch_runtime::StorageClass::PersistentConstant,
            location: effect_torch_runtime::Location::Persistent { .. }
        }
    )
}
fn safe_gap(command: &Command) -> bool {
    if !matches!(command.overlap, CommandOverlap::Primary) {
        return false;
    }
    match &command.kind {
        CommandKind::Input { .. }
        | CommandKind::Value(_)
        | CommandKind::Alias { .. }
        | CommandKind::Prepare
        | CommandKind::PlannedAlias => true,
        CommandKind::Kernel {
            name,
            checked,
            status,
            state,
            kv_matmul,
            ..
        } => {
            !checked
                && status.is_none()
                && matches!(state, crate::executable::StateAccess::None)
                && kv_matmul.is_none()
                && !name.starts_with("et_random_")
        }
        CommandKind::FusedElementwise { .. } => true,
        _ => false,
    }
}
// Advance only independent V work, leaving all Q/K consumers in their original
// order. The only moved producers are a direct Input and workspace Prepare.
fn early_pairs(
    program: &mut CudaLoweredProgram,
    commands: &mut [Command],
) -> Result<usize, String> {
    let mut count = 0;
    let mut first = 0;
    while first < commands.len() {
        let Some(a) = candidate(&commands[first]) else {
            first += 1;
            continue;
        };
        let Some(second) = (first + 1..commands.len()).find(|&p| !safe_gap(&commands[p])) else {
            first += 1;
            continue;
        };
        let Some(b) = candidate(&commands[second]) else {
            first += 1;
            continue;
        };
        if commands[first + 1..second].iter().any(|c| {
            matches!(c.kind, CommandKind::Value(_))
                && !c.output.is_some_and(|v| constant_storage(program, v))
        }) {
            first += 1;
            continue;
        }
        // Leave the no-GPU-gap case to the simpler delayed path.
        if !commands[first + 1..second].iter().any(|c| !host_only(c)) {
            first += 1;
            continue;
        }
        if a.x != b.x || a.plan != b.plan || !metadata_valid(program, a, b) {
            first += 1;
            continue;
        }
        let aliases = normalize_aliases(&program.values).map_err(|e| e.to_string())?;
        let root = |v: ValueId| aliases[v.index()].root;
        if root(a.weight) == root(b.weight)
            || root(a.output) == root(b.output)
            || [a.output, b.output].iter().any(|o| {
                [a.x, a.weight, b.weight]
                    .iter()
                    .any(|i| root(*o) == root(*i))
            })
        {
            first += 1;
            continue;
        }
        if [a.x, a.weight, b.weight, a.output, b.output]
            .iter()
            .any(|v| {
                program.values[v.index()].dtype != DType::BF16
                    || program.values[v.index()].storage.representation
                        != StorageRepresentation::Dense
            })
            || [a.output, b.output].iter().any(|v| {
                !matches!(
                    program.values[v.index()].decl.storage,
                    ValueStorage::Planned { .. }
                )
            })
        {
            first += 1;
            continue;
        }
        let positions = program
            .instructions
            .iter()
            .enumerate()
            .filter_map(|(p, i)| {
                (!matches!(
                    i.kind,
                    "state_prepare" | "state_commit" | "state_discard" | "status_check"
                ))
                .then_some(p)
            })
            .collect::<Vec<_>>();
        if positions.len() != commands.len() {
            return Err("compile: KV pair mapping mismatch".into());
        }
        let mut moves = Vec::new();
        let mut legal = true;
        for (value, is_weight) in [(b.weight, true), (b.workspace, false)] {
            let Some(command) = commands.iter().position(|c| c.output == Some(value)) else {
                legal = false;
                break;
            };
            if command < first {
                continue;
            }
            let instruction = &program.instructions[positions[command]];
            if command >= second
                || instruction.outputs.len() != usize::from(!is_weight)
                || !instruction.inputs.is_empty()
                || !instruction.scratch.is_empty()
                || !instruction.status.is_empty()
                || !instruction.state.is_empty()
                || !instruction.staging.is_empty()
                || instruction.effects.has_side_effects
                || !(if is_weight {
                    matches!(commands[command].kind, CommandKind::Input { .. })
                        || (matches!(commands[command].kind, CommandKind::Value(_))
                            && constant_storage(program, value))
                } else {
                    matches!(commands[command].kind, CommandKind::Prepare)
                })
            {
                legal = false;
                break;
            }
            moves.push(command);
        }
        if !legal {
            first += 1;
            continue;
        }
        // Preserve the relative order of independent binding validation errors:
        // move all direct input bindings in this gap in their original order.
        for command in first + 1..second {
            if matches!(commands[command].kind, CommandKind::Input { .. }) {
                let i = &program.instructions[positions[command]];
                if !i.inputs.is_empty()
                    || !i.outputs.is_empty()
                    || !i.scratch.is_empty()
                    || !i.staging.is_empty()
                    || !i.status.is_empty()
                    || !i.state.is_empty()
                    || i.effects.has_side_effects
                {
                    legal = false;
                    break;
                }
                moves.push(command);
            }
        }
        if !legal {
            first += 1;
            continue;
        }
        moves.sort_unstable();
        moves.dedup();
        for position in positions[first] + 1..positions[second] {
            let i = &program.instructions[position];
            if matches!(
                i.kind,
                "state_prepare" | "state_commit" | "state_discard" | "status_check"
            ) || i.effects.has_side_effects
                || !i.status.is_empty()
                || !i.state.is_empty()
            {
                legal = false;
                break;
            }
            // The selected V binding is a definition we deliberately hoist. All
            // other writes to shared operands, or early uses of V, forbid fusion.
            if moves.iter().any(|p| positions[*p] == position) {
                continue;
            }
            if i.resource_uses().any(|u| {
                root(u.value) == root(b.output)
                    || (u.access.writes()
                        && !matches!(
                            program.values[u.value.index()].decl.storage,
                            ValueStorage::Alias { .. }
                        )
                        && [a.x, a.weight, b.weight]
                            .iter()
                            .any(|v| root(*v) == root(u.value)))
            }) {
                legal = false;
                break;
            }
        }
        if !legal {
            first += 1;
            continue;
        }
        moves.sort_unstable();
        // Moving each declaration just before K shifts K one slot right. The
        // later V slot remains fixed because rotations stay inside the gap.
        for from in moves {
            let start = positions[first];
            let end = positions[from];
            commands[first..=from].rotate_right(1);
            program.instructions[start..=end].rotate_right(1);
            first += 1;
        }
        for (i, instruction) in program.instructions.iter_mut().enumerate() {
            instruction.id = effect_torch_runtime::InstructionId::from_index(i)
                .ok_or("compile: KV pair instruction overflow")?;
        }
        let i = &mut program.instructions[positions[first]];
        i.kind = "kv_pair_bf16_gemm";
        i.inputs = vec![
            ValueUse::read(a.x),
            ValueUse::read(a.weight),
            ValueUse::read(b.weight),
        ]
        .into_boxed_slice();
        i.outputs = vec![OutputDecl::new(a.output), OutputDecl::new(b.output)].into_boxed_slice();
        i.scratch = vec![
            ValueUse::read_write(a.workspace),
            ValueUse::read_write(b.workspace),
        ]
        .into_boxed_slice();
        commands[first].kind = CommandKind::GemmPair {
            x: a.x,
            weights: [a.weight, b.weight],
            second_output: b.output,
            plan: a.plan,
            workspaces: [a.workspace, b.workspace],
        };
        let i = &mut program.instructions[positions[second]];
        i.kind = "kv_pair_output";
        i.inputs = vec![ValueUse::read(b.output)].into_boxed_slice();
        i.outputs = Box::new([]);
        i.scratch = Box::new([]);
        i.effects = InstructionEffects::default();
        commands[second].kind = CommandKind::PlannedAlias;
        count += 1;
        first += 1;
    }
    Ok(count)
}

/// Delay a pure first GEMM through independent binding/alias/allocation bookkeeping.
/// No GPU computation, checked status, state, random, or first-result consumer is
/// crossed. Binding validation may fail before a now-unneeded pure launch; the
/// failed invocation still publishes no output and follows its existing fence.
pub(super) fn fuse(
    program: &mut CudaLoweredProgram,
    commands: &mut [Command],
) -> Result<usize, String> {
    let early_count = early_pairs(program, commands)?;
    let positions = program
        .instructions
        .iter()
        .enumerate()
        .filter_map(|(p, i)| {
            (!matches!(
                i.kind,
                "state_prepare" | "state_commit" | "state_discard" | "status_check"
            ))
            .then_some(p)
        })
        .collect::<Vec<_>>();
    if positions.len() != commands.len() {
        return Err("compile: KV pair command/instruction mapping mismatch".into());
    }
    let aliases = normalize_aliases(&program.values).map_err(|e| e.to_string())?;
    let root = |id: ValueId| aliases[id.index()].root;
    let mut count = early_count;
    for first in 0..commands.len() {
        let Some(a) = candidate(&commands[first]) else {
            continue;
        };
        let Some(second) = (first + 1..commands.len()).find(|&i| !host_only(&commands[i])) else {
            continue;
        };
        let Some(b) = candidate(&commands[second]) else {
            continue;
        };
        if a.x != b.x
            || a.plan != b.plan
            || root(a.weight) == root(b.weight)
            || !metadata_valid(program, a, b)
        {
            continue;
        }
        if [a.x, a.weight, b.weight, a.output, b.output]
            .iter()
            .any(|v| {
                let m = &program.values[v.index()];
                m.dtype != DType::BF16 || m.storage.representation != StorageRepresentation::Dense
            })
        {
            continue;
        }
        if [a.output, b.output].iter().any(|v| {
            !matches!(
                program.values[v.index()].decl.storage,
                ValueStorage::Planned { .. }
            )
        }) {
            continue;
        }
        if root(a.output) == root(b.output)
            || [a.output, b.output].iter().any(|o| {
                [a.x, a.weight, b.weight]
                    .iter()
                    .any(|i| root(*o) == root(*i))
            })
        {
            continue;
        }
        let between = &program.instructions[positions[first] + 1..positions[second]];
        if between.iter().any(|i| {
            matches!(
                i.kind,
                "state_prepare" | "state_commit" | "state_discard" | "status_check"
            ) || i.effects.has_side_effects
                || !i.status.is_empty()
                || !i.state.is_empty()
                || i.resource_uses().any(|u| root(u.value) == root(a.output))
        }) {
            continue;
        }
        if commands[first + 1..second].iter().any(|c| {
            !matches!(c.overlap, CommandOverlap::Primary)
                || !(matches!(
                    c.kind,
                    CommandKind::Input { .. }
                        | CommandKind::Alias { .. }
                        | CommandKind::Prepare
                        | CommandKind::PlannedAlias
                ) || (matches!(c.kind, CommandKind::Value(_))
                    && c.output.is_some_and(|v| constant_storage(program, v))))
        }) {
            continue;
        }
        // Both fallback workspaces have already been declared at their original
        // prepares; the merged scratch uses extend both lives through the pair.
        let i = &mut program.instructions[positions[first]];
        i.kind = "kv_pair_deferred";
        i.inputs = Box::new([]);
        i.outputs = Box::new([]);
        i.scratch = Box::new([]);
        i.staging = Box::new([]);
        i.status = Box::new([]);
        i.state = Box::new([]);
        i.effects = InstructionEffects::default();
        commands[first].kind = CommandKind::Prepare;
        commands[first].output = None;
        let i = &mut program.instructions[positions[second]];
        i.kind = "kv_pair_bf16_gemm";
        i.inputs = vec![
            ValueUse::read(a.x),
            ValueUse::read(a.weight),
            ValueUse::read(b.weight),
        ]
        .into_boxed_slice();
        i.outputs = vec![OutputDecl::new(a.output), OutputDecl::new(b.output)].into_boxed_slice();
        i.scratch = vec![
            ValueUse::read_write(a.workspace),
            ValueUse::read_write(b.workspace),
        ]
        .into_boxed_slice();
        commands[second].kind = CommandKind::GemmPair {
            x: a.x,
            weights: [a.weight, b.weight],
            second_output: b.output,
            plan: a.plan,
            workspaces: [a.workspace, b.workspace],
        };
        commands[second].output = Some(a.output);
        count += 1;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lowering::CudaValueMeta;
    use crate::workspace::CudaMemorySpace;
    use effect_torch_compiler::{
        analyze_liveness, plan_memory, LoweredInstruction, LoweredProgram, MemoryPlannerConfig,
        ValueDecl,
    };
    use effect_torch_runtime::{InstructionId, SegmentOwnership, StorageMetadata};
    fn plan() -> Bf16GemmPlan {
        Bf16GemmPlan {
            m: 256,
            n: 2048,
            k: 2816,
            batch: 1,
            stride_x: 256 * 2816,
            stride_weight: 2048 * 2816,
            stride_out: 256 * 2048,
        }
    }
    fn fixture() -> (CudaLoweredProgram, Vec<Command>) {
        let specs = [
            (DType::BF16, 256 * 2816),
            (DType::BF16, 2048 * 2816),
            (DType::BF16, 2048 * 2816),
            (DType::U8, crate::cublas::CUBLAS_WORKSPACE_BYTES),
            (DType::BF16, 256 * 2048),
            (DType::U8, crate::cublas::CUBLAS_WORKSPACE_BYTES),
            (DType::BF16, 256 * 2048),
        ];
        let values: Vec<_> = specs
            .into_iter()
            .enumerate()
            .map(|(i, (dtype, n))| CudaValueMeta {
                decl: ValueDecl::planned(
                    ValueId::new(i as u32),
                    format!("v{i}"),
                    n * dtype.size_in_bytes(),
                    256,
                    CudaMemorySpace::Device,
                    SegmentOwnership::Workspace,
                ),
                shape: vec![n],
                dtype,
                storage: StorageMetadata::dense(),
            })
            .collect();
        let id = ValueId::new;
        let instructions = vec![
            LoweredInstruction::new(
                InstructionId::new(0),
                "prepare",
                vec![],
                vec![
                    OutputDecl::new(id(0)),
                    OutputDecl::new(id(1)),
                    OutputDecl::new(id(2)),
                ],
            ),
            LoweredInstruction::new(
                InstructionId::new(1),
                "prepare",
                vec![],
                vec![OutputDecl::new(id(3))],
            ),
            LoweredInstruction::new(
                InstructionId::new(2),
                "gemm",
                vec![ValueUse::read(id(0)), ValueUse::read(id(1))],
                vec![OutputDecl::new(id(4))],
            )
            .with_resources(vec![ValueUse::read_write(id(3))], vec![], vec![], vec![]),
            LoweredInstruction::new(
                InstructionId::new(3),
                "prepare",
                vec![],
                vec![OutputDecl::new(id(5))],
            ),
            LoweredInstruction::new(
                InstructionId::new(4),
                "gemm",
                vec![ValueUse::read(id(0)), ValueUse::read(id(2))],
                vec![OutputDecl::new(id(6))],
            )
            .with_resources(vec![ValueUse::read_write(id(5))], vec![], vec![], vec![]),
        ];
        let gemm = |weight, workspace| CommandKind::Gemm {
            x: id(0),
            weight: id(weight),
            weight_transposed: true,
            plan: plan(),
            out_f32: false,
            workspace: id(workspace),
        };
        let commands: Vec<_> = [
            (None, CommandKind::Prepare),
            (Some(id(3)), CommandKind::Prepare),
            (Some(id(4)), gemm(1, 3)),
            (Some(id(5)), CommandKind::Prepare),
            (Some(id(6)), gemm(2, 5)),
        ]
        .into_iter()
        .map(|(output, kind)| Command {
            output,
            kind,
            overlap: CommandOverlap::Primary,
        })
        .collect();
        (
            LoweredProgram::new(values, instructions, vec![id(4), id(6)]),
            commands,
        )
    }
    #[test]
    fn paired_outputs_and_both_fallback_workspaces_have_distinct_live_storage() {
        let (mut p, mut c) = fixture();
        assert_eq!(fuse(&mut p, &mut c).unwrap(), 1);
        assert_eq!(p.instructions[4].outputs.len(), 2);
        assert!(matches!(c[4].kind, CommandKind::GemmPair { .. }));
        let live = analyze_liveness(&p).unwrap();
        assert_eq!(live.intervals[6].unwrap().start.index(), 4);
        assert_eq!(live.intervals[5].unwrap().start.index(), 3);
        let memory = plan_memory(
            &p,
            &MemoryPlannerConfig::uniform(CudaMemorySpace::Device, 1 << 30, 256, 256),
        )
        .unwrap();
        for a in [3, 4, 5, 6] {
            for b in [3, 4, 5, 6] {
                if a != b {
                    assert_ne!(memory.locations[a], memory.locations[b]);
                }
            }
        }
        assert_eq!(fuse(&mut p, &mut c).unwrap(), 0);
    }
    #[test]
    fn incompatible_geometry_dtypes_aliases_and_ordering_are_rejected() {
        for case in 0..12 {
            let (mut p, mut c) = fixture();
            match case {
                0 => {
                    if let CommandKind::Gemm { plan, .. } = &mut c[4].kind {
                        plan.m = 64;
                    }
                }
                1 => {
                    if let CommandKind::Gemm { plan, .. } = &mut c[4].kind {
                        plan.n = 2112;
                    }
                }
                2 => {
                    if let CommandKind::Gemm { out_f32, .. } = &mut c[4].kind {
                        *out_f32 = true;
                    }
                }
                3 => {
                    if let CommandKind::Gemm {
                        weight_transposed, ..
                    } = &mut c[4].kind
                    {
                        *weight_transposed = false;
                    }
                }
                4 => {
                    if let CommandKind::Gemm { x, .. } = &mut c[4].kind {
                        *x = ValueId::new(1);
                    }
                }
                5 => p.values[2].dtype = DType::F16,
                6 => {
                    p.values[2].decl.storage = ValueStorage::Alias {
                        source: ValueId::new(1),
                        byte_offset: 0,
                    }
                }
                7 => {
                    p.values[6].decl.storage = ValueStorage::Alias {
                        source: ValueId::new(4),
                        byte_offset: 0,
                    }
                }
                8 => {
                    p.instructions[3].inputs =
                        vec![ValueUse::read(ValueId::new(4))].into_boxed_slice();
                }
                9 => c[3].kind = CommandKind::Cursor { tensor: true },
                10 => {
                    p.instructions[3].status =
                        vec![ValueUse::read(ValueId::new(0))].into_boxed_slice()
                }
                _ => p.instructions[3].effects.has_side_effects = true,
            }
            assert_eq!(fuse(&mut p, &mut c).unwrap(), 0, "case {case}");
        }
    }
    #[test]
    fn range_guards_prevent_input_output_and_output_output_overlap() {
        let valid = [0x1000, 0x10000000, 0x20000000, 0x30000000, 0x40000000];
        assert!(pointers_valid(plan(), valid));
        for (i, v) in [
            (0, 0),
            (1, 17),
            (2, 0x10000000),
            (3, 0x1000),
            (4, 0x30000000),
            (4, u64::MAX - 15),
        ] {
            let mut p = valid;
            p[i] = v;
            assert!(!pointers_valid(plan(), p));
        }
        assert_eq!(std::mem::size_of::<Arguments>(), 40);
    }
    #[test]
    fn constants_require_immutable_persistent_storage() {
        let (mut p, _) = fixture();
        let v = ValueId::new(2);
        assert!(!constant_storage(&p, v));
        p.values[2].decl.storage = ValueStorage::Fixed {
            class: effect_torch_runtime::StorageClass::PersistentConstant,
            location: effect_torch_runtime::Location::Persistent { slot: 2 },
        };
        assert!(constant_storage(&p, v));
        p.values[2].decl.storage = ValueStorage::Fixed {
            class: effect_torch_runtime::StorageClass::ExternalInput,
            location: effect_torch_runtime::Location::External { slot: 2 },
        };
        assert!(!constant_storage(&p, v));
        p.values[2].decl.storage = ValueStorage::Alias {
            source: ValueId::new(1),
            byte_offset: 0,
        };
        assert!(!constant_storage(&p, v));
    }
}
