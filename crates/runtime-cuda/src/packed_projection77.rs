//! Opt-in immutable BF16 projection packing. Original contiguous outputs remain
//! explicit IR definitions; no model or tensor-layout contract changes.
use super::*;
use crate::buffer::CudaBuffer;
use cudarc::driver::sys;
use std::sync::{Mutex, OnceLock, Weak};

struct Cached {
    sources: Vec<Weak<CudaBuffer<u8>>>,
    addresses: Vec<u64>,
    widths: Vec<usize>,
    packed: Weak<CudaBuffer<u8>>,
}
static CACHE: OnceLock<Mutex<Vec<Cached>>> = OnceLock::new();

pub(super) fn admitted_policy(constant_weights: bool, requested: bool) -> bool {
    // A captured Value owns storage but may still be an optimizer parameter.
    // Only the compile option guarantees immutable bytes for the packed cache.
    constant_weights && requested
}

fn packed_weight(weights: &[CudaValue], widths: &[usize]) -> Result<Arc<CudaValue>, String> {
    let device = &weights[0].device;
    let _gate = device
        .graph_execution
        .lock()
        .map_err(|_| "packed77 device gate poisoned")?;
    let mut cache = CACHE
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .map_err(|_| "packed77 cache poisoned")?;
    cache.retain(|entry| entry.packed.strong_count() != 0);
    if let Some(value) = cache.iter().find_map(|entry| {
        (entry.widths == widths
            && entry.sources.len() == weights.len()
            && entry.sources.iter().zip(weights).zip(&entry.addresses).all(
                |((owner, value), address)| {
                    *address == value.storage_address()
                        && owner
                            .upgrade()
                            .is_some_and(|owner| Arc::ptr_eq(&owner, &value.buffer))
                },
            ))
        .then(|| entry.packed.upgrade())
        .flatten()
    }) {
        return Ok(Arc::new(CudaValue::from_planned_buffer(
            device.clone(),
            ValueSpec::dense(DType::BF16, &[widths.iter().sum(), 2816]),
            (*value).clone(),
        )?));
    }
    let columns: usize = widths.iter().sum();
    let buffer = CudaBuffer::from_slice(
        device
            .stream
            .alloc_zeros::<u8>(columns * 2816 * 2)
            .map_err(|e| e.to_string())?,
    );
    let mut offset = 0;
    let result = (|| {
        for (weight, width) in weights.iter().zip(widths) {
            let bytes = width * 2816 * 2;
            unsafe {
                sys::cuMemcpyDtoDAsync_v2(
                    buffer.address() + offset,
                    weight.storage_address(),
                    bytes,
                    device.stream.cu_stream(),
                )
            }
            .result()
            .map_err(|e| e.to_string())?;
            offset += bytes as u64;
        }
        Ok::<_, String>(())
    })();
    if let Err(error) = device.stream.synchronize() {
        std::mem::forget(buffer);
        std::mem::forget(weights.to_vec());
        return Err(format!(
            "packed77 packing drain failed; owners quarantined: {error}"
        ));
    }
    result?;
    let value = Arc::new(CudaValue::from_planned_buffer(
        device.clone(),
        ValueSpec::dense(DType::BF16, &[columns, 2816]),
        buffer,
    )?);
    cache.push(Cached {
        sources: weights
            .iter()
            .map(|value| Arc::downgrade(&value.buffer))
            .collect(),
        addresses: weights.iter().map(CudaValue::storage_address).collect(),
        widths: widths.to_vec(),
        packed: Arc::downgrade(&value.buffer),
    });
    Ok(value)
}

#[derive(Clone)]
struct Projection {
    position: usize,
    x: ValueId,
    weight: ValueId,
    output: ValueId,
    workspace: ValueId,
    plan: Bf16GemmPlan,
}
fn candidate(command: &Command, position: usize) -> Option<Projection> {
    let CommandKind::Gemm {
        x,
        weight,
        weight_transposed: true,
        plan,
        out_f32: false,
        workspace,
    } = command.kind
    else {
        return None;
    };
    (command.overlap == CommandOverlap::Primary
        && plan.m == 256
        && plan.k == 2816
        && plan.batch == 1
        && matches!(plan.n, 2112 | 4096 | 2048))
    .then(|| Projection {
        position,
        x,
        weight,
        output: command.output.unwrap(),
        workspace,
        plan,
    })
}
fn admitted_widths(widths: &[usize]) -> bool {
    matches!(widths, [2112, 2112] | [4096, 2048, 2048])
}

impl CudaProgramBuilder {
    pub(super) fn pack_projections77(&mut self) -> Result<(), String> {
        let mut first = 0;
        while first < self.commands.len() {
            let Some(a) = candidate(&self.commands[first], first) else {
                first += 1;
                continue;
            };
            if !matches!(a.plan.n, 2112 | 4096) {
                first += 1;
                continue;
            }
            let wanted = if a.plan.n == 2112 { 2 } else { 3 };
            let selected = self
                .commands
                .iter()
                .enumerate()
                .skip(first)
                .filter_map(|(position, command)| candidate(command, position))
                .filter(|p| p.x == a.x)
                .take(wanted)
                .collect::<Vec<_>>();
            let widths = selected.iter().map(|p| p.plan.n).collect::<Vec<_>>();
            if !admitted_widths(&widths) {
                first += 1;
                continue;
            }
            let weights = selected
                .iter()
                .map(|p| {
                    self.commands.iter().find_map(|command| {
                        if command.output != Some(p.weight) {
                            return None;
                        }
                        match &command.kind {
                            CommandKind::Value(value)
                                if value.dtype() == DType::BF16
                                    && value.shape() == [p.plan.n, 2816] =>
                            {
                                Some(value.clone())
                            }
                            _ => None,
                        }
                    })
                })
                .collect::<Option<Vec<_>>>();
            let Some(weights) = weights else {
                first += 1;
                continue;
            };
            if weights
                .iter()
                .any(|w| !Arc::ptr_eq(&w.device, &weights[0].device))
            {
                first += 1;
                continue;
            }
            let positions = self
                .lowered
                .iter()
                .enumerate()
                .filter_map(|(i, inst)| {
                    (!matches!(
                        inst.kind,
                        "state_prepare" | "state_commit" | "state_discard" | "status_check"
                    ))
                    .then_some(i)
                })
                .collect::<Vec<_>>();
            if positions.len() != self.commands.len() {
                return Err("packed77 command mapping mismatch".into());
            }
            let aliases = effect_torch_compiler::normalize_aliases(&self.values)
                .map_err(|e| e.to_string())?;
            let root = |id: ValueId| aliases[id.index()].root;
            let last = selected.last().unwrap().position;
            // Pure projections may run before independent normalization/checks;
            // no externally visible effect or activation mutation is crossed.
            // A failure still drains the invocation before discarding outputs.
            if self.lowered[positions[first]..=positions[last]]
                .iter()
                .any(|i| {
                    i.effects.has_side_effects
                        || i.resource_uses()
                            .any(|u| u.access.writes() && root(u.value) == root(a.x))
                })
            {
                first += 1;
                continue;
            }
            let packed = packed_weight(&weights, &widths)?;
            let columns = widths.iter().sum();
            let weight_id = self.value(
                vec![columns, 2816],
                DType::BF16,
                StorageMetadata::dense(),
                "packed77_weight",
                ValueStorage::Fixed {
                    class: StorageClass::PersistentConstant,
                    location: Location::Persistent {
                        slot: self.values.len() as u32,
                    },
                },
            )?;
            let temporary = self.planned(vec![256, columns], DType::BF16, "packed77_temporary")?;
            let mut plan = a.plan;
            plan.n = columns;
            plan.stride_weight = columns * 2816;
            plan.stride_out = 256 * columns;
            let outputs = selected.iter().map(|p| p.output).collect::<Vec<_>>();
            let instruction = &mut self.lowered[positions[first]];
            instruction.kind = "packed_projection77";
            instruction.inputs =
                vec![ValueUse::read(a.x), ValueUse::read(weight_id)].into_boxed_slice();
            instruction.outputs = outputs
                .iter()
                .copied()
                .chain(std::iter::once(temporary))
                .map(OutputDecl::new)
                .collect::<Vec<_>>()
                .into_boxed_slice();
            instruction.scratch = vec![ValueUse::read_write(a.workspace)].into_boxed_slice();
            self.commands[first].kind = CommandKind::PackedProjection77 {
                skip_split: false,
                x: a.x,
                weight: weight_id,
                outputs,
                widths,
                temporary,
                workspace: a.workspace,
                plan,
            };
            for p in selected.iter().skip(1) {
                let i = &mut self.lowered[positions[p.position]];
                i.kind = "packed77_output";
                i.inputs = vec![ValueUse::read(p.output)].into_boxed_slice();
                i.outputs = Box::new([]);
                i.scratch = Box::new([]);
                i.effects = InstructionEffects::default();
                self.commands[p.position].kind = CommandKind::PlannedAlias;
            }
            // Store a normal persistent value declaration, so ordinary graph
            // storage and lowering lifetime accounting retain its allocation.
            self.commands.insert(
                first,
                Command {
                    output: Some(weight_id),
                    kind: CommandKind::Value((*packed).clone()),
                    overlap: CommandOverlap::Primary,
                },
            );
            self.lowered.insert(
                positions[first],
                LoweredInstruction::new(
                    InstructionId::from_index(0).unwrap(),
                    "value",
                    Vec::new(),
                    Vec::new(),
                ),
            );
            // The declaration keeps the backing buffer alive; cache the same
            // owner separately via the value's retained allocation, not addresses.
            first += 2;
        }
        for (index, instruction) in self.lowered.iter_mut().enumerate() {
            instruction.id =
                InstructionId::from_index(index).ok_or("packed77 instruction overflow")?;
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mutable_weights_cannot_enter_the_packed_cache_even_when_requested() {
        assert!(!admitted_policy(false, true));
        assert!(!admitted_policy(false, false));
        assert!(!admitted_policy(true, false));
        assert!(admitted_policy(true, true));
    }
    #[test]
    fn projection_shapes_exclude_full_attention_and_partial_groups() {
        assert!(admitted_widths(&[2112, 2112]));
        assert!(admitted_widths(&[4096, 2048, 2048]));
        for widths in [
            &[8192, 1024][..],
            &[4096, 2048][..],
            &[2048, 2048, 4096][..],
            &[2112][..],
        ] {
            assert!(!admitted_widths(widths));
        }
    }
}
