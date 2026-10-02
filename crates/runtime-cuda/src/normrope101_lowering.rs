//! Late private packed-QKV normalization with original public output leases.
use super::*;

impl CudaProgramBuilder {
    pub(super) fn fuse_normrope101(&mut self, roots: &[ValueId]) -> Result<(), String> {
        if std::env::var_os(crate::triton_normrope101::DIRECTORY_ENV).is_none() {
            return Ok(());
        }
        let diagnostic_path = std::env::var_os("EFFECT_TORCH_CUDA_NORMROPE101_DIAGNOSTICS");
        let mut considered = 0usize;
        let mut admitted = 0usize;
        let mut image = None;
        for packed_command in 0..self.commands.len() {
            if !matches!(&self.commands[packed_command].kind, CommandKind::PackedProjection77 { widths, .. } if widths.as_slice() == [4096,2048,2048])
            {
                continue;
            }
            considered += 1;
            let group = if let Some(path) = &diagnostic_path {
                let mut records = Vec::new();
                let group = crate::normrope101_admission::admit_diagnostic(
                    &self.commands,
                    &self.lowered,
                    &self.values,
                    roots,
                    packed_command,
                    Some(&mut records),
                );
                let window = self.commands.iter().enumerate().skip(packed_command).take(64).map(|(position, command)| {
                    let kind = match &command.kind {
                        CommandKind::Kernel { name, .. } => *name,
                        CommandKind::Input { .. } => "input",
                        CommandKind::Value(_) => "value",
                        CommandKind::PlannedAlias => "planned_alias",
                        CommandKind::Alias { .. } => "alias",
                        CommandKind::Prepare => "prepare",
                        CommandKind::PackedProjection77 { .. } => "packed77",
                        _ => "other",
                    };
                    serde_json::json!({"command":position,"kind":kind,"output":command.output.map(|id|serde_json::json!({"id":id.get(),"shape":self.values[id.index()].shape,"dtype":format!("{:?}",self.values[id.index()].dtype),"bytes":self.values[id.index()].decl.bytes,"storage":format!("{:?}",self.values[id.index()].decl.storage)}))})
                }).collect::<Vec<_>>();
                let row = serde_json::json!({"event":"normrope101_admission","packedCommand":packed_command,"commands":self.commands.len(),"instructions":self.lowered.len(),"admitted":group.is_some(),"roots":roots.iter().map(|id|id.get()).collect::<Vec<_>>(),"records":records,"window":window});
                use std::io::Write;
                let mut file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .map_err(|e| format!("normrope101 diagnostic open: {e}"))?;
                writeln!(file, "{row}")
                    .map_err(|e| format!("normrope101 diagnostic write: {e}"))?;
                group
            } else {
                crate::normrope101_admission::admit(
                    &self.commands,
                    &self.lowered,
                    &self.values,
                    roots,
                    packed_command,
                )
            };
            let Some(group) = group else {
                continue;
            };
            admitted += 1;
            let plan = if let Some(plan) = &image {
                Arc::clone(plan)
            } else {
                let device = self
                    .commands
                    .iter()
                    .find_map(|command| match &command.kind {
                        CommandKind::Value(value) => Some(&value.device),
                        _ => None,
                    })
                    .ok_or("compile: normrope101 device missing")?;
                let Some(plan) =
                    crate::triton_normrope101::NormRope101::from_env(device.stream.context())?
                else {
                    return Ok(());
                };
                image = Some(plan.clone());
                plan
            };
            let scratch = [
                self.planned(vec![256, 256], DType::BF16, "normrope101_table")?,
                self.planned(vec![256], DType::I64, "normrope101_positions")?,
                self.planned(vec![256, 16], DType::F32, "normrope101_sum_q")?,
                self.planned(vec![256, 8], DType::F32, "normrope101_sum_k")?,
                self.planned(vec![256, 16, 256], DType::BF16, "normrope101_raw_q")?,
                self.planned(vec![256, 8, 256], DType::BF16, "normrope101_raw_k")?,
                self.planned(vec![256, 8, 256], DType::BF16, "normrope101_raw_v")?,
            ];
            let mut definitions = self.lowered[group.prepare_instruction].outputs.to_vec();
            definitions.extend(scratch.iter().copied().map(OutputDecl::new));
            self.lowered[group.prepare_instruction].outputs = definitions.into();
            for (&command, &instruction) in group.norm_commands.iter().zip(&group.norm_instructions)
            {
                if command == group.dispatch_command {
                    continue;
                }
                self.commands[command].kind = CommandKind::PlannedAlias;
                let lowered = &mut self.lowered[instruction];
                lowered.kind = "normrope101_output";
                lowered.inputs = Box::new([]);
                lowered.scratch = Box::new([]);
                lowered.effects = InstructionEffects::default();
            }
            let last = self.commands[group.dispatch_command]
                .output
                .ok_or("compile: normrope101 output missing")?;
            let instruction = &mut self.lowered[group.dispatch_instruction];
            instruction.kind = "triton_normrope101";
            instruction.inputs = std::iter::once(group.packed_temporary)
                .chain(group.borrowed)
                .map(ValueUse::read)
                .chain(
                    group
                        .outputs
                        .into_iter()
                        .filter(|id| *id != last)
                        .map(ValueUse::write),
                )
                .collect::<Vec<_>>()
                .into();
            instruction.scratch = scratch
                .into_iter()
                .map(ValueUse::read_write)
                .collect::<Vec<_>>()
                .into();
            self.commands[group.dispatch_command].kind = CommandKind::NormRope101 {
                packed: group.packed_temporary,
                borrowed: group.borrowed,
                table_strides: group.table_strides,
                scratch,
                outputs: group.outputs,
                plan,
            };
            let CommandKind::PackedProjection77 { skip_split, .. } =
                &mut self.commands[group.packed_command].kind
            else {
                unreachable!()
            };
            *skip_split = true;
        }
        if std::env::var("EFFECT_TORCH_CUDA_NORMROPE101_REQUIRE").as_deref() == Ok("1")
            && considered > 0
            && admitted == 0
        {
            return Err(format!(
                "compile: normrope101 required but all {considered} packed local groups declined; inspect diagnostics before generation"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "normrope101_compiled_tests.rs"]
mod tests;

/// Structural provenance only: each row contains two copies of the same half.
/// Reshaping may only regroup complete rows; arbitrary full-width inputs,
/// permutations, slices, and unrelated equal-looking constants are not proofs.
pub(super) fn repeated_half_table(mut node: &Node, width: usize) -> bool {
    if width == 0 || width % 2 != 0 || node.shape.last() != Some(&width) {
        return false;
    }
    for _ in 0..16 {
        match &node.kind {
            NodeKind::Concat { a, b, dim } => {
                return *dim + 1 == node.shape.len()
                    && a.id == b.id
                    && a.shape.len() == node.shape.len()
                    && a.shape.last() == Some(&(width / 2))
                    && a.shape[..a.shape.len() - 1] == node.shape[..node.shape.len() - 1];
            }
            NodeKind::Cast { a, .. } | NodeKind::Cos { a } | NodeKind::Sin { a }
                if a.shape == node.shape =>
            {
                node = a
            }
            NodeKind::Reshape { a, .. } if a.shape.last() == Some(&width) => node = a,
            _ => return false,
        }
    }
    false
}

#[cfg(test)]
mod repeat_proof_tests {
    use super::*;
    use effect_torch_graph::Device;
    fn input(slot: u32, shape: Vec<usize>) -> Arc<Node> {
        Node::new(NodeKind::Input {
            slot,
            shape,
            dtype: DType::F32,
            device: Device::Cuda(0),
            storage: StorageMetadata::dense(),
        })
        .unwrap()
    }
    #[test]
    fn repeated_half_provenance_accepts_actual_phase_trig_cast_reshape_chain() {
        let phase = input(0, vec![1, 256, 128]);
        let doubled = Node::new(NodeKind::Concat {
            a: phase.clone(),
            b: phase,
            dim: 2,
        })
        .unwrap();
        for trig in [
            NodeKind::Cos { a: doubled.clone() },
            NodeKind::Sin { a: doubled.clone() },
        ] {
            let table = Node::new(trig).unwrap();
            let table = Node::new(NodeKind::Cast {
                a: table,
                dtype: DType::BF16,
            })
            .unwrap();
            let table = Node::new(NodeKind::Reshape {
                a: table,
                shape: vec![1, 1, 256, 256],
            })
            .unwrap();
            assert!(repeated_half_table(&table, 256));
        }
        assert!(!repeated_half_table(&doubled, 512));
    }
    #[test]
    fn repeated_half_provenance_rejects_arbitrary_full_tables_wrong_axis_and_reordering() {
        let full = input(0, vec![1, 1, 256, 256]);
        assert!(!repeated_half_table(&full, 256));
        let a = input(1, vec![1, 256, 128]);
        let b = input(2, vec![1, 256, 128]);
        let distinct = Node::new(NodeKind::Concat {
            a: a.clone(),
            b,
            dim: 2,
        })
        .unwrap();
        assert!(!repeated_half_table(&distinct, 256));
        let repeated = Node::new(NodeKind::Concat {
            a: a.clone(),
            b: a,
            dim: 2,
        })
        .unwrap();
        let permuted = Node::new(NodeKind::Permute {
            a: repeated.clone(),
            dims: vec![0, 2, 1],
        })
        .unwrap();
        assert!(!repeated_half_table(&permuted, 256));
        let half = input(3, vec![1, 128, 256]);
        let rows = Node::new(NodeKind::Concat {
            a: half.clone(),
            b: half,
            dim: 1,
        })
        .unwrap();
        assert!(!repeated_half_table(&rows, 256));
        let mut deep = repeated;
        for _ in 0..17 {
            deep = Node::new(NodeKind::Cos { a: deep }).unwrap();
        }
        assert!(!repeated_half_table(&deep, 256));
    }
}
