use effect_torch_runtime::{InstructionId, Location, SegmentOwnership, StorageClass, ValueId};
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueAccess {
    Read,
    Write,
    ReadWrite,
}

impl ValueAccess {
    pub const fn reads(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }

    pub const fn writes(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueUse {
    pub value: ValueId,
    pub access: ValueAccess,
}

impl ValueUse {
    pub const fn read(value: ValueId) -> Self {
        Self {
            value,
            access: ValueAccess::Read,
        }
    }

    pub const fn write(value: ValueId) -> Self {
        Self {
            value,
            access: ValueAccess::Write,
        }
    }

    pub const fn read_write(value: ValueId) -> Self {
        Self {
            value,
            access: ValueAccess::ReadWrite,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OutputDecl {
    pub value: ValueId,
}

impl OutputDecl {
    pub const fn new(value: ValueId) -> Self {
        Self { value }
    }
}

/// Backend-neutral effects which are not represented by declared value uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct InstructionEffects {
    /// The instruction can report an execution failure.
    pub may_fail: bool,
    /// The instruction has observable effects beyond its declared resource writes.
    pub has_side_effects: bool,
}

/// Storage which is supplied outside the planner or is already assigned.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValueStorage<M> {
    Fixed {
        class: StorageClass,
        location: Location,
    },
    Planned {
        class: StorageClass,
        alignment: usize,
        memory_space: M,
        ownership: SegmentOwnership,
    },
    Alias {
        source: ValueId,
        byte_offset: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ValueDecl<M> {
    pub id: ValueId,
    pub name: String,
    pub bytes: usize,
    pub storage: ValueStorage<M>,
}

/// A backend value record containing the declaration authoritative for planning.
pub trait LoweredValue<M> {
    fn value_decl(&self) -> &ValueDecl<M>;
}

impl<M> LoweredValue<M> for ValueDecl<M> {
    fn value_decl(&self) -> &ValueDecl<M> {
        self
    }
}

impl<M> ValueDecl<M> {
    pub fn planned(
        id: ValueId,
        name: impl Into<String>,
        bytes: usize,
        alignment: usize,
        memory_space: M,
        ownership: SegmentOwnership,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            bytes,
            storage: ValueStorage::Planned {
                class: StorageClass::Workspace,
                alignment,
                memory_space,
                ownership,
            },
        }
    }

    pub fn alias(
        id: ValueId,
        name: impl Into<String>,
        source: ValueId,
        byte_offset: usize,
        bytes: usize,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            bytes,
            storage: ValueStorage::Alias {
                source,
                byte_offset,
            },
        }
    }

    pub const fn storage_class(&self) -> Option<StorageClass> {
        match &self.storage {
            ValueStorage::Fixed { class, .. } | ValueStorage::Planned { class, .. } => Some(*class),
            ValueStorage::Alias { .. } => None,
        }
    }
}

/// A backend-lowered logical instruction. `K` remains backend-defined.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LoweredInstruction<K> {
    pub id: InstructionId,
    pub kind: K,
    pub inputs: Box<[ValueUse]>,
    pub outputs: Box<[OutputDecl]>,
    pub scratch: Box<[ValueUse]>,
    pub staging: Box<[ValueUse]>,
    pub status: Box<[ValueUse]>,
    pub state: Box<[ValueUse]>,
    pub effects: InstructionEffects,
}

impl<K> LoweredInstruction<K> {
    pub fn new(
        id: InstructionId,
        kind: K,
        inputs: impl Into<Box<[ValueUse]>>,
        outputs: impl Into<Box<[OutputDecl]>>,
    ) -> Self {
        Self {
            id,
            kind,
            inputs: inputs.into(),
            outputs: outputs.into(),
            scratch: Box::new([]),
            staging: Box::new([]),
            status: Box::new([]),
            state: Box::new([]),
            effects: InstructionEffects::default(),
        }
    }

    pub fn with_resources(
        mut self,
        scratch: impl Into<Box<[ValueUse]>>,
        staging: impl Into<Box<[ValueUse]>>,
        status: impl Into<Box<[ValueUse]>>,
        state: impl Into<Box<[ValueUse]>>,
    ) -> Self {
        self.scratch = scratch.into();
        self.staging = staging.into();
        self.status = status.into();
        self.state = state.into();
        self
    }

    pub const fn with_effects(mut self, effects: InstructionEffects) -> Self {
        self.effects = effects;
        self
    }

    /// All values touched by this instruction. Outputs are defining writes.
    pub fn resource_uses(&self) -> impl Iterator<Item = ValueUse> + '_ {
        self.inputs
            .iter()
            .copied()
            .chain(
                self.outputs
                    .iter()
                    .map(|output| ValueUse::write(output.value)),
            )
            .chain(self.scratch.iter().copied())
            .chain(self.staging.iter().copied())
            .chain(self.status.iter().copied())
            .chain(self.state.iter().copied())
    }
}

/// Authoritative dense compiler IR consumed by planning and execution.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LoweredProgram<K, M, V = ValueDecl<M>> {
    pub values: Box<[V]>,
    pub instructions: Box<[LoweredInstruction<K>]>,
    pub outputs: Box<[ValueId]>,
    memory_space: PhantomData<fn() -> M>,
}

impl<K, M, V> LoweredProgram<K, M, V>
where
    V: LoweredValue<M>,
{
    pub fn new(
        values: impl Into<Box<[V]>>,
        instructions: impl Into<Box<[LoweredInstruction<K>]>>,
        outputs: impl Into<Box<[ValueId]>>,
    ) -> Self {
        Self {
            values: values.into(),
            instructions: instructions.into(),
            outputs: outputs.into(),
            memory_space: PhantomData,
        }
    }
}
