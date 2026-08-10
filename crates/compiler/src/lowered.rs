use effect_torch_runtime::{InstructionId, Location, SegmentOwnership, StorageClass, ValueId};

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
        }
    }
}

/// Dense compiler IR consumed by liveness and backend command planning.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LoweredSchedule<K, M> {
    pub values: Box<[ValueDecl<M>]>,
    pub instructions: Box<[LoweredInstruction<K>]>,
    pub outputs: Box<[ValueId]>,
}

impl<K, M> LoweredSchedule<K, M> {
    pub fn new(
        values: impl Into<Box<[ValueDecl<M>]>>,
        instructions: impl Into<Box<[LoweredInstruction<K>]>>,
        outputs: impl Into<Box<[ValueId]>>,
    ) -> Self {
        Self {
            values: values.into(),
            instructions: instructions.into(),
            outputs: outputs.into(),
        }
    }
}
