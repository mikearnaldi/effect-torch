//! Backend-neutral value and instruction tables between graph optimization and
//! physical memory planning.
//!
//! Planner validation enforces the [`LoweredProgram`] contract:
//!
//! - `values[i]` carries `ValueId::from_index(i)`, and `instructions[i]`
//!   carries `InstructionId::from_index(i)`. IDs are table positions. This
//!   gives O(1) lookup and makes table mismatches structural errors.
//! - Every value an instruction touches appears in exactly one resource
//!   category. Logical uses are `inputs` and `outputs`. Backend resources
//!   use `scratch`, `staging`, `status`, and `state`. Liveness and
//!   scheduling use all categories. Effects with no declared value use belong
//!   in [`InstructionEffects`].
//! - Each value declares `Fixed`, `Planned`, or `Alias` storage. The backend
//!   assigns fixed storage, the planner packs planned storage, and an alias
//!   refers to another value at a byte offset.
//!
//! Backends define the type parameters. `K` is the instruction kind, `M` is
//! the memory-space type, and `V` is the value record. `V` may wrap the
//! planning [`ValueDecl`] with backend metadata.

use effect_torch_runtime::{InstructionId, Location, SegmentOwnership, StorageClass, ValueId};
use std::marker::PhantomData;

/// How an instruction accesses one value.
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

/// One declared use of a value by an instruction.
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

/// Names a value defined by an instruction.
/// Outputs are the only category treated as definitions for liveness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OutputDecl {
    pub value: ValueId,
}

impl OutputDecl {
    pub const fn new(value: ValueId) -> Self {
        Self { value }
    }
}

/// Backend-neutral effects with no declared value use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct InstructionEffects {
    /// The instruction can report an execution failure.
    pub may_fail: bool,
    /// The instruction has observable effects beyond its declared resource writes.
    pub has_side_effects: bool,
}

/// Storage assigned outside the planner.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValueStorage<M> {
    /// The backend placed this value as an external input, persistent constant
    /// or state, escaping output, or device status. The planner counts its
    /// bytes but never moves it.
    Fixed {
        class: StorageClass,
        location: Location,
    },
    /// The planner must pack this value into a segment of `memory_space`
    /// with the given byte alignment and ownership class.
    Planned {
        class: StorageClass,
        alignment: usize,
        memory_space: M,
        ownership: SegmentOwnership,
    },
    /// A view into another value at `byte_offset`. An alias has no storage of
    /// its own. Alias normalization folds its uses into the root value's live
    /// interval.
    Alias { source: ValueId, byte_offset: usize },
}

/// Declares a lowered value's dense ID, diagnostic name, byte size, and storage.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ValueDecl<M> {
    pub id: ValueId,
    pub name: String,
    pub bytes: usize,
    pub storage: ValueStorage<M>,
}

/// A backend value record that provides its planning declaration.
pub trait LoweredValue<M> {
    fn value_decl(&self) -> &ValueDecl<M>;
}

impl<M> LoweredValue<M> for ValueDecl<M> {
    fn value_decl(&self) -> &ValueDecl<M> {
        self
    }
}

impl<M> ValueDecl<M> {
    /// A workspace value the planner will pack into a segment of
    /// `memory_space`.
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

    /// Declares a `bytes`-sized view into `source` at `byte_offset`. Alias
    /// normalization checks the bounds.
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

    /// The storage accounting class. Aliases return `None` because their root
    /// accounts for the bytes.
    pub const fn storage_class(&self) -> Option<StorageClass> {
        match &self.storage {
            ValueStorage::Fixed { class, .. } | ValueStorage::Planned { class, .. } => Some(*class),
            ValueStorage::Alias { .. } => None,
        }
    }
}

/// A backend-lowered logical instruction. `K` remains backend-defined.
///
/// Resource categories partition every value the instruction touches.
/// `inputs` are logical reads, and `outputs` are defining writes. `scratch`
/// is transient per-invocation workspace, `staging` is invocation staging,
/// `status` is device status, and `state` is persistent state. Every touched
/// value must appear in one list. Other effects belong to
/// [`InstructionEffects`].
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
    /// An instruction with no resource uses beyond its logical inputs and
    /// outputs and no extra effects.
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

    /// Attaches scratch, staging, status, and state uses in declaration order.
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

/// Dense compiler IR used by planning and execution.
///
/// `values` and `instructions` are index-parallel to their dense IDs.
/// `outputs` lists values that remain materialized through invocation.
/// `PhantomData<fn() -> M>` ties the memory-space type to value records
/// without owning a value of that type.
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
