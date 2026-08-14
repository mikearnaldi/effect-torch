//! The backend-neutral lowered IR: dense value and instruction tables that
//! sit between graph optimization and physical memory planning.
//!
//! A [`LoweredProgram`] is the authoritative artifact a backend lowering
//! produces. Its contract, enforced by the planner's validation:
//!
//! - **Dense identity.** `values[i]` must carry `ValueId::from_index(i)` and
//!   `instructions[i]` must carry `InstructionId::from_index(i)`. IDs are
//!   table positions, so lookups are O(1) and mismatched tables are
//!   structural errors, not runtime surprises.
//! - **Declared effects.** Every value an instruction touches is listed in
//!   exactly one resource category: logical `inputs`/`outputs`, plus
//!   `scratch`, `staging`, `status`, and `state` for backend resources.
//!   Liveness and scheduling see the union; effects that no declared use can
//!   express live in [`InstructionEffects`].
//! - **Storage intent.** Each value declares whether its storage is `Fixed`
//!   (backend-assigned, outside the planner), `Planned` (packed into
//!   planner-owned segments), or an `Alias` of another value at a byte
//!   offset.
//!
//! The type parameters stay fully backend-defined: `K` is the instruction
//! kind, `M` the memory-space type, and `V` the value record (which may wrap
//! the authoritative [`ValueDecl`] with backend-only metadata).

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

/// A defining write: each entry names a value the instruction defines.
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
    /// The backend already placed this value (external inputs, persistent
    /// constants/state, escaping outputs, device status); the planner only
    /// accounts for its bytes and never moves it.
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
    /// A view into another value at `byte_offset`. Aliases receive no
    /// storage of their own; their uses fold into the root value's live
    /// interval after alias normalization.
    Alias { source: ValueId, byte_offset: usize },
}

/// Declaration of one lowered value: its dense identity, a human-readable
/// name for diagnostics, its byte size, and its storage intent.
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

    /// A value declared as a `bytes`-sized view into `source` at
    /// `byte_offset`. Bounds are checked during alias normalization.
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

    /// The accounting class of this value's storage, or `None` for aliases
    /// (whose bytes are accounted through their root).
    pub const fn storage_class(&self) -> Option<StorageClass> {
        match &self.storage {
            ValueStorage::Fixed { class, .. } | ValueStorage::Planned { class, .. } => Some(*class),
            ValueStorage::Alias { .. } => None,
        }
    }
}

/// A backend-lowered logical instruction. `K` remains backend-defined.
///
/// The resource categories partition every value the instruction touches:
/// `inputs` are logical reads, `outputs` are defining writes, `scratch` is
/// transient per-invocation workspace, `staging` is invocation staging,
/// `status` is device status, and `state` is persistent state. Nothing may
/// be touched without appearing in one of these lists; undeclared effects
/// belong to [`InstructionEffects`].
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

    /// Attaches the non-logical resource uses (scratch, staging, status,
    /// state) in declaration order.
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
///
/// `values` and `instructions` are index-parallel to their dense IDs;
/// `outputs` lists the values that must remain materialized through
/// invocation completion. The `PhantomData<fn() -> M>` marker ties the
/// memory-space type to the value records without owning one.
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
