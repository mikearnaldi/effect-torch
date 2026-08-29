//! Compiler-produced memory plans and their validation.
//!
//! A [`MemoryPlan`] is the static output of the compiler's allocation pass.
//! It declares the flat byte segments a backend must reserve. Each
//! intermediate value has a [`Location`] in a segment, an external or
//! persistent slot, an inline scalar, or another value's storage. The plan
//! also fixes output slots. Reuse edges allow storage recycling after a
//! producer's last consumer executes. Backends follow these addresses rather
//! than computing their own.
//!
//! [`SegmentOwnership`] records who controls a segment's lifetime.
//! `Workspace` is transient scratch pooled per invocation.
//! `ProvisionalOutput` becomes visible to the caller on success.
//! `InvocationStaging` and `StateTransaction` hold per-call staging and
//! transactional state updates. [`StorageClass`] categorizes allocations in
//! [`MemoryReport`].
//!
//! [`MemoryPlan::validate`] checks the plan before a backend sees it.
//! Segment alignments must be non-zero powers of two, and their total byte
//! count must fit in `usize`. Each `Location::Segment` must refer to an
//! existing segment and fit within its bounds using checked end arithmetic.
//! Each output slot must refer to an existing location. Validation returns a
//! [`MemoryPlanError`] on failure.
//!
//! Plan types use [`NativeMemorySpace`] by default. A backend can supply its
//! own memory-space type as `M` without changing the rest of the plan.

use crate::{InstructionId, LocationId, OutputId, SegmentId, ValueId};
use std::error::Error;
use std::fmt;

/// Allocation lifecycle category used by [`MemoryReport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageClass {
    /// Caller-provided input storage that the runtime does not own.
    ExternalInput,
    /// Constants retained for the lifetime of the executable.
    PersistentConstant,
    /// Mutable state retained across invocations.
    PersistentState,
    /// Storage returned to the caller as an output.
    EscapingOutput,
    /// Transient scratch reused across invocations.
    Workspace,
    /// Device-visible status/auxiliary storage.
    DeviceStatus,
}

/// Segment memory space for backends without a richer model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeMemorySpace {
    Cpu,
    /// Metal shared memory, accessible to the CPU and GPU.
    MetalShared,
    /// Metal private memory, accessible only to the GPU.
    MetalPrivate,
}

/// Who controls the lifetime of a segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentOwnership {
    /// Transient scratch that may be pooled and recycled between invocations.
    Workspace,
    /// Execution writes this segment, then the runtime returns it as an output.
    ProvisionalOutput,
    /// Per-invocation staging area, such as host/device marshalling.
    InvocationStaging,
    /// Staging for state updates applied transactionally on success.
    StateTransaction,
}

/// Declaration of one flat segment a backend must allocate.
///
/// `alignment` must be a non-zero power of two (checked by
/// [`MemoryPlan::validate`]). `bytes` is the segment's total capacity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SegmentDecl<M = NativeMemorySpace> {
    pub bytes: usize,
    pub alignment: usize,
    pub memory_space: M,
    pub ownership: SegmentOwnership,
}

/// Where a value's bytes live.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Location {
    /// Caller-provided external input in the given slot.
    External { slot: u32 },
    /// Persistent (constant or state) storage in the given slot.
    Persistent { slot: u32 },
    /// A scalar value passed inline, with no backing storage.
    InlineScalar { slot: u32 },
    /// A byte range inside a declared segment. [`MemoryPlan::validate`]
    /// checks the range against the segment's capacity.
    Segment {
        segment: SegmentId,
        offset: usize,
        bytes: usize,
    },
    /// A byte range inside an output buffer.
    Output {
        slot: OutputId,
        byte_offset: usize,
        bytes: usize,
    },
    /// A byte range inside a persistent state buffer.
    State {
        slot: u32,
        byte_offset: usize,
        bytes: usize,
    },
    /// Reuses the storage of another value at a byte offset.
    Alias { root: ValueId, byte_offset: usize },
}

/// Binds an output value to its destination location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OutputSlot {
    pub value: ValueId,
    pub location: LocationId,
}

/// Contract for reusing storage. After instruction `after`,
/// `next_value` may reuse `previous_value`'s storage. The runtime must write
/// `next_value` before instruction `before`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReuseEdge {
    pub previous_value: ValueId,
    pub next_value: ValueId,
    pub after: InstructionId,
    pub before: InstructionId,
}

/// One named allocation in a [`MemoryReport`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AllocationReport {
    pub name: String,
    pub class: StorageClass,
    pub bytes: usize,
}

/// Per-executable memory totals grouped by storage role.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct MemoryReport {
    pub external_bytes: usize,
    pub persistent_bytes: usize,
    pub state_bytes: usize,
    pub output_bytes: usize,
    pub workspace_bytes: usize,
    pub transaction_bytes: usize,
    pub peak_live_bytes: usize,
    pub packing_overhead_bytes: usize,
    pub largest_allocations: Box<[AllocationReport]>,
}

/// Per-invocation memory totals for the static plan, leased workspace, and
/// backend-reserved headroom.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct InvocationMemoryReport {
    pub logical: MemoryReport,
    pub leased_workspace_bytes: usize,
    pub opaque_headroom_bytes: usize,
}

impl MemoryReport {
    /// Returns the bytes the runtime owns for the executable, excluding
    /// external inputs, or `None` on overflow.
    pub fn checked_runtime_owned_bytes(&self) -> Option<usize> {
        [
            self.persistent_bytes,
            self.state_bytes,
            self.output_bytes,
            self.workspace_bytes,
            self.transaction_bytes,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
}

impl InvocationMemoryReport {
    /// Returns the bytes used by one invocation, including logical
    /// allocations, leased workspace, and headroom, or `None` on overflow.
    pub fn checked_accounted_bytes(&self) -> Option<usize> {
        [
            self.logical.external_bytes,
            self.logical.persistent_bytes,
            self.logical.state_bytes,
            self.logical.output_bytes,
            self.logical.transaction_bytes,
            self.leased_workspace_bytes,
            self.opaque_headroom_bytes,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
}

/// Static memory plan of a compiled executable.
///
/// `segments`, `locations`, and `outputs` are dense tables indexed by
/// [`SegmentId`], [`LocationId`], and [`OutputId`]. The `Default`
/// implementation returns an empty valid plan.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemoryPlan<M = NativeMemorySpace> {
    pub segments: Box<[SegmentDecl<M>]>,
    pub locations: Box<[Location]>,
    pub outputs: Box<[OutputSlot]>,
    pub reuse_edges: Box<[ReuseEdge]>,
    pub report: MemoryReport,
}

impl<M> Default for MemoryPlan<M> {
    fn default() -> Self {
        Self {
            segments: Box::new([]),
            locations: Box::new([]),
            outputs: Box::new([]),
            reuse_edges: Box::new([]),
            report: MemoryReport::default(),
        }
    }
}

impl<M> MemoryPlan<M> {
    /// Total capacity of all segments, or `None` on overflow.
    pub fn checked_segment_bytes(&self) -> Option<usize> {
        self.segments
            .iter()
            .try_fold(0usize, |total, segment| total.checked_add(segment.bytes))
    }

    /// Checks segment alignments, total segment bytes, segment-backed
    /// location bounds, and output location references.
    ///
    /// This does not prove alias source ranges, state or output byte ranges,
    /// reuse-edge ordering, or consistency of the diagnostic [`MemoryReport`].
    /// The compiler and backend must enforce those invariants. Validation does
    /// not mutate the plan.
    pub fn validate(&self) -> Result<(), MemoryPlanError> {
        for (index, segment) in self.segments.iter().enumerate() {
            if segment.alignment == 0 || !segment.alignment.is_power_of_two() {
                return Err(MemoryPlanError::InvalidAlignment {
                    segment: SegmentId::from_index(index)
                        .ok_or(MemoryPlanError::TooManySegments)?,
                    alignment: segment.alignment,
                });
            }
        }
        self.checked_segment_bytes()
            .ok_or(MemoryPlanError::ByteSizeOverflow)?;

        for (index, location) in self.locations.iter().enumerate() {
            if let Location::Segment {
                segment,
                offset,
                bytes,
            } = location
            {
                let decl =
                    self.segments
                        .get(segment.index())
                        .ok_or(MemoryPlanError::UnknownSegment {
                            location: LocationId::from_index(index)
                                .ok_or(MemoryPlanError::TooManyLocations)?,
                            segment: *segment,
                        })?;
                let end = offset
                    .checked_add(*bytes)
                    .ok_or(MemoryPlanError::ByteSizeOverflow)?;
                if end > decl.bytes {
                    return Err(MemoryPlanError::LocationOutOfBounds {
                        location: LocationId::from_index(index)
                            .ok_or(MemoryPlanError::TooManyLocations)?,
                        segment: *segment,
                        end,
                        capacity: decl.bytes,
                    });
                }
            }
        }

        for output in &self.outputs {
            if self.locations.get(output.location.index()).is_none() {
                return Err(MemoryPlanError::UnknownOutputLocation {
                    output: output.value,
                    location: output.location,
                });
            }
        }
        Ok(())
    }
}

/// Why a [`MemoryPlan`] failed validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryPlanError {
    /// More segments than fit in the `u32` id space.
    TooManySegments,
    /// More locations than fit in the `u32` id space.
    TooManyLocations,
    /// Byte arithmetic overflowed `usize`.
    ByteSizeOverflow,
    /// Segment alignment was zero or not a power of two.
    InvalidAlignment {
        segment: SegmentId,
        alignment: usize,
    },
    /// A location points at a segment id not present in the plan.
    UnknownSegment {
        location: LocationId,
        segment: SegmentId,
    },
    /// A location's byte range extends past its segment's capacity.
    LocationOutOfBounds {
        location: LocationId,
        segment: SegmentId,
        end: usize,
        capacity: usize,
    },
    /// An output slot points at a location id not present in the plan.
    UnknownOutputLocation {
        output: ValueId,
        location: LocationId,
    },
}

impl fmt::Display for MemoryPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryPlanError::TooManySegments => f.write_str("memory plan has too many segments"),
            MemoryPlanError::TooManyLocations => f.write_str("memory plan has too many locations"),
            MemoryPlanError::ByteSizeOverflow => f.write_str("memory plan byte size overflow"),
            MemoryPlanError::InvalidAlignment { segment, alignment } => {
                write!(f, "segment {segment} has invalid alignment {alignment}")
            }
            MemoryPlanError::UnknownSegment { location, segment } => {
                write!(
                    f,
                    "location {location} references unknown segment {segment}"
                )
            }
            MemoryPlanError::LocationOutOfBounds {
                location,
                segment,
                end,
                capacity,
            } => write!(
                f,
                "location {location} ends at {end} in segment {segment} with capacity {capacity}"
            ),
            MemoryPlanError::UnknownOutputLocation { output, location } => write!(
                f,
                "output value {output} references unknown location {location}"
            ),
        }
    }
}

impl Error for MemoryPlanError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_plan_validation_checks_ranges_and_overflow() {
        let mut plan = MemoryPlan {
            segments: vec![SegmentDecl {
                bytes: 64,
                alignment: 16,
                memory_space: NativeMemorySpace::Cpu,
                ownership: SegmentOwnership::Workspace,
            }]
            .into_boxed_slice(),
            locations: vec![Location::Segment {
                segment: SegmentId::new(0),
                offset: 32,
                bytes: 32,
            }]
            .into_boxed_slice(),
            ..MemoryPlan::default()
        };
        assert!(plan.validate().is_ok());

        plan.locations[0] = Location::Segment {
            segment: SegmentId::new(0),
            offset: usize::MAX,
            bytes: 1,
        };
        assert_eq!(plan.validate(), Err(MemoryPlanError::ByteSizeOverflow));
    }
}
