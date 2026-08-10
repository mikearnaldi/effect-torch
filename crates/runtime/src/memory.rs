use crate::{InstructionId, LocationId, OutputId, SegmentId, ValueId};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageClass {
    ExternalInput,
    PersistentConstant,
    PersistentState,
    EscapingOutput,
    Workspace,
    DeviceStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeMemorySpace {
    Cpu,
    MetalShared,
    MetalPrivate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentOwnership {
    Workspace,
    ProvisionalOutput,
    InvocationStaging,
    StateTransaction,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SegmentDecl<M = NativeMemorySpace> {
    pub bytes: usize,
    pub alignment: usize,
    pub memory_space: M,
    pub ownership: SegmentOwnership,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Location {
    External {
        slot: u32,
    },
    Persistent {
        slot: u32,
    },
    InlineScalar {
        slot: u32,
    },
    Segment {
        segment: SegmentId,
        offset: usize,
        bytes: usize,
    },
    Output {
        slot: OutputId,
        byte_offset: usize,
        bytes: usize,
    },
    State {
        slot: u32,
        byte_offset: usize,
        bytes: usize,
    },
    Alias {
        root: ValueId,
        byte_offset: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OutputSlot {
    pub value: ValueId,
    pub location: LocationId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReuseEdge {
    pub previous_value: ValueId,
    pub next_value: ValueId,
    pub after: InstructionId,
    pub before: InstructionId,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AllocationReport {
    pub name: String,
    pub class: StorageClass,
    pub bytes: usize,
}

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

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct InvocationMemoryReport {
    pub logical: MemoryReport,
    pub leased_workspace_bytes: usize,
    pub opaque_headroom_bytes: usize,
}

impl MemoryReport {
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
    pub fn checked_segment_bytes(&self) -> Option<usize> {
        self.segments
            .iter()
            .try_fold(0usize, |total, segment| total.checked_add(segment.bytes))
    }

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryPlanError {
    TooManySegments,
    TooManyLocations,
    ByteSizeOverflow,
    InvalidAlignment {
        segment: SegmentId,
        alignment: usize,
    },
    UnknownSegment {
        location: LocationId,
        segment: SegmentId,
    },
    LocationOutOfBounds {
        location: LocationId,
        segment: SegmentId,
        end: usize,
        capacity: usize,
    },
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
