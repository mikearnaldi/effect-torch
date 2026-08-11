use crate::{LoweredProgram, LoweredValue, ValueStorage};
use effect_torch_runtime::{
    AllocationReport, InstructionId, Location, LocationId, MemoryPlan, MemoryReport, OutputSlot,
    ReuseEdge, SegmentDecl, SegmentId, SegmentOwnership, StorageClass, ValueId,
};
use std::cmp::Reverse;
use std::error::Error;
use std::fmt;
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NormalizedAlias {
    pub root: ValueId,
    pub byte_offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LiveInterval {
    pub value: ValueId,
    pub start: InstructionId,
    pub end: InstructionId,
}

impl LiveInterval {
    pub fn overlaps(self, other: Self) -> bool {
        self.start.get() < other.end.get() && other.start.get() < self.end.get()
    }

    pub fn contains(self, instruction: InstructionId) -> bool {
        self.start.get() <= instruction.get() && instruction.get() < self.end.get()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Liveness {
    pub aliases: Box<[NormalizedAlias]>,
    /// Alias entries are `None`; their uses are folded into the root entry.
    pub intervals: Box<[Option<LiveInterval>]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemorySpaceConfig<M> {
    pub memory_space: M,
    pub max_segment_bytes: usize,
    pub segment_alignment: usize,
    pub segment_granularity: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemoryPlannerConfig<M> {
    pub memory_spaces: Vec<MemorySpaceConfig<M>>,
    pub largest_allocations: usize,
}

impl<M> MemoryPlannerConfig<M> {
    pub fn new(memory_spaces: Vec<MemorySpaceConfig<M>>) -> Self {
        Self {
            memory_spaces,
            largest_allocations: 10,
        }
    }

    pub fn uniform(
        memory_space: M,
        max_segment_bytes: usize,
        segment_alignment: usize,
        segment_granularity: usize,
    ) -> Self {
        Self::new(vec![MemorySpaceConfig {
            memory_space,
            max_segment_bytes,
            segment_alignment,
            segment_granularity,
        }])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlannerError {
    TooManyValues,
    TooManyInstructions,
    TooManyLocations,
    TooManySegments,
    ValueIdMismatch {
        index: usize,
        actual: ValueId,
    },
    InstructionIdMismatch {
        index: usize,
        actual: InstructionId,
    },
    UnknownValue {
        instruction: Option<InstructionId>,
        value: ValueId,
    },
    AliasCycle {
        value: ValueId,
    },
    AliasOutOfBounds {
        value: ValueId,
        source: ValueId,
        end: usize,
        capacity: usize,
    },
    ByteSizeOverflow {
        value: Option<ValueId>,
    },
    InvalidValueAlignment {
        value: ValueId,
        alignment: usize,
    },
    InvalidMemorySpaceConfig {
        index: usize,
    },
    DuplicateMemorySpaceConfig {
        first: usize,
        second: usize,
    },
    MissingMemorySpaceConfig {
        value: ValueId,
    },
    ValueTooLarge {
        value: ValueId,
        bytes: usize,
        required_segment_bytes: usize,
        max_segment_bytes: usize,
    },
    ReadBeforeDefinition {
        value: ValueId,
        read: InstructionId,
        definition: InstructionId,
    },
    UndefinedPlannedValue {
        value: ValueId,
    },
    PackingConflict {
        first: ValueId,
        second: ValueId,
    },
    InvalidMemoryPlan(String),
}

impl fmt::Display for PlannerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyValues => f.write_str("lowered schedule has too many values"),
            Self::TooManyInstructions => f.write_str("lowered schedule has too many instructions"),
            Self::TooManyLocations => f.write_str("lowered schedule has too many locations"),
            Self::TooManySegments => f.write_str("memory plan has too many segments"),
            Self::ValueIdMismatch { index, actual } => {
                write!(f, "value table entry {index} has dense id {actual}")
            }
            Self::InstructionIdMismatch { index, actual } => {
                write!(f, "instruction table entry {index} has dense id {actual}")
            }
            Self::UnknownValue { instruction, value } => match instruction {
                Some(instruction) => {
                    write!(f, "instruction {instruction} references unknown value {value}")
                }
                None => write!(f, "lowered schedule references unknown value {value}"),
            },
            Self::AliasCycle { value } => write!(f, "alias cycle contains value {value}"),
            Self::AliasOutOfBounds {
                value,
                source,
                end,
                capacity,
            } => write!(
                f,
                "alias value {value} ends at byte {end} in value {source} with capacity {capacity}"
            ),
            Self::ByteSizeOverflow { value: Some(value) } => {
                write!(f, "byte size overflow while planning value {value}")
            }
            Self::ByteSizeOverflow { value: None } => {
                f.write_str("byte size overflow while constructing memory report")
            }
            Self::InvalidValueAlignment { value, alignment } => {
                write!(f, "value {value} has invalid alignment {alignment}")
            }
            Self::InvalidMemorySpaceConfig { index } => {
                write!(f, "memory-space configuration {index} is invalid")
            }
            Self::DuplicateMemorySpaceConfig { first, second } => write!(
                f,
                "memory-space configurations {first} and {second} name the same space"
            ),
            Self::MissingMemorySpaceConfig { value } => {
                write!(f, "value {value} has no memory-space configuration")
            }
            Self::ValueTooLarge {
                value,
                bytes,
                required_segment_bytes,
                max_segment_bytes,
            } => write!(
                f,
                "value {value} requires {bytes} bytes ({required_segment_bytes} after segment rounding), exceeding segment maximum {max_segment_bytes}"
            ),
            Self::ReadBeforeDefinition {
                value,
                read,
                definition,
            } => write!(
                f,
                "value {value} is read by instruction {read} before definition at instruction {definition}"
            ),
            Self::UndefinedPlannedValue { value } => {
                write!(f, "planned value {value} has no defining write")
            }
            Self::PackingConflict { first, second } => write!(
                f,
                "live values {first} and {second} overlap in one segment slice"
            ),
            Self::InvalidMemoryPlan(message) => write!(f, "invalid memory plan: {message}"),
        }
    }
}

impl Error for PlannerError {}

fn validate_dense_values<M, V>(values: &[V]) -> Result<(), PlannerError>
where
    V: LoweredValue<M>,
{
    for (index, value) in values.iter().enumerate() {
        let value = value.value_decl();
        let expected = ValueId::from_index(index).ok_or(PlannerError::TooManyValues)?;
        if value.id != expected {
            return Err(PlannerError::ValueIdMismatch {
                index,
                actual: value.id,
            });
        }
    }
    Ok(())
}

/// Validates dense table identities and every lowered value reference.
pub fn validate_lowered_program<K, M, V>(
    schedule: &LoweredProgram<K, M, V>,
) -> Result<(), PlannerError>
where
    V: LoweredValue<M>,
{
    validate_dense_values(&schedule.values)?;

    for value in &schedule.values {
        if let ValueStorage::Alias { source, .. } = &value.value_decl().storage {
            if schedule.values.get(source.index()).is_none() {
                return Err(PlannerError::UnknownValue {
                    instruction: None,
                    value: *source,
                });
            }
        }
    }
    for (index, instruction) in schedule.instructions.iter().enumerate() {
        let expected = InstructionId::from_index(index).ok_or(PlannerError::TooManyInstructions)?;
        if instruction.id != expected {
            return Err(PlannerError::InstructionIdMismatch {
                index,
                actual: instruction.id,
            });
        }
        for value_use in instruction.resource_uses() {
            if schedule.values.get(value_use.value.index()).is_none() {
                return Err(PlannerError::UnknownValue {
                    instruction: Some(instruction.id),
                    value: value_use.value,
                });
            }
        }
    }
    for output in &schedule.outputs {
        if schedule.values.get(output.index()).is_none() {
            return Err(PlannerError::UnknownValue {
                instruction: None,
                value: *output,
            });
        }
    }
    Ok(())
}

impl<K, M, V> LoweredProgram<K, M, V>
where
    V: LoweredValue<M>,
{
    pub fn validate(&self) -> Result<(), PlannerError> {
        validate_lowered_program(self)
    }
}

pub fn normalize_aliases<M, V>(values: &[V]) -> Result<Box<[NormalizedAlias]>, PlannerError>
where
    V: LoweredValue<M>,
{
    validate_dense_values(values)?;

    let mut state = vec![0u8; values.len()];
    let mut normalized: Vec<Option<NormalizedAlias>> = vec![None; values.len()];
    for start in 0..values.len() {
        if state[start] == 2 {
            continue;
        }
        let mut path = Vec::new();
        let mut cursor = start;
        loop {
            if state[cursor] == 2 {
                break;
            }
            if state[cursor] == 1 {
                return Err(PlannerError::AliasCycle {
                    value: values[cursor].value_decl().id,
                });
            }

            state[cursor] = 1;
            path.push(cursor);
            let value = values[cursor].value_decl();
            match &value.storage {
                ValueStorage::Alias {
                    source,
                    byte_offset,
                } => {
                    let source_decl =
                        values
                            .get(source.index())
                            .ok_or(PlannerError::UnknownValue {
                                instruction: None,
                                value: *source,
                            })?;
                    let source_decl = source_decl.value_decl();
                    let end = byte_offset.checked_add(value.bytes).ok_or(
                        PlannerError::ByteSizeOverflow {
                            value: Some(value.id),
                        },
                    )?;
                    if end > source_decl.bytes {
                        return Err(PlannerError::AliasOutOfBounds {
                            value: value.id,
                            source: *source,
                            end,
                            capacity: source_decl.bytes,
                        });
                    }
                    cursor = source.index();
                }
                _ => {
                    normalized[cursor] = Some(NormalizedAlias {
                        root: value.id,
                        byte_offset: 0,
                    });
                    state[cursor] = 2;
                    path.pop();
                    break;
                }
            }
        }

        while let Some(index) = path.pop() {
            let value = values[index].value_decl();
            let ValueStorage::Alias {
                source,
                byte_offset,
            } = &value.storage
            else {
                unreachable!("non-alias terminal was removed from the alias path")
            };
            let parent = normalized[source.index()].expect("alias parent must be normalized");
            let byte_offset = parent.byte_offset.checked_add(*byte_offset).ok_or(
                PlannerError::ByteSizeOverflow {
                    value: Some(value.id),
                },
            )?;
            normalized[index] = Some(NormalizedAlias {
                root: parent.root,
                byte_offset,
            });
            state[index] = 2;
        }
    }

    Ok(normalized
        .into_iter()
        .map(|alias| alias.expect("every value must be normalized"))
        .collect())
}

pub fn analyze_liveness<K, M, V>(
    schedule: &LoweredProgram<K, M, V>,
) -> Result<Liveness, PlannerError>
where
    V: LoweredValue<M>,
{
    validate_lowered_program(schedule)?;
    let aliases = normalize_aliases(&schedule.values)?;
    let mut first_read = vec![None::<usize>; schedule.values.len()];
    let mut first_write = vec![None::<usize>; schedule.values.len()];
    let mut last_access = vec![None::<usize>; schedule.values.len()];

    let mut record = |instruction: InstructionId,
                      value: ValueId,
                      reads: bool,
                      writes: bool|
     -> Result<(), PlannerError> {
        let normalized = aliases
            .get(value.index())
            .ok_or(PlannerError::UnknownValue {
                instruction: Some(instruction),
                value,
            })?;
        let root = normalized.root.index();
        let index = instruction.index();
        if reads {
            first_read[root].get_or_insert(index);
        }
        if writes {
            first_write[root].get_or_insert(index);
        }
        last_access[root] = Some(last_access[root].map_or(index, |last| last.max(index)));
        Ok(())
    };

    for instruction in &schedule.instructions {
        for value_use in instruction.resource_uses() {
            record(
                instruction.id,
                value_use.value,
                value_use.access.reads(),
                value_use.access.writes(),
            )?;
        }
    }

    for output in &schedule.outputs {
        if schedule.values.get(output.index()).is_none() {
            return Err(PlannerError::UnknownValue {
                instruction: None,
                value: *output,
            });
        }
        let root = aliases[output.index()].root.index();
        if let Some(last_instruction) = schedule.instructions.len().checked_sub(1) {
            last_access[root] =
                Some(last_access[root].map_or(last_instruction, |last| last.max(last_instruction)));
        }
    }

    let mut intervals = vec![None; schedule.values.len()];
    for (index, value) in schedule.values.iter().enumerate() {
        let value = value.value_decl();
        if matches!(value.storage, ValueStorage::Alias { .. }) {
            continue;
        }
        let start = match &value.storage {
            ValueStorage::Planned { alignment, .. } => {
                if *alignment == 0 || !alignment.is_power_of_two() {
                    return Err(PlannerError::InvalidValueAlignment {
                        value: value.id,
                        alignment: *alignment,
                    });
                }
                let definition = first_write[index]
                    .ok_or(PlannerError::UndefinedPlannedValue { value: value.id })?;
                if let Some(read) = first_read[index] {
                    if read < definition {
                        return Err(PlannerError::ReadBeforeDefinition {
                            value: value.id,
                            read: InstructionId::from_index(read)
                                .ok_or(PlannerError::TooManyInstructions)?,
                            definition: InstructionId::from_index(definition)
                                .ok_or(PlannerError::TooManyInstructions)?,
                        });
                    }
                }
                Some(definition)
            }
            _ => match (first_read[index], first_write[index]) {
                (Some(read), Some(write)) => Some(read.min(write)),
                (Some(read), None) => Some(read),
                (None, Some(write)) => Some(write),
                (None, None) => None,
            },
        };
        if let Some(start) = start {
            let end = last_access[index]
                .and_then(|last| last.checked_add(1))
                .ok_or(PlannerError::TooManyInstructions)?;
            intervals[index] = Some(LiveInterval {
                value: value.id,
                start: InstructionId::from_index(start).ok_or(PlannerError::TooManyInstructions)?,
                end: InstructionId::from_index(end).ok_or(PlannerError::TooManyInstructions)?,
            });
        }
    }

    Ok(Liveness {
        aliases,
        intervals: intervals.into_boxed_slice(),
    })
}

#[derive(Debug)]
struct ActiveAllocation {
    end: usize,
    offset: usize,
    bytes: usize,
}

#[derive(Debug)]
struct WorkingSegment<M> {
    memory_space: M,
    ownership: SegmentOwnership,
    max_bytes: usize,
    granularity: usize,
    alignment: usize,
    used_end: usize,
    free: Vec<Range<usize>>,
    active: Vec<ActiveAllocation>,
}

impl<M> WorkingSegment<M> {
    fn retire(&mut self, before: usize) {
        let active = std::mem::take(&mut self.active);
        for allocation in active {
            if allocation.end <= before {
                if allocation.bytes != 0 {
                    self.free
                        .push(allocation.offset..allocation.offset + allocation.bytes);
                }
            } else {
                self.active.push(allocation);
            }
        }
        coalesce(&mut self.free);
    }
}

#[derive(Debug, Clone, Copy)]
struct Assignment {
    value: ValueId,
    segment: usize,
    offset: usize,
    bytes: usize,
    interval: LiveInterval,
}

fn round_up(value: usize, multiple: usize, id: Option<ValueId>) -> Result<usize, PlannerError> {
    let remainder = value % multiple;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(multiple - remainder)
            .ok_or(PlannerError::ByteSizeOverflow { value: id })
    }
}

fn align_up(value: usize, alignment: usize, id: ValueId) -> Result<usize, PlannerError> {
    round_up(value, alignment, Some(id))
}

fn coalesce(ranges: &mut Vec<Range<usize>>) {
    ranges.sort_by_key(|range| (range.start, range.end));
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(ranges.len());
    for range in ranges.drain(..) {
        match merged.last_mut() {
            Some(previous) if range.start <= previous.end => {
                previous.end = previous.end.max(range.end);
            }
            _ => merged.push(range),
        }
    }
    *ranges = merged;
}

fn consume_free_range(
    segment: &mut WorkingSegment<impl Sized>,
    range_index: usize,
    offset: usize,
    bytes: usize,
) {
    let range = segment.free.remove(range_index);
    if range.start < offset {
        segment.free.push(range.start..offset);
    }
    let end = offset + bytes;
    if end < range.end {
        segment.free.push(end..range.end);
    }
    coalesce(&mut segment.free);
}

fn checked_add(target: &mut usize, amount: usize) -> Result<(), PlannerError> {
    *target = target
        .checked_add(amount)
        .ok_or(PlannerError::ByteSizeOverflow { value: None })?;
    Ok(())
}

fn add_fixed_bytes(
    report: &mut MemoryReport,
    class: StorageClass,
    bytes: usize,
) -> Result<(), PlannerError> {
    let target = match class {
        StorageClass::ExternalInput => &mut report.external_bytes,
        StorageClass::PersistentConstant => &mut report.persistent_bytes,
        StorageClass::PersistentState => &mut report.state_bytes,
        StorageClass::EscapingOutput => &mut report.output_bytes,
        StorageClass::Workspace | StorageClass::DeviceStatus => &mut report.workspace_bytes,
    };
    checked_add(target, bytes)
}

fn build_report<K, M, V>(
    schedule: &LoweredProgram<K, M, V>,
    liveness: &Liveness,
    segments: &[SegmentDecl<M>],
    largest_allocations: usize,
) -> Result<MemoryReport, PlannerError>
where
    V: LoweredValue<M>,
{
    let mut report = MemoryReport::default();
    let mut largest = Vec::new();
    for value in &schedule.values {
        let value = value.value_decl();
        match &value.storage {
            ValueStorage::Fixed { class, .. } => add_fixed_bytes(&mut report, *class, value.bytes)?,
            ValueStorage::Planned { class, .. } => {
                largest.push((
                    Reverse(value.bytes),
                    value.name.clone(),
                    value.id,
                    AllocationReport {
                        name: value.name.clone(),
                        class: *class,
                        bytes: value.bytes,
                    },
                ));
            }
            ValueStorage::Alias { .. } => continue,
        }
        if let Some(class) = value.storage_class() {
            if matches!(value.storage, ValueStorage::Fixed { .. }) {
                largest.push((
                    Reverse(value.bytes),
                    value.name.clone(),
                    value.id,
                    AllocationReport {
                        name: value.name.clone(),
                        class,
                        bytes: value.bytes,
                    },
                ));
            }
        }
    }

    for segment in segments {
        let target = match segment.ownership {
            SegmentOwnership::Workspace | SegmentOwnership::InvocationStaging => {
                &mut report.workspace_bytes
            }
            SegmentOwnership::ProvisionalOutput => &mut report.output_bytes,
            SegmentOwnership::StateTransaction => &mut report.transaction_bytes,
        };
        checked_add(target, segment.bytes)?;
    }

    let mut starts = vec![0usize; schedule.instructions.len() + 1];
    let mut ends = vec![0usize; schedule.instructions.len() + 1];
    for (value, interval) in schedule.values.iter().zip(liveness.intervals.iter()) {
        let value = value.value_decl();
        if !matches!(value.storage, ValueStorage::Planned { .. }) {
            continue;
        }
        let interval = interval.expect("planned values have checked intervals");
        starts[interval.start.index()] = starts[interval.start.index()]
            .checked_add(value.bytes)
            .ok_or(PlannerError::ByteSizeOverflow { value: None })?;
        ends[interval.end.index()] = ends[interval.end.index()]
            .checked_add(value.bytes)
            .ok_or(PlannerError::ByteSizeOverflow { value: None })?;
    }
    let mut live = 0usize;
    for instruction in 0..schedule.instructions.len() {
        live = live
            .checked_sub(ends[instruction])
            .ok_or(PlannerError::ByteSizeOverflow { value: None })?;
        live = live
            .checked_add(starts[instruction])
            .ok_or(PlannerError::ByteSizeOverflow { value: None })?;
        report.peak_live_bytes = report.peak_live_bytes.max(live);
    }
    let segment_bytes = segments
        .iter()
        .try_fold(0usize, |total, segment| total.checked_add(segment.bytes));
    report.packing_overhead_bytes = segment_bytes
        .ok_or(PlannerError::ByteSizeOverflow { value: None })?
        .checked_sub(report.peak_live_bytes)
        .ok_or(PlannerError::ByteSizeOverflow { value: None })?;

    largest.sort_by(|a, b| (a.0, &a.1, a.2).cmp(&(b.0, &b.1, b.2)));
    largest.truncate(largest_allocations);
    report.largest_allocations = largest
        .into_iter()
        .map(|(_, _, _, allocation)| allocation)
        .collect();
    Ok(report)
}

pub fn plan_memory<K, M, V>(
    schedule: &LoweredProgram<K, M, V>,
    config: &MemoryPlannerConfig<M>,
) -> Result<MemoryPlan<M>, PlannerError>
where
    M: Clone + Eq,
    V: LoweredValue<M>,
{
    let liveness = analyze_liveness(schedule)?;
    plan_memory_with_liveness(schedule, config, &liveness)
}

pub(crate) fn plan_memory_with_liveness<K, M, V>(
    schedule: &LoweredProgram<K, M, V>,
    config: &MemoryPlannerConfig<M>,
    liveness: &Liveness,
) -> Result<MemoryPlan<M>, PlannerError>
where
    M: Clone + Eq,
    V: LoweredValue<M>,
{
    for (index, space) in config.memory_spaces.iter().enumerate() {
        if space.max_segment_bytes == 0
            || space.segment_alignment == 0
            || !space.segment_alignment.is_power_of_two()
            || space.segment_granularity == 0
        {
            return Err(PlannerError::InvalidMemorySpaceConfig { index });
        }
        for previous in 0..index {
            if config.memory_spaces[previous].memory_space == space.memory_space {
                return Err(PlannerError::DuplicateMemorySpaceConfig {
                    first: previous,
                    second: index,
                });
            }
        }
    }

    let mut allocation_order: Vec<usize> = schedule
        .values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            matches!(value.value_decl().storage, ValueStorage::Planned { .. }).then_some(index)
        })
        .collect();
    allocation_order.sort_by_key(|index| {
        let interval = liveness.intervals[*index].expect("planned value interval");
        (interval.start, schedule.values[*index].value_decl().id)
    });

    let mut working: Vec<WorkingSegment<M>> = Vec::new();
    let mut assignments: Vec<Assignment> = Vec::new();
    let mut assigned_by_value: Vec<Option<Assignment>> = vec![None; schedule.values.len()];

    for value_index in allocation_order {
        let value = schedule.values[value_index].value_decl();
        let ValueStorage::Planned {
            alignment,
            memory_space,
            ownership,
            ..
        } = &value.storage
        else {
            unreachable!()
        };
        let interval = liveness.intervals[value_index].expect("planned value interval");
        let space = config
            .memory_spaces
            .iter()
            .find(|space| &space.memory_space == memory_space)
            .ok_or(PlannerError::MissingMemorySpaceConfig { value: value.id })?;
        let rounded = round_up(value.bytes, space.segment_granularity, Some(value.id))?;
        if value.bytes > space.max_segment_bytes || rounded > space.max_segment_bytes {
            return Err(PlannerError::ValueTooLarge {
                value: value.id,
                bytes: value.bytes,
                required_segment_bytes: rounded,
                max_segment_bytes: space.max_segment_bytes,
            });
        }

        for segment in &mut working {
            segment.retire(interval.start.index());
        }

        let mut free_candidate: Option<(usize, usize, usize, usize)> = None;
        for (segment_index, segment) in working.iter().enumerate() {
            if &segment.memory_space != memory_space || segment.ownership != *ownership {
                continue;
            }
            for (range_index, range) in segment.free.iter().enumerate() {
                let offset = align_up(range.start, *alignment, value.id)?;
                let end =
                    offset
                        .checked_add(value.bytes)
                        .ok_or(PlannerError::ByteSizeOverflow {
                            value: Some(value.id),
                        })?;
                if end <= range.end {
                    let waste = (range.end - range.start) - value.bytes;
                    let candidate = (waste, segment_index, offset, range_index);
                    if free_candidate.map_or(true, |current| candidate < current) {
                        free_candidate = Some(candidate);
                    }
                }
            }
        }

        let (segment_index, offset) =
            if let Some((_, segment_index, offset, range_index)) = free_candidate {
                consume_free_range(
                    &mut working[segment_index],
                    range_index,
                    offset,
                    value.bytes,
                );
                (segment_index, offset)
            } else {
                let mut tail_candidate: Option<(usize, usize, usize)> = None;
                for (segment_index, segment) in working.iter().enumerate() {
                    if &segment.memory_space != memory_space || segment.ownership != *ownership {
                        continue;
                    }
                    let offset = align_up(segment.used_end, *alignment, value.id)?;
                    let end =
                        offset
                            .checked_add(value.bytes)
                            .ok_or(PlannerError::ByteSizeOverflow {
                                value: Some(value.id),
                            })?;
                    let capacity = round_up(end, segment.granularity, Some(value.id))?;
                    if capacity <= segment.max_bytes {
                        let growth = capacity.saturating_sub(round_up(
                            segment.used_end,
                            segment.granularity,
                            Some(value.id),
                        )?);
                        let candidate = (growth, segment_index, offset);
                        if tail_candidate.map_or(true, |current| candidate < current) {
                            tail_candidate = Some(candidate);
                        }
                    }
                }

                if let Some((_, segment_index, offset)) = tail_candidate {
                    let segment = &mut working[segment_index];
                    if segment.used_end < offset {
                        segment.free.push(segment.used_end..offset);
                        coalesce(&mut segment.free);
                    }
                    segment.used_end =
                        offset
                            .checked_add(value.bytes)
                            .ok_or(PlannerError::ByteSizeOverflow {
                                value: Some(value.id),
                            })?;
                    (segment_index, offset)
                } else {
                    let segment_index = working.len();
                    SegmentId::from_index(segment_index).ok_or(PlannerError::TooManySegments)?;
                    working.push(WorkingSegment {
                        memory_space: memory_space.clone(),
                        ownership: *ownership,
                        max_bytes: space.max_segment_bytes,
                        granularity: space.segment_granularity,
                        alignment: space.segment_alignment.max(*alignment),
                        used_end: value.bytes,
                        free: Vec::new(),
                        active: Vec::new(),
                    });
                    (segment_index, 0)
                }
            };

        let segment = &mut working[segment_index];
        segment.alignment = segment.alignment.max(*alignment);
        segment.active.push(ActiveAllocation {
            end: interval.end.index(),
            offset,
            bytes: value.bytes,
        });
        let assignment = Assignment {
            value: value.id,
            segment: segment_index,
            offset,
            bytes: value.bytes,
            interval,
        };
        assignments.push(assignment);
        assigned_by_value[value_index] = Some(assignment);
    }

    let mut segments = Vec::with_capacity(working.len());
    for segment in &working {
        let bytes = round_up(segment.used_end, segment.granularity, None)?;
        debug_assert!(bytes <= segment.max_bytes);
        segments.push(SegmentDecl {
            bytes,
            alignment: segment.alignment,
            memory_space: segment.memory_space.clone(),
            ownership: segment.ownership,
        });
    }

    let mut reuse_edges = Vec::new();
    for (next_index, next) in assignments.iter().enumerate() {
        for previous in &assignments[..next_index] {
            if previous.segment != next.segment {
                continue;
            }
            let previous_end = previous.offset.checked_add(previous.bytes).ok_or(
                PlannerError::ByteSizeOverflow {
                    value: Some(previous.value),
                },
            )?;
            let next_end =
                next.offset
                    .checked_add(next.bytes)
                    .ok_or(PlannerError::ByteSizeOverflow {
                        value: Some(next.value),
                    })?;
            let address_overlap = previous.offset < next_end && next.offset < previous_end;
            if !address_overlap {
                continue;
            }
            if previous.interval.overlaps(next.interval) {
                return Err(PlannerError::PackingConflict {
                    first: previous.value,
                    second: next.value,
                });
            }
            if previous.interval.end.get() <= next.interval.start.get() {
                reuse_edges.push(ReuseEdge {
                    previous_value: previous.value,
                    next_value: next.value,
                    after: InstructionId::new(previous.interval.end.get() - 1),
                    before: next.interval.start,
                });
            }
        }
    }
    reuse_edges.sort_by_key(|edge| {
        (
            edge.before,
            edge.after,
            edge.next_value,
            edge.previous_value,
        )
    });
    reuse_edges.dedup();

    let mut locations = Vec::with_capacity(schedule.values.len());
    for (index, value) in schedule.values.iter().enumerate() {
        let value = value.value_decl();
        LocationId::from_index(index).ok_or(PlannerError::TooManyLocations)?;
        let location = match &value.storage {
            ValueStorage::Fixed { location, .. } => location.clone(),
            ValueStorage::Planned { .. } => {
                let assignment = assigned_by_value[index].expect("planned value assignment");
                Location::Segment {
                    segment: SegmentId::from_index(assignment.segment)
                        .ok_or(PlannerError::TooManySegments)?,
                    offset: assignment.offset,
                    bytes: assignment.bytes,
                }
            }
            ValueStorage::Alias { .. } => Location::Alias {
                root: liveness.aliases[index].root,
                byte_offset: liveness.aliases[index].byte_offset,
            },
        };
        locations.push(location);
    }

    let mut outputs = Vec::with_capacity(schedule.outputs.len());
    for value in &schedule.outputs {
        if schedule.values.get(value.index()).is_none() {
            return Err(PlannerError::UnknownValue {
                instruction: None,
                value: *value,
            });
        }
        outputs.push(OutputSlot {
            value: *value,
            location: LocationId::from_index(value.index())
                .ok_or(PlannerError::TooManyLocations)?,
        });
    }

    let report = build_report(schedule, &liveness, &segments, config.largest_allocations)?;
    let plan = MemoryPlan {
        segments: segments.into_boxed_slice(),
        locations: locations.into_boxed_slice(),
        outputs: outputs.into_boxed_slice(),
        reuse_edges: reuse_edges.into_boxed_slice(),
        report,
    };
    plan.validate()
        .map_err(|error| PlannerError::InvalidMemoryPlan(error.to_string()))?;
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LoweredInstruction, OutputDecl, ValueDecl, ValueUse};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum Space {
        A,
        B,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct BackendValue {
        declaration: ValueDecl<Space>,
        _backend_tag: &'static str,
    }

    impl LoweredValue<Space> for BackendValue {
        fn value_decl(&self) -> &ValueDecl<Space> {
            &self.declaration
        }
    }

    fn value(
        id: u32,
        bytes: usize,
        alignment: usize,
        space: Space,
        ownership: SegmentOwnership,
    ) -> ValueDecl<Space> {
        ValueDecl::planned(
            ValueId::new(id),
            format!("v{id}"),
            bytes,
            alignment,
            space,
            ownership,
        )
    }

    fn instruction(
        id: u32,
        inputs: Vec<ValueUse>,
        outputs: &[u32],
    ) -> LoweredInstruction<&'static str> {
        LoweredInstruction::new(
            InstructionId::new(id),
            "fixture",
            inputs,
            outputs
                .iter()
                .copied()
                .map(|value| OutputDecl::new(ValueId::new(value)))
                .collect::<Vec<_>>(),
        )
    }

    fn schedule(
        values: Vec<ValueDecl<Space>>,
        instructions: Vec<LoweredInstruction<&'static str>>,
    ) -> LoweredProgram<&'static str, Space> {
        LoweredProgram::new(values, instructions, Vec::new())
    }

    fn config(max_segment_bytes: usize) -> MemoryPlannerConfig<Space> {
        MemoryPlannerConfig::new(vec![
            MemorySpaceConfig {
                memory_space: Space::A,
                max_segment_bytes,
                segment_alignment: 16,
                segment_granularity: 16,
            },
            MemorySpaceConfig {
                memory_space: Space::B,
                max_segment_bytes,
                segment_alignment: 16,
                segment_granularity: 16,
            },
        ])
    }

    fn segment_location(plan: &MemoryPlan<Space>, value: usize) -> (SegmentId, usize, usize) {
        match plan.locations[value] {
            Location::Segment {
                segment,
                offset,
                bytes,
            } => (segment, offset, bytes),
            ref location => panic!("expected segment location, got {location:?}"),
        }
    }

    #[test]
    fn backend_values_are_authoritative_for_every_resource_and_storage_class() {
        let fixed = |id: u32, name: &str, bytes: usize, class: StorageClass| BackendValue {
            declaration: ValueDecl {
                id: ValueId::new(id),
                name: name.into(),
                bytes,
                storage: ValueStorage::Fixed {
                    class,
                    location: Location::Persistent { slot: id },
                },
            },
            _backend_tag: "backend-only metadata",
        };
        let values = vec![
            fixed(0, "input", 1, StorageClass::ExternalInput),
            BackendValue {
                declaration: value(1, 17, 16, Space::A, SegmentOwnership::Workspace),
                _backend_tag: "planned backend value",
            },
            fixed(2, "constant", 2, StorageClass::PersistentConstant),
            fixed(3, "state", 3, StorageClass::PersistentState),
            fixed(4, "output", 4, StorageClass::EscapingOutput),
            fixed(5, "status", 5, StorageClass::DeviceStatus),
        ];
        let instructions = vec![LoweredInstruction::new(
            InstructionId::new(0),
            "backend",
            vec![ValueUse::read(ValueId::new(0))],
            vec![OutputDecl::new(ValueId::new(1))],
        )
        .with_resources(
            vec![ValueUse::read(ValueId::new(2))],
            vec![ValueUse::read(ValueId::new(4))],
            vec![ValueUse::read(ValueId::new(5))],
            vec![ValueUse::read(ValueId::new(3))],
        )];
        let schedule = LoweredProgram::new(values, instructions, vec![ValueId::new(1)]);

        schedule.validate().unwrap();
        let liveness = analyze_liveness(&schedule).unwrap();
        assert!(liveness.intervals.iter().all(Option::is_some));

        let plan = plan_memory(&schedule, &config(128)).unwrap();
        assert_eq!(segment_location(&plan, 1).2, 17);
        assert_eq!(plan.report.external_bytes, 1);
        assert_eq!(plan.report.persistent_bytes, 2);
        assert_eq!(plan.report.state_bytes, 3);
        assert_eq!(plan.report.output_bytes, 4);
        assert_eq!(plan.report.workspace_bytes, 37);
    }

    #[test]
    fn every_resource_category_extends_liveness() {
        let resource_instruction = instruction(2, vec![], &[]).with_resources(
            vec![ValueUse::read(ValueId::new(0))],
            vec![ValueUse::read(ValueId::new(1))],
            vec![ValueUse::read(ValueId::new(2))],
            vec![ValueUse::read(ValueId::new(3))],
        );
        let schedule = schedule(
            (0..4)
                .map(|id| value(id, 16, 16, Space::A, SegmentOwnership::Workspace))
                .collect(),
            vec![
                instruction(0, vec![], &[0, 1, 2, 3]),
                instruction(1, vec![], &[]),
                resource_instruction,
            ],
        );

        let liveness = analyze_liveness(&schedule).unwrap();
        for interval in &liveness.intervals {
            assert_eq!(interval.unwrap().end, InstructionId::new(3));
        }
    }

    #[test]
    fn output_and_resource_writes_do_not_change_interval_semantics() {
        let instruction = instruction(0, vec![], &[0]).with_resources(
            vec![ValueUse::write(ValueId::new(0))],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        let schedule = schedule(
            vec![value(0, 16, 16, Space::A, SegmentOwnership::Workspace)],
            vec![instruction],
        );

        assert_eq!(
            analyze_liveness(&schedule).unwrap().intervals[0],
            Some(LiveInterval {
                value: ValueId::new(0),
                start: InstructionId::new(0),
                end: InstructionId::new(1),
            })
        );
    }

    #[test]
    fn resource_references_are_validated() {
        let instruction = instruction(0, vec![], &[]).with_resources(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![ValueUse::read(ValueId::new(7))],
        );
        let schedule = schedule(Vec::new(), vec![instruction]);

        assert_eq!(
            analyze_liveness(&schedule),
            Err(PlannerError::UnknownValue {
                instruction: Some(InstructionId::new(0)),
                value: ValueId::new(7),
            })
        );
    }

    #[test]
    fn overlapping_intervals_never_overlap_in_address_space() {
        let schedule = schedule(
            vec![
                value(0, 32, 16, Space::A, SegmentOwnership::Workspace),
                value(1, 32, 16, Space::A, SegmentOwnership::Workspace),
            ],
            vec![
                instruction(0, vec![], &[0]),
                instruction(1, vec![], &[1]),
                instruction(2, vec![ValueUse::read(ValueId::new(0))], &[]),
            ],
        );
        let plan = plan_memory(&schedule, &config(128)).unwrap();
        let (s0, o0, b0) = segment_location(&plan, 0);
        let (s1, o1, b1) = segment_location(&plan, 1);
        assert_eq!(s0, s1);
        assert!(o0 + b0 <= o1 || o1 + b1 <= o0);
        assert_eq!(plan.report.peak_live_bytes, 64);
    }

    #[test]
    fn disjoint_intervals_reuse_deterministically_and_emit_an_edge() {
        let schedule = schedule(
            vec![
                value(0, 32, 16, Space::A, SegmentOwnership::Workspace),
                value(1, 32, 16, Space::A, SegmentOwnership::Workspace),
            ],
            vec![
                instruction(0, vec![], &[0]),
                instruction(1, vec![ValueUse::read(ValueId::new(0))], &[]),
                instruction(2, vec![], &[1]),
            ],
        );
        let first = plan_memory(&schedule, &config(128)).unwrap();
        let second = plan_memory(&schedule, &config(128)).unwrap();
        assert_eq!(first, second);
        assert_eq!(segment_location(&first, 0), segment_location(&first, 1));
        assert_eq!(
            first.reuse_edges.as_ref(),
            [ReuseEdge {
                previous_value: ValueId::new(0),
                next_value: ValueId::new(1),
                after: InstructionId::new(1),
                before: InstructionId::new(2),
            }]
        );
    }

    #[test]
    fn escaping_outputs_remain_live_through_invocation_completion() {
        let schedule = LoweredProgram::new(
            vec![
                value(0, 32, 16, Space::A, SegmentOwnership::ProvisionalOutput),
                value(1, 32, 16, Space::A, SegmentOwnership::ProvisionalOutput),
            ],
            vec![instruction(0, vec![], &[0]), instruction(1, vec![], &[1])],
            vec![ValueId::new(0), ValueId::new(1)],
        );
        let liveness = analyze_liveness(&schedule).unwrap();
        assert_eq!(liveness.intervals[0].unwrap().end, InstructionId::new(2));
        assert_eq!(liveness.intervals[1].unwrap().end, InstructionId::new(2));

        let plan = plan_memory(&schedule, &config(128)).unwrap();
        let (first_segment, first_offset, first_bytes) = segment_location(&plan, 0);
        let (second_segment, second_offset, second_bytes) = segment_location(&plan, 1);
        assert_eq!(first_segment, second_segment);
        assert!(
            first_offset + first_bytes <= second_offset
                || second_offset + second_bytes <= first_offset
        );
    }

    #[test]
    fn last_read_and_next_write_in_one_instruction_cannot_reuse() {
        let schedule = schedule(
            vec![
                value(0, 32, 16, Space::A, SegmentOwnership::Workspace),
                value(1, 32, 16, Space::A, SegmentOwnership::Workspace),
            ],
            vec![
                instruction(0, vec![], &[0]),
                instruction(1, vec![ValueUse::read(ValueId::new(0))], &[1]),
            ],
        );
        let liveness = analyze_liveness(&schedule).unwrap();
        assert_eq!(
            liveness.intervals[0],
            Some(LiveInterval {
                value: ValueId::new(0),
                start: InstructionId::new(0),
                end: InstructionId::new(2),
            })
        );
        let plan = plan_memory(&schedule, &config(128)).unwrap();
        let (_, first, _) = segment_location(&plan, 0);
        let (_, second, _) = segment_location(&plan, 1);
        assert_ne!(first, second);
        assert!(plan.reuse_edges.is_empty());
    }

    #[test]
    fn aliases_normalize_and_extend_the_root_lifetime() {
        let schedule = schedule(
            vec![
                value(0, 64, 16, Space::A, SegmentOwnership::Workspace),
                ValueDecl::alias(ValueId::new(1), "view", ValueId::new(0), 8, 32),
                ValueDecl::alias(ValueId::new(2), "nested", ValueId::new(1), 4, 16),
                value(3, 64, 16, Space::A, SegmentOwnership::Workspace),
            ],
            vec![
                instruction(0, vec![], &[0]),
                instruction(1, vec![], &[3]),
                instruction(2, vec![ValueUse::read(ValueId::new(2))], &[]),
            ],
        );
        let liveness = analyze_liveness(&schedule).unwrap();
        assert_eq!(
            liveness.aliases[2],
            NormalizedAlias {
                root: ValueId::new(0),
                byte_offset: 12,
            }
        );
        assert_eq!(liveness.intervals[0].unwrap().end, InstructionId::new(3));
        let plan = plan_memory(&schedule, &config(128)).unwrap();
        assert_eq!(
            plan.locations[2],
            Location::Alias {
                root: ValueId::new(0),
                byte_offset: 12,
            }
        );
        let (_, root_offset, _) = segment_location(&plan, 0);
        let (_, competing_offset, _) = segment_location(&plan, 3);
        assert_ne!(root_offset, competing_offset);
    }

    #[test]
    fn alias_cycles_are_rejected() {
        let schedule = schedule(
            vec![
                ValueDecl::alias(ValueId::new(0), "a", ValueId::new(1), 0, 16),
                ValueDecl::alias(ValueId::new(1), "b", ValueId::new(0), 0, 16),
            ],
            vec![],
        );
        assert!(matches!(
            analyze_liveness(&schedule),
            Err(PlannerError::AliasCycle { .. })
        ));
    }

    #[test]
    fn out_of_bounds_aliases_are_rejected() {
        let schedule = schedule(
            vec![
                value(0, 16, 16, Space::A, SegmentOwnership::Workspace),
                ValueDecl::alias(ValueId::new(1), "bad", ValueId::new(0), 8, 12),
            ],
            vec![],
        );
        assert_eq!(
            analyze_liveness(&schedule),
            Err(PlannerError::AliasOutOfBounds {
                value: ValueId::new(1),
                source: ValueId::new(0),
                end: 20,
                capacity: 16,
            })
        );
    }

    #[test]
    fn segment_maximum_splits_overlapping_values_and_rejects_large_values() {
        let split = schedule(
            vec![
                value(0, 48, 16, Space::A, SegmentOwnership::Workspace),
                value(1, 48, 16, Space::A, SegmentOwnership::Workspace),
            ],
            vec![
                instruction(0, vec![], &[0]),
                instruction(1, vec![ValueUse::read(ValueId::new(0))], &[1]),
            ],
        );
        let plan = plan_memory(&split, &config(64)).unwrap();
        assert_eq!(plan.segments.len(), 2);
        assert!(plan.segments.iter().all(|segment| segment.bytes == 48));

        let too_large = schedule(
            vec![value(0, 65, 16, Space::A, SegmentOwnership::Workspace)],
            vec![instruction(0, vec![], &[0])],
        );
        assert_eq!(
            plan_memory(&too_large, &config(64)),
            Err(PlannerError::ValueTooLarge {
                value: ValueId::new(0),
                bytes: 65,
                required_segment_bytes: 80,
                max_segment_bytes: 64,
            })
        );
    }

    #[test]
    fn segments_are_right_sized_and_honor_value_alignment() {
        let schedule = schedule(
            vec![
                value(0, 17, 16, Space::A, SegmentOwnership::Workspace),
                value(1, 9, 32, Space::A, SegmentOwnership::Workspace),
            ],
            vec![
                instruction(0, vec![], &[0]),
                instruction(1, vec![ValueUse::read(ValueId::new(0))], &[1]),
            ],
        );
        let plan = plan_memory(&schedule, &config(128)).unwrap();
        assert_eq!(plan.segments.len(), 1);
        assert_eq!(plan.segments[0].bytes, 48);
        assert_eq!(plan.segments[0].alignment, 32);
        assert_eq!(segment_location(&plan, 0).1, 0);
        assert_eq!(segment_location(&plan, 1).1, 32);
    }

    #[test]
    fn memory_space_and_ownership_never_share_segments() {
        let schedule = schedule(
            vec![
                value(0, 16, 16, Space::A, SegmentOwnership::Workspace),
                value(1, 16, 16, Space::A, SegmentOwnership::ProvisionalOutput),
                value(2, 16, 16, Space::B, SegmentOwnership::Workspace),
            ],
            vec![
                instruction(0, vec![], &[0]),
                instruction(1, vec![], &[1]),
                instruction(2, vec![], &[2]),
            ],
        );
        let plan = plan_memory(&schedule, &config(128)).unwrap();
        assert_eq!(plan.segments.len(), 3);
        assert_ne!(segment_location(&plan, 0).0, segment_location(&plan, 1).0);
        assert_ne!(segment_location(&plan, 0).0, segment_location(&plan, 2).0);
        assert_eq!(plan.report.workspace_bytes, 32);
        assert_eq!(plan.report.output_bytes, 16);
    }
}
