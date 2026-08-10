use std::fmt;
use std::marker::PhantomData;
use std::ops::Range;

pub trait DenseId: Copy + Eq {
    fn from_u32(value: u32) -> Self;
    fn as_u32(self) -> u32;

    fn from_index(index: usize) -> Option<Self> {
        u32::try_from(index).ok().map(Self::from_u32)
    }

    fn index(self) -> usize {
        self.as_u32() as usize
    }
}

macro_rules! dense_ids {
    ($($name:ident),+ $(,)?) => {
        $(
            #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
            pub struct $name(u32);

            impl $name {
                pub const fn new(value: u32) -> Self {
                    Self(value)
                }

                pub const fn get(self) -> u32 {
                    self.0
                }

                pub const fn index(self) -> usize {
                    self.0 as usize
                }

                pub fn from_index(index: usize) -> Option<Self> {
                    u32::try_from(index).ok().map(Self)
                }
            }

            impl DenseId for $name {
                fn from_u32(value: u32) -> Self {
                    Self(value)
                }

                fn as_u32(self) -> u32 {
                    self.0
                }
            }

            impl fmt::Display for $name {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    self.0.fmt(f)
                }
            }
        )+
    };
}

dense_ids!(
    BindingId,
    ScalarId,
    RuntimeValueId,
    ValueId,
    InstructionId,
    PipelineId,
    CommandId,
    SegmentId,
    LocationId,
    OutputId,
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TableRange<I> {
    start: u32,
    len: u32,
    marker: PhantomData<fn() -> I>,
}

impl<I: DenseId> TableRange<I> {
    pub fn new(start: I, len: u32) -> Self {
        Self {
            start: start.as_u32(),
            len,
            marker: PhantomData,
        }
    }

    pub fn from_indexes(start: usize, len: usize) -> Option<Self> {
        let start = u32::try_from(start).ok()?;
        let len = u32::try_from(len).ok()?;
        start.checked_add(len)?;
        Some(Self {
            start,
            len,
            marker: PhantomData,
        })
    }

    pub fn start(self) -> I {
        I::from_u32(self.start)
    }

    pub const fn len(self) -> usize {
        self.len as usize
    }

    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    pub fn end(self) -> Option<I> {
        self.start.checked_add(self.len).map(I::from_u32)
    }

    pub fn as_range(self) -> Option<Range<usize>> {
        Some(self.start as usize..self.end()?.index())
    }

    pub fn get<'a, T>(self, table: &'a [T]) -> Option<&'a [T]> {
        table.get(self.as_range()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_ranges_are_checked() {
        let range = TableRange::new(CommandId::new(1), 2);
        assert_eq!(range.as_range(), Some(1..3));
        assert_eq!(range.get(&[0, 1, 2, 3]), Some(&[1, 2][..]));
        assert_eq!(TableRange::<CommandId>::from_indexes(0, usize::MAX), None);

        let overflow = TableRange::new(CommandId::new(u32::MAX), 1);
        assert_eq!(overflow.end(), None);
        assert_eq!(overflow.get(&[0]), None);
    }
}
