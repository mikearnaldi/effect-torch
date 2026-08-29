//! Tensor shape/stride/offset geometry.
//!
//! A [`Layout`] maps logical coordinates to element offsets in a backing
//! buffer. The offset of coordinate `c` is
//! `offset + Σ c[d] * strides[d]`, expressed in *elements*, not bytes.
//!
//! # Invariants
//!
//! - `shape.len() == strides.len()`. [`Layout::new`] panics on a mismatch
//!   because the compiler constructs layouts and a mismatched rank is a bug.
//! - Zero-sized dimensions are allowed. [`Layout::checked_max_index`] returns
//!   0 for a layout containing a zero dimension, regardless of its offset or
//!   strides. An empty tensor therefore has no readable storage.
//! - Checked size methods (`checked_numel`, `checked_max_index`, and
//!   `checked_byte_size`) return `None` on overflow. Their panicking
//!   counterparts (`numel`, `max_index`, and `byte_size`) panic.
//!
//! Views (`permute`, `narrow`, and `broadcast_to`) share the parent's offset
//! and strides. `broadcast_to` uses stride 0 for broadcast dimensions.

use crate::DType;

/// Shape, strides (in elements) and element offset of a tensor view.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Layout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl Layout {
    /// Creates a layout from explicit parts.
    ///
    /// # Panics
    ///
    /// Panics if `shape.len() != strides.len()`.
    pub fn new(shape: Vec<usize>, strides: Vec<usize>, offset: usize) -> Self {
        assert_eq!(shape.len(), strides.len());
        Layout {
            shape,
            strides,
            offset,
        }
    }

    /// Creates a contiguous (row-major, zero-offset) layout for `shape`.
    ///
    /// # Panics
    ///
    /// Panics if the stride computation overflows `usize`.
    pub fn contiguous(shape: Vec<usize>) -> Self {
        let mut strides = vec![0usize; shape.len()];
        let mut acc = 1usize;
        for d in (0..shape.len()).rev() {
            strides[d] = acc;
            if d != 0 {
                acc = acc.checked_mul(shape[d]).expect("layout stride overflow");
            }
        }
        Layout {
            shape,
            strides,
            offset: 0,
        }
    }

    /// Size of each dimension, outermost first.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Element stride of each dimension, matching [`shape`](Self::shape).
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Element offset of coordinate zero within the backing buffer.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Number of dimensions. Scalars have rank 0.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Total element count.
    ///
    /// # Panics
    ///
    /// Panics on `usize` overflow. Use
    /// [`checked_numel`](Self::checked_numel) for untrusted shapes.
    pub fn numel(&self) -> usize {
        self.checked_numel().expect("layout element count overflow")
    }

    /// Returns the total element count, or `None` on overflow. Empty layouts
    /// have 0 elements. Scalars have 1.
    pub fn checked_numel(&self) -> Option<usize> {
        self.shape
            .iter()
            .try_fold(1usize, |count, &dim| count.checked_mul(dim))
    }

    /// Returns `true` for a densely packed row-major layout. Strides do not
    /// matter for dimensions of size 1. Returns `false` if the running stride
    /// product overflows.
    pub fn is_contiguous(&self) -> bool {
        let mut acc = 1usize;
        for d in (0..self.shape.len()).rev() {
            if self.shape[d] != 1 && self.strides[d] != acc {
                return false;
            }
            if d != 0 {
                let Some(next) = acc.checked_mul(self.shape[d]) else {
                    return false;
                };
                acc = next;
            }
        }
        true
    }

    /// Returns a view with dimensions reordered by `axes`. `axes[i]` is the
    /// source dimension of output dimension `i`. The view permutes shape and
    /// strides but keeps the offset.
    ///
    /// # Panics
    ///
    /// Panics if `axes.len() != self.rank()` or any axis is out of range.
    pub fn permute(&self, axes: &[usize]) -> Self {
        assert_eq!(axes.len(), self.rank());
        Layout {
            shape: axes.iter().map(|&a| self.shape[a]).collect(),
            strides: axes.iter().map(|&a| self.strides[a]).collect(),
            offset: self.offset,
        }
    }

    /// Returns a view broadcast to a `shape` of equal or higher rank.
    /// Expanded dimensions of size 1 and new leading dimensions get stride 0.
    ///
    /// # Panics
    ///
    /// Panics if `shape` has lower rank, or if a non-1 source dimension
    /// disagrees with the target.
    pub fn broadcast_to(&self, shape: &[usize]) -> Self {
        assert!(shape.len() >= self.rank());
        let extra = shape.len() - self.rank();
        let mut strides = vec![0usize; shape.len()];
        for d in 0..self.rank() {
            let src = self.shape[d];
            let dst = shape[extra + d];
            assert!(src == dst || src == 1, "cannot broadcast {src} to {dst}");
            strides[extra + d] = if src == 1 && dst != 1 {
                0
            } else {
                self.strides[d]
            };
        }
        Layout {
            shape: shape.to_vec(),
            strides,
            offset: self.offset,
        }
    }

    /// Returns a view restricted to `start..start + len` along `dim`. The
    /// method advances the offset by `start * strides[dim]` and preserves the
    /// strides.
    ///
    /// # Panics
    ///
    /// Panics if `dim` is out of range or `start + len > shape[dim]`.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Self {
        assert!(start + len <= self.shape[dim]);
        let mut shape = self.shape.clone();
        shape[dim] = len;
        Layout {
            shape,
            strides: self.strides.clone(),
            offset: self.offset + start * self.strides[dim],
        }
    }

    /// Number of elements a backing buffer must have for this layout,
    /// `offset + Σ (shape[d] - 1) * strides[d] + 1`.
    ///
    /// # Panics
    ///
    /// Panics on `usize` overflow. Use
    /// [`checked_max_index`](Self::checked_max_index) for untrusted layouts.
    pub fn max_index(&self) -> usize {
        self.checked_max_index()
            .expect("layout maximum index overflow")
    }

    /// Checked form of [`max_index`](Self::max_index). Returns `Some(0)`
    /// for a layout with any zero-sized dimension because it cannot address
    /// storage. Returns `None` on arithmetic overflow.
    pub fn checked_max_index(&self) -> Option<usize> {
        if self.shape.contains(&0) {
            return Some(0);
        }
        let mut index = self.offset;
        for (&dim, &stride) in self.shape.iter().zip(&self.strides) {
            index = index.checked_add((dim - 1).checked_mul(stride)?)?;
        }
        index.checked_add(1)
    }

    /// Checked byte size of the smallest buffer holding this layout at the
    /// given dtype (`checked_max_index * dtype.size_in_bytes()`).
    pub fn checked_byte_size(&self, dtype: DType) -> Option<usize> {
        self.checked_max_index()?.checked_mul(dtype.size_in_bytes())
    }

    /// Byte size of the smallest buffer holding this layout at `dtype`.
    ///
    /// # Panics
    ///
    /// Panics on `usize` overflow. Use
    /// [`checked_byte_size`](Self::checked_byte_size) for untrusted layouts.
    pub fn byte_size(&self, dtype: DType) -> usize {
        self.checked_byte_size(dtype)
            .expect("layout byte size overflow")
    }
}

/// Broadcasts two shapes using NumPy rules. Dimensions align from the right,
/// must be equal or 1, and missing leading dimensions count as 1.
///
/// # Panics
///
/// Panics if any pair of dimensions is incompatible.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let rank = a.len().max(b.len());
    let mut out = vec![1usize; rank];
    for d in 0..rank {
        let ad = if d < rank - a.len() {
            1
        } else {
            a[d - (rank - a.len())]
        };
        let bd = if d < rank - b.len() {
            1
        } else {
            b[d - (rank - b.len())]
        };
        assert!(
            ad == bd || ad == 1 || bd == 1,
            "shape mismatch: {a:?} vs {b:?}"
        );
        out[d] = ad.max(bd);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_strides() {
        let l = Layout::contiguous(vec![2, 3, 4]);
        assert_eq!(l.strides(), &[12, 4, 1]);
        assert!(l.is_contiguous());
        assert_eq!(l.numel(), 24);
    }

    #[test]
    fn permute_is_not_contiguous() {
        let l = Layout::contiguous(vec![2, 3]).permute(&[1, 0]);
        assert_eq!(l.shape(), &[3, 2]);
        assert_eq!(l.strides(), &[1, 3]);
        assert!(!l.is_contiguous());
        assert_eq!(l.max_index(), 6);
    }

    #[test]
    fn broadcast_rules() {
        let l = Layout::contiguous(vec![3, 1]).broadcast_to(&[2, 3, 4]);
        assert_eq!(l.strides(), &[0, 1, 0]);
        assert_eq!(broadcast_shape(&[2, 1, 3], &[4, 3]), vec![2, 4, 3]);
    }

    #[test]
    fn narrow_offsets() {
        let l = Layout::contiguous(vec![4, 5]).narrow(1, 2, 2);
        assert_eq!(l.shape(), &[4, 2]);
        assert_eq!(l.offset(), 2);
        assert!(!l.is_contiguous());
        assert_eq!(l.max_index(), 19);
    }

    #[test]
    fn checked_layout_sizes_detect_overflow() {
        let numel_overflow = Layout::new(vec![usize::MAX, 2], vec![2, 1], 0);
        assert_eq!(numel_overflow.checked_numel(), None);
        assert_eq!(numel_overflow.checked_max_index(), None);

        let index_overflow = Layout::new(vec![2], vec![usize::MAX], 0);
        assert_eq!(index_overflow.checked_max_index(), None);

        let byte_overflow = Layout::new(vec![usize::MAX / 2], vec![1], 0);
        assert_eq!(byte_overflow.checked_byte_size(DType::F32), None);
    }

    #[test]
    fn checked_layout_sizes_handle_empty_and_strided_layouts() {
        let empty = Layout::new(vec![3, 0], vec![8, usize::MAX], usize::MAX);
        assert_eq!(empty.checked_numel(), Some(0));
        assert_eq!(empty.checked_max_index(), Some(0));
        assert_eq!(empty.checked_byte_size(DType::F32), Some(0));

        let strided = Layout::new(vec![2, 3], vec![8, 2], 1);
        assert_eq!(strided.checked_numel(), Some(6));
        assert_eq!(strided.checked_max_index(), Some(14));
        assert_eq!(strided.checked_byte_size(DType::F32), Some(56));
        assert_eq!(strided.byte_size(DType::F32), 56);
    }
}
