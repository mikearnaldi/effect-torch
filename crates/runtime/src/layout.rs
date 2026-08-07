#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl Layout {
    pub fn new(shape: Vec<usize>, strides: Vec<usize>, offset: usize) -> Self {
        assert_eq!(shape.len(), strides.len());
        Layout {
            shape,
            strides,
            offset,
        }
    }

    pub fn contiguous(shape: Vec<usize>) -> Self {
        let mut strides = vec![0usize; shape.len()];
        let mut acc = 1usize;
        for d in (0..shape.len()).rev() {
            strides[d] = acc;
            acc *= shape[d];
        }
        Layout {
            shape,
            strides,
            offset: 0,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_contiguous(&self) -> bool {
        let mut acc = 1usize;
        for d in (0..self.shape.len()).rev() {
            if self.shape[d] != 1 && self.strides[d] != acc {
                return false;
            }
            acc *= self.shape[d];
        }
        true
    }

    pub fn permute(&self, axes: &[usize]) -> Self {
        assert_eq!(axes.len(), self.rank());
        Layout {
            shape: axes.iter().map(|&a| self.shape[a]).collect(),
            strides: axes.iter().map(|&a| self.strides[a]).collect(),
            offset: self.offset,
        }
    }

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

    pub fn max_index(&self) -> usize {
        if self.numel() == 0 {
            return 0;
        }
        let mut idx = self.offset;
        for d in 0..self.rank() {
            if self.shape[d] > 0 {
                idx += (self.shape[d] - 1) * self.strides[d];
            }
        }
        idx + 1
    }
}

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
}
