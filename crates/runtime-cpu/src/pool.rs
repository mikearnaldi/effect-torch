//! Dtype-generic KV-cache slabs used by stateful CPU decoding.
//!
//! A [`Slab`] is a two-dimensional (`rows` × `row_width`) buffer of one of the
//! supported KV dtypes (`f32`, `f16`, `bf16`, or quantized `u8`). Rows are
//! addressed by token position and read/written in batches behind an
//! [`RwLock`], so decode sequences can share a slab across the NAPI worker
//! pool without holding references into the raw storage.
//!
//! This module also hosts the symmetric int8 quantization helpers
//! ([`quantize_int8`]/[`dequantize_int8`]) used to shrink KV state. Values are
//! quantized per `(row, head)` group with an absmax scale and stored biased by
//! +128 so they fit in `u8`; the small `1e-12` scale epsilon keeps all-zero
//! groups finite.

use effect_torch_runtime::DType;
use half::{bf16, f16};
use std::sync::RwLock;

/// Backing storage of a [`Slab`], tagged by element type.
pub enum SlabData {
    F32(Vec<f32>),
    F16(Vec<f16>),
    BF16(Vec<bf16>),
    U8(Vec<u8>),
}

/// A row-addressable, lock-protected cache slab.
///
/// `rows * row_width` elements are stored contiguously, row-major. All access
/// goes through closure-based [`Slab::read`]/[`Slab::write`] or the row
/// batch helpers so lock scope never escapes into callers.
pub struct Slab {
    data: RwLock<SlabData>,
    pub dtype: DType,
    pub rows: usize,
    pub row_width: usize,
}

/// Read-only typed view handed to [`Slab::read`] closures.
pub enum SlabReader<'a> {
    F32(&'a [f32]),
    F16(&'a [f16]),
    BF16(&'a [bf16]),
    U8(&'a [u8]),
}

impl SlabReader<'_> {
    /// Reads one element, converting half-precision and quantized storage to
    /// `f32` (u8 values are returned as their raw biased code).
    pub fn get_f32(&self, index: usize) -> f32 {
        match self {
            Self::F32(values) => values[index],
            Self::F16(values) => values[index].to_f32(),
            Self::BF16(values) => values[index].to_f32(),
            Self::U8(values) => values[index] as f32,
        }
    }
}

/// Mutable typed view handed to [`Slab::write`] closures.
pub enum SlabWriter<'a> {
    F32(&'a mut [f32]),
    F16(&'a mut [f16]),
    BF16(&'a mut [bf16]),
    U8(&'a mut [u8]),
}

impl SlabWriter<'_> {
    /// Writes one element, rounding into half precision when the slab stores
    /// `f16`/`bf16`. Panics on a `u8` slab: quantized rows must be encoded
    /// first (see [`quantize_int8`]) and written with [`SlabWriter::set_u8`].
    pub fn set_f32(&mut self, index: usize, value: f32) {
        match self {
            Self::F32(values) => values[index] = value,
            Self::F16(values) => values[index] = f16::from_f32(value),
            Self::BF16(values) => values[index] = bf16::from_f32(value),
            Self::U8(_) => panic!("set_f32 on a u8 slab"),
        }
    }

    /// Writes one raw quantized byte. Panics on non-`u8` slabs.
    pub fn set_u8(&mut self, index: usize, value: u8) {
        let Self::U8(values) = self else {
            panic!("set_u8 on a non-u8 slab")
        };
        values[index] = value;
    }
}

impl Slab {
    /// Allocates a zero-initialized slab. Only `F32`, `F16`, `BF16`, and `U8`
    /// are supported; other dtypes panic.
    pub fn new(rows: usize, row_width: usize, dtype: DType) -> Self {
        let n = rows * row_width;
        let data = match dtype {
            DType::F32 => SlabData::F32(vec![0.0; n]),
            DType::F16 => SlabData::F16(vec![f16::ZERO; n]),
            DType::BF16 => SlabData::BF16(vec![bf16::ZERO; n]),
            DType::U8 => SlabData::U8(vec![0; n]),
            d => panic!("kv slab: unsupported dtype {}", d.name()),
        };
        Slab {
            data: RwLock::new(data),
            dtype,
            rows,
            row_width,
        }
    }

    /// Runs `f` with a read-locked typed view of the whole slab.
    pub fn read<R>(&self, f: impl FnOnce(SlabReader<'_>) -> R) -> R {
        let data = self.data.read().unwrap();
        match &*data {
            SlabData::F32(values) => f(SlabReader::F32(values)),
            SlabData::F16(values) => f(SlabReader::F16(values)),
            SlabData::BF16(values) => f(SlabReader::BF16(values)),
            SlabData::U8(values) => f(SlabReader::U8(values)),
        }
    }

    /// Runs `f` with a write-locked typed view of the whole slab.
    pub fn write<R>(&self, f: impl FnOnce(SlabWriter<'_>) -> R) -> R {
        let mut data = self.data.write().unwrap();
        match &mut *data {
            SlabData::F32(values) => f(SlabWriter::F32(values)),
            SlabData::F16(values) => f(SlabWriter::F16(values)),
            SlabData::BF16(values) => f(SlabWriter::BF16(values)),
            SlabData::U8(values) => f(SlabWriter::U8(values)),
        }
    }

    /// Overwrites the listed rows from `src` (`rows.len() * row_width` f32
    /// values), converting to the slab dtype. Panics on `u8` slabs.
    pub fn write_rows_f32(&self, rows: &[u32], src: &[f32]) {
        assert_eq!(src.len(), rows.len() * self.row_width);
        let w = self.row_width;
        let mut data = self.data.write().unwrap();
        match &mut *data {
            SlabData::F32(buf) => {
                for (i, &r) in rows.iter().enumerate() {
                    let dst = r as usize * w;
                    buf[dst..dst + w].copy_from_slice(&src[i * w..(i + 1) * w]);
                }
            }
            SlabData::F16(buf) => {
                for (i, &r) in rows.iter().enumerate() {
                    let dst = r as usize * w;
                    for j in 0..w {
                        buf[dst + j] = f16::from_f32(src[i * w + j]);
                    }
                }
            }
            SlabData::BF16(buf) => {
                for (i, &r) in rows.iter().enumerate() {
                    let dst = r as usize * w;
                    for j in 0..w {
                        buf[dst + j] = bf16::from_f32(src[i * w + j]);
                    }
                }
            }
            SlabData::U8(_) => panic!("write_rows_f32 on a u8 slab: quantize first"),
        }
    }

    /// Overwrites the listed rows with raw quantized bytes. Panics on
    /// non-`u8` slabs.
    pub fn write_rows_u8(&self, rows: &[u32], src: &[u8]) {
        assert_eq!(src.len(), rows.len() * self.row_width);
        let w = self.row_width;
        let mut data = self.data.write().unwrap();
        let SlabData::U8(buf) = &mut *data else {
            panic!("write_rows_u8 on a non-u8 slab")
        };
        for (i, &r) in rows.iter().enumerate() {
            let dst = r as usize * w;
            buf[dst..dst + w].copy_from_slice(&src[i * w..(i + 1) * w]);
        }
    }

    /// Copies the listed rows out as `f32`, in the order requested.
    pub fn read_rows_f32(&self, rows: &[u32]) -> Vec<f32> {
        let w = self.row_width;
        let mut out = vec![0f32; rows.len() * w];
        let data = self.data.read().unwrap();
        match &*data {
            SlabData::F32(buf) => {
                for (i, &r) in rows.iter().enumerate() {
                    let src = r as usize * w;
                    out[i * w..(i + 1) * w].copy_from_slice(&buf[src..src + w]);
                }
            }
            SlabData::F16(buf) => {
                for (i, &r) in rows.iter().enumerate() {
                    let src = r as usize * w;
                    for j in 0..w {
                        out[i * w + j] = buf[src + j].to_f32();
                    }
                }
            }
            SlabData::BF16(buf) => {
                for (i, &r) in rows.iter().enumerate() {
                    let src = r as usize * w;
                    for j in 0..w {
                        out[i * w + j] = buf[src + j].to_f32();
                    }
                }
            }
            SlabData::U8(buf) => {
                for (i, &r) in rows.iter().enumerate() {
                    let src = r as usize * w;
                    for j in 0..w {
                        out[i * w + j] = buf[src + j] as f32;
                    }
                }
            }
        }
        out
    }
}

/// Symmetrically quantizes a `[rows, h, d]` f32 tensor to int8 codes biased
/// by +128 inside `u8` storage, returning the codes and one absmax scale per
/// `(row, head)` group. Values are rounded and clamped to ±127.
pub fn quantize_int8(x: &[f32], rows: usize, h: usize, d: usize) -> (Vec<u8>, Vec<f32>) {
    let mut q = vec![0u8; rows * h * d];
    let mut scales = vec![0f32; rows * h];
    for r in 0..rows {
        for head in 0..h {
            let base = (r * h + head) * d;
            let amax = x[base..base + d].iter().fold(0f32, |a, &v| a.max(v.abs()));
            let scale = amax / 127.0 + 1e-12;
            scales[r * h + head] = scale;
            for j in 0..d {
                let v = (x[base + j] / scale).round().clamp(-127.0, 127.0) + 128.0;
                q[base + j] = v as u8;
            }
        }
    }
    (q, scales)
}

/// Inverse of [`quantize_int8`]. `q` holds the biased codes as f32 (e.g.
/// freshly widened from the stored `u8`s) and `scales` the per-group absmax
/// scales produced by quantization.
pub fn dequantize_int8(q: &[f32], scales: &[f32], rows: usize, h: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * h * d];
    for r in 0..rows {
        for head in 0..h {
            let s = scales[r * h + head];
            let base = (r * h + head) * d;
            for j in 0..d {
                out[base + j] = (q[base + j] - 128.0) * s;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slab_write_read_f16() {
        let slab = Slab::new(4, 3, DType::F16);
        slab.write_rows_f32(&[1, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let got = slab.read_rows_f32(&[3, 0, 1]);
        assert_eq!(got, vec![4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn int8_roundtrip() {
        let x = vec![0.5f32, -1.0, 0.25, 2.0, 0.0, -0.5, 1.0, 1.0];
        let (q, s) = quantize_int8(&x, 2, 2, 2);
        let back = dequantize_int8(
            &q.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            &s,
            2,
            2,
            2,
        );
        for (a, b) in x.iter().zip(&back) {
            assert!((a - b).abs() < 0.02, "{a} vs {b}");
        }
    }
}
