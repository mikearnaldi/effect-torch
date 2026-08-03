use crate::dev::Device;
use crate::err::{err, Res};
use crate::runtime;
use crate::val::Val;
use std::collections::HashMap;

fn dtype_name(d: runtime::dtype::DType) -> &'static str {
    match d {
        runtime::dtype::DType::F32 => "F32",
        runtime::dtype::DType::F64 => "F64",
        runtime::dtype::DType::F16 => "F16",
        runtime::dtype::DType::BF16 => "BF16",
        runtime::dtype::DType::U8 => "U8",
        runtime::dtype::DType::U32 => "U32",
        runtime::dtype::DType::I64 => "I64",
    }
}

fn dtype_of(name: &str) -> Res<runtime::dtype::DType> {
    match name {
        "F32" => Ok(runtime::dtype::DType::F32),
        "F64" => Ok(runtime::dtype::DType::F64),
        "F16" => Ok(runtime::dtype::DType::F16),
        "BF16" => Ok(runtime::dtype::DType::BF16),
        "U8" => Ok(runtime::dtype::DType::U8),
        "U32" => Ok(runtime::dtype::DType::U32),
        "I64" => Ok(runtime::dtype::DType::I64),
        other => err(format!("safetensors: unsupported dtype {other}")),
    }
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn tensor_bytes(t: &Val) -> Res<Vec<u8>> {
    match t {
        Val::Cpu(t) => {
            let t = t.contiguous();
            let n = t.numel();
            let size = t.dtype().size_in_bytes();
            let mut out = Vec::with_capacity(n * size);
            macro_rules! go {
                ($v:expr) => {
                    for x in $v.iter() {
                        out.extend_from_slice(&x.to_le_bytes());
                    }
                };
            }
            match &t.buffer {
                runtime::cpu::CpuBuffer::F32(v) => go!(v),
                runtime::cpu::CpuBuffer::F64(v) => go!(v),
                runtime::cpu::CpuBuffer::F16(v) => go!(v),
                runtime::cpu::CpuBuffer::BF16(v) => go!(v),
                runtime::cpu::CpuBuffer::U8(v) => out.extend_from_slice(v),
                runtime::cpu::CpuBuffer::U32(v) => go!(v),
                runtime::cpu::CpuBuffer::I64(v) => go!(v),
            }
            Ok(out)
        }
        Val::Metal(t) => {
            runtime::metal::device::MetalDevice::get().synchronize();
            let n = t.numel();
            let size = t.dtype.size_in_bytes();
            let ptr = unsafe { t.buffer.contents_ptr().cast::<u8>().add(t.layout.offset() * size) };
            Ok(unsafe { std::slice::from_raw_parts(ptr, n * size) }.to_vec())
        }
    }
}

pub fn save(map: &HashMap<String, Val>, path: &str) -> Res<()> {
    let mut names: Vec<&String> = map.keys().collect();
    names.sort();
    let mut header = String::from("{");
    let mut offset = 0usize;
    let mut blobs: Vec<Vec<u8>> = Vec::with_capacity(names.len());
    for (i, name) in names.iter().enumerate() {
        let t = &map[*name];
        let bytes = tensor_bytes(t)?;
        let shape: Vec<String> = t.shape().iter().map(|d| d.to_string()).collect();
        if i > 0 {
            header.push(',');
        }
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            json_escape(name),
            dtype_name(t.dtype()),
            shape.join(","),
            offset,
            offset + bytes.len()
        ));
        offset += bytes.len();
        blobs.push(bytes);
    }
    header.push('}');
    let mut out = Vec::with_capacity(8 + header.len() + offset);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for blob in blobs {
        out.extend_from_slice(&blob);
    }
    std::fs::write(path, out).map_err(|e| e.to_string())
}

fn parse_header(json: &str) -> Res<Vec<(String, runtime::dtype::DType, Vec<usize>, usize, usize)>> {
    let mut out = Vec::new();
    let mut rest = json.trim();
    if !rest.starts_with('{') || !rest.ends_with('}') {
        return err("safetensors: malformed header");
    }
    rest = &rest[1..rest.len() - 1];
    for entry in split_top(rest, ',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let colon = entry.find("\":").ok_or_else(|| "safetensors: malformed entry".to_string())?;
        let name = entry[1..colon].to_string();
        let body = &entry[colon + 2..];
        let dtype = extract_str(body, "\"dtype\":\"").ok_or_else(|| "safetensors: missing dtype".to_string())?;
        let shape = extract_nums(body, "\"shape\":[", ']').ok_or_else(|| "safetensors: missing shape".to_string())?;
        let offsets = extract_nums(body, "\"data_offsets\":[", ']').ok_or_else(|| "safetensors: missing offsets".to_string())?;
        if offsets.len() != 2 {
            return err("safetensors: malformed data_offsets");
        }
        out.push((name, dtype_of(&dtype)?, shape, offsets[0], offsets[1]));
    }
    Ok(out)
}

fn split_top(s: &str, sep: char) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '{' | '[' => {
                depth += 1;
                cur.push(c);
            }
            '}' | ']' => {
                depth -= 1;
                cur.push(c);
            }
            c if c == sep && depth == 0 => {
                parts.push(std::mem::take(&mut cur));
            }
            c => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        parts.push(cur);
    }
    parts
}

fn extract_str(body: &str, key: &str) -> Option<String> {
    let start = body.find(key)? + key.len();
    let end = body[start..].find('"')? + start;
    Some(body[start..end].to_string())
}

fn extract_nums(body: &str, key: &str, close: char) -> Option<Vec<usize>> {
    let start = body.find(key)? + key.len();
    let end = body[start..].find(close)? + start;
    Some(
        body[start..end]
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse().ok())
            .collect::<Option<Vec<_>>>()?,
    )
}

pub fn load(path: &str, device: &Device) -> Res<Vec<(String, Val)>> {
    let raw = std::fs::read(path).map_err(|e| e.to_string())?;
    if raw.len() < 8 {
        return err("safetensors: file too small");
    }
    let header_len = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    if raw.len() < 8 + header_len {
        return err("safetensors: truncated header");
    }
    let header = std::str::from_utf8(&raw[8..8 + header_len]).map_err(|e| e.to_string())?;
    let data_base = 8 + header_len;
    let mut out = Vec::new();
    for (name, dtype, shape, start, end) in parse_header(header)? {
        let bytes = &raw[data_base + start..data_base + end];
        let expected: usize = shape.iter().product::<usize>() * dtype.size_in_bytes();
        if bytes.len() != expected {
            return err(format!(
                "safetensors: {name}: expected {expected} bytes, got {}",
                bytes.len()
            ));
        }
        let val = match device {
            Device::Cpu => cpu_from_bytes(bytes, &shape, dtype)?,
            Device::Metal => Val::Metal(runtime::metal::run::MetalTensor {
                buffer: runtime::metal::device::MetalDevice::get().upload_bytes(bytes),
                layout: runtime::layout::Layout::contiguous(shape),
                dtype,
            }),
        };
        out.push((name, val));
    }
    Ok(out)
}

fn cpu_from_bytes(bytes: &[u8], shape: &[usize], dtype: runtime::dtype::DType) -> Res<Val> {
    let t = match dtype {
        runtime::dtype::DType::F32 => runtime::cpu::Tensor::from_vec(
            bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            shape.to_vec(),
        ),
        runtime::dtype::DType::F64 => runtime::cpu::Tensor::from_vec(
            bytes.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect(),
            shape.to_vec(),
        ),
        runtime::dtype::DType::F16 => runtime::cpu::Tensor::from_vec(
            bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]])).collect(),
            shape.to_vec(),
        ),
        runtime::dtype::DType::BF16 => runtime::cpu::Tensor::from_vec(
            bytes.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]])).collect(),
            shape.to_vec(),
        ),
        runtime::dtype::DType::U8 => runtime::cpu::Tensor::from_vec(bytes.to_vec(), shape.to_vec()),
        runtime::dtype::DType::U32 => runtime::cpu::Tensor::from_vec(
            bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            shape.to_vec(),
        ),
        runtime::dtype::DType::I64 => runtime::cpu::Tensor::from_vec(
            bytes.chunks_exact(8).map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect(),
            shape.to_vec(),
        ),
    };
    Ok(Val::Cpu(t))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_cpu() {
        let mut map = HashMap::new();
        map.insert(
            "w".to_string(),
            Val::Cpu(runtime::cpu::Tensor::from_vec(vec![1f32, 2., 3., 4.], vec![2, 2])),
        );
        map.insert(
            "ids".to_string(),
            Val::Cpu(runtime::cpu::Tensor::from_vec(vec![1u32, 2, 3], vec![3])),
        );
        let path = std::env::temp_dir().join("et_st_test.safetensors");
        save(&map, path.to_str().unwrap()).unwrap();
        let entries = load(path.to_str().unwrap(), &Device::Cpu).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, "ids");
        assert_eq!(entries[1].0, "w");
        let Val::Cpu(w) = &entries[1].1 else { panic!() };
        assert_eq!(w.shape(), &[2, 2]);
        std::fs::remove_file(&path).ok();
    }
}
