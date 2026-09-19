//! Sparse GGUF fixture shared by parser and native backend tests.

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

pub struct SparseGguf {
    pub file: File,
    pub path: PathBuf,
    pub data_start: u64,
    pub dense: Vec<u8>,
    pub packed: Vec<u8>,
    unused_format_offset: u64,
}

fn string(bytes: &mut Vec<u8>, text: &str) {
    bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
    bytes.extend_from_slice(text.as_bytes());
}

impl SparseGguf {
    pub fn new(unused_bytes: u64) -> Self {
        assert!(unused_bytes > 0 && unused_bytes % 4 == 0);
        let mut header = b"GGUF".to_vec();
        header.extend_from_slice(&3u32.to_le_bytes());
        header.extend_from_slice(&3u64.to_le_bytes());
        header.extend_from_slice(&2u64.to_le_bytes());
        string(&mut header, "general.architecture");
        header.extend_from_slice(&8u32.to_le_bytes());
        string(&mut header, "test");
        string(&mut header, "general.quantization_version");
        header.extend_from_slice(&4u32.to_le_bytes());
        header.extend_from_slice(&2u32.to_le_bytes());
        let mut unused_format_offset = 0;
        // Archive order deliberately differs from physical payload order.
        for (name, dimensions, format, offset) in [
            ("unused", vec![unused_bytes / 4], 0u32, 4096u64),
            ("dense", vec![2], 0, 32),
            ("packed", vec![256, 2], 12, 128),
        ] {
            string(&mut header, name);
            header.extend_from_slice(&(dimensions.len() as u32).to_le_bytes());
            for dimension in dimensions {
                header.extend_from_slice(&dimension.to_le_bytes());
            }
            if name == "unused" {
                unused_format_offset = header.len() as u64;
            }
            header.extend_from_slice(&format.to_le_bytes());
            header.extend_from_slice(&offset.to_le_bytes());
        }
        header.resize(header.len().next_multiple_of(32), 0);
        let data_start = header.len() as u64;
        let path = std::env::temp_dir().join(format!(
            "effect-torch-selected-gguf-{}-{}.gguf",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed),
        ));
        let mut file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.write_all(&header).unwrap();
        file.set_len(data_start + 4096 + unused_bytes).unwrap();
        let dense = [0x80000000u32, 0x7f800123]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let packed = (0..288).map(|i| (i * 37 + 19) as u8).collect::<Vec<_>>();
        file.seek(SeekFrom::Start(data_start + 32)).unwrap();
        file.write_all(&dense).unwrap();
        file.seek(SeekFrom::Start(data_start + 128)).unwrap();
        file.write_all(&packed).unwrap();
        Self {
            file,
            path,
            data_start,
            dense,
            packed,
            unused_format_offset,
        }
    }

    /// Remove the unused payload after parsing, so any attempt to read it fails.
    pub fn truncate_unused(&self) {
        self.file
            .set_len(self.data_start + 128 + self.packed.len() as u64)
            .unwrap();
    }

    pub fn set_unused_format(&mut self, format: u32) {
        self.file
            .seek(SeekFrom::Start(self.unused_format_offset))
            .unwrap();
        self.file.write_all(&format.to_le_bytes()).unwrap();
    }
}

impl Drop for SparseGguf {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}
