//! Independent invocation-owned descriptors for overlapping exact second M1.
//! As with the other CUDA policy flags, configuration stays fixed from device
//! creation and compilation through execution; the capability fingerprint
//! separates compiled plans that reserve the appended descriptor bank.
use crate::cublas::{
    CUBLAS_WORKSPACE_BYTES, EXPERT_BLAS_STREAMS, EXPERT_GROUPED_POINTER_BYTES,
    EXPERT_SPLITK_DESCRIPTOR_BYTES,
};

pub(crate) const ENV: &str = "EFFECT_TORCH_CUDA_EXPERT_GEMV_OVERLAP";
pub(crate) const DESCRIPTOR_BYTES: usize = EXPERT_SPLITK_DESCRIPTOR_BYTES;

pub(crate) fn enabled() -> bool {
    std::env::var(ENV).as_deref() == Ok("1")
}

/// The planner appends this bank after every existing workspace component,
/// including optional split partials. It never aliases the merged descriptors.
pub(crate) fn descriptor_offset(workspace: u64, bytes: usize) -> Result<usize, String> {
    let minimum = CUBLAS_WORKSPACE_BYTES * EXPERT_BLAS_STREAMS
        + EXPERT_GROUPED_POINTER_BYTES
        + EXPERT_SPLITK_DESCRIPTOR_BYTES
        + DESCRIPTOR_BYTES;
    if workspace == 0 || workspace % 16 != 0 || bytes < minimum {
        return Err("CUDA overlapping GEMV descriptor workspace bounds/alignment exceeded".into());
    }
    workspace
        .checked_add(bytes as u64)
        .ok_or("CUDA overlapping GEMV workspace address overflow")?;
    let offset = bytes - DESCRIPTOR_BYTES;
    if offset % 16 != 0 {
        return Err("CUDA overlapping GEMV descriptor offset is unaligned".into());
    }
    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_descriptor_bank_checks_extent_alignment_and_old_layout() {
        let old = CUBLAS_WORKSPACE_BYTES * EXPERT_BLAS_STREAMS
            + EXPERT_GROUPED_POINTER_BYTES
            + EXPERT_SPLITK_DESCRIPTOR_BYTES;
        assert!(descriptor_offset(16, old).is_err());
        assert_eq!(descriptor_offset(16, old + DESCRIPTOR_BYTES).unwrap(), old);
        assert_eq!(
            descriptor_offset(16, old + DESCRIPTOR_BYTES + 4096).unwrap(),
            old + 4096
        );
        for (address, bytes) in [
            (0, old + DESCRIPTOR_BYTES),
            (17, old + DESCRIPTOR_BYTES),
            (16, old + DESCRIPTOR_BYTES - 1),
            (16, old + DESCRIPTOR_BYTES + 1),
            (u64::MAX - 15, old + DESCRIPTOR_BYTES),
        ] {
            assert!(descriptor_offset(address, bytes).is_err());
        }
    }
}
