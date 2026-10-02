//! Experimental categorical entropy identity over already-recognized F32 rows.
//! Explicit opt-in changes F32 rounding and can change entropy ordering/policy
//! decisions. Natural generation quality must gate use; the default five-stage
//! tensor contract and the compiler's private-intermediate proof are unchanged.
use cudarc::driver::LaunchConfig;

pub(crate) const ENV: &str = "EFFECT_TORCH_CUDA_RELAXED_ENTROPY81";
pub(crate) const KERNEL: &str = "et_entropy_relaxed81_moment_finish";
const EXACT: &[&str] = &[
    "et_entropy_max",
    "et_entropy_sum",
    "et_entropy_normalized_max",
    "et_entropy_normalized_sum",
    "et_entropy_finish",
];
const RELAXED: &[&str] = &["et_entropy_max", KERNEL];

pub(crate) fn enabled() -> bool {
    std::env::var(ENV).as_deref() == Ok("1")
}

pub(crate) fn stages(relaxed: bool) -> &'static [&'static str] {
    if relaxed {
        RELAXED
    } else {
        EXACT
    }
}

/// One CTA owns each row; no clamp may silently omit rows.
pub(crate) fn geometry(rows: u64, width: u64) -> Result<LaunchConfig, String> {
    if !(1..=65535).contains(&rows) || width < 4096 || rows.checked_mul(width).is_none() {
        return Err("entropy81: requires nonempty wide F32 rows within one-CTA grid limits".into());
    }
    Ok(LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (1024, 1, 1),
        shared_mem_bytes: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stages_preserve_exact_default_and_explicit_relaxed_identity() {
        assert_eq!(stages(false).len(), 5);
        assert_eq!(stages(false)[4], "et_entropy_finish");
        assert_eq!(stages(true), ["et_entropy_max", KERNEL]);
        assert!(!stages(false).contains(&KERNEL));
    }

    #[test]
    fn geometry_covers_all_rows_and_rejects_empty_narrow_overflow_and_truncated_grids() {
        for (rows, width) in [(1, 4096), (256, 262144), (65535, 4097)] {
            let config = geometry(rows, width).unwrap();
            assert_eq!(config.grid_dim, (rows as u32, 1, 1));
            assert_eq!(config.block_dim, (1024, 1, 1));
            assert_eq!(config.shared_mem_bytes, 0);
        }
        for (rows, width) in [(0, 4096), (1, 0), (1, 4095), (65536, 4096), (2, u64::MAX)] {
            assert!(geometry(rows, width).is_err(), "rows={rows} width={width}");
        }
    }
}
