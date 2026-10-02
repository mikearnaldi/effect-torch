//! Proven TopK routing for the opt-in device-controlled expert path.
use effect_torch_graph::{Node, NodeKind};
use effect_torch_runtime::DType;

pub(crate) const ENV: &str = "EFFECT_TORCH_CUDA_EXPERT_DEVICE";
pub(crate) const BYTES: usize = 8192 + 1536 + 528;
#[cfg(test)]
thread_local! { static TEST_POLICY: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) }; }
#[cfg(test)]
pub(crate) fn with_test_policy<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_POLICY.with(|p| p.set(self.0));
        }
    }
    let _restore = Restore(TEST_POLICY.with(|p| p.replace(Some(enabled))));
    run()
}
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = TEST_POLICY.with(|p| p.get()) {
        return value;
    }
    std::env::var(ENV).as_deref() == Ok("1")
        && ![
            "EFFECT_TORCH_CUDA_GRAPHS",
            "EFFECT_TORCH_CUDA_OVERLAP_PRIMARY_GRAPHS",
        ]
        .iter()
        .any(|key| std::env::var(key).as_deref() == Ok("1"))
}

/// Every expert occurs at most once in each token's TopK. Only bijective,
/// exact model views are accepted; arbitrary externally supplied routes retain
/// the general host-controlled implementation.
pub(crate) fn topk_rows(node: &Node) -> Option<usize> {
    let NodeKind::Reshape { a: permute, shape } = &node.kind else {
        return None;
    };
    let NodeKind::Permute { a: topk, dims } = &permute.kind else {
        return None;
    };
    let NodeKind::TopKIndices { a: scores, k: 8 } = &topk.kind else {
        return None;
    };
    let [tokens @ (64 | 256), 128] = scores.shape.as_slice() else {
        return None;
    };
    (node.dtype == DType::U32
        && dims == &[1, 0]
        && shape == &[tokens * 8]
        && topk.shape == [*tokens, 8]
        && permute.shape == [8, *tokens])
    .then_some(*tokens)
}

pub(crate) fn earlier_status(failure: u64, boundary: Option<usize>) -> bool {
    failure != 0 && boundary.is_some_and(|position| (failure >> 32) as usize <= position)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoint_preserves_host_error_before_observation() {
        assert!(!earlier_status((2 << 32) | 5, None));
        assert!(!earlier_status(0, Some(4)));
        assert!(earlier_status((2 << 32) | 5, Some(4)));
        assert!(earlier_status((4 << 32) | 1, Some(4)));
        assert!(!earlier_status((5 << 32) | 5, Some(4)));
    }
}

#[cfg(test)]
thread_local! { static FAIL_STATUS_READ: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
#[cfg(test)]
pub(crate) fn with_failed_status_read<T>(run: impl FnOnce() -> T) -> T {
    FAIL_STATUS_READ.with(|flag| flag.set(true));
    let result = run();
    FAIL_STATUS_READ.with(|flag| flag.set(false));
    result
}
#[cfg(test)]
pub(crate) fn fail_status_read() -> bool {
    FAIL_STATUS_READ.with(|flag| flag.replace(false))
}
