//! Error taxonomy shared by all backends.
//!
//! [`BackendError`] classifies failures by *phase* and *cause* rather than
//! by backend, so callers can react uniformly: `Unavailable` (device or
//! backend missing), `Unsupported` (operation/dtype/layout the backend
//! cannot do), `InvalidHandle` (a buffer owned by another runtime — see the
//! ownership model in the `backend` module), `Compilation`, `Execution`,
//! `Transfer`, `Cancelled` (cooperative cancellation via
//! [`CancellationFlag`](crate::CancellationFlag)) and `Closed` (use after
//! runtime shutdown).
//!
//! Variants carry structured context fields (dtype, layout, placement, ...)
//! instead of pre-formatted strings so embedders can both match on the
//! category and render rich diagnostics.

use crate::{DType, DeviceId, Layout, Placement, RuntimeId};
use std::error::Error;
use std::fmt;

/// Result alias for backend operations.
pub type BackendResult<T> = Result<T, BackendError>;

/// A failure reported by a backend, classified by phase and cause.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    /// The backend or requested device is not available on this system.
    Unavailable {
        backend: String,
        device: Option<DeviceId>,
        message: String,
    },
    /// The backend cannot perform the requested operation, dtype, layout
    /// or placement (whichever fields are present identify the mismatch).
    Unsupported {
        operation: Option<String>,
        dtype: Option<DType>,
        layout: Option<Layout>,
        placement: Option<Placement>,
        message: String,
    },
    /// A buffer or handle belongs to a different runtime than the one it
    /// was used with. `actual_runtime` is `None` when the offending handle
    /// carries no runtime id at all.
    InvalidHandle {
        expected_runtime: RuntimeId,
        actual_runtime: Option<RuntimeId>,
    },
    /// Program compilation failed.
    Compilation {
        operation: Option<String>,
        message: String,
    },
    /// Execution of a compiled program failed.
    Execution {
        operation: Option<String>,
        message: String,
    },
    /// A host/device or device/device transfer failed.
    Transfer {
        source: Option<Placement>,
        destination: Option<Placement>,
        message: String,
    },
    /// The operation observed a set cancellation flag and aborted.
    Cancelled { operation: Option<String> },
    /// The runtime has been closed and no longer accepts work.
    Closed { runtime: RuntimeId },
}

impl BackendError {
    /// Builds an [`InvalidHandle`](Self::InvalidHandle) error naming both
    /// the expected and the actual owning runtime.
    pub fn invalid_handle(expected_runtime: RuntimeId, actual_runtime: RuntimeId) -> Self {
        Self::InvalidHandle {
            expected_runtime,
            actual_runtime: Some(actual_runtime),
        }
    }
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::Unavailable {
                backend,
                device,
                message,
            } => {
                write!(f, "backend {backend}")?;
                if let Some(device) = device {
                    write!(f, " device {device}")?;
                }
                write!(f, " is unavailable: {message}")
            }
            BackendError::Unsupported {
                operation,
                dtype,
                layout,
                placement,
                message,
            } => {
                f.write_str("unsupported")?;
                if let Some(operation) = operation {
                    write!(f, " operation {operation}")?;
                }
                if let Some(dtype) = dtype {
                    write!(f, " for dtype {dtype}")?;
                }
                if let Some(layout) = layout {
                    write!(f, " with layout {layout:?}")?;
                }
                if let Some(placement) = placement {
                    write!(f, " on {placement}")?;
                }
                write!(f, ": {message}")
            }
            BackendError::InvalidHandle {
                expected_runtime,
                actual_runtime,
            } => match actual_runtime {
                Some(actual_runtime) => write!(
                    f,
                    "invalid handle owned by runtime {actual_runtime}; expected runtime {expected_runtime}"
                ),
                None => write!(f, "invalid handle; expected runtime {expected_runtime}"),
            },
            BackendError::Compilation { operation, message } => {
                write_phase_error(f, "compilation", operation.as_deref(), message)
            }
            BackendError::Execution { operation, message } => {
                write_phase_error(f, "execution", operation.as_deref(), message)
            }
            BackendError::Transfer {
                source,
                destination,
                message,
            } => {
                f.write_str("transfer")?;
                if let Some(source) = source {
                    write!(f, " from {source}")?;
                }
                if let Some(destination) = destination {
                    write!(f, " to {destination}")?;
                }
                write!(f, " failed: {message}")
            }
            BackendError::Cancelled { operation } => {
                f.write_str("operation")?;
                if let Some(operation) = operation {
                    write!(f, " {operation}")?;
                }
                f.write_str(" was cancelled")
            }
            BackendError::Closed { runtime } => write!(f, "runtime {runtime} is closed"),
        }
    }
}

impl Error for BackendError {}

fn write_phase_error(
    f: &mut fmt::Formatter<'_>,
    phase: &str,
    operation: Option<&str>,
    message: &str,
) -> fmt::Result {
    f.write_str(phase)?;
    if let Some(operation) = operation {
        write!(f, " of {operation}")?;
    }
    write!(f, " failed: {message}")
}
