use crate::{DType, DeviceId, Layout, Placement, RuntimeId};
use std::error::Error;
use std::fmt;

pub type BackendResult<T> = Result<T, BackendError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    Unavailable {
        backend: String,
        device: Option<DeviceId>,
        message: String,
    },
    Unsupported {
        operation: Option<String>,
        dtype: Option<DType>,
        layout: Option<Layout>,
        placement: Option<Placement>,
        message: String,
    },
    InvalidHandle {
        expected_runtime: RuntimeId,
        actual_runtime: Option<RuntimeId>,
    },
    Compilation {
        operation: Option<String>,
        message: String,
    },
    Execution {
        operation: Option<String>,
        message: String,
    },
    Transfer {
        source: Option<Placement>,
        destination: Option<Placement>,
        message: String,
    },
    Cancelled {
        operation: Option<String>,
    },
    Closed {
        runtime: RuntimeId,
    },
}

impl BackendError {
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
