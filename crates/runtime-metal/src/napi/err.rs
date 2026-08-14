//! Error adapters used by the Metal Node-API boundary.
//!
//! Internal Metal helpers use `Result<T, String>` to keep kernel/planner code
//! independent of Node. Exported methods convert those diagnostics to generic
//! NAPI failures only at the JavaScript boundary.

/// String-based result used by native Metal helpers.
pub type Res<T> = Result<T, String>;

/// Constructs a failed native result from any string-like diagnostic.
pub fn err<T>(message: impl Into<String>) -> Res<T> {
    Err(message.into())
}

/// Converts a string-like diagnostic without changing its contents.
pub fn err_str(message: impl Into<String>) -> String {
    message.into()
}

/// Maps an internal diagnostic to the generic NAPI failure channel.
pub fn to_napi_err(error: String) -> napi::Error {
    napi::Error::new(napi::Status::GenericFailure, error)
}
