//! Build-script entry point for the optional Node-API addon.
//!
//! Pure Rust consumers need no linker or export setup. With `napi-addon`
//! enabled, `napi-build` emits the platform directives for the Node module.
//! This matches the feature gate around `crate::napi`.

fn main() {
    if std::env::var_os("CARGO_FEATURE_NAPI_ADDON").is_some() {
        napi_build::setup();
    }
}
