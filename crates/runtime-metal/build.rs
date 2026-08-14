//! Build-script entry point for the optional Node-API addon.
//!
//! Pure Rust consumers need no linker/export setup. Enabling `napi-addon`
//! asks `napi-build` to emit the platform directives required by the Node
//! module, matching the feature gate around `crate::napi`.

fn main() {
    if std::env::var_os("CARGO_FEATURE_NAPI_ADDON").is_some() {
        napi_build::setup();
    }
}
