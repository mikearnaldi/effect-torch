//! Build script: only wires up the napi-rs linker configuration when the
//! `napi-addon` feature is enabled; otherwise the crate needs no build steps.

fn main() {
    if std::env::var_os("CARGO_FEATURE_NAPI_ADDON").is_some() {
        napi_build::setup();
    }
}
