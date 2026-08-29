//! Builds the tokenizer Node-API addon.
//!
//! `napi-build` emits the platform linker and export directives for the
//! `.node` cdylib used by the TypeScript facade.

fn main() {
    napi_build::setup();
}
