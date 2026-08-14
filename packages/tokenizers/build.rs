//! Build-script setup for the tokenizer Node-API addon.
//!
//! `napi-build` emits the platform linker/export directives required to
//! produce the `.node` cdylib consumed by the TypeScript facade.

fn main() {
    napi_build::setup();
}
