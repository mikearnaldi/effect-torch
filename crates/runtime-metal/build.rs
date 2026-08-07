fn main() {
    if std::env::var_os("CARGO_FEATURE_NAPI_ADDON").is_some() {
        napi_build::setup();
    }
}
