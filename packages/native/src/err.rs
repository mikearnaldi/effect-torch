pub type Res<T> = Result<T, String>;

pub fn err<T>(msg: impl Into<String>) -> Res<T> {
    Err(msg.into())
}

pub fn err_str(msg: impl Into<String>) -> String {
    msg.into()
}

pub fn to_napi_err(e: String) -> napi::Error {
    napi::Error::new(napi::Status::GenericFailure, e)
}

