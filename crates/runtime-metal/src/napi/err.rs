pub type Res<T> = Result<T, String>;

pub fn err<T>(message: impl Into<String>) -> Res<T> {
    Err(message.into())
}

pub fn err_str(message: impl Into<String>) -> String {
    message.into()
}

pub fn to_napi_err(error: String) -> napi::Error {
    napi::Error::new(napi::Status::GenericFailure, error)
}
