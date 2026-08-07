pub type Res<T> = Result<T, String>;

pub fn err<T>(message: impl Into<String>) -> Res<T> {
    Err(message.into())
}

pub fn to_napi_err(error: String) -> napi::Error {
    napi::Error::new(napi::Status::GenericFailure, error)
}
