use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Default)]
pub struct CancellationFlag(AtomicBool);

impl CancellationFlag {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

impl Deref for CancellationFlag {
    type Target = AtomicBool;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
