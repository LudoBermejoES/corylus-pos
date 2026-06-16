use thiserror::Error;

#[derive(Debug, Error)]
pub enum PosError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },
    #[error("model not installed for language: {0}")]
    NotInstalled(String),
    #[error("model error: {0}")]
    Model(String),
}
