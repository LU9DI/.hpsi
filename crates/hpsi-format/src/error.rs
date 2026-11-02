use core::fmt;

pub type Result<T> = core::result::Result<T, FormatError>;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FormatError {
    Io(String),
    BadMagic,
    UnsupportedVersion { major: u16, minor: u16 },
    UnsupportedFeature { flags: u32 },
    InvalidHeader(&'static str),
    InvalidRegion(&'static str),
    InvalidDirectory(&'static str),
    InvalidMetadata(&'static str),
    InvalidTensor(&'static str),
    InvalidBlock(&'static str),
    InvalidReference(&'static str),
    ChecksumMismatch,
    LimitExceeded(&'static str),
    Truncated { expected: usize, actual: usize },
    ArithmeticOverflow(&'static str),
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(f, "I/O error: {message}"),
            Self::BadMagic => f.write_str("invalid HPSI magic"),
            Self::UnsupportedVersion { major, minor } => {
                write!(f, "unsupported HPSI format version {major}.{minor}")
            }
            Self::UnsupportedFeature { flags } => {
                write!(f, "unsupported mandatory header flags: 0x{flags:08x}")
            }
            Self::InvalidHeader(message) => write!(f, "invalid HPSI header: {message}"),
            Self::InvalidRegion(message) => write!(f, "invalid HPSI region: {message}"),
            Self::InvalidDirectory(message) => write!(f, "invalid HPSI directory: {message}"),
            Self::InvalidMetadata(message) => write!(f, "invalid HPSI metadata: {message}"),
            Self::InvalidTensor(message) => write!(f, "invalid HPSI tensor: {message}"),
            Self::InvalidBlock(message) => write!(f, "invalid HPSI block: {message}"),
            Self::InvalidReference(message) => write!(f, "invalid HPSI reference: {message}"),
            Self::ChecksumMismatch => f.write_str("HPSI integrity checksum mismatch"),
            Self::LimitExceeded(message) => write!(f, "reader limit exceeded: {message}"),
            Self::Truncated { expected, actual } => {
                write!(
                    f,
                    "truncated input: expected {expected} bytes, got {actual}"
                )
            }
            Self::ArithmeticOverflow(context) => {
                write!(f, "integer overflow while calculating {context}")
            }
        }
    }
}

impl std::error::Error for FormatError {}
