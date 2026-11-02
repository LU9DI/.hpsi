//! Safe, byte-oriented primitives for the HPSI v0.1 container.
//!
//! This crate deliberately does not decode codecs or materialize tensors.
//! It validates the fixed header and region layout before payload access.

mod error;
mod header;
mod limits;
mod region;
mod source;

pub use error::{FormatError, Result};
pub use header::{HEADER_SIZE, Header};
pub use limits::ReaderLimits;
pub use region::{Region, validate_regions};
pub use source::{FileSource, MemorySource, RandomAccessSource};
