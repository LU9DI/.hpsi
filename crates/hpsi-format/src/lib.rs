//! Safe, byte-oriented primitives for the HPSI v0.1 container.
//!
//! The format crate validates structure and exposes opaque payload blocks.
//! Codec decoding and neural-network execution live in separate layers.

mod block;
mod error;
mod header;
mod integrity;
mod limits;
mod metadata;
mod reader;
mod region;
mod source;
mod tensor;
mod writer;

pub use block::BlockDescriptor;
pub use error::{FormatError, Result};
pub use header::{HEADER_SIZE, Header};
pub use integrity::{IntegrityRecord, IntegrityTarget};
pub use limits::ReaderLimits;
pub use metadata::Metadata;
pub use reader::{BlockData, ModelReader, ReaderStats};
pub use region::{Region, validate_regions};
pub use source::{FileSource, MemorySource, RandomAccessSource};
pub use tensor::{DType, TensorDescriptor};
pub use writer::{ModelWriter, PayloadBlock, StreamingTensorInput, TensorInput, WriterStats};
