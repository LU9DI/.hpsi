//! Random-access byte sources. No source implementation decompresses data.

use std::fs::File;
use std::path::Path;

use crate::{FormatError, Result};

/// A thread-safe source supporting exact reads at absolute byte offsets.
pub trait RandomAccessSource: Send + Sync {
    fn len(&self) -> Result<u64>;
    fn read_exact_at(&self, offset: u64, buffer: &mut [u8]) -> Result<()>;

    fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
}

#[derive(Debug)]
pub struct FileSource {
    file: File,
    len: u64,
}

impl FileSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path).map_err(|error| FormatError::Io(error.to_string()))?;
        let len = file
            .metadata()
            .map_err(|error| FormatError::Io(error.to_string()))?
            .len();
        Ok(Self { file, len })
    }
}

impl RandomAccessSource for FileSource {
    fn len(&self) -> Result<u64> {
        Ok(self.len)
    }

    fn read_exact_at(&self, offset: u64, buffer: &mut [u8]) -> Result<()> {
        let end = offset
            .checked_add(buffer.len() as u64)
            .ok_or(FormatError::ArithmeticOverflow("source read range"))?;
        if end > self.len {
            return Err(FormatError::InvalidRegion("read extends beyond source"));
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            let mut completed = 0;
            while completed < buffer.len() {
                let position = offset + completed as u64;
                let count = self
                    .file
                    .read_at(&mut buffer[completed..], position)
                    .map_err(|error| FormatError::Io(error.to_string()))?;
                if count == 0 {
                    return Err(FormatError::Truncated {
                        expected: buffer.len(),
                        actual: completed,
                    });
                }
                completed += count;
            }
        }

        #[cfg(windows)]
        {
            use std::os::windows::fs::FileExt;
            let mut completed = 0;
            while completed < buffer.len() {
                let position = offset + completed as u64;
                let count = self
                    .file
                    .seek_read(&mut buffer[completed..], position)
                    .map_err(|error| FormatError::Io(error.to_string()))?;
                if count == 0 {
                    return Err(FormatError::Truncated {
                        expected: buffer.len(),
                        actual: completed,
                    });
                }
                completed += count;
            }
        }

        #[cfg(not(any(unix, windows)))]
        {
            let _ = (offset, buffer);
            return Err(FormatError::Io(
                "random-access file reads unsupported on this platform".into(),
            ));
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MemorySource {
    bytes: std::sync::Arc<[u8]>,
}

impl MemorySource {
    pub fn new(bytes: impl Into<std::sync::Arc<[u8]>>) -> Self {
        Self {
            bytes: bytes.into(),
        }
    }
}

impl RandomAccessSource for MemorySource {
    fn len(&self) -> Result<u64> {
        Ok(self.bytes.len() as u64)
    }

    fn read_exact_at(&self, offset: u64, buffer: &mut [u8]) -> Result<()> {
        let start = usize::try_from(offset)
            .map_err(|_| FormatError::InvalidRegion("offset cannot fit address space"))?;
        let end = start
            .checked_add(buffer.len())
            .ok_or(FormatError::ArithmeticOverflow("memory source read range"))?;
        let source = self
            .bytes
            .get(start..end)
            .ok_or(FormatError::InvalidRegion(
                "read extends beyond memory source",
            ))?;
        buffer.copy_from_slice(source);
        Ok(())
    }
}
