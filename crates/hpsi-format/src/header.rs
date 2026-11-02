use crate::{FormatError, ReaderLimits, Result, Region, validate_regions};

pub const HEADER_SIZE: usize = 128;
const TENSOR_ENTRY_SIZE: u32 = 128;
const BLOCK_ENTRY_SIZE: u32 = 64;
const INTEGRITY_ENTRY_SIZE: u32 = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    pub format_major: u16,
    pub format_minor: u16,
    pub flags: u32,
    pub file_size: u64,
    pub metadata: Region,
    pub tensor_directory: Region,
    pub tensor_count: u32,
    pub block_directory: Region,
    pub block_count: u32,
    pub payload: Region,
    pub integrity_directory: Region,
    pub integrity_count: u32,
    pub artifact_id: [u8; 16],
}

impl Header {
    /// Parse and validate a complete 128-byte header against the actual source length.
    pub fn parse(bytes: &[u8], actual_file_size: u64, limits: &ReaderLimits) -> Result<Self> {
        if bytes.len() < HEADER_SIZE {
            return Err(FormatError::Truncated {
                expected: HEADER_SIZE,
                actual: bytes.len(),
            });
        }
        let bytes = &bytes[..HEADER_SIZE];

        if &bytes[0..4] != b"HPSI" {
            return Err(FormatError::BadMagic);
        }

        let major = u16_at(bytes, 4);
        let minor = u16_at(bytes, 6);
        if major != 0 || minor != 1 {
            return Err(FormatError::UnsupportedVersion { major, minor });
        }
        if u32_at(bytes, 8) != HEADER_SIZE as u32 {
            return Err(FormatError::InvalidHeader("header_size must be 128"));
        }

        let flags = u32_at(bytes, 12);
        if flags != 0 {
            return Err(FormatError::UnsupportedFeature { flags });
        }

        let file_size = u64_at(bytes, 16);
        if file_size != actual_file_size {
            return Err(FormatError::InvalidHeader("declared file size differs from actual size"));
        }
        if file_size > limits.max_file_size {
            return Err(FormatError::LimitExceeded("maximum file size"));
        }

        let metadata = Region {
            name: "metadata",
            offset: u64_at(bytes, 24),
            length: u64_at(bytes, 32),
        };
        let tensor_directory = Region {
            name: "tensor directory",
            offset: u64_at(bytes, 40),
            length: checked_table_len(u32_at(bytes, 48), u32_at(bytes, 52), TENSOR_ENTRY_SIZE)?,
        };
        let tensor_count = u32_at(bytes, 48);
        let block_directory = Region {
            name: "block directory",
            offset: u64_at(bytes, 56),
            length: checked_table_len(u32_at(bytes, 64), u32_at(bytes, 68), BLOCK_ENTRY_SIZE)?,
        };
        let block_count = u32_at(bytes, 64);
        let payload = Region {
            name: "payload",
            offset: u64_at(bytes, 72),
            length: u64_at(bytes, 80),
        };
        let integrity_count = u32_at(bytes, 96);
        let integrity_entry_size = u32_at(bytes, 100);
        let integrity_directory = Region {
            name: "integrity directory",
            offset: u64_at(bytes, 88),
            length: if integrity_count == 0 {
                if integrity_entry_size != 0 {
                    return Err(FormatError::InvalidHeader("empty integrity directory must have entry size zero"));
                }
                0
            } else {
                checked_table_len(integrity_count, integrity_entry_size, INTEGRITY_ENTRY_SIZE)?
            },
        };

        if tensor_count > limits.max_tensors {
            return Err(FormatError::LimitExceeded("maximum tensor count"));
        }
        if block_count > limits.max_blocks {
            return Err(FormatError::LimitExceeded("maximum block count"));
        }
        if metadata.length > limits.max_metadata_bytes {
            return Err(FormatError::LimitExceeded("maximum metadata bytes"));
        }

        if u32_at(bytes, 52) != TENSOR_ENTRY_SIZE || u32_at(bytes, 68) != BLOCK_ENTRY_SIZE {
            return Err(FormatError::InvalidHeader("unexpected directory entry size"));
        }
        if integrity_count > 0 && integrity_entry_size != INTEGRITY_ENTRY_SIZE {
            return Err(FormatError::InvalidHeader("unexpected integrity entry size"));
        }
        if bytes[120..128].iter().any(|byte| *byte != 0) {
            return Err(FormatError::InvalidHeader("reserved header bytes must be zero"));
        }

        if tensor_count == 0 || block_count == 0 {
            return Err(FormatError::InvalidHeader("v0.1 model requires at least one tensor and block"));
        }
        if metadata.length == 0 || payload.length == 0 {
            return Err(FormatError::InvalidHeader("metadata and payload regions must be non-empty"));
        }

        let directory_bytes = tensor_directory.length
            .checked_add(block_directory.length)
            .and_then(|n| n.checked_add(integrity_directory.length))
            .ok_or(FormatError::ArithmeticOverflow("total directory bytes"))?;
        if directory_bytes > limits.max_total_directory_bytes {
            return Err(FormatError::LimitExceeded("total directory bytes"));
        }

        validate_regions(
            &[metadata, tensor_directory, block_directory, payload, integrity_directory],
            file_size,
        )?;

        let mut artifact_id = [0_u8; 16];
        artifact_id.copy_from_slice(&bytes[104..120]);

        Ok(Self {
            format_major: major,
            format_minor: minor,
            flags,
            file_size,
            metadata,
            tensor_directory,
            tensor_count,
            block_directory,
            block_count,
            payload,
            integrity_directory,
            integrity_count,
            artifact_id,
        })
    }
}

fn checked_table_len(count: u32, entry_size: u32, expected_entry_size: u32) -> Result<u64> {
    if count == 0 {
        if entry_size != 0 && entry_size != expected_entry_size {
            return Err(FormatError::InvalidHeader("invalid empty directory entry size"));
        }
        return Ok(0);
    }
    if entry_size != expected_entry_size {
        return Err(FormatError::InvalidHeader("unexpected fixed directory entry size"));
    }
    u64::from(count)
        .checked_mul(u64::from(entry_size))
        .ok_or(FormatError::ArithmeticOverflow("directory length"))
}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("fixed header slice"))
}

fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("fixed header slice"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_header() -> [u8; HEADER_SIZE] {
        let mut b = [0_u8; HEADER_SIZE];
        b[0..4].copy_from_slice(b"HPSI");
        b[4..6].copy_from_slice(&0_u16.to_le_bytes());
        b[6..8].copy_from_slice(&1_u16.to_le_bytes());
        b[8..12].copy_from_slice(&128_u32.to_le_bytes());
        b[16..24].copy_from_slice(&400_u64.to_le_bytes());
        b[24..32].copy_from_slice(&128_u64.to_le_bytes());
        b[32..40].copy_from_slice(&16_u64.to_le_bytes());
        b[40..48].copy_from_slice(&144_u64.to_le_bytes());
        b[48..52].copy_from_slice(&1_u32.to_le_bytes());
        b[52..56].copy_from_slice(&128_u32.to_le_bytes());
        b[56..64].copy_from_slice(&272_u64.to_le_bytes());
        b[64..68].copy_from_slice(&1_u32.to_le_bytes());
        b[68..72].copy_from_slice(&64_u32.to_le_bytes());
        b[72..80].copy_from_slice(&336_u64.to_le_bytes());
        b[80..88].copy_from_slice(&64_u64.to_le_bytes());
        b
    }

    #[test]
    fn parses_valid_header() {
        let bytes = valid_header();
        let header = Header::parse(&bytes, 400, &ReaderLimits::default()).unwrap();
        assert_eq!(header.tensor_count, 1);
        assert_eq!(header.block_count, 1);
        assert_eq!(header.payload.length, 64);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = valid_header();
        bytes[0] = b'X';
        assert_eq!(
            Header::parse(&bytes, 400, &ReaderLimits::default()),
            Err(FormatError::BadMagic)
        );
    }

    #[test]
    fn rejects_declared_file_size_mismatch() {
        let bytes = valid_header();
        assert!(matches!(
            Header::parse(&bytes, 401, &ReaderLimits::default()),
            Err(FormatError::InvalidHeader(_))
        ));
    }

    #[test]
    fn rejects_nonzero_reserved_bytes() {
        let mut bytes = valid_header();
        bytes[127] = 1;
        assert!(matches!(
            Header::parse(&bytes, 400, &ReaderLimits::default()),
            Err(FormatError::InvalidHeader(_))
        ));
    }
}
