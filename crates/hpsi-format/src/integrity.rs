use crate::{FormatError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntegrityTarget {
    Metadata,
    BlockPayload(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegrityRecord {
    pub target: IntegrityTarget,
    pub crc32c: u32,
}

impl IntegrityRecord {
    pub const ENTRY_SIZE: usize = 32;

    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != Self::ENTRY_SIZE {
            return Err(FormatError::Truncated {
                expected: Self::ENTRY_SIZE,
                actual: bytes.len(),
            });
        }
        let target_kind = u32::from_le_bytes(bytes[0..4].try_into().expect("fixed integrity entry slice"));
        let target_index = u32::from_le_bytes(bytes[4..8].try_into().expect("fixed integrity entry slice"));
        let algorithm = u16::from_le_bytes([bytes[8], bytes[9]]);
        let flags = u16::from_le_bytes([bytes[10], bytes[11]]);
        let crc32c = u32::from_le_bytes(bytes[12..16].try_into().expect("fixed integrity entry slice"));
        let reserved = u32::from_le_bytes(bytes[28..32].try_into().expect("fixed integrity entry slice"));

        if algorithm != 1 || flags != 0 || reserved != 0 || bytes[16..28].iter().any(|byte| *byte != 0) {
            return Err(FormatError::InvalidDirectory("invalid integrity algorithm, flags, or reserved bytes"));
        }
        let target = match target_kind {
            1 if target_index == 0 => IntegrityTarget::Metadata,
            2 => IntegrityTarget::BlockPayload(target_index),
            _ => return Err(FormatError::InvalidReference("invalid integrity target")),
        };
        Ok(Self { target, crc32c })
    }
}
