use crate::{FormatError, Result};

const NONE_REF: u64 = u64::MAX;
const NONE_CHECKSUM: u32 = u32::MAX;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockDescriptor {
    pub tensor_id: u32,
    pub block_id: u32,
    pub codec: u16,
    pub layout: u16,
    pub logical_offset: u64,
    pub logical_length: u64,
    pub payload_offset: u64,
    pub payload_length: u64,
    pub auxiliary_ref: Option<u64>,
    pub checksum_index: Option<u32>,
}

impl BlockDescriptor {
    pub const ENTRY_SIZE: usize = 64;

    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != Self::ENTRY_SIZE {
            return Err(FormatError::Truncated {
                expected: Self::ENTRY_SIZE,
                actual: bytes.len(),
            });
        }
        let flags = u32_at(bytes, 8);
        let codec = u16_at(bytes, 12);
        let layout = u16_at(bytes, 14);
        let logical_offset = u64_at(bytes, 16);
        let logical_length = u64_at(bytes, 24);
        let payload_offset = u64_at(bytes, 32);
        let payload_length = u64_at(bytes, 40);
        let auxiliary_raw = u64_at(bytes, 48);
        let checksum_raw = u32_at(bytes, 56);
        let reserved = u32_at(bytes, 60);

        if flags != 0 || reserved != 0 || layout != 0 {
            return Err(FormatError::InvalidBlock(
                "unsupported flags, layout, or reserved field",
            ));
        }
        if logical_length == 0 || payload_length == 0 {
            return Err(FormatError::InvalidBlock(
                "logical and payload lengths must be non-zero",
            ));
        }
        if payload_offset.checked_add(payload_length).is_none() {
            return Err(FormatError::ArithmeticOverflow("block payload range"));
        }
        if logical_offset.checked_add(logical_length).is_none() {
            return Err(FormatError::ArithmeticOverflow("block logical range"));
        }

        Ok(Self {
            tensor_id: u32_at(bytes, 0),
            block_id: u32_at(bytes, 4),
            codec,
            layout,
            logical_offset,
            logical_length,
            payload_offset,
            payload_length,
            auxiliary_ref: (auxiliary_raw != NONE_REF).then_some(auxiliary_raw),
            checksum_index: (checksum_raw != NONE_CHECKSUM).then_some(checksum_raw),
        })
    }
}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}
fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("fixed block entry slice"),
    )
}
fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("fixed block entry slice"),
    )
}
