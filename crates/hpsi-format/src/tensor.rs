use crate::{FormatError, Result};

const NONE_REF: u64 = u64::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DType {
    U8 = 1,
    I8 = 2,
    F16 = 3,
    Bf16 = 4,
    F32 = 5,
    F64 = 6,
    I16 = 7,
    I32 = 8,
    I64 = 9,
    PackedU4 = 10,
    PackedI4 = 11,
    PackedU2 = 12,
    PackedI2 = 13,
    PackedU1 = 14,
    PackedI1 = 15,
    Opaque = 16,
}

impl DType {
    pub fn from_id(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::U8),
            2 => Ok(Self::I8),
            3 => Ok(Self::F16),
            4 => Ok(Self::Bf16),
            5 => Ok(Self::F32),
            6 => Ok(Self::F64),
            7 => Ok(Self::I16),
            8 => Ok(Self::I32),
            9 => Ok(Self::I64),
            10 => Ok(Self::PackedU4),
            11 => Ok(Self::PackedI4),
            12 => Ok(Self::PackedU2),
            13 => Ok(Self::PackedI2),
            14 => Ok(Self::PackedU1),
            15 => Ok(Self::PackedI1),
            16 => Ok(Self::Opaque),
            _ => Err(FormatError::InvalidTensor("unknown dtype identifier")),
        }
    }

    fn bits_per_element(self) -> Option<u64> {
        match self {
            Self::U8 | Self::I8 => Some(8),
            Self::F16 | Self::Bf16 | Self::I16 => Some(16),
            Self::F32 | Self::I32 => Some(32),
            Self::F64 | Self::I64 => Some(64),
            Self::PackedU4 | Self::PackedI4 => Some(4),
            Self::PackedU2 | Self::PackedI2 => Some(2),
            Self::PackedU1 | Self::PackedI1 => Some(1),
            Self::Opaque => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    pub tensor_id: u32,
    pub rank: u8,
    pub dtype: DType,
    pub codec: u16,
    pub block_count: u32,
    pub first_block_index: u64,
    pub logical_nbytes: u64,
    pub dimensions: [u64; 8],
    pub name_ref: u32,
    pub representation_ref: Option<u64>,
}

impl TensorDescriptor {
    pub const ENTRY_SIZE: usize = 128;

    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != Self::ENTRY_SIZE {
            return Err(FormatError::Truncated {
                expected: Self::ENTRY_SIZE,
                actual: bytes.len(),
            });
        }
        let rank = bytes[4];
        if rank > 8 {
            return Err(FormatError::InvalidTensor("rank exceeds 8"));
        }
        let dtype = DType::from_id(bytes[5])?;
        let codec = u16_at(bytes, 6);
        let flags = u32_at(bytes, 8);
        let block_count = u32_at(bytes, 12);
        let first_block_index = u64_at(bytes, 16);
        let logical_nbytes = u64_at(bytes, 24);
        let mut dimensions = [0_u64; 8];
        for (index, dimension) in dimensions.iter_mut().enumerate() {
            *dimension = u64_at(bytes, 32 + index * 8);
        }
        let name_ref = u32_at(bytes, 96);
        let reserved0 = u32_at(bytes, 100);
        let representation_raw = u64_at(bytes, 104);

        if flags != 0 || reserved0 != 0 || bytes[112..128].iter().any(|byte| *byte != 0) {
            return Err(FormatError::InvalidTensor(
                "reserved tensor fields must be zero",
            ));
        }
        if codec == 0 || block_count == 0 || logical_nbytes == 0 {
            return Err(FormatError::InvalidTensor(
                "tensor codec, block count, and logical size must be non-zero",
            ));
        }
        if dimensions[..usize::from(rank)].contains(&0) {
            return Err(FormatError::InvalidTensor(
                "active dimensions must be non-zero",
            ));
        }
        if dimensions[usize::from(rank)..]
            .iter()
            .any(|dimension| *dimension != 0)
        {
            return Err(FormatError::InvalidTensor("unused dimensions must be zero"));
        }

        if dtype != DType::Opaque {
            let elements =
                dimensions[..usize::from(rank)]
                    .iter()
                    .try_fold(1_u64, |product, dimension| {
                        product
                            .checked_mul(*dimension)
                            .ok_or(FormatError::ArithmeticOverflow("tensor element count"))
                    })?;
            let bits = elements
                .checked_mul(dtype.bits_per_element().unwrap_or(0))
                .ok_or(FormatError::ArithmeticOverflow("tensor logical bit count"))?;
            let expected_bytes = bits.checked_add(7).ok_or(FormatError::ArithmeticOverflow(
                "packed tensor byte rounding",
            ))? / 8;
            if expected_bytes != logical_nbytes {
                return Err(FormatError::InvalidTensor(
                    "logical_nbytes disagrees with dtype and dimensions",
                ));
            }
        }

        Ok(Self {
            tensor_id: u32_at(bytes, 0),
            rank,
            dtype,
            codec,
            block_count,
            first_block_index,
            logical_nbytes,
            dimensions,
            name_ref,
            representation_ref: (representation_raw != NONE_REF).then_some(representation_raw),
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
            .expect("fixed tensor entry slice"),
    )
}
fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("fixed tensor entry slice"),
    )
}
