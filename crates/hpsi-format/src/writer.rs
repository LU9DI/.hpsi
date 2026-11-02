use crate::{DType, FormatError, Result};
use serde_json::json;
use std::io::{Seek, SeekFrom, Write};

#[derive(Debug, Clone)]
pub struct PayloadBlock {
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct TensorInput {
    pub tensor_id: u32,
    pub name: String,
    pub dtype: DType,
    pub dimensions: Vec<u64>,
    pub blocks: Vec<PayloadBlock>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WriterStats {
    pub metadata_bytes: u64,
    pub directory_bytes: u64,
    pub payload_bytes: u64,
    pub total_bytes: u64,
    pub tensor_count: u32,
    pub block_count: u32,
}

/// Writes the interoperable RAW subset of HPSI v0.1. Payload blocks are
/// emitted sequentially; no second full-model payload buffer is constructed.
pub struct ModelWriter;

impl ModelWriter {
    pub fn write<W: Write + Seek>(out: &mut W, tensors: &[TensorInput]) -> Result<WriterStats> {
        let existing_len = out.seek(SeekFrom::End(0)).map_err(ioerr)?;
        if existing_len != 0 {
            return Err(FormatError::InvalidRegion(
                "writer requires an empty output stream",
            ));
        }
        if tensors.is_empty() || tensors.len() > u32::MAX as usize {
            return Err(FormatError::InvalidTensor(
                "writer requires a valid non-empty tensor set",
            ));
        }
        let mut names = Vec::new();
        let mut name_set = std::collections::HashSet::new();
        let mut previous_id = None;
        let mut block_count = 0usize;
        let mut payload_bytes = 0u64;
        for t in tensors {
            if previous_id.is_some_and(|id| t.tensor_id <= id) {
                return Err(FormatError::InvalidTensor(
                    "tensor IDs must be strictly increasing",
                ));
            }
            previous_id = Some(t.tensor_id);
            if t.name.is_empty()
                || t.name.contains('\0')
                || t.name.len() > 1024
                || !name_set.insert(t.name.as_str())
            {
                return Err(FormatError::InvalidMetadata(
                    "invalid or duplicate tensor name",
                ));
            }
            if t.dimensions.is_empty()
                || t.dimensions.len() > 8
                || t.dimensions.contains(&0)
                || t.blocks.is_empty()
            {
                return Err(FormatError::InvalidTensor(
                    "invalid tensor rank, dimensions, or blocks",
                ));
            }
            block_count = block_count
                .checked_add(t.blocks.len())
                .ok_or(FormatError::ArithmeticOverflow("block count"))?;
            for b in &t.blocks {
                if b.bytes.is_empty() {
                    return Err(FormatError::InvalidBlock("empty payload block"));
                }
                payload_bytes = payload_bytes
                    .checked_add(b.bytes.len() as u64)
                    .ok_or(FormatError::ArithmeticOverflow("payload size"))?;
            }
            names.push(t.name.clone());
        }
        if block_count > u32::MAX as usize {
            return Err(FormatError::LimitExceeded("writer block count"));
        }

        let metadata =
            serde_json::to_vec(&json!({"format":"hpsi","metadata_version":1,"tensor_names":names}))
                .map_err(|e| FormatError::Io(e.to_string()))?;
        let metadata_off = 128u64;
        let tensor_off = align8(metadata_off + metadata.len() as u64)?;
        let tensor_len = tensors.len() as u64 * 128;
        let block_off = align8(
            tensor_off
                .checked_add(tensor_len)
                .ok_or(FormatError::ArithmeticOverflow("tensor directory"))?,
        )?;
        let block_len = block_count as u64 * 64;
        let integrity_off = align8(
            block_off
                .checked_add(block_len)
                .ok_or(FormatError::ArithmeticOverflow("block directory"))?,
        )?;
        let integrity_count = block_count + 1;
        let integrity_len = integrity_count as u64 * 32;
        let payload_off = align8(
            integrity_off
                .checked_add(integrity_len)
                .ok_or(FormatError::ArithmeticOverflow("integrity directory"))?,
        )?;
        let file_size = payload_off
            .checked_add(payload_bytes)
            .ok_or(FormatError::ArithmeticOverflow("file size"))?;

        let mut tensor_dir = Vec::with_capacity(tensors.len() * 128);
        let mut block_dir = Vec::with_capacity(block_count * 64);
        let mut integrity_dir = Vec::with_capacity(integrity_count * 32);
        let mut payload_pos = payload_off;
        let mut global_block = 0u32;
        for (name_ref, t) in tensors.iter().enumerate() {
            let logical = t.blocks.iter().map(|b| b.bytes.len() as u64).sum::<u64>();
            validate_shape(t.dtype, &t.dimensions, logical)?;
            let mut e = [0u8; 128];
            put32(&mut e, 0, t.tensor_id);
            e[4] = t.dimensions.len() as u8;
            e[5] = t.dtype as u8;
            put16(&mut e, 6, 1);
            put32(&mut e, 12, t.blocks.len() as u32);
            put64(&mut e, 16, global_block as u64);
            put64(&mut e, 24, logical);
            for (i, d) in t.dimensions.iter().enumerate() {
                put64(&mut e, 32 + i * 8, *d);
            }
            put32(&mut e, 96, name_ref as u32);
            put64(&mut e, 104, u64::MAX);
            tensor_dir.extend_from_slice(&e);
            let mut logical_pos = 0u64;
            for (block_id, b) in t.blocks.iter().enumerate() {
                let mut be = [0u8; 64];
                put32(&mut be, 0, t.tensor_id);
                put32(&mut be, 4, block_id as u32);
                put64(&mut be, 16, logical_pos);
                put64(&mut be, 24, b.bytes.len() as u64);
                put64(&mut be, 32, payload_pos);
                put64(&mut be, 40, b.bytes.len() as u64);
                put64(&mut be, 48, u64::MAX);
                put32(&mut be, 56, global_block + 1);
                block_dir.extend_from_slice(&be);
                let mut ir = [0u8; 32];
                put32(&mut ir, 0, 2);
                put32(&mut ir, 4, global_block);
                put16(&mut ir, 8, 1);
                put32(&mut ir, 12, crc32c(&b.bytes));
                integrity_dir.extend_from_slice(&ir);
                logical_pos += b.bytes.len() as u64;
                payload_pos += b.bytes.len() as u64;
                global_block += 1;
            }
        }
        let mut mr = [0u8; 32];
        put32(&mut mr, 0, 1);
        put16(&mut mr, 8, 1);
        put32(&mut mr, 12, crc32c(&metadata));
        integrity_dir.splice(0..0, mr);

        let mut h = [0u8; 128];
        h[0..4].copy_from_slice(b"HPSI");
        put16(&mut h, 6, 1);
        put32(&mut h, 8, 128);
        put64(&mut h, 16, file_size);
        put64(&mut h, 24, metadata_off);
        put64(&mut h, 32, metadata.len() as u64);
        put64(&mut h, 40, tensor_off);
        put32(&mut h, 48, tensors.len() as u32);
        put32(&mut h, 52, 128);
        put64(&mut h, 56, block_off);
        put32(&mut h, 64, block_count as u32);
        put32(&mut h, 68, 64);
        put64(&mut h, 72, payload_off);
        put64(&mut h, 80, payload_bytes);
        put64(&mut h, 88, integrity_off);
        put32(&mut h, 96, integrity_count as u32);
        put32(&mut h, 100, 32);
        out.seek(SeekFrom::Start(0)).map_err(ioerr)?;
        out.write_all(&h).map_err(ioerr)?;
        pad_to(out, metadata_off)?;
        out.write_all(&metadata).map_err(ioerr)?;
        pad_to(out, tensor_off)?;
        out.write_all(&tensor_dir).map_err(ioerr)?;
        pad_to(out, block_off)?;
        out.write_all(&block_dir).map_err(ioerr)?;
        pad_to(out, integrity_off)?;
        out.write_all(&integrity_dir).map_err(ioerr)?;
        pad_to(out, payload_off)?;
        for t in tensors {
            for b in &t.blocks {
                out.write_all(&b.bytes).map_err(ioerr)?;
            }
        }
        out.flush().map_err(ioerr)?;
        Ok(WriterStats {
            metadata_bytes: metadata.len() as u64,
            directory_bytes: tensor_len + block_len + integrity_len,
            payload_bytes,
            total_bytes: file_size,
            tensor_count: tensors.len() as u32,
            block_count: block_count as u32,
        })
    }
}

fn validate_shape(dtype: DType, dims: &[u64], bytes: u64) -> Result<()> {
    let bits = match dtype {
        DType::U8 | DType::I8 => 8,
        DType::F16 | DType::Bf16 | DType::I16 => 16,
        DType::F32 | DType::I32 => 32,
        DType::F64 | DType::I64 => 64,
        DType::PackedU4 | DType::PackedI4 => 4,
        DType::PackedU2 | DType::PackedI2 => 2,
        DType::PackedU1 | DType::PackedI1 => 1,
        DType::Opaque => return Ok(()),
    };
    let n = dims.iter().try_fold(1u64, |a, d| {
        a.checked_mul(*d)
            .ok_or(FormatError::ArithmeticOverflow("tensor elements"))
    })?;
    let expected = n
        .checked_mul(bits)
        .and_then(|v| v.checked_add(7))
        .ok_or(FormatError::ArithmeticOverflow("tensor bytes"))?
        / 8;
    if expected != bytes {
        return Err(FormatError::InvalidTensor(
            "payload size disagrees with dtype and dimensions",
        ));
    }
    Ok(())
}
fn pad_to<W: Write + Seek>(w: &mut W, target: u64) -> Result<()> {
    let pos = w.stream_position().map_err(ioerr)?;
    if pos > target {
        return Err(FormatError::InvalidRegion("writer offset exceeded target"));
    }
    let zero = [0u8; 8];
    let mut n = target - pos;
    while n > 0 {
        let k = n.min(8) as usize;
        w.write_all(&zero[..k]).map_err(ioerr)?;
        n -= k as u64;
    }
    Ok(())
}
fn align8(v: u64) -> Result<u64> {
    v.checked_add(7)
        .map(|x| x & !7)
        .ok_or(FormatError::ArithmeticOverflow("alignment"))
}
fn ioerr(e: std::io::Error) -> FormatError {
    FormatError::Io(e.to_string())
}
fn put16(b: &mut [u8], o: usize, v: u16) {
    b[o..o + 2].copy_from_slice(&v.to_le_bytes())
}
fn put32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes())
}
fn put64(b: &mut [u8], o: usize, v: u64) {
    b[o..o + 8].copy_from_slice(&v.to_le_bytes())
}
fn crc32c(b: &[u8]) -> u32 {
    let mut c = !0u32;
    for x in b {
        c ^= u32::from(*x);
        for _ in 0..8 {
            c = if c & 1 == 1 {
                (c >> 1) ^ 0x82f63b78
            } else {
                c >> 1
            };
        }
    }
    !c
}
