use crate::{DType, FileSource, FormatError, ModelReader, ReaderLimits, Result};
use serde_json::json;
use std::fs::{self, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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

/// Metadata and exact payload lengths for a tensor written block-by-block.
#[derive(Debug, Clone)]
pub struct StreamingTensorInput {
    pub tensor_id: u32,
    pub name: String,
    pub dtype: DType,
    pub dimensions: Vec<u64>,
    pub block_lengths: Vec<u64>,
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

const MAX_WRITER_TENSORS: usize = 250_000;
const MAX_WRITER_BLOCKS: usize = 1_000_000;
const MAX_WRITER_METADATA_BYTES: usize = 16 * 1024 * 1024;
const MAX_WRITER_DIRECTORY_BYTES: u64 = 256 * 1024 * 1024;
const MAX_WRITER_BLOCK_PAYLOAD_BYTES: u64 = 64 * 1024 * 1024;
const MAX_WRITER_FILE_BYTES: u64 = 1_u64 << 40;
const MAX_WRITER_TENSOR_LOGICAL_BYTES: u64 = 1_u64 << 40;

fn clone_writer_name(name: &str) -> Result<String> {
    let mut cloned = String::new();
    cloned
        .try_reserve_exact(name.len())
        .map_err(|_| FormatError::LimitExceeded("writer tensor name allocation"))?;
    cloned.push_str(name);
    Ok(cloned)
}

impl ModelWriter {
    /// Stream tensor payload blocks directly to the output.
    ///
    /// The layout is planned from exact block lengths before payload emission.
    /// The callback must write exactly the declared number of bytes. Per-block
    /// CRC32C values are computed as bytes pass through and patched into the
    /// integrity directory afterwards.
    pub fn write_streaming<W, F>(
        out: &mut W,
        tensors: &[StreamingTensorInput],
        write_block: F,
    ) -> Result<WriterStats>
    where
        W: Write + Seek,
        F: FnMut(usize, usize, &mut dyn Write) -> Result<u64>,
    {
        Self::write_streaming_with_metadata(out, tensors, None, write_block)
    }

    /// Stream tensor payloads with optional validated model/producer/user
    /// provenance. The tensor-name index is always derived from the actual
    /// directory, preventing metadata from describing a different tensor set.
    pub fn write_streaming_with_metadata<W, F>(
        out: &mut W,
        tensors: &[StreamingTensorInput],
        extra_metadata: Option<serde_json::Value>,
        mut write_block: F,
    ) -> Result<WriterStats>
    where
        W: Write + Seek,
        F: FnMut(usize, usize, &mut dyn Write) -> Result<u64>,
    {
        if out.seek(SeekFrom::End(0)).map_err(ioerr)? != 0 {
            return Err(FormatError::InvalidRegion(
                "writer requires an empty output stream",
            ));
        }
        if tensors.is_empty()
            || tensors.len() > MAX_WRITER_TENSORS
            || tensors.len() > u32::MAX as usize
        {
            return Err(FormatError::InvalidTensor(
                "writer requires a valid non-empty tensor set",
            ));
        }
        let mut names = Vec::new();
        names
            .try_reserve_exact(tensors.len())
            .map_err(|_| FormatError::LimitExceeded("writer tensor names"))?;
        let mut name_set = std::collections::HashSet::new();
        name_set
            .try_reserve(tensors.len())
            .map_err(|_| FormatError::LimitExceeded("writer tensor name index allocation"))?;
        let mut previous_id = None;
        let mut block_count = 0usize;
        let mut payload_bytes = 0u64;
        for tensor in tensors {
            if previous_id.is_some_and(|id| tensor.tensor_id <= id) {
                return Err(FormatError::InvalidTensor(
                    "tensor IDs must be strictly increasing",
                ));
            }
            previous_id = Some(tensor.tensor_id);
            if tensor.name.is_empty()
                || tensor.name.contains('\0')
                || tensor.name.len() > 1024
                || !name_set.insert(tensor.name.as_str())
            {
                return Err(FormatError::InvalidMetadata(
                    "invalid or duplicate tensor name",
                ));
            }
            if tensor.dimensions.is_empty()
                || tensor.dimensions.len() > 8
                || tensor.dimensions.contains(&0)
                || tensor.block_lengths.is_empty()
                || tensor.block_lengths.iter().any(|length| *length == 0)
            {
                return Err(FormatError::InvalidTensor(
                    "invalid tensor rank, dimensions, or blocks",
                ));
            }
            block_count = block_count
                .checked_add(tensor.block_lengths.len())
                .ok_or(FormatError::ArithmeticOverflow("block count"))?;
            let tensor_logical_bytes =
                tensor.block_lengths.iter().try_fold(0_u64, |sum, length| {
                    sum.checked_add(*length)
                        .ok_or(FormatError::ArithmeticOverflow("tensor logical bytes"))
                })?;
            if tensor_logical_bytes > MAX_WRITER_TENSOR_LOGICAL_BYTES {
                return Err(FormatError::LimitExceeded("writer tensor logical bytes"));
            }
            for length in &tensor.block_lengths {
                if *length > MAX_WRITER_BLOCK_PAYLOAD_BYTES {
                    return Err(FormatError::LimitExceeded("writer payload bytes per block"));
                }
                payload_bytes = payload_bytes
                    .checked_add(*length)
                    .ok_or(FormatError::ArithmeticOverflow("payload size"))?;
            }
            names.push(clone_writer_name(&tensor.name)?);
        }
        if block_count > MAX_WRITER_BLOCKS || block_count > u32::MAX as usize {
            return Err(FormatError::LimitExceeded("writer block count"));
        }

        let mut metadata_value = extra_metadata.unwrap_or_else(|| json!({}));
        let metadata_object =
            metadata_value
                .as_object_mut()
                .ok_or(FormatError::InvalidMetadata(
                    "writer metadata must be a JSON object",
                ))?;
        for key in metadata_object.keys() {
            if !matches!(key.as_str(), "model" | "producer" | "user") {
                return Err(FormatError::InvalidMetadata(
                    "custom writer metadata may contain only model, producer, and user",
                ));
            }
        }
        metadata_object.insert("format".into(), json!("hpsi"));
        metadata_object.insert("metadata_version".into(), json!(1));
        metadata_object.insert("tensor_names".into(), json!(names));
        let metadata = serde_json::to_vec(&metadata_value)
            .map_err(|error| FormatError::Io(error.to_string()))?;
        if metadata.len() > MAX_WRITER_METADATA_BYTES {
            return Err(FormatError::LimitExceeded("writer metadata bytes"));
        }
        crate::Metadata::parse(&metadata, &ReaderLimits::default())?;
        let metadata_off = 128_u64;
        let tensor_off = align8(
            metadata_off
                .checked_add(metadata.len() as u64)
                .ok_or(FormatError::ArithmeticOverflow("metadata end"))?,
        )?;
        let tensor_len = (tensors.len() as u64)
            .checked_mul(128)
            .ok_or(FormatError::ArithmeticOverflow("tensor directory length"))?;
        let block_off = align8(
            tensor_off
                .checked_add(tensor_len)
                .ok_or(FormatError::ArithmeticOverflow("tensor directory"))?,
        )?;
        let block_len = (block_count as u64)
            .checked_mul(64)
            .ok_or(FormatError::ArithmeticOverflow("block directory length"))?;
        let integrity_off = align8(
            block_off
                .checked_add(block_len)
                .ok_or(FormatError::ArithmeticOverflow("block directory"))?,
        )?;
        let integrity_count = block_count
            .checked_add(1)
            .ok_or(FormatError::ArithmeticOverflow("integrity count"))?;
        if integrity_count > u32::MAX as usize {
            return Err(FormatError::LimitExceeded("writer integrity record count"));
        }
        let integrity_len =
            (integrity_count as u64)
                .checked_mul(32)
                .ok_or(FormatError::ArithmeticOverflow(
                    "integrity directory length",
                ))?;
        let total_directory_bytes = tensor_len
            .checked_add(block_len)
            .and_then(|v| v.checked_add(integrity_len))
            .ok_or(FormatError::ArithmeticOverflow(
                "total writer directory length",
            ))?;
        if total_directory_bytes > MAX_WRITER_DIRECTORY_BYTES {
            return Err(FormatError::LimitExceeded("writer total directory bytes"));
        }
        let payload_off = align8(
            integrity_off
                .checked_add(integrity_len)
                .ok_or(FormatError::ArithmeticOverflow("integrity directory"))?,
        )?;
        let file_size = payload_off
            .checked_add(payload_bytes)
            .ok_or(FormatError::ArithmeticOverflow("file size"))?;
        if file_size > MAX_WRITER_FILE_BYTES {
            return Err(FormatError::LimitExceeded("writer maximum file size"));
        }

        let mut tensor_dir = Vec::new();
        tensor_dir
            .try_reserve_exact(
                usize::try_from(tensor_len)
                    .map_err(|_| FormatError::LimitExceeded("tensor directory allocation"))?,
            )
            .map_err(|_| FormatError::LimitExceeded("tensor directory allocation"))?;
        let mut block_dir = Vec::new();
        block_dir
            .try_reserve_exact(
                usize::try_from(block_len)
                    .map_err(|_| FormatError::LimitExceeded("block directory allocation"))?,
            )
            .map_err(|_| FormatError::LimitExceeded("block directory allocation"))?;

        let mut payload_pos = payload_off;
        let mut global_block = 0_u32;
        for (name_ref, tensor) in tensors.iter().enumerate() {
            let logical = tensor.block_lengths.iter().try_fold(0_u64, |sum, length| {
                sum.checked_add(*length)
                    .ok_or(FormatError::ArithmeticOverflow("tensor logical bytes"))
            })?;
            validate_shape(tensor.dtype, &tensor.dimensions, logical)?;
            let mut entry = [0_u8; 128];
            put32(&mut entry, 0, tensor.tensor_id);
            entry[4] = tensor.dimensions.len() as u8;
            entry[5] = tensor.dtype as u8;
            put16(&mut entry, 6, 1);
            put32(&mut entry, 12, tensor.block_lengths.len() as u32);
            put64(&mut entry, 16, u64::from(global_block));
            put64(&mut entry, 24, logical);
            for (index, dimension) in tensor.dimensions.iter().enumerate() {
                put64(&mut entry, 32 + index * 8, *dimension);
            }
            put32(&mut entry, 96, name_ref as u32);
            put64(&mut entry, 104, u64::MAX);
            tensor_dir.extend_from_slice(&entry);

            let mut logical_pos = 0_u64;
            for (block_id, length) in tensor.block_lengths.iter().enumerate() {
                let mut entry = [0_u8; 64];
                put32(&mut entry, 0, tensor.tensor_id);
                put32(&mut entry, 4, block_id as u32);
                put64(&mut entry, 16, logical_pos);
                put64(&mut entry, 24, *length);
                put64(&mut entry, 32, payload_pos);
                put64(&mut entry, 40, *length);
                put64(&mut entry, 48, u64::MAX);
                put32(&mut entry, 56, global_block + 1);
                block_dir.extend_from_slice(&entry);
                logical_pos = logical_pos
                    .checked_add(*length)
                    .ok_or(FormatError::ArithmeticOverflow("logical block offset"))?;
                payload_pos = payload_pos
                    .checked_add(*length)
                    .ok_or(FormatError::ArithmeticOverflow("payload offset"))?;
                global_block += 1;
            }
        }

        let mut header = [0_u8; 128];
        header[0..4].copy_from_slice(b"HPSI");
        put16(&mut header, 6, 1);
        put32(&mut header, 8, 128);
        put64(&mut header, 16, file_size);
        put64(&mut header, 24, metadata_off);
        put64(&mut header, 32, metadata.len() as u64);
        put64(&mut header, 40, tensor_off);
        put32(&mut header, 48, tensors.len() as u32);
        put32(&mut header, 52, 128);
        put64(&mut header, 56, block_off);
        put32(&mut header, 64, block_count as u32);
        put32(&mut header, 68, 64);
        put64(&mut header, 72, payload_off);
        put64(&mut header, 80, payload_bytes);
        put64(&mut header, 88, integrity_off);
        put32(&mut header, 96, integrity_count as u32);
        put32(&mut header, 100, 32);

        out.seek(SeekFrom::Start(0)).map_err(ioerr)?;
        out.write_all(&header).map_err(ioerr)?;
        pad_to(out, metadata_off)?;
        out.write_all(&metadata).map_err(ioerr)?;
        pad_to(out, tensor_off)?;
        out.write_all(&tensor_dir).map_err(ioerr)?;
        pad_to(out, block_off)?;
        out.write_all(&block_dir).map_err(ioerr)?;
        pad_to(out, integrity_off)?;
        let mut metadata_record = [0_u8; 32];
        put32(&mut metadata_record, 0, 1);
        put16(&mut metadata_record, 8, 1);
        put32(&mut metadata_record, 12, crc32c(&metadata));
        out.write_all(&metadata_record).map_err(ioerr)?;
        let zeroes = [0_u8; 64 * 1024];
        let mut remaining_integrity =
            integrity_len
                .checked_sub(32)
                .ok_or(FormatError::ArithmeticOverflow(
                    "integrity directory remainder",
                ))?;
        while remaining_integrity > 0 {
            let count = remaining_integrity.min(zeroes.len() as u64) as usize;
            out.write_all(&zeroes[..count]).map_err(ioerr)?;
            remaining_integrity -= count as u64;
        }
        pad_to(out, payload_off)?;

        let mut integrity_index = 1_u64;
        for (tensor_index, tensor) in tensors.iter().enumerate() {
            for (block_index, expected_len) in tensor.block_lengths.iter().enumerate() {
                let mut crc_writer = Crc32cWriter::new(out, *expected_len);
                let written = write_block(tensor_index, block_index, &mut crc_writer)?;
                if written != *expected_len || crc_writer.bytes_written != *expected_len {
                    return Err(FormatError::InvalidBlock(
                        "stream callback wrote a different block length than declared",
                    ));
                }
                let mut record = [0_u8; 32];
                put32(&mut record, 0, 2);
                put32(&mut record, 4, (integrity_index - 1) as u32);
                put16(&mut record, 8, 1);
                put32(&mut record, 12, crc_writer.finish());
                let record_offset = integrity_off
                    .checked_add(integrity_index * 32)
                    .ok_or(FormatError::ArithmeticOverflow("integrity record offset"))?;
                out.seek(SeekFrom::Start(record_offset)).map_err(ioerr)?;
                out.write_all(&record).map_err(ioerr)?;
                out.seek(SeekFrom::End(0)).map_err(ioerr)?;
                integrity_index += 1;
            }
        }
        if out.seek(SeekFrom::End(0)).map_err(ioerr)? != file_size {
            return Err(FormatError::InvalidRegion(
                "streamed payload length disagrees with planned file size",
            ));
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

    pub fn write<W: Write + Seek>(out: &mut W, tensors: &[TensorInput]) -> Result<WriterStats> {
        let existing_len = out.seek(SeekFrom::End(0)).map_err(ioerr)?;
        if existing_len != 0 {
            return Err(FormatError::InvalidRegion(
                "writer requires an empty output stream",
            ));
        }
        if tensors.is_empty()
            || tensors.len() > MAX_WRITER_TENSORS
            || tensors.len() > u32::MAX as usize
        {
            return Err(FormatError::InvalidTensor(
                "writer requires a valid non-empty tensor set",
            ));
        }
        let mut names = Vec::new();
        names
            .try_reserve_exact(tensors.len())
            .map_err(|_| FormatError::LimitExceeded("writer tensor names"))?;
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
            let tensor_logical_bytes = t.blocks.iter().try_fold(0_u64, |sum, block| {
                sum.checked_add(block.bytes.len() as u64)
                    .ok_or(FormatError::ArithmeticOverflow("tensor logical bytes"))
            })?;
            if tensor_logical_bytes > MAX_WRITER_TENSOR_LOGICAL_BYTES {
                return Err(FormatError::LimitExceeded("writer tensor logical bytes"));
            }
            for b in &t.blocks {
                if b.bytes.is_empty() {
                    return Err(FormatError::InvalidBlock("empty payload block"));
                }
                if b.bytes.len() as u64 > MAX_WRITER_BLOCK_PAYLOAD_BYTES {
                    return Err(FormatError::LimitExceeded("writer payload bytes per block"));
                }
                payload_bytes = payload_bytes
                    .checked_add(b.bytes.len() as u64)
                    .ok_or(FormatError::ArithmeticOverflow("payload size"))?;
            }
            names.push(clone_writer_name(&t.name)?);
        }
        if block_count > MAX_WRITER_BLOCKS || block_count > u32::MAX as usize {
            return Err(FormatError::LimitExceeded("writer block count"));
        }

        let metadata =
            serde_json::to_vec(&json!({"format":"hpsi","metadata_version":1,"tensor_names":names}))
                .map_err(|e| FormatError::Io(e.to_string()))?;
        if metadata.len() > MAX_WRITER_METADATA_BYTES {
            return Err(FormatError::LimitExceeded("writer metadata bytes"));
        }
        let metadata_off = 128u64;
        let metadata_end = metadata_off
            .checked_add(metadata.len() as u64)
            .ok_or(FormatError::ArithmeticOverflow("metadata end"))?;
        let tensor_off = align8(metadata_end)?;
        let tensor_len = u64::try_from(tensors.len())
            .map_err(|_| FormatError::LimitExceeded("tensor directory length"))?
            .checked_mul(128)
            .ok_or(FormatError::ArithmeticOverflow("tensor directory length"))?;
        let block_off = align8(
            tensor_off
                .checked_add(tensor_len)
                .ok_or(FormatError::ArithmeticOverflow("tensor directory"))?,
        )?;
        let block_len = u64::try_from(block_count)
            .map_err(|_| FormatError::LimitExceeded("block directory length"))?
            .checked_mul(64)
            .ok_or(FormatError::ArithmeticOverflow("block directory length"))?;
        let integrity_off = align8(
            block_off
                .checked_add(block_len)
                .ok_or(FormatError::ArithmeticOverflow("block directory"))?,
        )?;
        let integrity_count = block_count
            .checked_add(1)
            .ok_or(FormatError::ArithmeticOverflow("integrity count"))?;
        if integrity_count > u32::MAX as usize {
            return Err(FormatError::LimitExceeded("writer integrity record count"));
        }
        let integrity_len = u64::try_from(integrity_count)
            .map_err(|_| FormatError::LimitExceeded("integrity directory length"))?
            .checked_mul(32)
            .ok_or(FormatError::ArithmeticOverflow(
                "integrity directory length",
            ))?;
        let payload_off = align8(
            integrity_off
                .checked_add(integrity_len)
                .ok_or(FormatError::ArithmeticOverflow("integrity directory"))?,
        )?;
        let total_directory_bytes = tensor_len
            .checked_add(block_len)
            .and_then(|v| v.checked_add(integrity_len))
            .ok_or(FormatError::ArithmeticOverflow(
                "total writer directory length",
            ))?;
        if total_directory_bytes > MAX_WRITER_DIRECTORY_BYTES {
            return Err(FormatError::LimitExceeded("writer total directory bytes"));
        }
        let file_size = payload_off
            .checked_add(payload_bytes)
            .ok_or(FormatError::ArithmeticOverflow("file size"))?;
        if file_size > MAX_WRITER_FILE_BYTES {
            return Err(FormatError::LimitExceeded("writer maximum file size"));
        }

        let mut tensor_dir = Vec::new();
        tensor_dir
            .try_reserve_exact(
                usize::try_from(tensor_len)
                    .map_err(|_| FormatError::LimitExceeded("tensor directory allocation"))?,
            )
            .map_err(|_| FormatError::LimitExceeded("tensor directory allocation"))?;
        let mut block_dir = Vec::new();
        block_dir
            .try_reserve_exact(
                usize::try_from(block_len)
                    .map_err(|_| FormatError::LimitExceeded("block directory allocation"))?,
            )
            .map_err(|_| FormatError::LimitExceeded("block directory allocation"))?;
        let mut integrity_dir = Vec::new();
        integrity_dir
            .try_reserve_exact(
                usize::try_from(integrity_len)
                    .map_err(|_| FormatError::LimitExceeded("integrity directory allocation"))?,
            )
            .map_err(|_| FormatError::LimitExceeded("integrity directory allocation"))?;
        let mut payload_pos = payload_off;
        let mut global_block = 0u32;
        for (name_ref, t) in tensors.iter().enumerate() {
            let logical = t.blocks.iter().try_fold(0u64, |sum, block| {
                sum.checked_add(block.bytes.len() as u64)
                    .ok_or(FormatError::ArithmeticOverflow("tensor logical bytes"))
            })?;
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
                logical_pos = logical_pos
                    .checked_add(b.bytes.len() as u64)
                    .ok_or(FormatError::ArithmeticOverflow("logical block offset"))?;
                payload_pos = payload_pos
                    .checked_add(b.bytes.len() as u64)
                    .ok_or(FormatError::ArithmeticOverflow("payload offset"))?;
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
struct Crc32cWriter<'a, W: Write> {
    inner: &'a mut W,
    state: u32,
    bytes_written: u64,
    limit: u64,
}

impl<'a, W: Write> Crc32cWriter<'a, W> {
    fn new(inner: &'a mut W, limit: u64) -> Self {
        Self {
            inner,
            state: !0_u32,
            bytes_written: 0,
            limit,
        }
    }

    fn finish(self) -> u32 {
        !self.state
    }
}

impl<W: Write> Write for Crc32cWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let remaining = self.limit.saturating_sub(self.bytes_written);
        if remaining == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "streaming block exceeded its declared length",
            ));
        }
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > remaining {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "streaming write exceeds its declared block length",
            ));
        }
        let written = self.inner.write(bytes)?;
        for byte in &bytes[..written] {
            self.state ^= u32::from(*byte);
            for _ in 0..8 {
                self.state = if self.state & 1 == 1 {
                    (self.state >> 1) ^ 0x82f63b78
                } else {
                    self.state >> 1
                };
            }
        }
        self.bytes_written = self.bytes_written.saturating_add(written as u64);
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
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

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn open_temporary_file(path: &Path) -> std::io::Result<std::fs::File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    options.open(path)
}

struct TemporaryFileGuard {
    path: std::path::PathBuf,
}

impl Drop for TemporaryFileGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn create_temporary_file(parent: &Path) -> Result<(std::path::PathBuf, std::fs::File)> {
    for _ in 0..128 {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(".hpsi-{}-{sequence}.tmp", std::process::id()));
        match open_temporary_file(&candidate) {
            Ok(file) => return Ok((candidate, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(ioerr(error)),
        }
    }
    Err(FormatError::Io(
        "could not allocate a unique temporary output file".into(),
    ))
}

impl ModelWriter {
    /// Stream, sync, reopen, and verify an artifact before publication.
    /// Existing destinations are never overwritten. A parent-directory sync failure
    /// after linking returns `PublicationDurabilityUncertain`; the destination may
    /// already be visible, but crash durability is not confirmed.
    ///
    /// The parent path is resolved by the operating system as supplied. This API
    /// does not pin the parent directory against concurrent rename or symlink
    /// substitution; callers requiring protection from a hostile directory namespace
    /// must provide a trusted, non-writable parent directory. On Unix, temporary
    /// files are created with mode 0600; on other platforms they inherit directory
    /// ACL/default permissions.
    pub fn write_path_streaming<F>(
        path: impl AsRef<Path>,
        tensors: &[StreamingTensorInput],
        mut write_block: F,
    ) -> Result<WriterStats>
    where
        F: FnMut(usize, usize, &mut dyn Write) -> Result<u64>,
    {
        let destination = path.as_ref();
        if destination.file_name().is_none() {
            return Err(FormatError::InvalidRegion("destination must name a file"));
        }
        let parent = destination
            .parent()
            .filter(|value| !value.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let (temporary_path, temporary_file) = create_temporary_file(parent)?;
        let _temporary_file_guard = TemporaryFileGuard {
            path: temporary_path.clone(),
        };
        let result = (|| {
            let mut file = temporary_file;
            let stats = Self::write_streaming(&mut file, tensors, |tensor, block, output| {
                write_block(tensor, block, output)
            })?;
            file.sync_all().map_err(ioerr)?;
            drop(file);
            let source = Arc::new(FileSource::open(&temporary_path)?);
            let reader = ModelReader::open(source, &ReaderLimits::default())?;
            reader.verify_all()?;
            fs::hard_link(&temporary_path, destination).map_err(ioerr)?;
            if let Err(error) = sync_parent_directory(parent) {
                return Err(FormatError::PublicationDurabilityUncertain {
                    path: destination.display().to_string(),
                    reason: error.to_string(),
                });
            }
            Ok(stats)
        })();
        result
    }

    /// Write, sync, reopen, and verify an artifact before publishing it.
    ///
    /// Publication uses a same-directory hard link and never overwrites an existing
    /// destination. Filesystems that do not support hard links return an I/O error.
    /// Parent-path resolution is not pinned against concurrent namespace changes;
    /// use a trusted parent directory when the namespace may be modified by an
    /// untrusted actor. Temporary-file permissions follow Unix mode 0600 or the
    /// platform's inherited directory ACLs. A parent-directory sync failure after linking returns
    /// `PublicationDurabilityUncertain`; the destination may already be visible.
    pub fn write_path(path: impl AsRef<Path>, tensors: &[TensorInput]) -> Result<WriterStats> {
        let destination = path.as_ref();
        if destination.file_name().is_none() {
            return Err(FormatError::InvalidRegion("destination must name a file"));
        }
        let parent = destination
            .parent()
            .filter(|value| !value.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));

        let (temporary_path, temporary_file) = create_temporary_file(parent)?;
        let _temporary_file_guard = TemporaryFileGuard {
            path: temporary_path.clone(),
        };

        let result = (|| {
            let mut file = temporary_file;
            let stats = Self::write(&mut file, tensors)?;
            file.sync_all().map_err(ioerr)?;
            drop(file);

            let source = Arc::new(FileSource::open(&temporary_path)?);
            let reader = ModelReader::open(source, &ReaderLimits::default())?;
            reader.verify_all()?;

            fs::hard_link(&temporary_path, destination).map_err(ioerr)?;
            if let Err(error) = sync_parent_directory(parent) {
                return Err(FormatError::PublicationDurabilityUncertain {
                    path: destination.display().to_string(),
                    reason: error.to_string(),
                });
            }
            Ok(stats)
        })();

        result
    }
}

#[cfg(unix)]
fn sync_parent_directory(parent: &Path) -> Result<()> {
    std::fs::File::open(parent)
        .map_err(ioerr)?
        .sync_all()
        .map_err(ioerr)
}

#[cfg(not(unix))]
fn sync_parent_directory(_parent: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod streaming_writer_tests {
    use super::*;
    use crate::MemorySource;
    use std::io::Cursor;

    fn test_output_path(label: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "hpsi-writer-test-{}-{sequence}-{label}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).unwrap();
        let destination = directory.join("model.hpsi");
        (directory, destination)
    }

    #[test]
    fn crc_writer_rejects_a_single_write_that_exceeds_remaining_length() {
        let mut output = Cursor::new(Vec::new());
        let mut writer = Crc32cWriter::new(&mut output, 1);

        let error = writer.write(&[1, 2]).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(writer.bytes_written, 0);
        assert!(output.get_ref().is_empty());
    }

    fn one_byte_streaming_tensor() -> Vec<StreamingTensorInput> {
        vec![StreamingTensorInput {
            tensor_id: 1,
            name: "payload".into(),
            dtype: DType::U8,
            dimensions: vec![1],
            block_lengths: vec![1],
        }]
    }

    #[cfg(unix)]
    #[test]
    fn path_streaming_temporary_file_is_private() {
        use std::os::unix::fs::PermissionsExt;

        let (directory, destination) = test_output_path("private-temp");
        let result = ModelWriter::write_path_streaming(
            &destination,
            &one_byte_streaming_tensor(),
            |_, _, writer| {
                writer.write_all(&[42]).map_err(ioerr)?;
                Ok(1)
            },
        );
        assert!(result.is_ok());
        assert_eq!(
            std::fs::metadata(&destination)
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        std::fs::remove_file(&destination).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn path_streaming_unwind_does_not_leave_temporary_file() {
        let (directory, destination) = test_output_path("callback-unwind");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = ModelWriter::write_path_streaming(
                &destination,
                &one_byte_streaming_tensor(),
                |_, _, _| -> Result<u64> {
                    std::panic::resume_unwind(Box::new("injected unwind"));
                },
            );
        }));

        assert!(result.is_err());
        assert!(!destination.exists());
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 0);
        std::fs::remove_dir(directory).unwrap();
    }

    fn one_byte_buffered_tensor() -> Vec<TensorInput> {
        vec![TensorInput {
            tensor_id: 1,
            name: "payload".into(),
            dtype: DType::U8,
            dimensions: vec![1],
            blocks: vec![PayloadBlock { bytes: vec![42] }],
        }]
    }

    #[test]
    fn buffered_and_streaming_path_writers_produce_equivalent_artifacts() {
        let (buffered_directory, buffered_destination) =
            test_output_path("buffered-streaming-parity-a");
        let (streaming_directory, streaming_destination) =
            test_output_path("buffered-streaming-parity-b");

        let buffered_stats =
            ModelWriter::write_path(&buffered_destination, &one_byte_buffered_tensor()).unwrap();
        let streaming_stats = ModelWriter::write_path_streaming(
            &streaming_destination,
            &one_byte_streaming_tensor(),
            |_, _, writer| {
                writer.write_all(&[42]).map_err(ioerr)?;
                Ok(1)
            },
        )
        .unwrap();

        assert_eq!(buffered_stats, streaming_stats);

        let buffered = ModelReader::open(
            Arc::new(FileSource::open(&buffered_destination).unwrap()),
            &ReaderLimits::default(),
        )
        .unwrap();
        let streaming = ModelReader::open(
            Arc::new(FileSource::open(&streaming_destination).unwrap()),
            &ReaderLimits::default(),
        )
        .unwrap();

        buffered.verify_all().unwrap();
        streaming.verify_all().unwrap();
        assert_eq!(buffered.metadata.value, streaming.metadata.value);
        assert_eq!(
            buffered.metadata.tensor_names,
            streaming.metadata.tensor_names
        );
        assert_eq!(buffered.header.tensor_count, streaming.header.tensor_count);
        assert_eq!(buffered.header.block_count, streaming.header.block_count);
        assert_eq!(
            buffered.read_block(1, 0).unwrap().payload,
            streaming.read_block(1, 0).unwrap().payload
        );

        std::fs::remove_file(&buffered_destination).unwrap();
        std::fs::remove_file(&streaming_destination).unwrap();
        std::fs::remove_dir(buffered_directory).unwrap();
        std::fs::remove_dir(streaming_directory).unwrap();
    }

    #[test]
    fn path_buffered_roundtrips_and_verifies_block_integrity() {
        let (directory, destination) = test_output_path("buffered-roundtrip");
        let stats = ModelWriter::write_path(&destination, &one_byte_buffered_tensor()).unwrap();
        assert_eq!(stats.payload_bytes, 1);
        assert_eq!(stats.tensor_count, 1);
        assert_eq!(stats.block_count, 1);

        let source = Arc::new(FileSource::open(&destination).unwrap());
        let reader = ModelReader::open(source, &ReaderLimits::default()).unwrap();
        reader.verify_all().unwrap();
        assert_eq!(reader.read_block(1, 0).unwrap().payload, [42]);

        std::fs::remove_file(&destination).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn path_buffered_failure_does_not_publish_or_leave_temporary_file() {
        let (directory, destination) = test_output_path("buffered-invalid-input");
        let invalid = vec![TensorInput {
            tensor_id: 1,
            name: "payload".into(),
            dtype: DType::U8,
            dimensions: vec![1],
            blocks: Vec::new(),
        }];

        let result = ModelWriter::write_path(&destination, &invalid);
        assert!(result.is_err());
        assert!(!destination.exists());
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 0);
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn path_buffered_does_not_overwrite_existing_destination() {
        let (directory, destination) = test_output_path("buffered-existing-destination");
        std::fs::write(&destination, b"keep original").unwrap();

        let result = ModelWriter::write_path(&destination, &one_byte_buffered_tensor());
        assert!(result.is_err());
        assert_eq!(std::fs::read(&destination).unwrap(), b"keep original");
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);

        std::fs::remove_file(&destination).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn path_streaming_callback_failure_does_not_publish_or_leave_temporary_file() {
        let (directory, destination) = test_output_path("callback-failure");
        let result = ModelWriter::write_path_streaming(
            &destination,
            &one_byte_streaming_tensor(),
            |_, _, _| Err(FormatError::InvalidBlock("injected callback failure")),
        );
        assert!(matches!(result, Err(FormatError::InvalidBlock(_))));
        assert!(!destination.exists());
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 0);
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn concurrent_path_streaming_writers_never_overwrite_destination() {
        use std::sync::{Arc, Barrier};

        let (directory, destination) = test_output_path("concurrent-publish");
        let barrier = Arc::new(Barrier::new(2));
        let mut workers = Vec::new();

        for byte in [17_u8, 34_u8] {
            let path = destination.clone();
            let barrier = Arc::clone(&barrier);
            workers.push(std::thread::spawn(move || {
                ModelWriter::write_path_streaming(
                    path,
                    &one_byte_streaming_tensor(),
                    move |_, _, writer| {
                        writer.write_all(&[byte]).map_err(ioerr)?;
                        barrier.wait();
                        Ok(1)
                    },
                )
            }));
        }

        let results: Vec<_> = workers
            .into_iter()
            .map(|worker| worker.join().expect("writer thread panicked"))
            .collect();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(results.iter().filter(|result| result.is_err()).count(), 1);

        let source = Arc::new(FileSource::open(&destination).unwrap());
        let reader = ModelReader::open(source, &ReaderLimits::default()).unwrap();
        reader.verify_all().unwrap();
        let payload = reader.read_block(1, 0).unwrap().payload;
        assert!(payload == [17] || payload == [34]);

        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
        std::fs::remove_file(&destination).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn temporary_file_creation_does_not_follow_symlinks() {
        use std::os::unix::fs::symlink;

        let (directory, _) = test_output_path("temporary-symlink");
        let target = directory.join("target");
        let candidate = directory.join("candidate.tmp");
        std::fs::write(&target, b"preserve target").unwrap();
        symlink(&target, &candidate).unwrap();

        let result = open_temporary_file(&candidate);
        assert_eq!(
            result.unwrap_err().kind(),
            std::io::ErrorKind::AlreadyExists
        );
        assert_eq!(std::fs::read(&target).unwrap(), b"preserve target");
        assert!(
            std::fs::symlink_metadata(&candidate)
                .unwrap()
                .file_type()
                .is_symlink()
        );

        std::fs::remove_file(&candidate).unwrap();
        std::fs::remove_file(&target).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn path_streaming_does_not_replace_symlink_destination_or_target() {
        use std::os::unix::fs::symlink;

        let (directory, destination) = test_output_path("symlink-destination");
        let target = directory.join("target");
        std::fs::write(&target, b"preserve target").unwrap();
        symlink(&target, &destination).unwrap();

        let result = ModelWriter::write_path_streaming(
            &destination,
            &one_byte_streaming_tensor(),
            |_, _, writer| {
                writer.write_all(&[42]).map_err(ioerr)?;
                Ok(1)
            },
        );

        assert!(result.is_err());
        assert_eq!(std::fs::read(&target).unwrap(), b"preserve target");
        assert!(
            std::fs::symlink_metadata(&destination)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 2);

        std::fs::remove_file(&destination).unwrap();
        std::fs::remove_file(&target).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn path_streaming_does_not_overwrite_existing_destination() {
        let (directory, destination) = test_output_path("existing-destination");
        std::fs::write(&destination, b"keep original").unwrap();
        let result = ModelWriter::write_path_streaming(
            &destination,
            &one_byte_streaming_tensor(),
            |_, _, writer| {
                writer.write_all(&[42]).map_err(ioerr)?;
                Ok(1)
            },
        );
        assert!(result.is_err());
        assert_eq!(std::fs::read(&destination).unwrap(), b"keep original");
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
        std::fs::remove_file(&destination).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn streaming_writer_roundtrips_and_verifies_block_integrity() {
        let tensors = vec![StreamingTensorInput {
            tensor_id: 7,
            name: "weights".to_owned(),
            dtype: DType::U8,
            dimensions: vec![6],
            block_lengths: vec![2, 4],
        }];
        let payloads = [vec![1_u8, 2], vec![3_u8, 4, 5, 6]];
        let mut output = Cursor::new(Vec::new());
        let stats = ModelWriter::write_streaming(
            &mut output,
            &tensors,
            |tensor_index, block_index, writer| {
                assert_eq!(tensor_index, 0);
                writer.write_all(&payloads[block_index]).map_err(ioerr)?;
                Ok(payloads[block_index].len() as u64)
            },
        )
        .unwrap();
        assert_eq!(stats.payload_bytes, 6);
        assert_eq!(stats.block_count, 2);
        let reader = ModelReader::open(
            Arc::new(MemorySource::new(Arc::<[u8]>::from(output.into_inner()))),
            &ReaderLimits::default(),
        )
        .unwrap();
        reader.verify_all().unwrap();
        let mut decoded = Vec::new();
        for block_id in 0..2 {
            decoded.extend_from_slice(&reader.read_block(7, block_id).unwrap().payload);
        }
        assert_eq!(decoded, [1_u8, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn streaming_writer_roundtrips_validated_provenance_metadata() {
        let tensors = vec![StreamingTensorInput {
            tensor_id: 3,
            name: "layer.weight".into(),
            dtype: DType::U8,
            dimensions: vec![2],
            block_lengths: vec![2],
        }];
        let mut output = Cursor::new(Vec::new());
        ModelWriter::write_streaming_with_metadata(
            &mut output,
            &tensors,
            Some(json!({
                "producer": {"name": "hpsi-import", "version": "0.1"},
                "model": {"source_format": "safetensors", "revision": "abc123"},
                "user": {"source_sha256": "deadbeef"}
            })),
            |_, _, writer| {
                writer.write_all(&[7, 9]).map_err(ioerr)?;
                Ok(2)
            },
        )
        .unwrap();
        let reader = ModelReader::open(
            Arc::new(MemorySource::new(Arc::<[u8]>::from(output.into_inner()))),
            &ReaderLimits::default(),
        )
        .unwrap();
        reader.verify_all().unwrap();
        assert_eq!(reader.metadata.value["producer"]["name"], "hpsi-import");
        assert_eq!(
            reader.metadata.value["model"]["source_format"],
            "safetensors"
        );
        assert_eq!(reader.metadata.value["user"]["source_sha256"], "deadbeef");
        assert_eq!(reader.metadata.tensor_names, vec!["layer.weight"]);
    }

    #[test]
    fn streaming_writer_rejects_unrecognized_provenance_fields() {
        let tensors = vec![StreamingTensorInput {
            tensor_id: 1,
            name: "payload".into(),
            dtype: DType::U8,
            dimensions: vec![1],
            block_lengths: vec![1],
        }];
        let mut output = Cursor::new(Vec::new());
        let result = ModelWriter::write_streaming_with_metadata(
            &mut output,
            &tensors,
            Some(json!({"arbitrary": true})),
            |_, _, writer| {
                writer.write_all(&[1]).map_err(ioerr)?;
                Ok(1)
            },
        );
        assert!(matches!(result, Err(FormatError::InvalidMetadata(_))));
    }

    #[test]
    fn streaming_writer_rejects_declared_length_mismatch() {
        let tensors = vec![StreamingTensorInput {
            tensor_id: 1,
            name: "payload".to_owned(),
            dtype: DType::U8,
            dimensions: vec![3],
            block_lengths: vec![3],
        }];
        let mut output = Cursor::new(Vec::new());
        let result = ModelWriter::write_streaming(&mut output, &tensors, |_, _, writer| {
            writer.write_all(&[1_u8, 2]).map_err(ioerr)?;
            Ok(2)
        });
        assert!(matches!(result, Err(FormatError::InvalidBlock(_))));
    }

    #[test]
    fn streaming_writer_caps_output_at_declared_block_length() {
        let tensors = vec![StreamingTensorInput {
            tensor_id: 1,
            name: "payload".to_owned(),
            dtype: DType::U8,
            dimensions: vec![2],
            block_lengths: vec![2],
        }];
        let mut output = Cursor::new(Vec::new());
        let result = ModelWriter::write_streaming(&mut output, &tensors, |_, _, writer| {
            writer.write_all(&[1_u8, 2, 3]).map_err(ioerr)?;
            Ok(2)
        });
        assert!(result.is_err());
        let planned_size = u64::from_le_bytes(output.get_ref()[16..24].try_into().unwrap());
        assert_eq!(output.get_ref().len() as u64, planned_size);
    }
    #[test]
    fn streaming_writer_rejects_block_over_reader_default_limit_before_callback() {
        let oversized = MAX_WRITER_BLOCK_PAYLOAD_BYTES + 1;
        let tensors = vec![StreamingTensorInput {
            tensor_id: 1,
            name: "oversized".into(),
            dtype: DType::U8,
            dimensions: vec![oversized],
            block_lengths: vec![oversized],
        }];
        let mut output = Cursor::new(Vec::new());
        let result = ModelWriter::write_streaming(&mut output, &tensors, |_, _, _| {
            panic!("preflight must reject oversized payload")
        });
        assert!(matches!(
            result,
            Err(FormatError::LimitExceeded("writer payload bytes per block"))
        ));
        assert!(output.get_ref().is_empty());
    }

    #[test]
    fn streaming_writer_rejects_excessive_block_directories_before_writing() {
        let tensors = vec![StreamingTensorInput {
            tensor_id: 1,
            name: "payload".into(),
            dtype: DType::U8,
            dimensions: vec![1],
            block_lengths: vec![1; MAX_WRITER_BLOCKS + 1],
        }];
        let mut output = Cursor::new(Vec::new());
        let result = ModelWriter::write_streaming(&mut output, &tensors, |_, _, _| {
            panic!("writer must reject the directory before invoking payload callbacks")
        });
        assert!(matches!(
            result,
            Err(FormatError::LimitExceeded("writer block count"))
        ));
        assert!(output.get_ref().is_empty());
    }
}
