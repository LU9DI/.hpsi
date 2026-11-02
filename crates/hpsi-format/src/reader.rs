use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    BlockDescriptor, FormatError, Header, IntegrityRecord, IntegrityTarget, Metadata,
    RandomAccessSource, ReaderLimits, Result, TensorDescriptor, HEADER_SIZE,
};

#[derive(Debug, Clone)]
pub struct BlockData {
    pub descriptor: BlockDescriptor,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReaderStats {
    pub header_bytes_read: u64,
    pub metadata_bytes_read: u64,
    pub directory_bytes_read: u64,
    pub payload_bytes_read: u64,
    pub read_operations: u64,
    pub verified_records: u64,
}

#[derive(Default)]
struct AtomicStats {
    header_bytes_read: AtomicU64,
    metadata_bytes_read: AtomicU64,
    directory_bytes_read: AtomicU64,
    payload_bytes_read: AtomicU64,
    read_operations: AtomicU64,
    verified_records: AtomicU64,
}

impl AtomicStats {
    fn snapshot(&self) -> ReaderStats {
        ReaderStats {
            header_bytes_read: self.header_bytes_read.load(Ordering::Relaxed),
            metadata_bytes_read: self.metadata_bytes_read.load(Ordering::Relaxed),
            directory_bytes_read: self.directory_bytes_read.load(Ordering::Relaxed),
            payload_bytes_read: self.payload_bytes_read.load(Ordering::Relaxed),
            read_operations: self.read_operations.load(Ordering::Relaxed),
            verified_records: self.verified_records.load(Ordering::Relaxed),
        }
    }
}

pub struct ModelReader {
    source: Arc<dyn RandomAccessSource>,
    pub header: Header,
    pub metadata: Metadata,
    tensors: Vec<TensorDescriptor>,
    blocks: Vec<BlockDescriptor>,
    integrity: Vec<IntegrityRecord>,
    tensor_lookup: HashMap<u32, usize>,
    stats: AtomicStats,
}

impl ModelReader {
    /// Open and fully validate the container directories without reading payload bytes.
    pub fn open(source: Arc<dyn RandomAccessSource>, limits: &ReaderLimits) -> Result<Self> {
        let actual_len = source.len()?;
        let mut header_bytes = [0_u8; HEADER_SIZE];
        source.read_exact_at(0, &mut header_bytes)?;
        let header = Header::parse(&header_bytes, actual_len, limits)?;

        let metadata_bytes = read_region(&*source, header.metadata, limits.max_metadata_bytes)?;
        let metadata = Metadata::parse(&metadata_bytes, limits)?;

        let tensor_bytes = read_region(
            &*source,
            header.tensor_directory,
            limits.max_total_directory_bytes,
        )?;
        let block_bytes = read_region(
            &*source,
            header.block_directory,
            limits.max_total_directory_bytes,
        )?;
        let integrity_bytes = read_region(
            &*source,
            header.integrity_directory,
            limits.max_total_directory_bytes,
        )?;

        let tensors = parse_table::<TensorDescriptor>(
            &tensor_bytes,
            header.tensor_count as usize,
            TensorDescriptor::ENTRY_SIZE,
            TensorDescriptor::parse,
        )?;
        let blocks = parse_table::<BlockDescriptor>(
            &block_bytes,
            header.block_count as usize,
            BlockDescriptor::ENTRY_SIZE,
            BlockDescriptor::parse,
        )?;
        let integrity = parse_table::<IntegrityRecord>(
            &integrity_bytes,
            header.integrity_count as usize,
            IntegrityRecord::ENTRY_SIZE,
            IntegrityRecord::parse,
        )?;

        let tensor_lookup = validate_directories(&header, &metadata, &tensors, &blocks, &integrity, limits)?;

        let stats = AtomicStats::default();
        stats.header_bytes_read.store(HEADER_SIZE as u64, Ordering::Relaxed);
        stats.metadata_bytes_read.store(metadata_bytes.len() as u64, Ordering::Relaxed);
        stats.directory_bytes_read.store(
            (tensor_bytes.len() + block_bytes.len() + integrity_bytes.len()) as u64,
            Ordering::Relaxed,
        );

        Ok(Self {
            source,
            header,
            metadata,
            tensors,
            blocks,
            integrity,
            tensor_lookup,
            stats,
        })
    }

    pub fn tensor(&self, tensor_id: u32) -> Result<&TensorDescriptor> {
        let index = self.tensor_lookup.get(&tensor_id)
            .ok_or(FormatError::InvalidReference("unknown tensor ID"))?;
        Ok(&self.tensors[*index])
    }

    pub fn block(&self, tensor_id: u32, block_id: u32) -> Result<&BlockDescriptor> {
        let tensor = self.tensor(tensor_id)?;
        if block_id >= tensor.block_count {
            return Err(FormatError::InvalidReference("block ID is outside tensor block range"));
        }
        let index = usize::try_from(tensor.first_block_index)
            .map_err(|_| FormatError::InvalidDirectory("block index exceeds address space"))?
            .checked_add(block_id as usize)
            .ok_or(FormatError::ArithmeticOverflow("block directory index"))?;
        self.blocks.get(index)
            .ok_or(FormatError::InvalidDirectory("block directory index is out of bounds"))
    }

    /// Read exactly one block's stored payload. No neighboring block is touched.
    pub fn read_block(&self, tensor_id: u32, block_id: u32) -> Result<BlockData> {
        let descriptor = self.block(tensor_id, block_id)?.clone();
        let length = usize::try_from(descriptor.payload_length)
            .map_err(|_| FormatError::LimitExceeded("block payload cannot fit address space"))?;
        let mut payload = Vec::new();
        payload.try_reserve_exact(length)
            .map_err(|_| FormatError::LimitExceeded("block payload allocation failed"))?;
        payload.resize(length, 0);
        self.source.read_exact_at(descriptor.payload_offset, &mut payload)?;
        self.stats.payload_bytes_read.fetch_add(length as u64, Ordering::Relaxed);
        self.stats.read_operations.fetch_add(1, Ordering::Relaxed);
        Ok(BlockData { descriptor, payload })
    }

    /// Verify every declared CRC32C record using bounded reads.
    pub fn verify_all(&self) -> Result<u64> {
        let mut verified = 0_u64;
        for record in &self.integrity {
            let (offset, length) = match record.target {
                IntegrityTarget::Metadata => (self.header.metadata.offset, self.header.metadata.length),
                IntegrityTarget::BlockPayload(index) => {
                    let block = self.blocks.get(index as usize)
                        .ok_or(FormatError::InvalidReference("integrity block index out of range"))?;
                    (block.payload_offset, block.payload_length)
                }
            };
            let mut remaining = length;
            let mut position = offset;
            let mut checksum = Crc32c::new();
            let mut buffer = [0_u8; 64 * 1024];
            while remaining > 0 {
                let chunk = usize::try_from(remaining.min(buffer.len() as u64))
                    .map_err(|_| FormatError::LimitExceeded("verification chunk"))?;
                self.source.read_exact_at(position, &mut buffer[..chunk])?;
                checksum.update(&buffer[..chunk]);
                position = position.checked_add(chunk as u64)
                    .ok_or(FormatError::ArithmeticOverflow("verification offset"))?;
                remaining -= chunk as u64;
                if matches!(record.target, IntegrityTarget::BlockPayload(_)) {
                    self.stats.payload_bytes_read.fetch_add(chunk as u64, Ordering::Relaxed);
                }
                self.stats.read_operations.fetch_add(1, Ordering::Relaxed);
            }
            if checksum.finish() != record.crc32c {
                return Err(FormatError::ChecksumMismatch);
            }
            verified += 1;
            self.stats.verified_records.fetch_add(1, Ordering::Relaxed);
        }
        Ok(verified)
    }

    pub fn stats(&self) -> ReaderStats {
        self.stats.snapshot()
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
}

fn read_region(
    source: &dyn RandomAccessSource,
    region: crate::Region,
    max_bytes: u64,
) -> Result<Vec<u8>> {
    if region.length > max_bytes {
        return Err(FormatError::LimitExceeded("region read size"));
    }
    let length = usize::try_from(region.length)
        .map_err(|_| FormatError::LimitExceeded("region cannot fit address space"))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length)
        .map_err(|_| FormatError::LimitExceeded("region allocation failed"))?;
    bytes.resize(length, 0);
    source.read_exact_at(region.offset, &mut bytes)?;
    Ok(bytes)
}

fn parse_table<T>(
    bytes: &[u8],
    count: usize,
    entry_size: usize,
    parse: fn(&[u8]) -> Result<T>,
) -> Result<Vec<T>> {
    let expected = count.checked_mul(entry_size)
        .ok_or(FormatError::ArithmeticOverflow("directory table size"))?;
    if bytes.len() != expected {
        return Err(FormatError::InvalidDirectory("directory byte length does not match entry count"));
    }
    let mut entries = Vec::new();
    entries.try_reserve_exact(count)
        .map_err(|_| FormatError::LimitExceeded("directory entry allocation failed"))?;
    for chunk in bytes.chunks_exact(entry_size) {
        entries.push(parse(chunk)?);
    }
    Ok(entries)
}

fn validate_directories(
    header: &Header,
    metadata: &Metadata,
    tensors: &[TensorDescriptor],
    blocks: &[BlockDescriptor],
    integrity: &[IntegrityRecord],
    limits: &ReaderLimits,
) -> Result<HashMap<u32, usize>> {
    if tensors.len() != header.tensor_count as usize || blocks.len() != header.block_count as usize {
        return Err(FormatError::InvalidDirectory("directory count mismatch"));
    }

    let mut lookup = HashMap::with_capacity(tensors.len());
    let mut expected_first_block = 0_u64;
    let mut previous_tensor_id = None;
    for (index, tensor) in tensors.iter().enumerate() {
        if previous_tensor_id.is_some_and(|previous| tensor.tensor_id <= previous) {
            return Err(FormatError::InvalidDirectory("tensor IDs must be strictly increasing"));
        }
        previous_tensor_id = Some(tensor.tensor_id);
        if tensor.name_ref as usize >= metadata.tensor_names.len() {
            return Err(FormatError::InvalidReference("tensor name reference is out of range"));
        }
        if tensor.logical_nbytes > limits.max_tensor_logical_bytes {
            return Err(FormatError::LimitExceeded("tensor logical bytes"));
        }
        if tensor.rank > limits.max_rank {
            return Err(FormatError::LimitExceeded("tensor rank"));
        }
        if tensor.first_block_index != expected_first_block {
            return Err(FormatError::InvalidDirectory("tensor block ranges must be contiguous"));
        }
        expected_first_block = expected_first_block.checked_add(u64::from(tensor.block_count))
            .ok_or(FormatError::ArithmeticOverflow("total tensor block count"))?;
        lookup.insert(tensor.tensor_id, index);
    }
    if expected_first_block != blocks.len() as u64 {
        return Err(FormatError::InvalidDirectory("tensor block ranges do not cover block directory"));
    }

    let mut payload_ranges = Vec::with_capacity(blocks.len());
    for tensor in tensors {
        let start = usize::try_from(tensor.first_block_index)
            .map_err(|_| FormatError::InvalidDirectory("first block index exceeds address space"))?;
        let end = start.checked_add(tensor.block_count as usize)
            .ok_or(FormatError::ArithmeticOverflow("tensor block range"))?;
        let tensor_blocks = blocks.get(start..end)
            .ok_or(FormatError::InvalidDirectory("tensor block range exceeds directory"))?;
        let mut expected_logical_offset = 0_u64;
        for (expected_block_id, block) in tensor_blocks.iter().enumerate() {
            if block.tensor_id != tensor.tensor_id || block.block_id != expected_block_id as u32 {
                return Err(FormatError::InvalidDirectory("block ownership or ID ordering mismatch"));
            }
            if block.logical_offset != expected_logical_offset {
                return Err(FormatError::InvalidDirectory("block logical ranges must be contiguous"));
            }
            expected_logical_offset = expected_logical_offset.checked_add(block.logical_length)
                .ok_or(FormatError::ArithmeticOverflow("logical block coverage"))?;
            let payload_end = block.payload_offset.checked_add(block.payload_length)
                .ok_or(FormatError::ArithmeticOverflow("block payload end"))?;
            let payload_region_end = header.payload.offset.checked_add(header.payload.length)
                .ok_or(FormatError::ArithmeticOverflow("payload region end"))?;
            if block.payload_offset < header.payload.offset || payload_end > payload_region_end {
                return Err(FormatError::InvalidBlock("block payload is outside payload region"));
            }
            if block.payload_length > limits.max_payload_bytes_per_block {
                return Err(FormatError::LimitExceeded("payload bytes per block"));
            }
            let codec = if block.codec == 0 { tensor.codec } else { block.codec };
            if codec == 1 && block.payload_length != block.logical_length {
                return Err(FormatError::InvalidBlock("RAW payload length must equal logical length"));
            }
            if let Some(reference) = block.auxiliary_ref {
                if !lookup.contains_key(&(reference as u32)) || reference > u64::from(u32::MAX) {
                    return Err(FormatError::InvalidReference("block auxiliary tensor does not exist"));
                }
            }
            payload_ranges.push((block.payload_offset, payload_end));
        }
        if expected_logical_offset != tensor.logical_nbytes {
            return Err(FormatError::InvalidDirectory("blocks do not cover tensor logical byte range"));
        }
    }

    payload_ranges.sort_unstable_by_key(|range| range.0);
    for pair in payload_ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(FormatError::InvalidDirectory("block payload ranges overlap"));
        }
    }

    let mut integrity_targets = HashMap::with_capacity(integrity.len());
    for (index, record) in integrity.iter().enumerate() {
        if integrity_targets.insert(record.target, index).is_some() {
            return Err(FormatError::InvalidDirectory("duplicate integrity target"));
        }
        if let IntegrityTarget::BlockPayload(block_index) = record.target {
            if block_index as usize >= blocks.len() {
                return Err(FormatError::InvalidReference("integrity block target is out of range"));
            }
        }
    }
    for (block_index, block) in blocks.iter().enumerate() {
        if let Some(checksum_index) = block.checksum_index {
            let record = integrity.get(checksum_index as usize)
                .ok_or(FormatError::InvalidReference("block checksum index is out of range"))?;
            if record.target != IntegrityTarget::BlockPayload(block_index as u32) {
                return Err(FormatError::InvalidReference("block checksum target mismatch"));
            }
        }
    }

    validate_reference_cycles(tensors, blocks, &lookup)?;
    Ok(lookup)
}

fn validate_reference_cycles(
    tensors: &[TensorDescriptor],
    blocks: &[BlockDescriptor],
    lookup: &HashMap<u32, usize>,
) -> Result<()> {
    let mut edges = vec![Vec::<usize>::new(); tensors.len()];
    for (index, tensor) in tensors.iter().enumerate() {
        if let Some(reference) = tensor.representation_ref {
            let target = u32::try_from(reference)
                .ok()
                .and_then(|id| lookup.get(&id).copied())
                .ok_or(FormatError::InvalidReference("tensor representation reference is missing"))?;
            edges[index].push(target);
        }
        let start = tensor.first_block_index as usize;
        let end = start + tensor.block_count as usize;
        for block in &blocks[start..end] {
            if let Some(reference) = block.auxiliary_ref {
                let target = u32::try_from(reference)
                    .ok()
                    .and_then(|id| lookup.get(&id).copied())
                    .ok_or(FormatError::InvalidReference("block auxiliary reference is missing"))?;
                edges[index].push(target);
            }
        }
    }

    let mut color = vec![0_u8; tensors.len()];
    for start in 0..tensors.len() {
        if color[start] != 0 {
            continue;
        }
        let mut stack = vec![(start, false)];
        while let Some((node, exiting)) = stack.pop() {
            if exiting {
                color[node] = 2;
                continue;
            }
            if color[node] == 1 {
                return Err(FormatError::InvalidReference("cyclic auxiliary tensor references"));
            }
            if color[node] == 2 {
                continue;
            }
            color[node] = 1;
            stack.push((node, true));
            for dependency in &edges[node] {
                if color[*dependency] == 1 {
                    return Err(FormatError::InvalidReference("cyclic auxiliary tensor references"));
                }
                if color[*dependency] == 0 {
                    stack.push((*dependency, false));
                }
            }
        }
    }
    Ok(())
}

struct Crc32c(u32);

impl Crc32c {
    fn new() -> Self {
        Self(!0_u32)
    }

    fn update(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u32::from(*byte);
            for _ in 0..8 {
                self.0 = if self.0 & 1 == 1 {
                    (self.0 >> 1) ^ 0x82f6_3b78
                } else {
                    self.0 >> 1
                };
            }
        }
    }

    fn finish(self) -> u32 {
        !self.0
    }
}
