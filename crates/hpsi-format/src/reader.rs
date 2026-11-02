use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    BlockDescriptor, FormatError, HEADER_SIZE, Header, IntegrityRecord, IntegrityTarget, Metadata,
    RandomAccessSource, ReaderLimits, Result, TensorDescriptor,
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

        let (metadata, metadata_bytes_read) = {
            let bytes = read_region(&*source, header.metadata, limits.max_metadata_bytes)?;
            let length = bytes.len() as u64;
            (Metadata::parse(&bytes, limits)?, length)
        };

        // Parse one directory at a time and release its raw byte buffer before
        // reading the next one. Keeping all raw directories alive alongside all
        // parsed descriptors needlessly multiplies peak memory for large files.
        let (tensors, tensor_directory_bytes_read) = {
            let bytes = read_region(
                &*source,
                header.tensor_directory,
                limits.max_total_directory_bytes,
            )?;
            let length = bytes.len() as u64;
            (
                parse_table::<TensorDescriptor>(
                    &bytes,
                    header.tensor_count as usize,
                    TensorDescriptor::ENTRY_SIZE,
                    TensorDescriptor::parse,
                )?,
                length,
            )
        };
        let (blocks, block_directory_bytes_read) = {
            let bytes = read_region(
                &*source,
                header.block_directory,
                limits.max_total_directory_bytes,
            )?;
            let length = bytes.len() as u64;
            (
                parse_table::<BlockDescriptor>(
                    &bytes,
                    header.block_count as usize,
                    BlockDescriptor::ENTRY_SIZE,
                    BlockDescriptor::parse,
                )?,
                length,
            )
        };
        let (integrity, integrity_directory_bytes_read) = {
            let bytes = read_region(
                &*source,
                header.integrity_directory,
                limits.max_total_directory_bytes,
            )?;
            let length = bytes.len() as u64;
            (
                parse_table::<IntegrityRecord>(
                    &bytes,
                    header.integrity_count as usize,
                    IntegrityRecord::ENTRY_SIZE,
                    IntegrityRecord::parse,
                )?,
                length,
            )
        };

        let tensor_lookup =
            validate_directories(&header, &metadata, &tensors, &blocks, &integrity, limits)?;

        let stats = AtomicStats::default();
        stats
            .header_bytes_read
            .store(HEADER_SIZE as u64, Ordering::Relaxed);
        stats
            .metadata_bytes_read
            .store(metadata_bytes_read, Ordering::Relaxed);
        stats.directory_bytes_read.store(
            tensor_directory_bytes_read
                + block_directory_bytes_read
                + integrity_directory_bytes_read,
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
        let index = self
            .tensor_lookup
            .get(&tensor_id)
            .ok_or(FormatError::InvalidReference("unknown tensor ID"))?;
        Ok(&self.tensors[*index])
    }

    pub fn block(&self, tensor_id: u32, block_id: u32) -> Result<&BlockDescriptor> {
        let tensor = self.tensor(tensor_id)?;
        if block_id >= tensor.block_count {
            return Err(FormatError::InvalidReference(
                "block ID is outside tensor block range",
            ));
        }
        let index = usize::try_from(tensor.first_block_index)
            .map_err(|_| FormatError::InvalidDirectory("block index exceeds address space"))?
            .checked_add(block_id as usize)
            .ok_or(FormatError::ArithmeticOverflow("block directory index"))?;
        self.blocks.get(index).ok_or(FormatError::InvalidDirectory(
            "block directory index is out of bounds",
        ))
    }

    /// Read exactly one block's stored payload. No neighboring block is touched.
    pub fn read_block(&self, tensor_id: u32, block_id: u32) -> Result<BlockData> {
        let descriptor = self.block(tensor_id, block_id)?.clone();
        let length = usize::try_from(descriptor.payload_length)
            .map_err(|_| FormatError::LimitExceeded("block payload cannot fit address space"))?;
        let mut payload = Vec::new();
        payload
            .try_reserve_exact(length)
            .map_err(|_| FormatError::LimitExceeded("block payload allocation failed"))?;
        payload.resize(length, 0);
        self.source
            .read_exact_at(descriptor.payload_offset, &mut payload)?;
        self.stats
            .payload_bytes_read
            .fetch_add(length as u64, Ordering::Relaxed);
        self.stats.read_operations.fetch_add(1, Ordering::Relaxed);
        Ok(BlockData {
            descriptor,
            payload,
        })
    }

    /// Verify every declared CRC32C record using bounded reads.
    pub fn verify_all(&self) -> Result<u64> {
        let mut verified = 0_u64;
        for record in &self.integrity {
            let (offset, length) =
                match record.target {
                    IntegrityTarget::Metadata => {
                        (self.header.metadata.offset, self.header.metadata.length)
                    }
                    IntegrityTarget::BlockPayload(index) => {
                        let block = self.blocks.get(index as usize).ok_or(
                            FormatError::InvalidReference("integrity block index out of range"),
                        )?;
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
                position = position
                    .checked_add(chunk as u64)
                    .ok_or(FormatError::ArithmeticOverflow("verification offset"))?;
                remaining -= chunk as u64;
                if matches!(record.target, IntegrityTarget::BlockPayload(_)) {
                    self.stats
                        .payload_bytes_read
                        .fetch_add(chunk as u64, Ordering::Relaxed);
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

    /// Verify zero-filled gaps between the header and declared top-level regions.
    ///
    /// The caller supplies an explicit byte budget so opening a sparse or adversarial
    /// container never triggers an unbounded scan. Payload and trailing file bytes
    /// are not read by this method.
    pub fn verify_padding(&self, max_bytes: u64) -> Result<u64> {
        verify_padding_regions(
            &*self.source,
            &[
                self.header.metadata,
                self.header.tensor_directory,
                self.header.block_directory,
                self.header.payload,
                self.header.integrity_directory,
            ],
            max_bytes,
        )
    }

    pub fn stats(&self) -> ReaderStats {
        self.stats.snapshot()
    }

    /// Return tensor descriptors in their validated directory order.
    pub fn tensors(&self) -> &[TensorDescriptor] {
        &self.tensors
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
}

/// Verify zero-filled gaps among validated top-level regions with a caller budget.
fn verify_padding_regions(
    source: &dyn RandomAccessSource,
    regions: &[crate::Region],
    max_bytes: u64,
) -> Result<u64> {
    let mut active: Vec<_> = regions
        .iter()
        .copied()
        .filter(|region| !region.is_empty())
        .collect();
    active.sort_unstable_by_key(|region| region.offset);

    let mut gaps = Vec::new();
    let mut cursor = HEADER_SIZE as u64;
    let mut total = 0_u64;
    for region in active {
        if region.offset > cursor {
            let length = region.offset - cursor;
            total = total
                .checked_add(length)
                .ok_or(FormatError::ArithmeticOverflow("total padding length"))?;
            gaps.push((cursor, length));
        }
        cursor = region.end()?;
    }
    if total > max_bytes {
        return Err(FormatError::LimitExceeded("padding verification budget"));
    }

    let mut buffer = [0_u8; 64 * 1024];
    for (mut offset, mut remaining) in gaps {
        while remaining > 0 {
            let chunk = usize::try_from(remaining.min(buffer.len() as u64))
                .map_err(|_| FormatError::LimitExceeded("padding verification chunk"))?;
            source.read_exact_at(offset, &mut buffer[..chunk])?;
            if buffer[..chunk].iter().any(|byte| *byte != 0) {
                return Err(FormatError::InvalidRegion("non-zero padding bytes"));
            }
            offset = offset
                .checked_add(chunk as u64)
                .ok_or(FormatError::ArithmeticOverflow("padding offset"))?;
            remaining -= chunk as u64;
        }
    }
    Ok(total)
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
    bytes
        .try_reserve_exact(length)
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
    let expected = count
        .checked_mul(entry_size)
        .ok_or(FormatError::ArithmeticOverflow("directory table size"))?;
    if bytes.len() != expected {
        return Err(FormatError::InvalidDirectory(
            "directory byte length does not match entry count",
        ));
    }
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(count)
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
    if tensors.len() != header.tensor_count as usize || blocks.len() != header.block_count as usize
    {
        return Err(FormatError::InvalidDirectory("directory count mismatch"));
    }

    let mut lookup = HashMap::new();
    lookup
        .try_reserve(tensors.len())
        .map_err(|_| FormatError::LimitExceeded("tensor lookup allocation failed"))?;
    let mut expected_first_block = 0_u64;
    let mut previous_tensor_id = None;
    for (index, tensor) in tensors.iter().enumerate() {
        if previous_tensor_id.is_some_and(|previous| tensor.tensor_id <= previous) {
            return Err(FormatError::InvalidDirectory(
                "tensor IDs must be strictly increasing",
            ));
        }
        previous_tensor_id = Some(tensor.tensor_id);
        if tensor.name_ref as usize >= metadata.tensor_names.len() {
            return Err(FormatError::InvalidReference(
                "tensor name reference is out of range",
            ));
        }
        if tensor.logical_nbytes > limits.max_tensor_logical_bytes {
            return Err(FormatError::LimitExceeded("tensor logical bytes"));
        }
        if tensor.rank > limits.max_rank {
            return Err(FormatError::LimitExceeded("tensor rank"));
        }
        if tensor.first_block_index != expected_first_block {
            return Err(FormatError::InvalidDirectory(
                "tensor block ranges must be contiguous",
            ));
        }
        expected_first_block = expected_first_block
            .checked_add(u64::from(tensor.block_count))
            .ok_or(FormatError::ArithmeticOverflow("total tensor block count"))?;
        lookup.insert(tensor.tensor_id, index);
    }
    if expected_first_block != blocks.len() as u64 {
        return Err(FormatError::InvalidDirectory(
            "tensor block ranges do not cover block directory",
        ));
    }

    let mut payload_ranges = Vec::new();
    payload_ranges
        .try_reserve_exact(blocks.len())
        .map_err(|_| FormatError::LimitExceeded("payload range allocation failed"))?;
    for tensor in tensors {
        let start = usize::try_from(tensor.first_block_index).map_err(|_| {
            FormatError::InvalidDirectory("first block index exceeds address space")
        })?;
        let end = start
            .checked_add(tensor.block_count as usize)
            .ok_or(FormatError::ArithmeticOverflow("tensor block range"))?;
        let tensor_blocks = blocks.get(start..end).ok_or(FormatError::InvalidDirectory(
            "tensor block range exceeds directory",
        ))?;
        let mut expected_logical_offset = 0_u64;
        for (expected_block_id, block) in tensor_blocks.iter().enumerate() {
            if block.tensor_id != tensor.tensor_id || block.block_id != expected_block_id as u32 {
                return Err(FormatError::InvalidDirectory(
                    "block ownership or ID ordering mismatch",
                ));
            }
            if block.logical_offset != expected_logical_offset {
                return Err(FormatError::InvalidDirectory(
                    "block logical ranges must be contiguous",
                ));
            }
            expected_logical_offset = expected_logical_offset
                .checked_add(block.logical_length)
                .ok_or(FormatError::ArithmeticOverflow("logical block coverage"))?;
            let payload_end = block
                .payload_offset
                .checked_add(block.payload_length)
                .ok_or(FormatError::ArithmeticOverflow("block payload end"))?;
            let payload_region_end = header
                .payload
                .offset
                .checked_add(header.payload.length)
                .ok_or(FormatError::ArithmeticOverflow("payload region end"))?;
            if block.payload_offset < header.payload.offset || payload_end > payload_region_end {
                return Err(FormatError::InvalidBlock(
                    "block payload is outside payload region",
                ));
            }
            if block.payload_length > limits.max_payload_bytes_per_block {
                return Err(FormatError::LimitExceeded("payload bytes per block"));
            }
            let codec = if block.codec == 0 {
                tensor.codec
            } else {
                block.codec
            };
            if codec == 1 && block.payload_length != block.logical_length {
                return Err(FormatError::InvalidBlock(
                    "RAW payload length must equal logical length",
                ));
            }
            if let Some(reference) = block.auxiliary_ref {
                let target_id = u32::try_from(reference).map_err(|_| {
                    FormatError::InvalidReference(
                        "block auxiliary tensor ID exceeds the supported ID width",
                    )
                })?;
                if !lookup.contains_key(&target_id) {
                    return Err(FormatError::InvalidReference(
                        "block auxiliary tensor does not exist",
                    ));
                }
            }
            payload_ranges.push((block.payload_offset, payload_end));
        }
        if expected_logical_offset != tensor.logical_nbytes {
            return Err(FormatError::InvalidDirectory(
                "blocks do not cover tensor logical byte range",
            ));
        }
    }

    payload_ranges.sort_unstable_by_key(|range| range.0);
    for pair in payload_ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(FormatError::InvalidDirectory(
                "block payload ranges overlap",
            ));
        }
    }

    validate_integrity_coverage(blocks, integrity)?;

    validate_reference_cycles(tensors, blocks, &lookup)?;
    Ok(lookup)
}

fn validate_integrity_coverage(
    blocks: &[BlockDescriptor],
    integrity: &[IntegrityRecord],
) -> Result<()> {
    let mut metadata_seen = false;
    let mut block_seen = Vec::new();
    block_seen
        .try_reserve_exact(blocks.len())
        .map_err(|_| FormatError::LimitExceeded("integrity coverage allocation failed"))?;
    block_seen.resize(blocks.len(), false);

    for (record_index, record) in integrity.iter().enumerate() {
        match record.target {
            IntegrityTarget::Metadata => {
                if metadata_seen {
                    return Err(FormatError::InvalidDirectory(
                        "metadata may have at most one integrity record",
                    ));
                }
                metadata_seen = true;
            }
            IntegrityTarget::BlockPayload(block_index) => {
                let seen = block_seen.get_mut(block_index as usize).ok_or(
                    FormatError::InvalidReference("integrity block index out of range"),
                )?;
                if *seen {
                    return Err(FormatError::InvalidDirectory(
                        "a block may have at most one integrity record",
                    ));
                }
                *seen = true;
                let block = &blocks[block_index as usize];
                let expected_index = u32::try_from(record_index)
                    .map_err(|_| FormatError::LimitExceeded("integrity record index"))?;
                if block.checksum_index != Some(expected_index) {
                    return Err(FormatError::InvalidReference(
                        "block checksum index is missing or inconsistent",
                    ));
                }
            }
        }
    }

    for (block_index, block) in blocks.iter().enumerate() {
        match block.checksum_index {
            Some(index) => {
                let record = integrity
                    .get(index as usize)
                    .ok_or(FormatError::InvalidReference(
                        "block checksum index is out of range",
                    ))?;
                if record.target != IntegrityTarget::BlockPayload(block_index as u32) {
                    return Err(FormatError::InvalidReference(
                        "block checksum target mismatch",
                    ));
                }
            }
            None if block_seen[block_index] => {
                return Err(FormatError::InvalidReference(
                    "integrity record is not referenced by its block",
                ));
            }
            None => {}
        }
    }
    Ok(())
}

fn validate_reference_cycles(
    tensors: &[TensorDescriptor],
    blocks: &[BlockDescriptor],
    lookup: &HashMap<u32, usize>,
) -> Result<()> {
    // Traverse references directly from descriptors. Materializing Vec<Vec<usize>>
    // here duplicates potentially millions of edges and allocates one Vec per tensor.
    validate_acyclic_references(tensors, blocks, lookup)
}

fn next_reference(
    tensors: &[TensorDescriptor],
    blocks: &[BlockDescriptor],
    lookup: &HashMap<u32, usize>,
    node: usize,
    representation_pending: &mut bool,
    next_block: &mut usize,
) -> Result<Option<usize>> {
    let tensor = tensors.get(node).ok_or(FormatError::InvalidReference(
        "reference graph node is out of range",
    ))?;

    if *representation_pending {
        *representation_pending = false;
        if let Some(reference) = tensor.representation_ref {
            let id = u32::try_from(reference).map_err(|_| {
                FormatError::InvalidReference("tensor representation ID exceeds supported width")
            })?;
            let target = lookup
                .get(&id)
                .copied()
                .ok_or(FormatError::InvalidReference(
                    "tensor representation reference is missing",
                ))?;
            return Ok(Some(target));
        }
    }

    let start = usize::try_from(tensor.first_block_index).map_err(|_| {
        FormatError::InvalidDirectory("reference block index exceeds address space")
    })?;
    let end = start
        .checked_add(tensor.block_count as usize)
        .ok_or(FormatError::ArithmeticOverflow("reference block range"))?;
    let tensor_blocks = blocks.get(start..end).ok_or(FormatError::InvalidDirectory(
        "reference block range exceeds directory",
    ))?;

    while *next_block < tensor_blocks.len() {
        let block = &tensor_blocks[*next_block];
        *next_block += 1;
        if let Some(reference) = block.auxiliary_ref {
            let id = u32::try_from(reference).map_err(|_| {
                FormatError::InvalidReference("block auxiliary ID exceeds supported width")
            })?;
            let target = lookup
                .get(&id)
                .copied()
                .ok_or(FormatError::InvalidReference(
                    "block auxiliary reference is missing",
                ))?;
            return Ok(Some(target));
        }
    }
    Ok(None)
}

fn validate_acyclic_references(
    tensors: &[TensorDescriptor],
    blocks: &[BlockDescriptor],
    lookup: &HashMap<u32, usize>,
) -> Result<()> {
    // 0 = unseen, 1 = active DFS path, 2 = fully processed.
    // The stack stores cursors into descriptors, not a materialized adjacency list.
    let mut color = Vec::new();
    color
        .try_reserve_exact(tensors.len())
        .map_err(|_| FormatError::LimitExceeded("reference traversal allocation failed"))?;
    color.resize(tensors.len(), 0_u8);

    let mut stack: Vec<(usize, usize, bool)> = Vec::new();
    for start in 0..tensors.len() {
        if color[start] != 0 {
            continue;
        }
        color[start] = 1;
        stack.try_reserve(1).map_err(|_| {
            FormatError::LimitExceeded("reference traversal stack allocation failed")
        })?;
        stack.push((start, 0, true));

        while let Some((node, next_block, representation_pending)) = stack.last_mut() {
            let dependency = next_reference(
                tensors,
                blocks,
                lookup,
                *node,
                representation_pending,
                next_block,
            )?;
            let Some(dependency) = dependency else {
                let finished = *node;
                stack.pop();
                color[finished] = 2;
                continue;
            };

            if dependency >= tensors.len() {
                return Err(FormatError::InvalidReference(
                    "reference graph target is out of range",
                ));
            }
            match color[dependency] {
                1 => {
                    return Err(FormatError::InvalidReference(
                        "cyclic auxiliary tensor references",
                    ));
                }
                2 => {}
                _ => {
                    color[dependency] = 1;
                    stack.try_reserve(1).map_err(|_| {
                        FormatError::LimitExceeded("reference traversal stack allocation failed")
                    })?;
                    stack.push((dependency, 0, true));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn validate_acyclic_graph(edges: &[Vec<usize>]) -> Result<()> {
    let mut color = vec![0_u8; edges.len()];
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for start in 0..edges.len() {
        if color[start] != 0 {
            continue;
        }
        color[start] = 1;
        stack.push((start, 0));
        while let Some((node, next_edge)) = stack.last_mut() {
            if *next_edge == edges[*node].len() {
                let finished = *node;
                stack.pop();
                color[finished] = 2;
                continue;
            }
            let dependency = edges[*node][*next_edge];
            *next_edge += 1;
            if dependency >= edges.len() {
                return Err(FormatError::InvalidReference(
                    "reference graph target is out of range",
                ));
            }
            match color[dependency] {
                1 => {
                    return Err(FormatError::InvalidReference(
                        "cyclic auxiliary tensor references",
                    ));
                }
                2 => {}
                _ => {
                    color[dependency] = 1;
                    stack.push((dependency, 0));
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

#[cfg(test)]
mod tests {
    use super::{Crc32c, validate_acyclic_references, verify_padding_regions};
    use crate::{BlockDescriptor, DType, FormatError, RandomAccessSource, Region, Result, TensorDescriptor};
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct TrackingSource {
        bytes: Vec<u8>,
        reads: Mutex<Vec<(u64, usize)>>,
    }

    impl RandomAccessSource for TrackingSource {
        fn len(&self) -> Result<u64> {
            Ok(self.bytes.len() as u64)
        }

        fn read_exact_at(&self, offset: u64, buffer: &mut [u8]) -> Result<()> {
            self.reads.lock().unwrap().push((offset, buffer.len()));
            let start =
                usize::try_from(offset).map_err(|_| FormatError::InvalidRegion("test offset"))?;
            let end = start
                .checked_add(buffer.len())
                .ok_or(FormatError::ArithmeticOverflow("test read"))?;
            let source = self
                .bytes
                .get(start..end)
                .ok_or(FormatError::InvalidRegion("test read out of bounds"))?;
            buffer.copy_from_slice(source);
            Ok(())
        }
    }

    fn tracking_source(size: usize) -> TrackingSource {
        TrackingSource {
            bytes: vec![0; size],
            reads: Mutex::new(Vec::new()),
        }
    }

    fn regions_out_of_order() -> [Region; 2] {
        [
            Region {
                name: "later",
                offset: 300,
                length: 20,
            },
            Region {
                name: "earlier",
                offset: 160,
                length: 20,
            },
        ]
    }

    #[test]
    fn padding_verification_checks_zero_gaps_in_offset_order() {
        let source = tracking_source(400);
        let verified = verify_padding_regions(&source, &regions_out_of_order(), 152).unwrap();
        assert_eq!(verified, 152);
        assert_eq!(*source.reads.lock().unwrap(), vec![(128, 32), (180, 120)]);
    }

    #[test]
    fn padding_verification_rejects_nonzero_gap_byte() {
        let mut source = tracking_source(400);
        source.bytes[200] = 1;
        assert!(matches!(
            verify_padding_regions(&source, &regions_out_of_order(), 152),
            Err(FormatError::InvalidRegion("non-zero padding bytes"))
        ));
    }

    #[test]
    fn padding_budget_is_checked_before_any_source_read() {
        let source = tracking_source(400);
        assert!(matches!(
            verify_padding_regions(&source, &regions_out_of_order(), 151),
            Err(FormatError::LimitExceeded("padding verification budget"))
        ));
        assert!(source.reads.lock().unwrap().is_empty());
    }

    #[test]
    fn large_padding_gap_is_scanned_in_bounded_chunks() {
        let source = tracking_source(200_200);
        let regions = [Region {
            name: "last",
            offset: 200_000,
            length: 8,
        }];
        assert_eq!(
            verify_padding_regions(&source, &regions, 199_872).unwrap(),
            199_872
        );
        let reads = source.reads.lock().unwrap();
        assert_eq!(reads.len(), 4);
        assert!(reads.iter().all(|(_, length)| *length <= 64 * 1024));
        assert_eq!(
            reads.iter().map(|(_, length)| *length).sum::<usize>(),
            199_872
        );
    }

    #[test]
    fn reference_graph_accepts_duplicate_and_converging_edges() {
        let edges = vec![vec![1, 1, 2], vec![3], vec![3], vec![]];
        assert!(super::validate_acyclic_graph(&edges).is_ok());
    }

    #[test]
    fn reference_graph_rejects_direct_and_indirect_cycles() {
        assert!(matches!(
            super::validate_acyclic_graph(&[vec![0]]),
            Err(crate::FormatError::InvalidReference(_))
        ));
        assert!(matches!(
            super::validate_acyclic_graph(&[vec![1], vec![2], vec![0]]),
            Err(crate::FormatError::InvalidReference(_))
        ));
    }

    #[test]
    fn reference_graph_rejects_out_of_range_targets() {
        assert!(matches!(
            super::validate_acyclic_graph(&[vec![1]]),
            Err(crate::FormatError::InvalidReference(_))
        ));
    }

    fn graph_tensor(
        tensor_id: u32,
        first_block_index: u64,
        block_count: u32,
        representation_ref: Option<u64>,
    ) -> TensorDescriptor {
        TensorDescriptor {
            tensor_id,
            rank: 1,
            dtype: DType::U8,
            codec: 1,
            block_count,
            first_block_index,
            logical_nbytes: 1,
            dimensions: [1, 0, 0, 0, 0, 0, 0, 0],
            name_ref: 0,
            representation_ref,
        }
    }

    fn graph_block(tensor_id: u32, block_id: u32, auxiliary_ref: Option<u64>) -> BlockDescriptor {
        BlockDescriptor {
            tensor_id,
            block_id,
            codec: 1,
            layout: 0,
            logical_offset: 0,
            logical_length: 1,
            payload_offset: 256 + u64::from(block_id),
            payload_length: 1,
            auxiliary_ref,
            checksum_index: None,
        }
    }

    #[test]
    fn descriptor_reference_walk_accepts_converging_edges() {
        // 10 -> 20 and 30; both 20 and 30 -> 40.
        let tensors = [
            graph_tensor(10, 0, 2, None),
            graph_tensor(20, 2, 1, None),
            graph_tensor(30, 3, 1, None),
            graph_tensor(40, 4, 0, None),
        ];
        let blocks = [
            graph_block(10, 0, Some(20)),
            graph_block(10, 1, Some(30)),
            graph_block(20, 0, Some(40)),
            graph_block(30, 0, Some(40)),
        ];
        let lookup = tensors.iter().enumerate().map(|(i, t)| (t.tensor_id, i)).collect::<HashMap<_, _>>();
        assert!(validate_acyclic_references(&tensors, &blocks, &lookup).is_ok());
    }

    #[test]
    fn descriptor_reference_walk_rejects_representation_and_auxiliary_cycles() {
        let representation_cycle = [
            graph_tensor(10, 0, 0, Some(20)),
            graph_tensor(20, 0, 0, Some(10)),
        ];
        let lookup = representation_cycle.iter().enumerate().map(|(i, t)| (t.tensor_id, i)).collect::<HashMap<_, _>>();
        assert!(matches!(
            validate_acyclic_references(&representation_cycle, &[], &lookup),
            Err(FormatError::InvalidReference("cyclic auxiliary tensor references"))
        ));

        let auxiliary_cycle = [
            graph_tensor(10, 0, 1, None),
            graph_tensor(20, 1, 1, None),
        ];
        let blocks = [graph_block(10, 0, Some(20)), graph_block(20, 0, Some(10))];
        let lookup = auxiliary_cycle.iter().enumerate().map(|(i, t)| (t.tensor_id, i)).collect::<HashMap<_, _>>();
        assert!(matches!(
            validate_acyclic_references(&auxiliary_cycle, &blocks, &lookup),
            Err(FormatError::InvalidReference("cyclic auxiliary tensor references"))
        ));
    }

    #[test]
    fn descriptor_reference_walk_rejects_missing_target() {
        let tensors = [graph_tensor(10, 0, 0, Some(99))];
        let lookup = HashMap::from([(10, 0)]);
        assert!(matches!(
            validate_acyclic_references(&tensors, &[], &lookup),
            Err(FormatError::InvalidReference("tensor representation reference is missing"))
        ));
    }

    #[test]
    fn crc32c_uses_castagnoli_polynomial() {
        let mut crc = Crc32c::new();
        crc.update(b"123456789");
        assert_eq!(crc.finish(), 0xe306_9283);
    }
    #[test]
    fn integrity_coverage_allows_optional_records() {
        use crate::BlockDescriptor;
        let block = BlockDescriptor {
            tensor_id: 7,
            block_id: 0,
            codec: 1,
            layout: 0,
            logical_offset: 0,
            logical_length: 4,
            payload_offset: 256,
            payload_length: 4,
            auxiliary_ref: None,
            checksum_index: None,
        };
        assert!(super::validate_integrity_coverage(&[block], &[]).is_ok());
    }

    #[test]
    fn integrity_coverage_allows_metadata_only_record() {
        use crate::{BlockDescriptor, IntegrityRecord, IntegrityTarget};
        let block = BlockDescriptor {
            tensor_id: 7,
            block_id: 0,
            codec: 1,
            layout: 0,
            logical_offset: 0,
            logical_length: 4,
            payload_offset: 256,
            payload_length: 4,
            auxiliary_ref: None,
            checksum_index: None,
        };
        let records = [IntegrityRecord {
            target: IntegrityTarget::Metadata,
            crc32c: 0,
        }];
        assert!(super::validate_integrity_coverage(&[block], &records).is_ok());
    }

    #[test]
    fn integrity_coverage_rejects_missing_block_checksum_index() {
        use crate::{BlockDescriptor, IntegrityRecord, IntegrityTarget};
        let block = BlockDescriptor {
            tensor_id: 7,
            block_id: 0,
            codec: 1,
            layout: 0,
            logical_offset: 0,
            logical_length: 4,
            payload_offset: 256,
            payload_length: 4,
            auxiliary_ref: None,
            checksum_index: None,
        };
        let records = [
            IntegrityRecord {
                target: IntegrityTarget::Metadata,
                crc32c: 0,
            },
            IntegrityRecord {
                target: IntegrityTarget::BlockPayload(0),
                crc32c: 0,
            },
        ];
        assert!(matches!(
            super::validate_integrity_coverage(&[block], &records),
            Err(crate::FormatError::InvalidReference(_))
        ));
    }
}
