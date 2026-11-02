/// Resource limits applied before allocating or reading untrusted regions.
#[derive(Debug, Clone)]
pub struct ReaderLimits {
    pub max_file_size: u64,
    pub max_metadata_bytes: u64,
    pub max_tensors: u32,
    pub max_blocks: u32,
    pub max_total_directory_bytes: u64,
    pub max_tensor_logical_bytes: u64,
    pub max_payload_bytes_per_block: u64,
    pub max_rank: u8,
    pub max_json_depth: usize,
    pub max_json_string_bytes: usize,
    pub max_json_container_items: usize,
    pub max_tensor_name_bytes: usize,
}

impl Default for ReaderLimits {
    fn default() -> Self {
        Self {
            max_file_size: 1_u64 << 40,
            max_metadata_bytes: 16 * 1024 * 1024,
            max_tensors: 1_000_000,
            max_blocks: 16_000_000,
            max_total_directory_bytes: 2 * 1024 * 1024 * 1024,
            max_tensor_logical_bytes: 1_u64 << 40,
            max_payload_bytes_per_block: 64 * 1024 * 1024,
            max_rank: 8,
            max_json_depth: 32,
            max_json_string_bytes: 1024 * 1024,
            max_json_container_items: 1_000_000,
            max_tensor_name_bytes: 1024,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::ReaderLimits;

    #[test]
    fn defaults_match_v01_reader_resource_limits() {
        let limits = ReaderLimits::default();
        assert_eq!(limits.max_file_size, 1_u64 << 40);
        assert_eq!(limits.max_metadata_bytes, 16 * 1024 * 1024);
        assert_eq!(limits.max_tensors, 1_000_000);
        assert_eq!(limits.max_blocks, 16_000_000);
        assert_eq!(limits.max_total_directory_bytes, 2 * 1024 * 1024 * 1024);
        assert_eq!(limits.max_tensor_logical_bytes, 1_u64 << 40);
        assert_eq!(limits.max_payload_bytes_per_block, 64 * 1024 * 1024);
        assert_eq!(limits.max_rank, 8);
        assert_eq!(limits.max_json_depth, 32);
        assert_eq!(limits.max_json_string_bytes, 1024 * 1024);
        assert_eq!(limits.max_tensor_name_bytes, 1024);
    }
}
