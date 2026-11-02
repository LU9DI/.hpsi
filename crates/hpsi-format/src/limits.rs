/// Resource limits applied before allocating or reading untrusted regions.
#[derive(Debug, Clone)]
pub struct ReaderLimits {
    pub max_file_size: u64,
    pub max_metadata_bytes: u64,
    pub max_tensors: u32,
    pub max_blocks: u32,
    pub max_total_directory_bytes: u64,
}

impl Default for ReaderLimits {
    fn default() -> Self {
        Self {
            max_file_size: 1_u64 << 40,
            max_metadata_bytes: 16 * 1024 * 1024,
            max_tensors: 1_000_000,
            max_blocks: 16_000_000,
            max_total_directory_bytes: 2 * 1024 * 1024 * 1024,
        }
    }
}
