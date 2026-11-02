#![no_main]

use std::sync::Arc;

use hpsi_format::{MemorySource, ModelReader, ReaderLimits};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let source = Arc::new(MemorySource::new(data.to_vec()));
    if let Ok(reader) = ModelReader::open(source, &ReaderLimits::default()) {
        let _ = reader.verify_all();
    }
});
