#![no_main]

use hpsi_format::{Header, ReaderLimits};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let limits = ReaderLimits::default();
    let _ = Header::parse(data, data.len() as u64, &limits);
});
