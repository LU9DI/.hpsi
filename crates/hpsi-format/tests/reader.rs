use std::sync::Arc;

use hpsi_format::{MemorySource, ModelReader, ReaderLimits};

fn fixture() -> Vec<u8> {
    let metadata = br#"{"format":"hpsi","metadata_version":1,"tensor_names":["weights"]}"#;
    let metadata_offset = 128_usize;
    let tensor_offset = 200_usize;
    let block_offset = 328_usize;
    let payload_offset = 392_usize;
    let payload = [0x11_u8, 0x22, 0x33, 0x44];
    let file_size = payload_offset + payload.len();
    let mut file = vec![0_u8; file_size];

    file[0..4].copy_from_slice(b"HPSI");
    file[4..6].copy_from_slice(&0_u16.to_le_bytes());
    file[6..8].copy_from_slice(&1_u16.to_le_bytes());
    file[8..12].copy_from_slice(&128_u32.to_le_bytes());
    file[16..24].copy_from_slice(&(file_size as u64).to_le_bytes());
    file[24..32].copy_from_slice(&(metadata_offset as u64).to_le_bytes());
    file[32..40].copy_from_slice(&(metadata.len() as u64).to_le_bytes());
    file[40..48].copy_from_slice(&(tensor_offset as u64).to_le_bytes());
    file[48..52].copy_from_slice(&1_u32.to_le_bytes());
    file[52..56].copy_from_slice(&128_u32.to_le_bytes());
    file[56..64].copy_from_slice(&(block_offset as u64).to_le_bytes());
    file[64..68].copy_from_slice(&1_u32.to_le_bytes());
    file[68..72].copy_from_slice(&64_u32.to_le_bytes());
    file[72..80].copy_from_slice(&(payload_offset as u64).to_le_bytes());
    file[80..88].copy_from_slice(&(payload.len() as u64).to_le_bytes());

    file[metadata_offset..metadata_offset + metadata.len()].copy_from_slice(metadata);

    let tensor = &mut file[tensor_offset..tensor_offset + 128];
    tensor[0..4].copy_from_slice(&7_u32.to_le_bytes());
    tensor[4] = 1;
    tensor[5] = 1;
    tensor[6..8].copy_from_slice(&1_u16.to_le_bytes());
    tensor[12..16].copy_from_slice(&1_u32.to_le_bytes());
    tensor[16..24].copy_from_slice(&0_u64.to_le_bytes());
    tensor[24..32].copy_from_slice(&4_u64.to_le_bytes());
    tensor[32..40].copy_from_slice(&4_u64.to_le_bytes());
    tensor[96..100].copy_from_slice(&0_u32.to_le_bytes());
    tensor[104..112].copy_from_slice(&u64::MAX.to_le_bytes());

    let block = &mut file[block_offset..block_offset + 64];
    block[0..4].copy_from_slice(&7_u32.to_le_bytes());
    block[4..8].copy_from_slice(&0_u32.to_le_bytes());
    block[12..14].copy_from_slice(&0_u16.to_le_bytes());
    block[14..16].copy_from_slice(&0_u16.to_le_bytes());
    block[16..24].copy_from_slice(&0_u64.to_le_bytes());
    block[24..32].copy_from_slice(&4_u64.to_le_bytes());
    block[32..40].copy_from_slice(&(payload_offset as u64).to_le_bytes());
    block[40..48].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    block[48..56].copy_from_slice(&u64::MAX.to_le_bytes());
    block[56..60].copy_from_slice(&u32::MAX.to_le_bytes());

    file[payload_offset..].copy_from_slice(&payload);
    file
}

#[test]
fn open_reads_directories_but_not_payload() {
    let source = Arc::new(MemorySource::new(fixture()));
    let reader = ModelReader::open(source, &ReaderLimits::default()).unwrap();

    let stats = reader.stats();
    assert_eq!(stats.payload_bytes_read, 0);
    assert_eq!(reader.tensor_count(), 1);
    assert_eq!(reader.block_count(), 1);
    assert_eq!(reader.tensor(7).unwrap().logical_nbytes, 4);
}

#[test]
fn read_block_fetches_only_selected_payload() {
    let source = Arc::new(MemorySource::new(fixture()));
    let reader = ModelReader::open(source, &ReaderLimits::default()).unwrap();

    let block = reader.read_block(7, 0).unwrap();
    assert_eq!(block.payload, [0x11, 0x22, 0x33, 0x44]);
    assert_eq!(reader.stats().payload_bytes_read, 4);
}

#[test]
fn rejects_unknown_tensor_id() {
    let source = Arc::new(MemorySource::new(fixture()));
    let reader = ModelReader::open(source, &ReaderLimits::default()).unwrap();

    assert!(reader.tensor(999).is_err());
}

#[test]
fn rejects_invalid_tensor_metadata_reference() {
    let mut bytes = fixture();
    bytes[200 + 96..200 + 100].copy_from_slice(&9_u32.to_le_bytes());
    let source = Arc::new(MemorySource::new(bytes));

    assert!(ModelReader::open(source, &ReaderLimits::default()).is_err());
}

