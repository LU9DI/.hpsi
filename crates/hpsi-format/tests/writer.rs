use hpsi_format::{
    DType, FileSource, MemorySource, ModelReader, ModelWriter, PayloadBlock, ReaderLimits,
    TensorInput,
};
use std::io::Cursor;
use std::sync::Arc;

#[test]
fn writer_round_trips_and_verifies_all_integrity_records() {
    let tensors = vec![TensorInput {
        tensor_id: 4,
        name: "layer.weight".into(),
        dtype: DType::F32,
        dimensions: vec![2],
        blocks: vec![
            PayloadBlock {
                bytes: vec![0, 0, 128, 63],
            },
            PayloadBlock {
                bytes: vec![0, 0, 0, 64],
            },
        ],
    }];
    let mut output = Cursor::new(Vec::new());
    let stats = ModelWriter::write(&mut output, &tensors).unwrap();
    let bytes = output.into_inner();
    assert_eq!(stats.tensor_count, 1);
    assert_eq!(stats.block_count, 2);
    assert_eq!(stats.total_bytes, bytes.len() as u64);
    let reader =
        ModelReader::open(Arc::new(MemorySource::new(bytes)), &ReaderLimits::default()).unwrap();
    assert_eq!(reader.metadata.tensor_names, ["layer.weight"]);
    assert_eq!(reader.read_block(4, 1).unwrap().payload, [0, 0, 0, 64]);
    assert_eq!(reader.verify_all().unwrap(), 3);
}

#[test]
fn writer_rejects_shape_mismatch() {
    let tensors = vec![TensorInput {
        tensor_id: 1,
        name: "bad".into(),
        dtype: DType::F32,
        dimensions: vec![2],
        blocks: vec![PayloadBlock {
            bytes: vec![1, 2, 3, 4],
        }],
    }];
    assert!(ModelWriter::write(&mut Cursor::new(Vec::new()), &tensors).is_err());
}

#[test]
fn writer_rejects_non_increasing_tensor_ids() {
    let make = |id| TensorInput {
        tensor_id: id,
        name: format!("tensor-{id}"),
        dtype: DType::U8,
        dimensions: vec![1],
        blocks: vec![PayloadBlock { bytes: vec![1] }],
    };
    assert!(ModelWriter::write(&mut Cursor::new(Vec::new()), &[make(2), make(1)]).is_err());
}

#[test]
fn writer_refuses_to_overwrite_non_empty_stream() {
    let tensor = TensorInput {
        tensor_id: 1,
        name: "x".into(),
        dtype: DType::U8,
        dimensions: vec![1],
        blocks: vec![PayloadBlock { bytes: vec![7] }],
    };
    let mut output = Cursor::new(vec![0xaa]);
    assert!(ModelWriter::write(&mut output, &[tensor]).is_err());
    assert_eq!(output.into_inner(), vec![0xaa]);
}

#[test]
fn verify_detects_payload_corruption() {
    let tensor = TensorInput {
        tensor_id: 1,
        name: "x".into(),
        dtype: DType::U8,
        dimensions: vec![2],
        blocks: vec![PayloadBlock { bytes: vec![1, 2] }],
    };
    let mut output = Cursor::new(Vec::new());
    ModelWriter::write(&mut output, &[tensor]).unwrap();
    let mut bytes = output.into_inner();
    let payload_offset = u64::from_le_bytes(bytes[72..80].try_into().unwrap()) as usize;
    bytes[payload_offset] ^= 0xff;
    let reader =
        ModelReader::open(Arc::new(MemorySource::new(bytes)), &ReaderLimits::default()).unwrap();
    assert!(matches!(
        reader.verify_all(),
        Err(hpsi_format::FormatError::ChecksumMismatch)
    ));
}

#[test]
fn write_path_verifies_before_publish_and_never_overwrites() {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "hpsi-writer-{}-{}.hpsi",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let tensor = TensorInput {
        tensor_id: 9,
        name: "atomic.weight".into(),
        dtype: DType::U8,
        dimensions: vec![3],
        blocks: vec![PayloadBlock {
            bytes: vec![3, 4, 5],
        }],
    };

    let stats = ModelWriter::write_path(&path, std::slice::from_ref(&tensor)).unwrap();
    assert_eq!(stats.payload_bytes, 3);
    let original = std::fs::read(&path).unwrap();

    assert!(ModelWriter::write_path(&path, &[tensor]).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), original);

    let invalid_tensor = TensorInput {
        tensor_id: 10,
        name: "invalid".into(),
        dtype: DType::F32,
        dimensions: vec![2],
        blocks: vec![PayloadBlock {
            bytes: vec![1, 2, 3, 4],
        }],
    };
    assert!(ModelWriter::write_path(&path, &[invalid_tensor]).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), original);

    let reader = ModelReader::open(
        Arc::new(FileSource::open(&path).unwrap()),
        &ReaderLimits::default(),
    )
    .unwrap();
    assert_eq!(reader.verify_all().unwrap(), 2);
    std::fs::remove_file(path).unwrap();
}


#[test]
fn rejects_overlapping_payload_blocks() {
    let tensors = vec![TensorInput {
        tensor_id: 1,
        name: "overlap".into(),
        dtype: DType::U8,
        dimensions: vec![4],
        blocks: vec![
            PayloadBlock { bytes: vec![1, 2] },
            PayloadBlock { bytes: vec![3, 4] },
        ],
    }];
    let mut output = Cursor::new(Vec::new());
    ModelWriter::write(&mut output, &tensors).unwrap();
    let mut bytes = output.into_inner();
    let block_directory = u64::from_le_bytes(bytes[56..64].try_into().unwrap()) as usize;
    let first_payload_offset = bytes[block_directory + 32..block_directory + 40].to_vec();
    bytes[block_directory + 64 + 32..block_directory + 64 + 40]
        .copy_from_slice(&first_payload_offset);
    let reader = ModelReader::open(
        Arc::new(MemorySource::new(bytes)),
        &ReaderLimits::default(),
    );
    assert!(reader.is_err());
}

#[test]
fn verify_detects_metadata_corruption() {
    let tensors = vec![TensorInput {
        tensor_id: 1,
        name: "checksum".into(),
        dtype: DType::U8,
        dimensions: vec![1],
        blocks: vec![PayloadBlock { bytes: vec![9] }],
    }];
    let mut output = Cursor::new(Vec::new());
    ModelWriter::write(&mut output, &tensors).unwrap();
    let mut bytes = output.into_inner();
    let name_offset = bytes
        .windows(b"checksum".len())
        .position(|window| window == b"checksum")
        .unwrap();
    bytes[name_offset] = b'C';
    let reader = ModelReader::open(
        Arc::new(MemorySource::new(bytes)),
        &ReaderLimits::default(),
    ).unwrap();
    assert!(matches!(
        reader.verify_all(),
        Err(hpsi_format::FormatError::ChecksumMismatch)
    ));
}
