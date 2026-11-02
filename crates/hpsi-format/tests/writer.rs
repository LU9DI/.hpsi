use hpsi_format::{
    DType, MemorySource, ModelReader, ModelWriter, PayloadBlock, ReaderLimits, TensorInput,
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
