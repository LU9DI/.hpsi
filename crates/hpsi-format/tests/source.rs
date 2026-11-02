use hpsi_format::{MemorySource, RandomAccessSource};

#[test]
fn memory_source_reads_exact_requested_range() {
    let source = MemorySource::new(vec![10_u8, 20, 30, 40, 50]);
    let mut output = [0_u8; 3];

    source.read_exact_at(1, &mut output).unwrap();

    assert_eq!(output, [20, 30, 40]);
}

#[test]
fn memory_source_rejects_out_of_bounds_range() {
    let source = MemorySource::new(vec![1_u8, 2, 3]);
    let mut output = [0_u8; 2];

    assert!(source.read_exact_at(2, &mut output).is_err());
}

#[test]
fn memory_source_reports_length() {
    let source = MemorySource::new(vec![1_u8, 2, 3, 4]);
    assert_eq!(source.len().unwrap(), 4);
}
