use hpsi_format::{FileSource, ModelReader, ReaderLimits};
use std::process::ExitCode;
use std::sync::Arc;

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("hpsi: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    let Some(command) = args.first().map(String::as_str) else {
        return Err(usage());
    };
    let Some(path) = args.get(1) else {
        return Err(usage());
    };
    if args.len() != 2 {
        return Err(usage());
    }
    let source = Arc::new(FileSource::open(path).map_err(|e| e.to_string())?);
    let reader = ModelReader::open(source, &ReaderLimits::default()).map_err(|e| e.to_string())?;

    match command {
        "inspect" => {
            println!(
                "HPSI {}.{}",
                reader.header.format_major, reader.header.format_minor
            );
            println!("file_size: {} bytes", reader.header.file_size);
            println!("tensors: {}", reader.tensor_count());
            println!("blocks: {}", reader.block_count());
            println!("metadata: {} bytes", reader.header.metadata.length);
            println!("payload_region: {} bytes", reader.header.payload.length);
            for tensor in reader.tensors() {
                let name = reader
                    .metadata
                    .tensor_names
                    .get(tensor.name_ref as usize)
                    .map(String::as_str)
                    .unwrap_or("<invalid-name-ref>");
                let shape = tensor.dimensions[..usize::from(tensor.rank)]
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join("x");
                let stored = (0..tensor.block_count).try_fold(0u64, |sum, block_id| {
                    let block = reader
                        .block(tensor.tensor_id, block_id)
                        .map_err(|e| e.to_string())?;
                    sum.checked_add(block.payload_length)
                        .ok_or_else(|| "stored byte count overflow".to_owned())
                })?;
                println!(
                    "tensor id={} name={} dtype={:?} shape={} codec={} blocks={} logical_bytes={} stored_bytes={}",
                    tensor.tensor_id,
                    name,
                    tensor.dtype,
                    shape,
                    tensor.codec,
                    tensor.block_count,
                    tensor.logical_nbytes,
                    stored
                );
            }
            println!(
                "payload_bytes_read_during_inspect: {}",
                reader.stats().payload_bytes_read
            );
            Ok(())
        }
        "verify" => {
            let verified = reader.verify_all().map_err(|e| e.to_string())?;
            let stats = reader.stats();
            println!("OK: verified {verified} CRC32C records");
            println!("payload_bytes_read: {}", stats.payload_bytes_read);
            println!("read_operations: {}", stats.read_operations);
            Ok(())
        }
        _ => Err(usage()),
    }
}

fn usage() -> String {
    "usage: hpsi <inspect|verify> <model.hpsi>".to_owned()
}
