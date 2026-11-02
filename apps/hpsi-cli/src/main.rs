use hpsi_format::{DType, FileSource, ModelReader, ReaderLimits};
use serde_json::json;
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
    if command == "safetensors-inspect" && args.len() == 2 {
        return inspect_safetensors(path);
    }
    let tensor_id = match command {
        "inspect" | "verify" if args.len() == 2 => None,
        "calibrate" if args.len() == 3 => Some(
            args[2]
                .parse::<u32>()
                .map_err(|_| "tensor-id must be an unsigned 32-bit integer".to_owned())?,
        ),
        _ => return Err(usage()),
    };

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
        "calibrate" => calibrate_activations(&reader, tensor_id.expect("validated command")),
        _ => Err(usage()),
    }
}

/// Compute diagonal activation second moments from a row-major F32 activation tensor.
/// Payload blocks are consumed one at a time; no complete activation matrix is assembled.
fn calibrate_activations(reader: &ModelReader, tensor_id: u32) -> Result<(), String> {
    let tensor = reader.tensor(tensor_id).map_err(|e| e.to_string())?;
    if tensor.dtype != DType::F32 || tensor.rank != 2 || tensor.codec != 1 {
        return Err("calibration requires a rank-2 F32 tensor using the RAW codec".to_owned());
    }
    let rows = tensor.dimensions[0];
    let cols = tensor.dimensions[1];
    let cols_usize = usize::try_from(cols).map_err(|_| "column count exceeds address space")?;
    let expected_values = rows
        .checked_mul(cols)
        .ok_or_else(|| "activation element count overflow".to_owned())?;
    let mut sum_squares = Vec::new();
    sum_squares
        .try_reserve_exact(cols_usize)
        .map_err(|_| "cannot allocate calibration accumulator".to_owned())?;
    sum_squares.resize(cols_usize, 0.0_f64);

    let mut expected_offset = 0_u64;
    let mut value_count = 0_u64;
    let mut pending = [0_u8; 4];
    let mut pending_len = 0_usize;

    for block_id in 0..tensor.block_count {
        let descriptor = reader
            .block(tensor_id, block_id)
            .map_err(|e| e.to_string())?;
        if descriptor.logical_offset != expected_offset
            || descriptor.logical_length != descriptor.payload_length
        {
            return Err("activation blocks must be contiguous RAW payloads".to_owned());
        }
        let block = reader
            .read_block(tensor_id, block_id)
            .map_err(|e| e.to_string())?;
        for byte in block.payload {
            pending[pending_len] = byte;
            pending_len += 1;
            if pending_len == 4 {
                let value = f32::from_le_bytes(pending);
                if !value.is_finite() {
                    return Err("activation tensor contains a non-finite F32 value".to_owned());
                }
                let column = usize::try_from(value_count % cols)
                    .map_err(|_| "column index exceeds address space")?;
                let value = f64::from(value);
                sum_squares[column] += value * value;
                value_count += 1;
                pending_len = 0;
            }
        }
        expected_offset = expected_offset
            .checked_add(descriptor.logical_length)
            .ok_or_else(|| "activation byte offset overflow".to_owned())?;
    }

    if pending_len != 0
        || value_count != expected_values
        || expected_offset != tensor.logical_nbytes
    {
        return Err("activation payload length does not match tensor shape".to_owned());
    }
    let sample_count = rows as f64;
    let second_moment: Vec<f64> = sum_squares
        .into_iter()
        .map(|sum| sum / sample_count)
        .collect();
    let name = reader
        .metadata
        .tensor_names
        .get(tensor.name_ref as usize)
        .ok_or_else(|| "activation tensor has an invalid name reference".to_owned())?;
    let stats = reader.stats();
    let calibration = json!({
        "format": "hpsi-activation-calibration",
        "version": 1,
        "source_tensor_id": tensor_id,
        "source_tensor_name": name,
        "shape": [rows, cols],
        "sample_count": rows,
        "statistic": "diagonal_second_moment",
        "activation_second_moment": second_moment,
        "source_payload_bytes_read": stats.payload_bytes_read,
        "source_read_operations": stats.read_operations
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&calibration).map_err(|e| e.to_string())?
    );
    Ok(())
}

fn inspect_safetensors(path: &str) -> Result<(), String> {
    let source = Arc::new(FileSource::open(path).map_err(|e| e.to_string())?);
    let file =
        hpsi_ops::SafeTensorFile::open(source, 100 * 1024 * 1024).map_err(|e| e.to_string())?;
    println!("safetensors: {}", path);
    println!("tensors: {}", file.tensors().len());
    for tensor in file.tensors() {
        let shape = tensor
            .shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join("x");
        println!(
            "name={} dtype={} shape={} data_bytes={}",
            tensor.name,
            tensor.dtype,
            shape,
            tensor.data_end - tensor.data_start
        );
    }
    let candidates = hpsi_ops::discover_projection_candidates(file.tensors());
    println!("projection_candidates: {}", candidates.len());
    for candidate in candidates {
        println!(
            "candidate name={} dtype={} shape={}x{} conventional_name={}",
            candidate.name,
            candidate.dtype,
            candidate.rows,
            candidate.cols,
            candidate.conventional_projection_name
        );
    }
    println!(
        "note: candidates are not automatically imported; calibration must match the exact tensor name."
    );
    Ok(())
}

fn usage() -> String {
    "usage: hpsi <inspect|verify> <model.hpsi> | hpsi calibrate <model.hpsi> <tensor-id> | hpsi safetensors-inspect <model.safetensors>".to_owned()
}
