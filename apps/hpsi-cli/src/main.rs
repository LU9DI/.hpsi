use hpsi_format::{DType, FileSource, ModelReader, ReaderLimits};
use hpsi_ops::{
    ActivationCalibration, ModelImportPlan, UnsupportedTensorPolicy, VqConfig,
    write_model_import_to_hpsi_path, write_model_import_to_hpsi_path_calibrated,
};
use serde_json::json;
use std::fs;
use std::io::Read;
use std::path::Path;

const MAX_CALIBRATION_FILE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CALIBRATION_TOTAL_BYTES: u64 = 128 * 1024 * 1024;
const MAX_CALIBRATION_FILES: usize = 1024;
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
    if command == "safetensors-import" {
        return import_safetensors_command(&args);
    }
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

fn import_safetensors_command(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err(usage());
    }
    let source_path = &args[1];
    let destination = &args[2];
    let mut calibration_dir: Option<&Path> = None;
    let mut reject_raw = false;
    let mut index = 3;
    while index < args.len() {
        match args[index].as_str() {
            "--calibration-dir" if index + 1 < args.len() && calibration_dir.is_none() => {
                calibration_dir = Some(Path::new(&args[index + 1]));
                index += 2;
            }
            "--reject-raw" if !reject_raw => {
                reject_raw = true;
                index += 1;
            }
            _ => return Err(usage()),
        }
    }

    let source = Arc::new(FileSource::open(source_path).map_err(|e| e.to_string())?);
    let safetensors =
        hpsi_ops::SafeTensorFile::open(source, 100 * 1024 * 1024).map_err(|e| e.to_string())?;
    let policy = if reject_raw {
        UnsupportedTensorPolicy::Reject
    } else {
        UnsupportedTensorPolicy::AllowRawPreservation
    };
    let mut calibrations = Vec::new();
    if let Some(directory) = calibration_dir {
        let mut files = Vec::new();
        for entry in fs::read_dir(directory).map_err(|e| {
            format!(
                "cannot read calibration directory {}: {e}",
                directory.display()
            )
        })? {
            let entry = entry.map_err(|e| e.to_string())?;
            let file_type = entry.file_type().map_err(|e| e.to_string())?;
            if file_type.is_file()
                && entry.path().extension().and_then(|value| value.to_str()) == Some("json")
                && entry.file_name() != "manifest.json"
            {
                files.push(entry.path());
            }
        }
        files.sort_unstable();
        if files.len() > MAX_CALIBRATION_FILES {
            return Err(format!(
                "calibration directory contains {} JSON files; limit is {}",
                files.len(),
                MAX_CALIBRATION_FILES
            ));
        }
        let mut total_bytes = 0_u64;
        for path in files {
            // Read through a hard byte cap rather than trusting metadata alone:
            // the file may change between metadata inspection and the read.
            let file = fs::File::open(&path).map_err(|e| e.to_string())?;
            let mut contents = String::new();
            file.take(MAX_CALIBRATION_FILE_BYTES + 1)
                .read_to_string(&mut contents)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            let file_bytes = u64::try_from(contents.len())
                .map_err(|_| "calibration file length exceeds u64".to_owned())?;
            if file_bytes > MAX_CALIBRATION_FILE_BYTES {
                return Err(format!(
                    "calibration file exceeds the 64 MiB limit: {}",
                    path.display()
                ));
            }
            total_bytes = total_bytes
                .checked_add(file_bytes)
                .ok_or_else(|| "total calibration byte count overflow".to_owned())?;
            if total_bytes > MAX_CALIBRATION_TOTAL_BYTES {
                return Err("calibration directory exceeds the 128 MiB aggregate limit".to_owned());
            }
            let calibration = ActivationCalibration::from_json(&contents)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            calibrations.push(calibration);
        }
    }

    let stats = if calibration_dir.is_some() {
        let plan = ModelImportPlan::build_with_provenance(&safetensors, &calibrations, policy)
            .map_err(|e| e.to_string())?;
        write_model_import_to_hpsi_path_calibrated(
            &safetensors,
            &plan,
            destination,
            VqConfig::default(),
            128,
        )
        .map_err(|e| e.to_string())?
    } else {
        let plan = ModelImportPlan::build(&safetensors, &[], policy).map_err(|e| e.to_string())?;
        write_model_import_to_hpsi_path(&safetensors, &plan, destination, VqConfig::default(), 128)
            .map_err(|e| e.to_string())?
    };
    println!("OK: imported {} HPSI tensors", stats.tensor_count);
    println!("calibration_files: {}", calibrations.len());
    println!("output: {}", destination);
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
    "usage: hpsi <inspect|verify> <model.hpsi> | hpsi calibrate <model.hpsi> <tensor-id> | hpsi safetensors-inspect <model.safetensors> | hpsi safetensors-import <model.safetensors> <output.hpsi> [--calibration-dir <dir>] [--reject-raw]".to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hpsi_format::{ModelReader, ReaderLimits};
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(std::path::PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let sequence = TEST_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "hpsi-cli-test-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&path).expect("create test directory");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write_raw_u8_safetensors(path: &Path) {
        let header = br#"{"embedding":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[3, 5, 7, 11]);
        fs::write(path, bytes).expect("write safetensors fixture");
    }

    #[test]
    fn cli_imports_raw_tensor_and_verifies_hpsi_output() {
        let directory = TestDirectory::new();
        let source = directory.path().join("model.safetensors");
        let destination = directory.path().join("model.hpsi");
        write_raw_u8_safetensors(&source);

        import_safetensors_command(&[
            "safetensors-import".to_owned(),
            source.display().to_string(),
            destination.display().to_string(),
        ])
        .expect("CLI import should succeed");

        let reader = ModelReader::open(
            Arc::new(FileSource::open(&destination).expect("open imported HPSI")),
            &ReaderLimits::default(),
        )
        .expect("parse imported HPSI");
        assert_eq!(reader.tensor_count(), 1);
        reader.verify_all().expect("verify imported HPSI");
    }

    #[test]
    fn cli_reject_raw_rejects_unsupported_tensor_without_output() {
        let directory = TestDirectory::new();
        let source = directory.path().join("model.safetensors");
        let destination = directory.path().join("model.hpsi");
        write_raw_u8_safetensors(&source);

        let result = import_safetensors_command(&[
            "safetensors-import".to_owned(),
            source.display().to_string(),
            destination.display().to_string(),
            "--reject-raw".to_owned(),
        ]);
        assert!(result.is_err());
        assert!(!destination.exists());
    }

    #[test]
    fn cli_does_not_overwrite_existing_destination() {
        let directory = TestDirectory::new();
        let source = directory.path().join("model.safetensors");
        let destination = directory.path().join("model.hpsi");
        write_raw_u8_safetensors(&source);
        fs::write(&destination, b"preserve-existing-file").expect("create destination");

        let result = import_safetensors_command(&[
            "safetensors-import".to_owned(),
            source.display().to_string(),
            destination.display().to_string(),
        ]);
        assert!(result.is_err());
        assert_eq!(
            fs::read(&destination).expect("read existing destination"),
            b"preserve-existing-file"
        );
    }
}
