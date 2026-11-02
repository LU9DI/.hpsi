use std::process::ExitCode;
use std::sync::Arc;
use hpsi_format::{FileSource, ModelReader, ReaderLimits};

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => { eprintln!("hpsi: {message}"); ExitCode::FAILURE }
    }
}
fn run(args: Vec<String>) -> Result<(), String> {
    let Some(command)=args.first().map(String::as_str) else { return Err(usage()); };
    let Some(path)=args.get(1) else { return Err(usage()); };
    if args.len()!=2 { return Err(usage()); }
    let source=Arc::new(FileSource::open(path).map_err(|e|e.to_string())?);
    let reader=ModelReader::open(source,&ReaderLimits::default()).map_err(|e|e.to_string())?;
    match command {
        "inspect" => {
            println!("HPSI {}.{}",reader.header.format_major,reader.header.format_minor);
            println!("file_size: {} bytes",reader.header.file_size);
            println!("tensors: {}",reader.tensor_count());
            println!("blocks: {}",reader.block_count());
            println!("metadata: {} bytes",reader.header.metadata.length);
            println!("payload: {} bytes",reader.header.payload.length);
            for (i,name) in reader.metadata.tensor_names.iter().enumerate(){println!("tensor_name[{i}]: {name}");}
            println!("open_payload_bytes_read: {}",reader.stats().payload_bytes_read);
            Ok(())
        }
        "verify" => {
            let verified=reader.verify_all().map_err(|e|e.to_string())?;
            let stats=reader.stats();
            println!("OK: verified {verified} CRC32C records");
            println!("payload_bytes_read: {}",stats.payload_bytes_read);
            println!("read_operations: {}",stats.read_operations);
            Ok(())
        }
        _=>Err(usage())
    }
}
fn usage()->String{"usage: hpsi <inspect|verify> <model.hpsi>".to_owned()}
