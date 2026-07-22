use std::fs::File;
use std::io::{self, Write};
use std::process::ExitCode;

use bench_runner::{run_suite, SuiteConfig};

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let output = args.next();
    if args.next().is_some() {
        eprintln!("usage: bench-runner [output.jsonl]");
        return ExitCode::from(2);
    }

    let config = SuiteConfig::default();
    let line_count = match output {
        Some(path) => match File::create(&path) {
            Ok(mut file) => run_suite(&mut file, &config),
            Err(error) => {
                eprintln!("failed to create {}: {error}", path.to_string_lossy());
                return ExitCode::FAILURE;
            }
        },
        None => run_suite(&mut io::stdout().lock(), &config),
    };

    if line_count == 0 {
        let _ = writeln!(io::stderr().lock(), "benchmark report could not be written");
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
