use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    println!(
        "cargo:rustc-env=BENCH_RUNNER_RUSTFLAGS={}",
        build_rustflags()
    );

    for git_path in ["HEAD", "refs/heads"] {
        if let Some(path) = git_output(["rev-parse", "--git-path", git_path]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }

    let commit = git_output(["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=BENCH_RUNNER_GIT_COMMIT={commit}");
}

fn build_rustflags() -> String {
    std::env::var("CARGO_ENCODED_RUSTFLAGS")
        .ok()
        .filter(|flags| !flags.is_empty())
        .map(|flags| flags.split('\x1f').collect::<Vec<_>>().join(" "))
        .or_else(|| std::env::var("RUSTFLAGS").ok())
        .unwrap_or_default()
}

fn git_output<const N: usize>(args: [&str; N]) -> Option<String> {
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|output| output.trim().to_owned())
}
