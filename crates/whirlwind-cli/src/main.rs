//! `whirlwind` binary: thin wrapper over the whirlwind-cli library.

fn main() {
    std::process::exit(whirlwind_cli::run(std::env::args_os()));
}
