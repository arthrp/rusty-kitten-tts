//! Link against `libespeak-ng` and help the linker find it (e.g. Homebrew on macOS).

fn main() {
    if cfg!(target_os = "macos") {
        if let Ok(prefix) = std::env::var("HOMEBREW_PREFIX") {
            println!("cargo:rustc-link-search=native={prefix}/lib");
        } else {
            for dir in ["/opt/homebrew/lib", "/usr/local/lib"] {
                if std::path::Path::new(dir).join("libespeak-ng.dylib").exists()
                    || std::path::Path::new(dir).join("libespeak-ng.a").exists()
                {
                    println!("cargo:rustc-link-search=native={dir}");
                    break;
                }
            }
        }
        println!("cargo:rerun-if-env-changed=HOMEBREW_PREFIX");
    } else if cfg!(target_os = "linux") {
        // Typical distro paths; `pkg-config --libs espeak-ng` also works if set up
        for dir in ["/usr/lib", "/usr/local/lib"] {
            if std::path::Path::new(dir).exists() {
                println!("cargo:rustc-link-search=native={dir}");
            }
        }
    }
    // `#[link(name = "espeak-ng")]` on the `extern` block in `espeak_native.rs` supplies `-lespeak-ng`.
}
