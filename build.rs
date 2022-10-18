fn main() {
    println!("cargo:rustc-link-search=/lib/intel64");
    println!("cargo:rustc-link-lib=mkl_intel_ilp64");
    println!("cargo:rustc-link-lib=mkl_intel_thread");
    println!("cargo:rustc-link-lib=mkl_core");
    println!("cargo:rustc-link-lib=iomp5");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=dl");
}
