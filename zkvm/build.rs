use risc0_build::embed_methods;

fn main() {
    // Generate Rust code to include the guest methods and image IDs
    embed_methods();
}