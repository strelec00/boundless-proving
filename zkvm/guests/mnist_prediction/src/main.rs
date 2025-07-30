// guest/src/main.rs
use risc0_zkvm::guest::env;

mod weights {
    include!("weights/W1.incl.rs");
    include!("weights/B1.incl.rs");
    include!("weights/W2.incl.rs");
    include!("weights/B2.incl.rs");
}

fn relu(x: i32) -> i32 {
    x.max(0)
}

fn main() {
    // Read input data as Vec<u32> directly (not U256)
    let input_data: Vec<u32> = env::read();

    // Debug print to see what we're getting
    eprintln!("Guest received {} inputs", input_data.len());

    // Validate input length
    if input_data.len() != 784 {
        panic!("Invalid input length: expected 784, got {}", input_data.len());
    }

    // Convert to i32 array for neural network processing
    let mut input: [i32; 784] = [0; 784];
    for (i, &val) in input_data.iter().enumerate() {
        if val > 1 {
            panic!("Invalid pixel value at index {}: {} (expected 0 or 1)", i, val);
        }
        input[i] = val as i32;
    }

    // Layer 1: 784 -> 64 with ReLU activation
    let mut h1 = [0i32; 64];
    for i in 0..64 {
        let mut sum = weights::B1[i];
        for j in 0..784 {
            sum += weights::W1[j][i] * input[j];
        }
        h1[i] = relu(sum);
    }

    // Layer 2: 64 -> 10 (output layer)
    let mut out = [0i32; 10];
    for i in 0..10 {
        let mut sum = weights::B2[i];
        for j in 0..64 {
            sum += weights::W2[j][i] * h1[j];
        }
        out[i] = sum;
    }

    // Find predicted digit (argmax)
    let mut predicted_digit = 0u32;
    let mut max_val = out[0];
    for i in 1..10 {
        if out[i] > max_val {
            max_val = out[i];
            predicted_digit = i as u32;
        }
    }

    // Commit the prediction
    env::commit(&predicted_digit);
}