use std::io::Read;
use alloy_primitives::U256;
use alloy_sol_types::SolValue;
use risc0_zkvm::guest::env;

mod weights {
    include!("weights/W1.incl.rs");
    include!("weights/B1.incl.rs");
    include!("weights/W2.incl.rs");
    include!("weights/B2.incl.rs");
}

// Updated scale to match training
const SCALE: i32 = 16384;  // Must match the SCALE in Python

fn relu(x: i64) -> i64 {
    x.max(0)
}

fn main() {
    // Read the input data for this application.
    let mut input_bytes = Vec::<u8>::new();
    env::stdin().read_to_end(&mut input_bytes).unwrap();

    // Decode the input - expecting array of 784 U256 values
    let input_data = <Vec<U256>>::abi_decode(&input_bytes).unwrap();

    // Convert U256 to i32 array for neural network processing
    let mut input: [i32; 784] = [0; 784];
    for (i, &val) in input_data.iter().enumerate() {
        if i < 784 {
            input[i] = val.to::<i32>();
        }
    }

    // Layer 1: 784 -> 128 with ReLU activation (UPDATED from 64 to 128)
    // Using i64 for intermediate calculations to prevent overflow
    let mut h1 = [0i64; 128]; // CHANGED: Array size from 64 to 128
    for i in 0..128 { // CHANGED: Loop from 64 to 128
        let mut sum = weights::B1[i] as i64; // Don't multiply by SCALE again
        for j in 0..784 {
            sum += (weights::W1[j][i] as i64) * (input[j] as i64);
        }
        h1[i] = relu(sum); // Already scaled by SCALE once
    }

    // Layer 2: 128 -> 10 (output layer) (UPDATED from 64 to 128)
    let mut out = [0i64; 10];
    for i in 0..10 {
        let mut sum = weights::B2[i] as i64;
        for j in 0..128 { // CHANGED: Loop from 64 to 128
            sum += (weights::W2[j][i] as i64) * h1[j] / (SCALE as i64);
        }
        out[i] = sum;
    }

    // Find predicted digit (argmax)
    let (mut predicted_digit, mut max_val) = (0, out[0]);
    for i in 1..10 {
        if out[i] > max_val {
            max_val = out[i];
            predicted_digit = i;
        }
    }

    // Debug output (optional - remove in production)
    eprintln!("Output scores (scaled):");
    for i in 0..10 {
        eprintln!("  Digit {}: {}", i, out[i]);
    }
    eprintln!("Predicted digit: {}", predicted_digit);

    // Commit the prediction as U256 for Solidity compatibility
    let prediction = U256::from(predicted_digit);
    env::commit_slice(prediction.abi_encode().as_slice());
}